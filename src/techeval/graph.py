"""LangGraph StateGraph (E). CONTRACTS §8 흐름을 그대로 구현한다.

- 에이전트 노드는 `Send` payload를 `AgentInput`으로 검증해 순수 함수 `run_xxx(inp, deps)`를 호출하고
  부분 State를 반환한다.
- 제어 노드는 `techeval.control.*`의 순수 함수를 감싼다. 의견을 만들지 않는다.
- 모든 루프는 `retry_counts`로 상한을 건다 (기술 재검색 2, 관점별 2, 반대 근거 1, 보고서 재생성 1).
- `operator.add` 키의 중복은 종합·보고서 입력 조립 시 `latest_by_criterion`/`latest_by_tech`로 제거한다.

State.missing_criteria 는 관점별 합집합(`"trl"`)에 더해 기술별 항목(`"trl:mla"`)도 같은 dict에 기록한다
(재실행 대상을 Send payload로 나눌 때 필요). 키 형식은 `retry_counts`와 같다.
"""

import functools
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from techeval.agents._deps import AgentInput, Deps, ReportInput, SynthesisInput, TechResearchOutput
from techeval.control._common import make_not_public_result
from techeval.control.counter_evidence import search_counter_evidence
from techeval.control.evidence_gap import check_evidence_gap
from techeval.control.judge import judge_report
from techeval.control.perspective_check import check_perspectives
from techeval.control.query_rewrite import rewrite_queries
from techeval.control.sources import SourceRegistry
from techeval.control.tech_evidence_check import check_tech_evidence
from techeval.schemas import (
    MAX_COUNTER_EVIDENCE_SEARCH,
    MAX_REPORT_REGENERATION,
    MAX_RETRY_PER_PERSPECTIVE,
    PERSPECTIVE_CRITERIA,
    CriterionResult,
    TechRef,
)
from techeval.state import GraphState, build_initial_state, latest_by_criterion, latest_by_tech
from techeval.stub_llm import DEFAULT_FIXTURES_DIR

logger = logging.getLogger(__name__)

PERSPECTIVE_NODE: dict[str, str] = {
    "trl": "tech_research",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}
PERSPECTIVE_KEY: dict[str, str] = {
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}
DEFAULT_RECURSION_LIMIT = 60


class Agents(BaseModel):
    """그래프에 연결할 에이전트 함수 묶음 (CONTRACTS §5 시그니처)."""

    model_config = {"arbitrary_types_allowed": True}

    run_tech_research: Callable[..., TechResearchOutput]
    run_domain_eval: Callable[..., list[CriterionResult]]
    run_market_eval: Callable[..., list[CriterionResult]]
    run_stakeholder_eval: Callable[..., list[CriterionResult]]
    run_synthesis: Callable[..., Any]
    run_report: Callable[..., str]
    render_pdf: Callable[[str, str], str]


_AGENT_SOURCES: dict[str, tuple[str, str, str]] = {
    # 필드 → (모듈, 함수, 소유 역할)
    "run_tech_research": ("techeval.agents.tech_research", "run_tech_research", "B"),
    "run_domain_eval": ("techeval.agents.domain", "run_domain_eval", "B"),
    "run_market_eval": ("techeval.agents.market", "run_market_eval", "C"),
    "run_stakeholder_eval": ("techeval.agents.stakeholder", "run_stakeholder_eval", "C"),
    "run_synthesis": ("techeval.agents.synthesis", "run_synthesis", "D"),
    "run_report": ("techeval.agents.report", "run_report", "D"),
    "render_pdf": ("techeval.report.pdf", "render_pdf", "D"),
}


def load_agents(stub: bool = False, *, allow_stub_fallback: bool = False) -> tuple[Agents, list[str]]:
    """실제 에이전트를 import한다. stub=True면 전부 스텁. 반환: (Agents, 스텁으로 대체된 필드 목록)."""
    from techeval import stub_agents

    if stub:
        return Agents(**{f: getattr(stub_agents, f) for f in _AGENT_SOURCES}), list(_AGENT_SOURCES)

    fields: dict[str, Any] = {}
    stubbed: list[str] = []
    for field, (module, fn, owner) in _AGENT_SOURCES.items():
        try:
            mod = __import__(module, fromlist=[fn])
            fields[field] = getattr(mod, fn)
        except (ImportError, AttributeError) as e:
            if not allow_stub_fallback:
                raise ImportError(f"{module}.{fn} (역할 {owner})를 불러올 수 없습니다: {e}") from e
            logger.warning("%s.%s (역할 %s) 없음 — 스텁으로 대체", module, fn, owner)
            fields[field] = getattr(stub_agents, field)
            stubbed.append(field)
    return Agents(**fields), stubbed


class GraphConfig(BaseModel):
    output_dir: str = "outputs"
    skip_pdf: bool = False
    stub: bool = False  # True면 query_rewrite에 LLM을 쓰지 않고, 웹 픽스처 URL을 출처 레지스트리에 미리 등록
    recursion_limit: int = DEFAULT_RECURSION_LIMIT


def _preload_fixture_urls() -> list[str]:
    path = Path(DEFAULT_FIXTURES_DIR) / "web_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [r["url"] for results in data.values() for r in results if r.get("url")]


def _tech_by_id(state: GraphState, tech_id: str) -> TechRef:
    return next(t for t in state["technologies"] if t.tech_id == tech_id)


def _latest_evals(state: GraphState) -> dict[str, list[CriterionResult]]:
    return {p: latest_by_criterion(state.get(key, [])) for p, key in PERSPECTIVE_KEY.items()}


def _log_node(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(state: Any) -> Any:
        logger.info("▶ %s", fn.__name__)
        out = fn(state)
        logger.info("◀ %s", fn.__name__)
        return out

    return wrapper


def build_graph(
    deps: Deps,
    agents: Agents | None = None,
    config: GraphConfig | None = None,
    agent_deps: dict[str, Deps] | None = None,
):
    """CompiledStateGraph를 만든다. `deps.web_search`는 출처 레지스트리로 감싸 V5 검사에 쓴다.

    `agent_deps`: 에이전트 이름(tech_research/domain/market/stakeholder/synthesis/report) → 그 노드에만 줄 Deps.
    없는 에이전트는 공용 `deps`를 쓴다 (에이전트별 모델 분리용, 계약 변경 없음).
    """
    cfg = config or GraphConfig()
    if agents is None:
        agents, _ = load_agents(stub=cfg.stub)
    registry = SourceRegistry(preload_urls=_preload_fixture_urls() if cfg.stub else None)
    wrapped_search = registry.wrap(deps.web_search)
    deps = deps.model_copy(update={"web_search": wrapped_search})
    per_agent = {k: v.model_copy(update={"web_search": wrapped_search}) for k, v in (agent_deps or {}).items()}

    def deps_for(agent: str) -> Deps:
        return per_agent.get(agent, deps)

    retriever = deps.retriever
    rewrite_llm = None if cfg.stub else deps.llm

    # ----------------------------------------------------------------- payloads

    def agent_payload(state: GraphState, tech: TechRef, perspective: str, **extra: Any) -> dict:
        profiles = {p.tech_id: p for p in latest_by_tech(state.get("tech_profiles", []))}
        trl = [r for r in latest_by_criterion(state.get("trl_eval", [])) if r.tech_id == tech.tech_id]
        payload = {
            "tech": tech,
            "domain": state["domain"],
            "tech_profile": profiles.get(tech.tech_id),
            "trl_eval": trl,
            "retry_count": state["retry_counts"].get(f"{perspective}:{tech.tech_id}", 0),
        }
        payload.update(extra)
        return payload

    # -------------------------------------------------------------------- nodes

    @_log_node
    def init(state: GraphState) -> dict:
        return dict(build_initial_state())

    def fan_out_tech(state: GraphState) -> list[Send]:
        return [Send("tech_research", agent_payload(state, t, "trl")) for t in state["technologies"]]

    @_log_node
    def tech_research(payload: dict) -> dict:
        inp = AgentInput.model_validate(payload)
        out = agents.run_tech_research(inp, deps_for("tech_research"))
        out = TechResearchOutput.model_validate(out if isinstance(out, dict) else out.model_dump())
        return {"tech_profiles": [out.tech_profile], "trl_eval": out.trl_eval}

    def route_after_tech_research(state: GraphState) -> str:
        # 관점 평가가 이미 시작된 뒤(perspective_check가 T 기준을 재요청한 경우)에는 검사 노드로 돌아가지 않는다
        return "perspective_check" if state.get("market_eval") else "tech_evidence_check"

    @_log_node
    def tech_evidence_check(state: GraphState) -> dict:
        res = check_tech_evidence(
            state["technologies"],
            latest_by_tech(state.get("tech_profiles", [])),
            latest_by_criterion(state.get("trl_eval", [])),
            retriever,
            registry,
        )
        retry = dict(state["retry_counts"])
        missing = {k: v for k, v in state.get("missing_criteria", {}).items() if not k.startswith("trl")}
        missing["trl"] = []
        confirmed: list[CriterionResult] = []
        for tech in state["technologies"]:
            tid = tech.tech_id
            items = res.missing.get(tid, [])
            key = f"trl:{tid}"
            if not items:
                missing[key] = []
                continue
            if retry.get(key, 0) < MAX_RETRY_PER_PERSPECTIVE:
                retry[key] = retry.get(key, 0) + 1
                missing[key] = items
                missing["trl"] = sorted(set(missing["trl"]) | {i for i in items if not i.startswith("PROFILE:")})
                logger.info(
                    "tech_evidence_check: %s 재검색 %d/%d — %s", tid, retry[key], MAX_RETRY_PER_PERSPECTIVE, items
                )
            else:
                missing[key] = []
                profile_items = [i for i in items if i.startswith("PROFILE:")]
                if profile_items:
                    logger.warning(
                        "tech_evidence_check: %s 프로필 항목 %s 은 상한 도달로 그대로 진행", tid, profile_items
                    )
                for cid in res.missing_criteria(tid):
                    confirmed.append(
                        make_not_public_result(tech, cid, queries=[], now=deps.now(), retry_count=retry.get(key, 0))
                    )
        return {"missing_criteria": missing, "retry_counts": retry, "trl_eval": confirmed}

    def route_after_tech_check(state: GraphState) -> str | list[Send]:
        mc = state["missing_criteria"]
        if any(mc.get(f"trl:{t.tech_id}") for t in state["technologies"]):
            return "query_rewrite"
        return [
            Send(PERSPECTIVE_NODE[p], agent_payload(state, t, p))
            for p in ("market", "stakeholder", "domain")
            for t in state["technologies"]
        ]

    @_log_node
    def query_rewrite(state: GraphState) -> dict:
        # 검색어 자체는 아래 fan_out_rewritten 에서 Send payload로만 전달한다 (State 키 없음)
        return {}

    def fan_out_rewritten(state: GraphState) -> list[Send]:
        profiles = {p.tech_id: p for p in latest_by_tech(state.get("tech_profiles", []))}
        sends = []
        for tech in state["technologies"]:
            items = state["missing_criteria"].get(f"trl:{tech.tech_id}") or []
            if not items:
                continue
            prev = profiles[tech.tech_id].search_queries_used if tech.tech_id in profiles else []
            retry = state["retry_counts"][f"trl:{tech.tech_id}"]
            queries = rewrite_queries(tech, items, prev, retry_count=retry, llm=rewrite_llm)
            criteria = [i for i in items if not i.startswith("PROFILE:")] or None
            sends.append(
                Send(
                    "tech_research",
                    agent_payload(state, tech, "trl", missing_criteria=criteria, rewritten_queries=queries),
                )
            )
        return sends

    def _perspective_node(name: str, key: str, fn_name: str):
        def node(payload: dict) -> dict:
            inp = AgentInput.model_validate(payload)
            results = getattr(agents, fn_name)(inp, deps_for(name))
            results = [CriterionResult.model_validate(r if isinstance(r, dict) else r.model_dump()) for r in results]
            expected = set(inp.missing_criteria) if inp.missing_criteria else set(PERSPECTIVE_CRITERIA[name])
            got = {r.criterion_id for r in results}
            if got != expected:
                logger.warning(
                    "%s(%s): 반환 기준 %s ≠ 요청 %s — perspective_check가 재실행을 건다",
                    key,
                    inp.tech.tech_id,
                    sorted(got),
                    sorted(expected),
                )
            return {key: results}

        node.__name__ = key
        return _log_node(node)

    market_eval = _perspective_node("market", "market_eval", "run_market_eval")
    stakeholder_eval = _perspective_node("stakeholder", "stakeholder_eval", "run_stakeholder_eval")
    domain_eval = _perspective_node("domain", "domain_eval", "run_domain_eval")

    @_log_node
    def perspective_check(state: GraphState) -> dict:
        res = check_perspectives(state["technologies"], _latest_evals(state), retriever, registry)
        retry = dict(state["retry_counts"])
        missing: dict[str, list[str]] = {k: v for k, v in state["missing_criteria"].items() if k.startswith("trl:")}
        updates: dict[str, list[CriterionResult]] = {key: [] for key in PERSPECTIVE_KEY.values()}
        for r in res.corrected:
            updates[PERSPECTIVE_KEY[r.perspective]].append(r)
        for p, techs in res.missing.items():
            missing[p] = []
            for tid, cids in techs.items():
                key = f"{p}:{tid}"
                if not cids:
                    missing[key] = []
                    continue
                if retry.get(key, 0) < MAX_RETRY_PER_PERSPECTIVE:
                    retry[key] = retry.get(key, 0) + 1
                    missing[key] = list(cids)
                    missing[p] = sorted(set(missing[p]) | set(cids))
                    logger.info(
                        "perspective_check: %s 재실행 %d/%d — %s", key, retry[key], MAX_RETRY_PER_PERSPECTIVE, cids
                    )
                else:
                    missing[key] = []
                    tech = _tech_by_id(state, tid)
                    for cid in cids:
                        updates[PERSPECTIVE_KEY[p]].append(
                            make_not_public_result(tech, cid, queries=[], now=deps.now(), retry_count=retry.get(key, 0))
                        )
        return {"missing_criteria": missing, "retry_counts": retry, **updates}

    def route_after_perspective_check(state: GraphState) -> str | list[Send]:
        mc = state["missing_criteria"]
        sends: list[Send] = []
        for p in PERSPECTIVE_CRITERIA:
            for tech in state["technologies"]:
                cids = mc.get(f"{p}:{tech.tech_id}") or []
                if not cids:
                    continue
                extra: dict[str, Any] = {"missing_criteria": cids}
                if p == "trl":
                    profiles = {pr.tech_id: pr for pr in latest_by_tech(state.get("tech_profiles", []))}
                    prev = profiles[tech.tech_id].search_queries_used if tech.tech_id in profiles else []
                    extra["rewritten_queries"] = rewrite_queries(
                        tech, cids, prev, retry_count=state["retry_counts"][f"trl:{tech.tech_id}"], llm=rewrite_llm
                    )
                else:
                    extra["previous_results"] = [
                        r for r in latest_by_criterion(state.get(PERSPECTIVE_KEY[p], [])) if r.tech_id == tech.tech_id
                    ]
                sends.append(Send(PERSPECTIVE_NODE[p], agent_payload(state, tech, p, **extra)))
        return sends or "synthesis"

    def synthesis_input(state: GraphState) -> SynthesisInput:
        ev = _latest_evals(state)
        return SynthesisInput(
            technologies=state["technologies"],
            tech_profiles=latest_by_tech(state.get("tech_profiles", [])),
            trl_eval=ev["trl"],
            market_eval=ev["market"],
            stakeholder_eval=ev["stakeholder"],
            domain_eval=ev["domain"],
            counter_evidence=state.get("counter_evidence", []),
        )

    @_log_node
    def synthesis(state: GraphState) -> dict:
        out = agents.run_synthesis(synthesis_input(state), deps_for("synthesis"))
        from techeval.schemas import SynthesisResult

        return {"synthesis": SynthesisResult.model_validate(out if isinstance(out, dict) else out.model_dump())}

    @_log_node
    def evidence_gap_check(state: GraphState) -> dict:
        gap = check_evidence_gap(
            state["technologies"], _latest_evals(state), state.get("synthesis"), state.get("counter_evidence", [])
        )
        return {"evidence_gap": gap}

    def route_after_gap_check(state: GraphState) -> str:
        gap = state["evidence_gap"]
        if gap.needs_counter_search and state["retry_counts"].get("counter", 0) < MAX_COUNTER_EVIDENCE_SEARCH:
            return "counter_evidence_search"
        return "report"

    @_log_node
    def counter_evidence_search(state: GraphState) -> dict:
        evs = search_counter_evidence(state["evidence_gap"], deps, technologies=state["technologies"])
        retry = dict(state["retry_counts"])
        retry["counter"] = retry.get("counter", 0) + 1
        return {"counter_evidence": [*state.get("counter_evidence", []), *evs], "retry_counts": retry}

    def report_input(state: GraphState) -> ReportInput:
        si = synthesis_input(state)
        return ReportInput(
            technologies=si.technologies,
            domain=state["domain"],
            tech_profiles=si.tech_profiles,
            trl_eval=si.trl_eval,
            market_eval=si.market_eval,
            stakeholder_eval=si.stakeholder_eval,
            domain_eval=si.domain_eval,
            counter_evidence=si.counter_evidence,
            synthesis=state["synthesis"],
            evidence_gap=state["evidence_gap"],
            judge_result=state.get("judge_result"),
            previous_report_md=state.get("report_md"),
        )

    @_log_node
    def report(state: GraphState) -> dict:
        retry = dict(state["retry_counts"])
        if state.get("judge_result") is not None:  # 재생성
            retry["report"] = retry.get("report", 0) + 1
            logger.info("report: 재생성 %d/%d", retry["report"], MAX_REPORT_REGENERATION)
        md = agents.run_report(report_input(state), deps_for("report"))
        if not isinstance(md, str) or not md.strip():
            raise ValueError("run_report must return non-empty markdown")
        return {"report_md": md, "retry_counts": retry}

    @_log_node
    def judge(state: GraphState) -> dict:
        judge_llm = deps.judge_llm
        if judge_llm is None:
            raise RuntimeError("deps.judge_llm (JUDGE_MODEL) 이 필요합니다 — AGENTS.md 규칙 9")
        return {"judge_result": judge_report(state["report_md"], report_input(state), judge_llm, deps.now())}

    def route_after_judge(state: GraphState) -> str:
        jr = state["judge_result"]
        if not jr.passed and state["retry_counts"].get("report", 0) < MAX_REPORT_REGENERATION:
            return "report"
        if not jr.passed:
            logger.warning(
                "judge 미달이지만 재생성 상한 도달 — 그대로 출력: scores=%s missing=%s", jr.scores, jr.missing_required
            )
        return "render_pdf"

    @_log_node
    def render_pdf(state: GraphState) -> dict:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(state["report_md"], encoding="utf-8")
        if cfg.skip_pdf:
            return {"report_pdf_path": ""}
        return {"report_pdf_path": agents.render_pdf(state["report_md"], str(out_dir / "report.pdf"))}

    # -------------------------------------------------------------------- graph

    g = StateGraph(GraphState)
    for name, fn in [
        ("init", init),
        ("tech_research", tech_research),
        ("tech_evidence_check", tech_evidence_check),
        ("query_rewrite", query_rewrite),
        ("market_eval", market_eval),
        ("stakeholder_eval", stakeholder_eval),
        ("domain_eval", domain_eval),
        ("perspective_check", perspective_check),
        ("synthesis", synthesis),
        ("evidence_gap_check", evidence_gap_check),
        ("counter_evidence_search", counter_evidence_search),
        ("report", report),
        ("judge", judge),
        ("render_pdf", render_pdf),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "init")
    g.add_conditional_edges("init", fan_out_tech, ["tech_research"])
    g.add_conditional_edges("tech_research", route_after_tech_research, ["tech_evidence_check", "perspective_check"])
    g.add_conditional_edges(
        "tech_evidence_check",
        route_after_tech_check,
        ["query_rewrite", "market_eval", "stakeholder_eval", "domain_eval"],
    )
    g.add_conditional_edges("query_rewrite", fan_out_rewritten, ["tech_research"])
    for n in ("market_eval", "stakeholder_eval", "domain_eval"):
        g.add_edge(n, "perspective_check")
    g.add_conditional_edges(
        "perspective_check",
        route_after_perspective_check,
        ["tech_research", "market_eval", "stakeholder_eval", "domain_eval", "synthesis"],
    )
    g.add_edge("synthesis", "evidence_gap_check")
    g.add_conditional_edges("evidence_gap_check", route_after_gap_check, ["counter_evidence_search", "report"])
    g.add_edge("counter_evidence_search", "synthesis")
    g.add_edge("report", "judge")
    g.add_conditional_edges("judge", route_after_judge, ["report", "render_pdf"])
    g.add_edge("render_pdf", END)

    compiled = g.compile()
    compiled.registry = registry  # type: ignore[attr-defined]  # 테스트·run.py에서 검색 호출 기록 확인용
    return compiled


def invoke_config(cfg: GraphConfig | None = None) -> dict:
    return {"recursion_limit": (cfg or GraphConfig()).recursion_limit}


def state_to_json(state: dict) -> str:
    """State 덤프용 JSON (Pydantic 객체 → dict)."""

    def default(o: Any) -> Any:
        if isinstance(o, BaseModel):
            return o.model_dump()
        if isinstance(o, tuple | set):
            return list(o)
        return str(o)

    return json.dumps(state, ensure_ascii=False, indent=2, default=default)
