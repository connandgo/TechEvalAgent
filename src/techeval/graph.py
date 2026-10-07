"""LangGraph StateGraph (E) — Supervisor 패턴.

Agent 실습에서 고정 순서(기술 조사 → 검사 → 관점 3개 병렬 → 종합 → 보고서)를 Supervisor 패턴으로 바꿨다.

- `supervisor`가 매번 현재 State(수집된 관점, 근거 충분도, 재시도 횟수)를 보고 `add_conditional_edges`로
  다음 담당을 고른다. 실행 순서를 코드에 박아 두지 않는다.
- 하위 에이전트(tech_research, market_eval, stakeholder_eval, domain_eval, counter_evidence, synthesis)는
  끝나면 항상 supervisor로 돌아온다. 하위 에이전트끼리는 직접 연결되지 않는다.
- 근거 충분도 판단은 기존 제어 함수(`check_tech_evidence`, `check_perspectives`, `check_evidence_gap`)를
  supervisor 안에서 호출해 코드로 결정한다(판정은 결정론, 의견 생성 없음).
- 근거가 부족하면 해당 하위 에이전트에 부족 기준만 재작업을 요청한다. 상한(관점·기술별 2회)에 도달하면 not_public으로 확정.
- 종료는 단계 수 고정이 아니라 상한으로 보장한다: 재시도 상한 + `MAX_SUPERVISOR_STEPS` + `recursion_limit`.
- 보고서·품질 평가도 supervisor로 돌아온다. `quality_eval`(Groundedness·중립성·편향 통제·관점 커버리지 + 기존 Judge)이
  미달이면 supervisor가 원인(`QualityResult.issues`)을 보고 재작업 위치를 고른다: 결과 누락 → 해당 관점 에이전트,
  반대 근거 미확보·근거 비대칭·벤더 편중 → 반대 근거 탐색(→ 재종합), 근거 약함 → 근거 부족 기준 재작업, 그 밖 → 보고서 재작성.
  보고서 재작성 상한(`MAX_REPORT_REGENERATION`)에 닿으면 PDF로 간다.
- supervisor 판단은 `SupervisorDecision`으로 검증한 뒤 State에 쓴다(잘못된 action·기술 ID·담당 밖 기준 차단).
- 하위 에이전트가 예외를 내면 `node_status`에 error로 기록하고 그래프는 계속 간다(fallback: 재시도 → not_public).
"""

import functools
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ValidationError

from techeval.agents._deps import (
    AgentInput,
    Deps,
    ReportInput,
    SynthesisInput,
    TechResearchOutput,
)
from techeval.control._common import make_not_public_result
from techeval.control.counter_evidence import search_counter_evidence
from techeval.control.evidence_gap import check_evidence_gap
from techeval.control.perspective_check import check_perspectives
from techeval.control.quality_eval import evaluate_report_quality
from techeval.control.query_rewrite import rewrite_queries
from techeval.control.sources import SourceRegistry
from techeval.control.tech_evidence_check import check_tech_evidence
from techeval.schemas import (
    MAX_COUNTER_EVIDENCE_SEARCH,
    MAX_REPORT_REGENERATION,
    MAX_RETRY_PER_PERSPECTIVE,
    MAX_SUPERVISOR_STEPS,
    PERSPECTIVE_CRITERIA,
    CriterionResult,
    QualityResult,
    SupervisorDecision,
    TechRef,
)
from techeval.state import (
    GraphState,
    build_initial_state,
    latest_by_criterion,
    latest_by_tech,
)
from techeval.stub_llm import DEFAULT_FIXTURES_DIR

logger = logging.getLogger(__name__)

PERSPECTIVE_NODE: dict[str, str] = {
    "trl": "tech_research",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}
NODE_PERSPECTIVE: dict[str, str] = {v: k for k, v in PERSPECTIVE_NODE.items()}
PERSPECTIVE_KEY: dict[str, str] = {
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}
# supervisor가 첫 실행을 배정하는 관점 순서. 순서 자체가 흐름을 정하지 않는다 — 매 판단은 State 조건으로 한다.
EVAL_PERSPECTIVES: tuple[str, ...] = ("market", "stakeholder", "domain")
DEFAULT_RECURSION_LIMIT = 150


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


class _QueryRecordingRetriever:
    """`search` 호출의 질의를 기록하고 나머지는 원래 검색기로 넘긴다 (not_public 기록에 실제 검색 이력을 남기기 위함)."""

    def __init__(self, inner: Any, log: list[str]):
        self._inner = inner
        self._log = log

    def search(self, query: str, *args: Any, **kwargs: Any) -> Any:
        self._log.append(query)
        return self._inner.search(query, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


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

    def agent_payload(state: GraphState, tech: TechRef, perspective: str, **extra: Any) -> dict:
        """하위 에이전트에 넘길 스코프 입력. 전체 State가 아니라 담당 기술·관점에 필요한 것만."""
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

    # ================================================================ init

    def init(state: GraphState, config: RunnableConfig) -> dict:
        trace_id = (config.get("metadata") or {}).get("trace_id")
        out = dict(build_initial_state(trace_id=trace_id))
        logger.info("[trace=%s] 시작", out["trace_id"])
        return out

    # ================================================================ supervisor

    def _post_tech_research(state: GraphState, retry: dict, missing: dict, status: dict) -> dict[str, list]:
        """방금 끝난 기술 조사 결과의 근거 충분도를 판정하고, 부족하면 재작업 대상으로 올린다."""
        res = check_tech_evidence(
            state["technologies"],
            latest_by_tech(state.get("tech_profiles", [])),
            latest_by_criterion(state.get("trl_eval", [])),
            retriever,
            registry,
        )
        latest_trl = {(r.tech_id, r.criterion_id): r for r in latest_by_criterion(state.get("trl_eval", []))}
        confirmed: list[CriterionResult] = []
        for tech in state["technologies"]:
            tid = tech.tech_id
            key = f"trl:{tid}"
            items = res.missing.get(tid, [])
            if not items or status.get(key) == "exhausted":
                missing[key] = []
                continue
            if retry.get(key, 0) < MAX_RETRY_PER_PERSPECTIVE:
                retry[key] = retry.get(key, 0) + 1
                missing[key] = items
                continue
            # 상한 도달: 근거를 끝내 못 찾은 기준은 not_public으로 확정하고 재작업 대상에서 뺀다
            missing[key] = []
            status[key] = "exhausted"
            profile = next((p for p in latest_by_tech(state.get("tech_profiles", [])) if p.tech_id == tid), None)
            used = list(
                dict.fromkeys(
                    [*(profile.search_queries_used if profile else []), *state.get("search_log", {}).get(key, [])]
                )
            )
            err = _run_error(state, key, state.get("node_status", {}).get(key))
            for cid in res.missing_criteria(tid):
                prev = latest_trl.get((tid, cid))
                if prev is not None and prev.level == "not_public":
                    continue
                confirmed.append(
                    make_not_public_result(
                        tech,
                        cid,
                        queries=used,
                        now=deps.now(),
                        retry_count=retry.get(key, 0),
                        run_error=err,
                        problems=[p for p in res.problems.get(tid, []) if p.startswith(f"{cid}:") or f"-{cid}-" in p],
                    )
                )
        return {"trl_eval": confirmed}

    def _searched(state: GraphState, key: str, results: list[CriterionResult]) -> list[str]:
        """이 작업(`관점:기술`)에서 실제 실행한 검색어: 노드가 기록한 이력 + 결과 Evidence의 search_query."""
        qs = list(state.get("search_log", {}).get(key, []))
        qs += [e.search_query for r in results for e in r.evidence if e.search_query and e.source_type != "not_public"]
        return list(dict.fromkeys(qs))

    def _run_error(state: GraphState, key: str, status_before: str | None) -> str | None:
        """마지막 실행이 예외로 끝났으면 그 오류 요약 (검색 미수행 not_public과 '공개 근거 없음'을 구분)."""
        if status_before != "error":
            return None
        err = state.get("last_error") or ""
        return err if err.startswith(f"{key}:") else f"{key}: 노드 예외"

    def _post_perspectives(
        state: GraphState, just_ran: list[str], retry: dict, missing: dict, status: dict
    ) -> dict[str, list]:
        """방금 돌아온 관점(`just_ran`)만 판정한다 (검사 함수는 한 번만 호출).

        배정되지 않았던 관점은 건드리지 않는다 — 재작업을 배정받기 전에 재시도 횟수만 올라가
        상한에 걸리는 일을 막는다.
        """
        evals = _latest_evals(state)
        res = check_perspectives(state["technologies"], evals, retriever, registry)
        updates: dict[str, list[CriterionResult]] = {key: [] for key in PERSPECTIVE_KEY.values()}
        for r in res.corrected:  # 신뢰도 재계산본 (V7)
            updates[PERSPECTIVE_KEY[r.perspective]].append(r)
        for p in just_ran:
            latest = {(r.tech_id, r.criterion_id): r for r in evals[p]}
            for tech in state["technologies"]:
                tid = tech.tech_id
                key = f"{p}:{tid}"
                if key not in status:  # 아직 배정 전
                    continue
                cids = res.missing[p].get(tid, [])
                if not cids or status.get(key) == "exhausted":
                    missing[key] = []
                    continue
                if retry.get(key, 0) < MAX_RETRY_PER_PERSPECTIVE:
                    retry[key] = retry.get(key, 0) + 1
                    missing[key] = list(cids)
                    continue
                missing[key] = []
                err = _run_error(state, key, status.get(key))
                status[key] = "exhausted"
                used = _searched(state, key, [r for r in evals[p] if r.tech_id == tid])
                for cid in cids:
                    prev = latest.get((tid, cid))
                    if prev is not None and prev.level == "not_public":
                        continue
                    updates[PERSPECTIVE_KEY[p]].append(
                        make_not_public_result(
                            tech,
                            cid,
                            queries=used,
                            now=deps.now(),
                            retry_count=retry.get(key, 0),
                            problems=[x for x in res.problems if x.startswith(f"{tid}/{cid}") or f"{tid}-{cid}-" in x],
                            run_error=err,
                        )
                    )
        return updates

    def _decide(state: GraphState, missing: dict, status: dict, retry: dict, step: int) -> tuple[str, dict, str]:
        """현재 State만 보고 다음 담당을 고른다.

        반환: (next, dispatch{노드: {tech_id: 부족 항목}}, 사유). 빈 리스트 = 첫 실행.
        기술 조사(TRL)를 먼저 두는 것은 순서 고정이 아니라 데이터 의존이다 — 도메인 평가가 `tech_profile`을,
        모든 관점이 `trl_eval`을 입력으로 받는다. 시장·이해관계자·도메인 사이에는 의존이 없으므로
        State상 대기 중인 관점(미수집 또는 근거 부족)을 한 번의 판단으로 모두 배정한다.
        """
        techs = state["technologies"]

        if step > MAX_SUPERVISOR_STEPS:
            stale = [k for k in ("synthesis_stale", "report_stale") if state.get(k)]
            cap = f"supervisor 상한({MAX_SUPERVISOR_STEPS}) 도달 — 남은 재작업 없이" + (
                f"(주의: {stale} — 재작업 결과가 보고서에 반영되지 않음)" if stale else ""
            )
            if not state.get("synthesis"):
                return "synthesis", {}, f"{cap} 종합으로 진행"
            if not state.get("report_md"):
                return "report", {}, f"{cap} 보고서 작성으로 진행"
            if state.get("quality_pending"):
                return "quality_eval", {}, f"{cap} 품질 평가로 진행"
            return "render_pdf", {}, f"{cap} 출력"

        not_run = [t.tech_id for t in techs if f"trl:{t.tech_id}" not in status]
        if not_run:
            return (
                "tech_research",
                {"tech_research": {tid: [] for tid in not_run}},
                f"기술 프로필·TRL 미수집: {not_run}",
            )

        rework = {t.tech_id: missing[f"trl:{t.tech_id}"] for t in techs if missing.get(f"trl:{t.tech_id}")}
        if rework:
            return "tech_research", {"tech_research": rework}, f"기술 근거 부족 → 재조사 요청 {rework}"

        plan: dict[str, dict[str, list[str]]] = {}
        why: list[str] = []
        for p in EVAL_PERSPECTIVES:
            for t in techs:
                key = f"{p}:{t.tech_id}"
                if key not in status:
                    plan.setdefault(PERSPECTIVE_NODE[p], {})[t.tech_id] = []
                    why.append(f"{key} 미수집")
                elif missing.get(key):
                    plan.setdefault(PERSPECTIVE_NODE[p], {})[t.tech_id] = list(missing[key])
                    why.append(f"{key} 근거 부족 {missing[key]} → 재작업")
        if plan:
            return "perspectives", plan, "관점 평가 배정: " + ", ".join(why)

        if not state.get("synthesis") or state.get("synthesis_stale"):
            why = "반대 근거 반영 위해 재종합" if state.get("synthesis_stale") else "4개 관점 근거 충분 → 종합"
            return "synthesis", {}, why

        gap = state.get("evidence_gap")
        if gap is not None and gap.needs_counter_search and retry.get("counter", 0) < MAX_COUNTER_EVIDENCE_SEARCH:
            causes = []
            if gap.asymmetry:
                causes.append(f"근거 수 비대칭 {gap.evidence_count}")
            if gap.opposing_missing:
                causes.append(f"반대 근거 없는 기준 {len(gap.opposing_missing)}개")
            return "counter_evidence", {}, "반대 근거 탐색 필요: " + (", ".join(causes) or gap.note[:80])

        if not state.get("report_md"):
            return "report", {}, "근거 충분·비대칭 해소 → 보고서 작성"
        if state.get("report_stale"):
            return "report", {}, "품질 미달 재작업 반영 → 보고서 재작성"
        if state.get("quality_pending"):
            return "quality_eval", {}, "새 보고서 → 품질 평가"

        q = state.get("quality_result")
        if q is None:
            return "render_pdf", {}, "품질 검수 실행 오류(재시도 포함) — 보고서는 보존, 검수 미완료로 출력"
        if q.passed:
            return "render_pdf", {}, "품질 평가 4개 항목·기존 Judge 통과 → 출력"
        if retry.get("report", 0) >= MAX_REPORT_REGENERATION:
            return "render_pdf", {}, f"품질 미달이지만 보고서 재작성 상한({MAX_REPORT_REGENERATION}) 도달 — 그대로 출력"
        return _quality_rework(state, q, missing, status, retry)

    def _quality_rework(
        state: GraphState, q: QualityResult, missing: dict, status: dict, retry: dict
    ) -> tuple[str, dict, str]:
        """품질 미달 원인별로 재작업 위치를 고른다. 고칠 수 있는 상류(조사·반대 근거)부터, 안 되면 보고서 재작성."""
        causes = sorted({i.cause for i in q.issues})

        # 1) State에 결과 자체가 없는 기준 → 그 관점 에이전트 (TRL이 있으면 TRL 먼저: 다른 관점의 입력)
        lacking: dict[str, dict[str, list[str]]] = {}
        for i in q.issues:
            if i.cause != "missing_result" or not i.tech_id or not i.criterion_id:
                continue
            p = next(p for p, cids in PERSPECTIVE_CRITERIA.items() if i.criterion_id in cids)
            if status.get(f"{p}:{i.tech_id}") != "exhausted":
                lacking.setdefault(PERSPECTIVE_NODE[p], {}).setdefault(i.tech_id, []).append(i.criterion_id)
        if "tech_research" in lacking:
            return (
                "tech_research",
                {"tech_research": lacking["tech_research"]},
                f"품질: 기술×기준 결과 누락 → 재조사 {lacking}",
            )
        if lacking:
            return "perspectives", lacking, f"품질: 기술×기준 결과 누락 → 관점 재평가 {lacking}"

        # 2) 편향 통제: 반대 근거 탐색 전이면 탐색 (→ 재종합 → 보고서)
        bias = sorted(
            {f"{i.cause}:{i.tech_id}" for i in q.issues if i.cause in ("asymmetry", "counter_missing", "vendor_heavy")}
        )
        if bias and retry.get("counter", 0) < MAX_COUNTER_EVIDENCE_SEARCH:
            return "counter_evidence", {}, f"품질: 편향 통제 미달 {bias} → 반대 근거 탐색"

        # 3) 근거 약함(LLM groundedness) → 신뢰도 low 판정 중 재작업 여유가 있는 기준
        if any(i.cause == "weak_evidence" for i in q.issues):
            weak: dict[str, dict[str, list[str]]] = {}
            for p, results in _latest_evals(state).items():
                for r in results:
                    key = f"{p}:{r.tech_id}"
                    if r.confidence != "low" or r.level == "not_public" or status.get(key) == "exhausted":
                        continue
                    if retry.get(key, 0) >= MAX_RETRY_PER_PERSPECTIVE:
                        continue
                    weak.setdefault(PERSPECTIVE_NODE[p], {}).setdefault(r.tech_id, []).append(r.criterion_id)
            if "tech_research" in weak:
                weak = {"tech_research": weak["tech_research"]}
            for node, by_tech in weak.items():
                for tid, cids in by_tech.items():
                    # 재시도 횟수는 올리지 않는다 — 재실행 결과가 여전히 부족하면 사후 검사(_post_*)가 올린다 (이중 계산 방지)
                    missing[f"{NODE_PERSPECTIVE[node]}:{tid}"] = cids
            if weak:
                nxt = "tech_research" if "tech_research" in weak else "perspectives"
                return (nxt, weak, f"품질: Groundedness 미달 → 신뢰도 low 기준 재작업 {weak}")

        # 4) 보고서 표현·구조 문제(각주·금칙어·어조·한계점 미공개·절 누락·기존 Judge) 또는 상류 재작업 여유 없음
        return "report", {}, f"품질 미달 {causes} → 수정 지시로 보고서 재작성"

    def supervisor(state: GraphState) -> dict:
        step = state.get("step_count", 0) + 1
        retry = dict(state["retry_counts"])
        missing = dict(state.get("missing_criteria", {}))
        status = dict(state.get("node_status", {}))
        prev = state.get("next")
        updates: dict[str, Any] = {}

        # 1) 방금 돌아온 하위 에이전트 결과를 판정한다 (근거 충분도 = 코드 판정)
        if prev == "tech_research":
            for k, v in _post_tech_research(state, retry, missing, status).items():
                updates[k] = [*updates.get(k, []), *v]
        elif prev == "perspectives":
            ran = [NODE_PERSPECTIVE[n] for n in state.get("dispatch", {})]
            for k, v in _post_perspectives(state, ran, retry, missing, status).items():
                updates[k] = [*updates.get(k, []), *v]
        elif prev == "synthesis":
            updates["synthesis_stale"] = False
            updates["evidence_gap"] = check_evidence_gap(
                state["technologies"], _latest_evals(state), state.get("synthesis"), state.get("counter_evidence", [])
            )
        elif prev == "report":
            updates["report_stale"] = False
            updates["quality_pending"] = True
        elif prev == "quality_eval":
            updates["quality_pending"] = False
        if prev in ("tech_research", "perspectives", "counter_evidence") and state.get("synthesis"):
            updates["synthesis_stale"] = True  # 종합 뒤에 근거가 바뀜 → 재종합

        # 2) 갱신된 State로 다음 담당을 고른다
        keys = ("synthesis_stale", "evidence_gap", "report_stale", "quality_pending")
        view = {**state, **{k: v for k, v in updates.items() if k in keys}}
        # 판단은 사본에 하고, 검증을 통과했을 때만 재시도·부족 기준 변경을 반영한다
        retry_d, missing_d = dict(retry), dict(missing)
        nxt, dispatch, reason = _decide(view, missing_d, status, retry_d, step)
        try:  # 배정 검증: 잘못된 action·기술 ID·담당 밖 기준은 State에 쓰기 전에 차단
            decision = SupervisorDecision(next=nxt, dispatch=dispatch, reason=reason)
            retry, missing = retry_d, missing_d
        except ValidationError as e:
            fallback = "render_pdf" if view.get("report_md") else ("report" if view.get("synthesis") else "synthesis")
            logger.error("[trace=%s] supervisor 판단 검증 실패 → %s: %s", state.get("trace_id"), fallback, e)
            decision = SupervisorDecision(
                next=fallback, dispatch={}, reason=f"판단 검증 실패로 {fallback} 진행: {e.errors()[0]['msg']}"
            )
            updates["last_error"] = f"supervisor: {e.errors()[0]['msg']}"
        nxt, dispatch, reason = decision.next, decision.dispatch, decision.reason
        q = view.get("quality_result")
        if q is not None and not q.passed and not view.get("quality_pending") and nxt not in ("report", "render_pdf"):
            updates["report_stale"] = True  # 품질 미달로 상류 재작업 → 끝나면 보고서를 다시 써야 함
        if nxt == "render_pdf":  # 보고서 생성과 품질 검수 통과를 구분해 남긴다
            passed = q is not None and q.passed and not view.get("report_stale") and not view.get("synthesis_stale")
            updates["run_status"] = "succeeded" if passed else "incomplete"
            updates["end_reason"] = reason
        if nxt == "counter_evidence":
            retry["counter"] = retry.get("counter", 0) + 1
        for node, techs_items in dispatch.items():  # 배정 표시 (재개 시 어디까지 갔는지 판단)
            for tid in techs_items:
                status.setdefault(f"{NODE_PERSPECTIVE[node]}:{tid}", "assigned")

        logger.info("[trace=%s] supervisor #%d → %s | %s", state.get("trace_id"), step, nxt, reason)
        return {
            **updates,
            "next": nxt,
            "dispatch": dispatch,
            "decision_reason": reason,
            "step_count": step,
            "retry_counts": retry,
            "missing_criteria": missing,
            "node_status": status,
        }

    def route_from_supervisor(state: GraphState) -> str | list[Send]:
        """supervisor가 State에 남긴 결정대로 분기한다. 평가 에이전트는 기술별 Send로 보낸다."""
        nxt = state["next"]
        if nxt not in ("tech_research", "perspectives"):
            return nxt
        sends: list[Send] = []
        for node, techs_items in state.get("dispatch", {}).items():
            for tid, items in techs_items.items():
                sends.append(_send_for(state, node, tid, items))
        if not sends:  # _decide가 빈 배정을 만들 수 없지만, 만약을 대비해 보고서로 진행 (그래프가 조용히 끝나지 않게)
            logger.error("빈 배정 — 보고서 작성으로 진행")
            return "report"
        return sends

    def _send_for(state: GraphState, nxt: str, tid: str, items: list[str]) -> Send:
        p = NODE_PERSPECTIVE[nxt]
        tech = _tech_by_id(state, tid)
        if not items:  # 첫 실행
            return Send(nxt, agent_payload(state, tech, p))
        criteria = [i for i in items if not i.startswith("PROFILE:")] or None
        extra: dict[str, Any] = {"missing_criteria": criteria}
        if p == "trl":
            profiles = {pr.tech_id: pr for pr in latest_by_tech(state.get("tech_profiles", []))}
            prev_q = profiles[tid].search_queries_used if tid in profiles else []
            retry_n = state["retry_counts"].get(f"trl:{tid}", 0)
            try:
                extra["rewritten_queries"] = rewrite_queries(tech, items, prev_q, retry_count=retry_n, llm=rewrite_llm)
            except Exception:
                logger.exception("rewrite_queries LLM 실패(%s) — 결정적 검색어로 대체", tid)
                extra["rewritten_queries"] = rewrite_queries(tech, items, prev_q, retry_count=retry_n, llm=None)
        else:
            extra["previous_results"] = [
                r for r in latest_by_criterion(state.get(PERSPECTIVE_KEY[p], [])) if r.tech_id == tid
            ]
        return Send(nxt, agent_payload(state, tech, p, **extra))

    # ================================================================ 하위 에이전트

    def _guard(key_of: Callable[[dict], str], name: str):
        """하위 에이전트 예외를 그래프 중단 대신 node_status/last_error로 기록 (fallback = 재시도 → not_public)."""

        def deco(fn: Callable[[dict, Deps], dict]) -> Callable[[dict], dict]:
            @functools.wraps(fn)
            def wrapper(payload: dict) -> dict:
                key = key_of(payload)
                logger.info("▶ %s (%s)", name, key)
                # 실제 실행한 검색어(웹·논문)를 기록하는 Deps — 상한 도달 시 not_public 기록에 그대로 쓴다
                queries: list[str] = []
                base = deps_for(name)

                def web_search(query: str, **kwargs: Any) -> list[Any]:
                    queries.append(query)
                    return base.web_search(query, **kwargs)

                tracked = base.model_copy(
                    update={"web_search": web_search, "retriever": _QueryRecordingRetriever(base.retriever, queries)}
                )
                log = {"search_log": {key: queries}}
                try:
                    out = fn(payload, tracked)
                except Exception as e:
                    logger.exception("%s(%s) 실패 — supervisor가 재작업/확정을 결정", name, key)
                    return {**log, "node_status": {key: "error"}, "last_error": f"{key}: {type(e).__name__}: {e}"}
                logger.info("◀ %s (%s)", name, key)
                return {**out, **log, "node_status": {key: "done"}}

            return wrapper

        return deco

    @_guard(lambda p: f"trl:{p['tech'].tech_id}", "tech_research")
    def tech_research(payload: dict, d: Deps) -> dict:
        inp = AgentInput.model_validate(payload)
        out = agents.run_tech_research(inp, d)
        out = TechResearchOutput.model_validate(out if isinstance(out, dict) else out.model_dump())
        return {"tech_profiles": [out.tech_profile], "trl_eval": out.trl_eval}

    def _perspective_node(name: str, key: str, fn_name: str):
        @_guard(lambda p: f"{name}:{p['tech'].tech_id}", name)
        def node(payload: dict, d: Deps) -> dict:
            inp = AgentInput.model_validate(payload)
            results = getattr(agents, fn_name)(inp, d)
            results = [CriterionResult.model_validate(r if isinstance(r, dict) else r.model_dump()) for r in results]
            return {key: results}

        node.__name__ = key
        return node

    market_eval = _perspective_node("market", "market_eval", "run_market_eval")
    stakeholder_eval = _perspective_node("stakeholder", "stakeholder_eval", "run_stakeholder_eval")
    domain_eval = _perspective_node("domain", "domain_eval", "run_domain_eval")

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
        from techeval.schemas import SynthesisResult

        try:
            out = agents.run_synthesis(synthesis_input(state), deps_for("synthesis"))
        except Exception:  # 일시 오류 대비 1회 재시도 (종합이 없으면 보고서를 만들 수 없음)
            logger.exception("synthesis 실패 — 1회 재시도")
            out = agents.run_synthesis(synthesis_input(state), deps_for("synthesis"))
        return {"synthesis": SynthesisResult.model_validate(out if isinstance(out, dict) else out.model_dump())}

    @_log_node
    def counter_evidence(state: GraphState) -> dict:
        try:
            evs = search_counter_evidence(state["evidence_gap"], deps, technologies=state["technologies"])
        except Exception as e:  # 반대 근거 탐색 실패는 보고서 한계점으로 남기고 진행 (재시도 횟수는 이미 소진 처리됨)
            logger.exception("counter_evidence 실패 — 반대 근거 없이 진행")
            return {"last_error": f"counter_evidence: {type(e).__name__}: {e}"}
        return {"counter_evidence": [*state.get("counter_evidence", []), *evs]}

    # ================================================================ 보고서 · 품질 평가

    def report_input(state: GraphState) -> ReportInput:
        si = synthesis_input(state)
        gap = state.get("evidence_gap") or check_evidence_gap(
            state["technologies"], _latest_evals(state), state.get("synthesis"), state.get("counter_evidence", [])
        )
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
            evidence_gap=gap,
            judge_result=state.get("judge_result"),
            previous_report_md=state.get("report_md"),
        )

    @_log_node
    def report(state: GraphState) -> dict:
        retry = dict(state["retry_counts"])
        if state.get("judge_result") is not None:  # 품질 평가 미달 → 재생성
            retry["report"] = retry.get("report", 0) + 1
            logger.info("report: 재생성 %d/%d", retry["report"], MAX_REPORT_REGENERATION)
        md = agents.run_report(report_input(state), deps_for("report"))
        if not isinstance(md, str) or not md.strip():
            raise ValueError("run_report must return non-empty markdown")
        return {"report_md": md, "retry_counts": retry}

    @_log_node
    def quality_eval(state: GraphState) -> dict:
        judge_llm = deps.judge_llm
        counter_attempts = state["retry_counts"].get("counter", 0)
        if judge_llm is None:
            raise RuntimeError("deps.judge_llm (JUDGE_MODEL) 이 필요합니다 — AGENTS.md 규칙 9")
        try:
            quality, judge = evaluate_report_quality(
                state["report_md"], report_input(state), judge_llm, deps.now(), counter_attempts=counter_attempts
            )
        except Exception:  # Judge LLM 일시 오류 대비 1회 재시도, 그래도 실패하면 평가 없이 출력 (보고서는 남긴다)
            logger.exception("quality_eval 실패 — 1회 재시도")
            try:
                quality, judge = evaluate_report_quality(
                    state["report_md"], report_input(state), judge_llm, deps.now(), counter_attempts=counter_attempts
                )
            except Exception as e:
                logger.exception("quality_eval 재시도 실패 — 평가 없이 PDF로 진행")
                return {"last_error": f"quality_eval: {type(e).__name__}: {e}", "quality_result": None}
        failed = [c for c, chk in quality.checks.items() if not chk.passed]
        logger.info(
            "[trace=%s] quality_eval passed=%s 미달=%s 기존 judge=%s 원인=%s",
            state.get("trace_id"),
            quality.passed,
            failed,
            quality.legacy_judge_passed,
            sorted({i.cause for i in quality.issues}),
        )
        return {"quality_result": quality, "judge_result": judge}

    @_log_node
    def render_pdf(state: GraphState) -> dict:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(state["report_md"], encoding="utf-8")
        if cfg.skip_pdf:
            return {"report_pdf_path": ""}
        return {"report_pdf_path": agents.render_pdf(state["report_md"], str(out_dir / "report.pdf"))}

    # ================================================================ graph

    g = StateGraph(GraphState)
    g.add_node("init", init)
    g.add_node("supervisor", supervisor)
    g.add_node("tech_research", tech_research)
    g.add_node("market_eval", market_eval)
    g.add_node("stakeholder_eval", stakeholder_eval)
    g.add_node("domain_eval", domain_eval)
    g.add_node("synthesis", synthesis)
    g.add_node("counter_evidence", counter_evidence)
    g.add_node("report", report)
    g.add_node("quality_eval", quality_eval)
    g.add_node("render_pdf", render_pdf)

    g.add_edge(START, "init")
    g.add_edge("init", "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        [
            "tech_research",
            "market_eval",
            "stakeholder_eval",
            "domain_eval",
            "counter_evidence",
            "synthesis",
            "report",
            "quality_eval",
            "render_pdf",
        ],
    )
    # 하위 에이전트·보고서·품질 평가는 항상 supervisor로만 복귀 (하위 에이전트 간 직접 통신 없음)
    for n in (
        "tech_research",
        "market_eval",
        "stakeholder_eval",
        "domain_eval",
        "counter_evidence",
        "synthesis",
        "report",
        "quality_eval",
    ):
        g.add_edge(n, "supervisor")
    g.add_edge("render_pdf", END)

    compiled = g.compile()
    compiled.registry = registry  # type: ignore[attr-defined]  # 테스트·run.py에서 검색 호출 기록 확인용
    return compiled


def invoke_config(cfg: GraphConfig | None = None, trace_id: str | None = None) -> dict:
    """recursion_limit + LangSmith 상관 메타데이터(trace_id)."""
    out: dict[str, Any] = {"recursion_limit": (cfg or GraphConfig()).recursion_limit}
    if trace_id:
        out["metadata"] = {"trace_id": trace_id}
        out["run_name"] = f"TechEvalAgent-supervisor-{trace_id}"
    return out


def state_to_json(state: dict) -> str:
    """State 덤프용 JSON (Pydantic 객체 → dict)."""

    def default(o: Any) -> Any:
        if isinstance(o, BaseModel):
            return o.model_dump()
        if isinstance(o, tuple | set):
            return list(o)
        return str(o)

    return json.dumps(state, ensure_ascii=False, indent=2, default=default)
