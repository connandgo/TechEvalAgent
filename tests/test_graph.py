"""스텁 deps로 그래프 흐름·분기·상한을 검증한다 (역할 E 문서 §6 (a)~(f)).

에이전트는 `techeval.stub_agents`를 감싼 카운팅 래퍼를 쓰고, 검색기·웹검색은 메모리 픽스처를 쓴다.
시나리오별로 특정 에이전트의 출력을 고의로 깨뜨려 제어 노드가 정확히 그 부분만 재실행하는지 본다.
"""

from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest

from techeval import stub_agents
from techeval.agents._deps import Deps
from techeval.graph import Agents, GraphConfig, build_graph, invoke_config
from techeval.schemas import MAX_RETRY_PER_PERSPECTIVE, JudgeResult, TechRef
from techeval.state import latest_by_criterion
from techeval.stub_llm import FakeStructuredLLM
from tests.helpers import MemRetriever, make_chunks, make_web_search

# --- 카운팅 에이전트 ---------------------------------------------------------------


class Harness:
    """stub_agents를 감싸 호출 횟수·입력을 기록하고, 시나리오별 override를 끼운다."""

    def __init__(self, tmp_path):
        self.calls: Counter[str] = Counter()
        self.inputs: dict[str, list[Any]] = {}
        self.overrides: dict[str, Callable[..., Any]] = {}
        self.retriever = MemRetriever(make_chunks())
        self.web_search = make_web_search()
        self.judge_llm = FakeStructuredLLM()
        clock = {"n": 0}

        def now() -> str:  # 호출마다 1초씩 증가하는 올바른 ISO 시각 (문자열 비교로 최신 판별 가능해야 함)
            clock["n"] += 1
            n = clock["n"]
            return f"2026-01-01T{n // 3600:02d}:{n // 60 % 60:02d}:{n % 60:02d}"

        self.deps = Deps(
            retriever=self.retriever,
            web_search=self.web_search,
            llm=FakeStructuredLLM(),
            judge_llm=self.judge_llm,
            now=now,
        )
        self.cfg = GraphConfig(output_dir=str(tmp_path), skip_pdf=True, stub=True)

    def _wrap(self, name: str):
        base = getattr(stub_agents, name)

        def fn(*args: Any):
            self.calls[name] += 1
            self.inputs.setdefault(name, []).append(args[0])
            out = base(*args)
            if name in self.overrides:
                out = self.overrides[name](args[0], out, self.calls[name])
            return out

        return fn

    def agents(self) -> Agents:
        return Agents(
            run_tech_research=self._wrap("run_tech_research"),
            run_domain_eval=self._wrap("run_domain_eval"),
            run_market_eval=self._wrap("run_market_eval"),
            run_stakeholder_eval=self._wrap("run_stakeholder_eval"),
            run_synthesis=self._wrap("run_synthesis"),
            run_report=self._wrap("run_report"),
            render_pdf=stub_agents.render_pdf,
        )

    def run(self) -> tuple[dict, Counter]:
        graph = build_graph(self.deps, self.agents(), self.cfg)
        node_counts: Counter[str] = Counter()
        final: dict = {}
        for mode, chunk in graph.stream({}, config=invoke_config(self.cfg), stream_mode=["updates", "values"]):
            if mode == "updates":
                for node in chunk:
                    node_counts[node] += 1
            else:
                final = chunk
        return final, node_counts


@pytest.fixture
def h(tmp_path) -> Harness:
    # stub_agents가 B/C/D 픽스처를 읽지 않도록 (있으면 픽스처 chunk_id가 MemRetriever와 안 맞는다)
    stub_agents.FIXTURES_DIR = tmp_path / "no_fixtures"
    yield Harness(tmp_path)
    stub_agents.FIXTURES_DIR = stub_agents.DEFAULT_FIXTURES_DIR


def _levels(state: dict, key: str, tech: str) -> dict[str, str]:
    return {r.criterion_id: r.level for r in latest_by_criterion(state[key]) if r.tech_id == tech}


# --- (a) 정상 경로 ---------------------------------------------------------------


def test_a_normal_path_reaches_end(h: Harness):
    final, nodes = h.run()
    assert final["report_md"] and (h.cfg.output_dir + "/report.md")
    assert final["judge_result"].passed is True
    assert all(v == 0 for v in final["retry_counts"].values())
    assert nodes["tech_research"] == 2 and nodes["market_eval"] == 2
    assert nodes["stakeholder_eval"] == 2 and nodes["domain_eval"] == 2
    assert nodes["perspective_check"] == 1 and nodes["synthesis"] == 1 and nodes["report"] == 1
    assert nodes["query_rewrite"] == 0 and nodes["counter_evidence_search"] == 0
    # 30개 (tech, criterion) 결과, 전부 실제 검색 결과 근거 (not_public 없음)
    results = [
        r for k in ("trl_eval", "market_eval", "stakeholder_eval", "domain_eval") for r in latest_by_criterion(final[k])
    ]
    assert len(results) == 30
    assert not [r for r in results if r.level == "not_public"]
    # 에이전트에 넘어간 입력은 AgentInput 규약을 따른다
    for inp in h.inputs["run_market_eval"]:
        assert isinstance(inp.tech, TechRef) and inp.tech_profile is not None and inp.missing_criteria is None


# --- (b) T3 누락 → query_rewrite 경유 후 2회 상한에서 진행 ------------------------------


def test_b_missing_t3_retries_then_not_public(h: Harness):
    def drop_t3(inp, out, n):
        if inp.tech.tech_id == "mla":
            out.trl_eval = [r for r in out.trl_eval if r.criterion_id != "T3"]
        return out

    h.overrides["run_tech_research"] = drop_t3
    final, nodes = h.run()

    assert nodes["query_rewrite"] == MAX_RETRY_PER_PERSPECTIVE
    assert nodes["tech_evidence_check"] == MAX_RETRY_PER_PERSPECTIVE + 1
    # mla만 재검색: 초기 2 + 재시도 2
    assert h.calls["run_tech_research"] == 2 + MAX_RETRY_PER_PERSPECTIVE
    assert [i.tech.tech_id for i in h.inputs["run_tech_research"][2:]] == ["mla"] * MAX_RETRY_PER_PERSPECTIVE
    # 재시도 입력에는 missing_criteria 와 새 검색어가 실린다
    retry_inputs = h.inputs["run_tech_research"][2:]
    assert all(i.missing_criteria == ["T3"] and i.rewritten_queries for i in retry_inputs)
    assert retry_inputs[0].retry_count == 1 and retry_inputs[1].retry_count == 2
    assert retry_inputs[0].rewritten_queries != retry_inputs[1].rewritten_queries
    # 상한 도달 → not_public 확정 후 진행, 나머지는 정상
    assert final["retry_counts"]["trl:mla"] == MAX_RETRY_PER_PERSPECTIVE
    assert _levels(final, "trl_eval", "mla")["T3"] == "not_public"
    assert _levels(final, "trl_eval", "pim_cxl")["T3"] != "not_public"
    assert final["missing_criteria"]["trl:mla"] == []
    assert final["judge_result"] is not None and nodes["render_pdf"] == 1


# --- (c) M2 누락 → market_eval만 재실행 --------------------------------------------


def test_c_missing_m2_reruns_only_market(h: Harness):
    def drop_m2_once(inp, out, n):
        if inp.tech.tech_id == "pim_cxl" and inp.missing_criteria is None:
            return [r for r in out if r.criterion_id != "M2"]
        return out

    h.overrides["run_market_eval"] = drop_m2_once
    final, nodes = h.run()

    assert nodes["perspective_check"] == 2
    assert nodes["market_eval"] == 3
    assert nodes["stakeholder_eval"] == 2 and nodes["domain_eval"] == 2 and nodes["tech_research"] == 2
    assert nodes["query_rewrite"] == 0
    rerun = h.inputs["run_market_eval"][2]
    assert rerun.tech.tech_id == "pim_cxl" and rerun.missing_criteria == ["M2"] and rerun.retry_count == 1
    assert {r.criterion_id for r in rerun.previous_results} == {"M1", "M3"}
    assert final["retry_counts"]["market:pim_cxl"] == 1
    assert _levels(final, "market_eval", "pim_cxl")["M2"] != "not_public"
    assert len(latest_by_criterion(final["market_eval"])) == 6


# --- (d) 비대칭 → counter_evidence_search 1회만 -------------------------------------


def test_d_asymmetry_triggers_single_counter_search(h: Harness):
    def thin_pim(inp, out, n):
        if inp.tech.tech_id == "pim_cxl":
            for r in out:
                r.evidence = r.evidence[:1]
        return out

    def fat_mla(inp, out, n):
        if inp.tech.tech_id == "mla":
            for r in out:
                extra = [
                    e.model_copy(update={"evidence_id": f"{e.evidence_id}-x{i}"}) for i, e in enumerate(r.evidence)
                ]
                r.evidence = [*r.evidence, *extra, *extra]
        return out

    for name in ("run_market_eval", "run_stakeholder_eval", "run_domain_eval"):
        h.overrides[name] = fat_mla if name != "run_domain_eval" else thin_pim
    final, nodes = h.run()

    # 최종 evidence_gap 은 반대 근거 탐색 뒤의 재검사 결과 → 이미 탐색한 기술은 다시 요청하지 않는다
    gap = final["evidence_gap"]
    assert gap.needs_counter_search is False
    assert nodes["counter_evidence_search"] == 1
    assert nodes["synthesis"] == 2 and nodes["evidence_gap_check"] == 2
    assert final["retry_counts"]["counter"] == 1
    assert final["counter_evidence"] and all("-COUNTER-" in e.evidence_id for e in final["counter_evidence"])
    assert all(e.unit == "family" for e in final["counter_evidence"])
    # 두 번째 synthesis 입력에 반대 근거가 들어간다
    assert h.inputs["run_synthesis"][1].counter_evidence == final["counter_evidence"]
    assert nodes["render_pdf"] == 1


# --- (e) judge 실패 → report 1회만 재실행 ------------------------------------------


def test_e_judge_failure_regenerates_once(h: Harness):
    def failing(text: str) -> JudgeResult:
        return JudgeResult(
            scores={"evidence": 1, "criteria_compliance": 3, "neutrality": 3, "specificity": 3, "format": 3},
            reasons={"evidence": "테스트용 실패"},
            missing_required=[],
            revision_instructions=["근거 각주를 보강하라"],
            passed=False,
            judged_at="2026-01-01T00:00:00",
        )

    h.judge_llm.register(JudgeResult, failing)
    final, nodes = h.run()

    assert nodes["report"] == 2 and nodes["judge"] == 2 and nodes["render_pdf"] == 1
    assert final["retry_counts"]["report"] == 1
    assert final["judge_result"].passed is False  # 상한 도달 후 그대로 출력
    second = h.inputs["run_report"][1]
    assert second.judge_result is not None and second.judge_result.revision_instructions == ["근거 각주를 보강하라"]
    assert second.previous_report_md == h.inputs["run_report"][0].previous_report_md or second.previous_report_md


# --- (f) operator.add 중복이 D 입력에서 제거됨 ---------------------------------------


def test_f_duplicates_removed_before_synthesis(h: Harness):
    def drop_t3(inp, out, n):
        if inp.tech.tech_id == "mla":
            out.trl_eval = [r for r in out.trl_eval if r.criterion_id != "T3"]
        return out

    h.overrides["run_tech_research"] = drop_t3
    final, _ = h.run()

    # State에는 재시도분이 누적된다 (mla T1/T2/T4 × 3회 + pim_cxl 4 + not_public 1)
    assert len(final["trl_eval"]) > 8
    assert len(final["tech_profiles"]) > 2
    # D가 받는 입력은 (tech, criterion) 당 1개, 프로필도 기술당 1개
    si = h.inputs["run_synthesis"][0]
    assert len(si.trl_eval) == 8 and len({(r.tech_id, r.criterion_id) for r in si.trl_eval}) == 8
    assert [p.tech_id for p in si.tech_profiles] == ["mla", "pim_cxl"]
    ri = h.inputs["run_report"][0]
    assert len(ri.trl_eval) == 8 and len(ri.tech_profiles) == 2
    # 최신(generated_at 최대) 것이 선택된다
    newest = max(
        (r for r in final["trl_eval"] if r.tech_id == "mla" and r.criterion_id == "T1"), key=lambda r: r.generated_at
    )
    chosen = next(r for r in si.trl_eval if r.tech_id == "mla" and r.criterion_id == "T1")
    assert chosen.generated_at == newest.generated_at


# --- 근거 실존 검사 (V5/V6) ----------------------------------------------------------


def test_fabricated_chunk_is_rejected_and_reran(h: Harness):
    """존재하지 않는 chunk_id를 단 근거는 검사 노드가 잡아 재실행을 건다."""

    faked = []

    def fake_chunk(inp, out, n):
        if inp.tech.tech_id == "mla" and not faked:
            faked.append(True)
            for r in out:
                r.evidence[0] = r.evidence[0].model_copy(update={"chunk_id": "deepseek_v2:999:99"})
        return out

    h.overrides["run_domain_eval"] = fake_chunk
    final, nodes = h.run()
    assert nodes["domain_eval"] == 3
    rerun = h.inputs["run_domain_eval"][2]
    assert rerun.tech.tech_id == "mla" and set(rerun.missing_criteria) == {"D1", "D2", "D3", "D4"}
    assert not [r for r in latest_by_criterion(final["domain_eval"]) if r.level == "not_public"]


def test_recursion_limit_is_configured():
    assert invoke_config(GraphConfig(recursion_limit=7)) == {"recursion_limit": 7}
