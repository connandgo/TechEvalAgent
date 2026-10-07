"""스텁 deps로 그래프 흐름·분기·상한을 검증한다 (역할 E 문서 §6 (a)~(f)).

에이전트는 `techeval.stub_agents`를 감싼 카운팅 래퍼를 쓰고, 검색기·웹검색은 메모리 픽스처를 쓴다.
시나리오별로 특정 에이전트의 출력을 고의로 깨뜨려 제어 노드가 정확히 그 부분만 재실행하는지 본다.
"""

from techeval import stub_agents
from techeval.graph import GraphConfig, build_graph, invoke_config
from techeval.schemas import MAX_RETRY_PER_PERSPECTIVE, JudgeResult, TechRef
from techeval.state import latest_by_criterion
from tests.graph_harness import Harness
from tests.graph_harness import levels as _levels

# --- (a) 정상 경로 ---------------------------------------------------------------


def test_a_normal_path_reaches_end(h: Harness):
    final, nodes = h.run()
    assert final["report_md"] and (h.cfg.output_dir + "/report.md")
    assert final["judge_result"].passed is True
    assert all(v == 0 for v in final["retry_counts"].values())
    assert nodes["tech_research"] == 2 and nodes["market_eval"] == 2
    assert nodes["stakeholder_eval"] == 2 and nodes["domain_eval"] == 2
    assert nodes["supervisor"] >= 1 and nodes["synthesis"] == 1 and nodes["report"] == 1
    assert nodes["judge"] == 1 and nodes["render_pdf"] == 1
    # 정상 경로에서는 재작업(missing_criteria)이 한 번도 나가지 않는다
    assert all(i.missing_criteria is None for n in ("run_tech_research", "run_market_eval") for i in h.inputs[n])
    assert nodes["counter_evidence"] == 0
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

    # query_rewrite 는 supervisor 내부 호출 → rewritten_queries 가 실린 재작업 tech_research 호출 수로 확인
    assert sum(1 for i in h.inputs["run_tech_research"] if i.rewritten_queries) == MAX_RETRY_PER_PERSPECTIVE
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
    assert final["node_status"]["trl:mla"] == "exhausted"
    assert final["judge_result"] is not None and nodes["render_pdf"] == 1


# --- (c) M2 누락 → market_eval만 재실행 --------------------------------------------


def test_c_missing_m2_reruns_only_market(h: Harness):
    def drop_m2_once(inp, out, n):
        if inp.tech.tech_id == "pim_cxl" and inp.missing_criteria is None:
            return [r for r in out if r.criterion_id != "M2"]
        return out

    h.overrides["run_market_eval"] = drop_m2_once
    final, nodes = h.run()

    assert nodes["market_eval"] == 3
    assert nodes["stakeholder_eval"] == 2 and nodes["domain_eval"] == 2 and nodes["tech_research"] == 2
    assert not any(i.rewritten_queries for i in h.inputs["run_tech_research"])  # 쿼리 재작성은 기술 조사 전용
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
    assert nodes["counter_evidence"] == 1
    assert nodes["synthesis"] == 2  # 반대 근거 반영을 위해 재종합
    # counter_evidence 직후 synthesis 가 supervisor 를 거쳐 다시 돈다
    assert h.order[h.order.index("counter_evidence") + 1] == "supervisor"
    assert h.order[h.order.index("counter_evidence") + 2] == "synthesis"
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
    instr = second.judge_result.revision_instructions
    assert second.judge_result is not None and instr
    assert any("각주" in i or "근거" in i for i in instr)  # groundedness 관련 지시
    assert second.previous_report_md == h.inputs["run_report"][0].previous_report_md or second.previous_report_md


# --- (f) operator.add 중복이 D 입력에서 제거됨 ---------------------------------------


def test_f_duplicates_removed_before_synthesis(h: Harness):
    def drop_t3_but_return_everything(inp, out, n):
        """missing_criteria를 무시하고 항상 T1/T2/T4를 다시 돌려주는(계약 위반) 에이전트를 흉내 낸다."""
        if inp.tech.tech_id == "mla":
            full = stub_agents.run_tech_research(inp.model_copy(update={"missing_criteria": None}), h.deps)
            out.trl_eval = [r for r in full.trl_eval if r.criterion_id != "T3"]
        return out

    h.overrides["run_tech_research"] = drop_t3_but_return_everything
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


# --- Supervisor 패턴 --------------------------------------------------------------

SUB_AGENTS = ("tech_research", "market_eval", "stakeholder_eval", "domain_eval", "synthesis", "counter_evidence")


def test_supervisor_is_hub(h: Harness):
    graph = build_graph(h.deps, h.agents(), h.cfg)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    for n in SUB_AGENTS:
        outgoing = {t for s, t in edges if s == n}
        assert outgoing == {"supervisor"}, (n, outgoing)
    assert not [(s, t) for s, t in edges if s in SUB_AGENTS and t in SUB_AGENTS]
    # supervisor 는 모든 하위 에이전트로 갈 수 있다
    assert {t for s, t in edges if s == "supervisor"} >= set(SUB_AGENTS)


def test_routing_follows_state_not_fixed_order(h: Harness):
    final, nodes = h.run()
    assert isinstance(final["decision_reason"], str) and final["decision_reason"].strip()
    assert final["step_count"] >= 1
    assert final["step_count"] == nodes["supervisor"]
    # 하위 에이전트 실행 묶음(같은 슈퍼스텝의 Send 병렬 포함)마다 직후에 supervisor 가 한 번 돈다
    order = h.order
    batches: list[set[str]] = []
    i = 0
    while i < len(order):
        if order[i] in SUB_AGENTS:
            j = i
            while j < len(order) and order[j] in SUB_AGENTS:
                j += 1
            batches.append(set(order[i:j]))
            assert order[j] == "supervisor", (order[i:j], order[j])
            i = j
        else:
            i += 1
    assert nodes["supervisor"] == len(batches) + 1
    # 시장·이해관계자·도메인은 서로 의존이 없어 State상 대기 중이면 한 번의 판단으로 함께 배정된다 (순서 고정 아님)
    assert {"market_eval", "stakeholder_eval", "domain_eval"} in batches
    assert order[:2] == ["init", "supervisor"] and order[-3:] == ["report", "judge", "render_pdf"]


def test_recursion_limit_is_configured():
    assert invoke_config(GraphConfig(recursion_limit=7)) == {"recursion_limit": 7}
