"""품질 평가 미달 → supervisor가 원인별 재작업 위치를 고르는지, 실행 상태(run_status)가 남는지 검증."""

from techeval.schemas import JudgeResult
from tests.graph_harness import Harness

# --- 품질 미달 → supervisor가 원인별 재작업 위치 선택 ---------------------------------


def _fail_quality_once(monkeypatch, *issues: dict) -> list[int]:
    """첫 품질 평가만 지정한 원인으로 탈락시키고, 이후는 실제 평가를 쓴다."""
    import techeval.graph as graph_mod
    from techeval.schemas import QualityIssue

    real = graph_mod.evaluate_report_quality
    calls = [0]

    def fake(*args, **kwargs):
        calls[0] += 1
        quality, judge = real(*args, **kwargs)
        if calls[0] == 1:
            quality = quality.model_copy(
                update={"passed": False, "issues": [QualityIssue(**i) for i in issues], "revision_instructions": ["x"]}
            )
            judge = judge.model_copy(update={"passed": False, "revision_instructions": ["x"]})
        return quality, judge

    monkeypatch.setattr(graph_mod, "evaluate_report_quality", fake)
    return calls


def _after_first_quality(order: list[str]) -> list[str]:
    i = order.index("quality_eval")
    return order[i + 1 :]


def test_quality_counter_missing_routes_to_counter_evidence(h: Harness, monkeypatch):
    _fail_quality_once(monkeypatch, {"criterion": "bias_control", "cause": "counter_missing", "tech_id": "mla"})
    final, _nodes = h.run()
    rest = [n for n in _after_first_quality(h.order) if n != "supervisor"]
    # 반대 근거 탐색 → 재종합 → 보고서 재작성 → 재평가 → 출력
    assert rest == ["counter_evidence", "synthesis", "report", "quality_eval", "render_pdf"]
    assert final["retry_counts"]["counter"] == 1 and final["retry_counts"]["report"] == 1
    assert final["quality_result"].passed is True


def test_quality_missing_result_routes_to_perspective_agent(h: Harness, monkeypatch):
    _fail_quality_once(
        monkeypatch,
        {"criterion": "perspective_coverage", "cause": "missing_result", "tech_id": "mla", "criterion_id": "M2"},
    )
    _final, _nodes = h.run()
    rest = [n for n in _after_first_quality(h.order) if n != "supervisor"]
    assert rest == ["market_eval", "synthesis", "report", "quality_eval", "render_pdf"]
    rerun = h.inputs["run_market_eval"][-1]
    assert rerun.tech.tech_id == "mla" and rerun.missing_criteria == ["M2"]


def test_quality_wording_problem_routes_to_report(h: Harness, monkeypatch):
    _fail_quality_once(monkeypatch, {"criterion": "neutrality", "cause": "banned_term"})
    _final, nodes = h.run()
    rest = [n for n in _after_first_quality(h.order) if n != "supervisor"]
    assert rest == ["report", "quality_eval", "render_pdf"]
    assert nodes["synthesis"] == 1 and nodes.get("counter_evidence", 0) == 0


def test_quality_counter_exhausted_falls_back_to_report(h: Harness, monkeypatch):
    import techeval.graph as graph_mod

    monkeypatch.setattr(graph_mod, "MAX_COUNTER_EVIDENCE_SEARCH", 0)
    _fail_quality_once(monkeypatch, {"criterion": "bias_control", "cause": "asymmetry", "tech_id": "pim_cxl"})
    _final, _nodes = h.run()
    rest = [n for n in _after_first_quality(h.order) if n != "supervisor"]
    assert rest == ["report", "quality_eval", "render_pdf"]


# --- 출력 시점 실행 상태 ---------------------------------------------------------------


def test_judge_error_marks_run_incomplete(h: Harness):
    """Judge가 계속 예외를 내도 그래프는 끝나고, 보고서는 남기되 검수 통과로 표시하지 않는다."""

    def boom(_text):
        raise RuntimeError("judge backend down")

    h.judge_llm.register(JudgeResult, boom)
    final, nodes = h.run()
    assert nodes["render_pdf"] == 1 and final["report_md"]  # 보고서 보존
    assert final.get("quality_result") is None
    assert final["run_status"] == "incomplete" and "검수 실행 오류" in final["end_reason"]
    assert final["last_error"].startswith("quality_eval")


def test_normal_run_marks_succeeded(h: Harness):
    final, _ = h.run()
    assert final["run_status"] == "succeeded" and "통과" in final["end_reason"]
