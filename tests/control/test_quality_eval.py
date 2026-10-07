"""보고서 품질 평가(quality_eval) 단위 테스트. 네트워크 없음 — FakeStructuredLLM(judge_result.json 픽스처)."""

import json

import pytest

from techeval.agents._deps import ReportInput
from techeval.control.quality_eval import evaluate_report_quality
from techeval.control.sources import SourceRegistry
from techeval.schemas import QUALITY_CRITERIA, EvidenceGap, JudgeResult
from techeval.stub_llm import DEFAULT_FIXTURES_DIR, FakeStructuredLLM
from tests.control.test_checks import GOOD_MD, _report_input, _retriever
from tests.helpers import NOW


def _inp(**gap_overrides) -> ReportInput:
    inp = _report_input(_retriever(), SourceRegistry())
    if gap_overrides:
        gap = EvidenceGap.model_validate({**inp.evidence_gap.model_dump(), **gap_overrides})
        inp = inp.model_copy(update={"evidence_gap": gap})
    return inp


def _failed(quality) -> set[str]:
    return {k for k, c in quality.checks.items() if not c.passed}


def test_passing_report():
    quality, judge = evaluate_report_quality(GOOD_MD, _inp(), FakeStructuredLLM(), NOW)
    assert set(quality.checks) == set(QUALITY_CRITERIA)
    assert quality.passed is True and _failed(quality) == set()
    assert quality.revision_instructions == [] and quality.evaluated_at == NOW
    assert quality.checks["groundedness"].method == "hybrid"
    assert quality.checks["bias_control"].method == "code"
    assert judge.passed is True and judge.revision_instructions == []
    assert judge.scores["evidence"] == 5  # 원 judge 점수 유지


def test_forbidden_word_fails_neutrality():
    md = GOOD_MD.replace("요약", "MLA가 더 우수하다")
    quality, judge = evaluate_report_quality(md, _inp(), FakeStructuredLLM(), NOW)
    assert _failed(quality) == {"neutrality"} and quality.passed is False
    assert any("우열" in i for i in quality.revision_instructions)
    assert judge.passed is False and judge.revision_instructions == quality.revision_instructions


def test_unknown_citation_fails_groundedness():
    md = GOOD_MD.replace("요약 [E: mla-T1-01]", "요약 [E: mla-T1-01] [E: ghost-X1-01]")
    quality, _ = evaluate_report_quality(md, _inp(), FakeStructuredLLM(), NOW)
    assert "groundedness" in _failed(quality)
    assert "ghost-X1-01" in quality.checks["groundedness"].detail
    assert any("ghost-X1-01" in i for i in quality.revision_instructions)


def test_missing_section_fails_coverage():
    md = GOOD_MD.replace("### 4.3 이해관계자\n", "")
    quality, _ = evaluate_report_quality(md, _inp(), FakeStructuredLLM(), NOW)
    assert "perspective_coverage" in _failed(quality)
    assert "4.3" in quality.checks["perspective_coverage"].detail
    assert any("4.3" in i for i in quality.revision_instructions)


def _causes(quality) -> set[str]:
    return {i.cause for i in quality.issues}


def _with_limits(text: str) -> str:
    return GOOD_MD.replace("## 6. 한계점\n", f"## 6. 한계점\n{text}\n\n")


def _judge_llm(**scores) -> FakeStructuredLLM:
    """judge_result.json 픽스처에서 일부 차원 점수만 바꾼 Judge LLM."""
    raw = json.loads((DEFAULT_FIXTURES_DIR / "judge_result.json").read_text(encoding="utf-8"))
    raw["scores"].update(scores)
    return FakeStructuredLLM(overrides={JudgeResult: lambda _p: raw})


def test_legacy_judge_failure_is_not_overridden():
    """기존 Judge가 specificity·criteria_compliance로 탈락시키면 4개 항목이 통과해도 최종 탈락."""
    for dim in ("specificity", "criteria_compliance"):
        quality, judge = evaluate_report_quality(GOOD_MD, _inp(), _judge_llm(**{dim: 1}), NOW)
        assert _failed(quality) == set()  # 4개 항목 자체는 통과
        assert quality.legacy_judge_passed is False
        assert quality.passed is False and judge.passed is False
        assert "legacy_judge" in _causes(quality)
        assert quality.revision_instructions  # 기존 Judge 수정 지시가 전달됨


def test_asymmetry_needs_counter_search_and_specific_disclosure():
    inp = _inp(evidence_count={"mla": 30, "pim_cxl": 10}, asymmetry=True)
    # 반대 근거 탐색 전: 한계점에 써 있어도 탈락, 원인 = asymmetry (supervisor가 탐색으로 보냄)
    md = _with_limits("PIM/CXL은 근거 수가 mla 30건, pim_cxl 10건으로 비대칭이다.")
    quality, _ = evaluate_report_quality(md, inp, FakeStructuredLLM(), NOW, counter_attempts=0)
    assert _failed(quality) == {"bias_control"} and "asymmetry" in _causes(quality)
    assert "3.00:1" in quality.checks["bias_control"].detail

    # 탐색 후 + 키워드만 있고 기술명이 다른 문단 → 공개 안 된 것으로 본다
    md_loose = _with_limits("근거가 비대칭이다.\n\n기타: PIM/CXL 관련 서술.")
    quality, _ = evaluate_report_quality(md_loose, inp, FakeStructuredLLM(), NOW, counter_attempts=1)
    assert _failed(quality) == {"bias_control"} and _causes(quality) == {"undisclosed"}
    assert any("비대칭" in i and "PIM/CXL" in i for i in quality.revision_instructions)

    # 탐색 후 + 같은 문단에 유형·기술명 → 통과
    quality, judge = evaluate_report_quality(md, inp, FakeStructuredLLM(), NOW, counter_attempts=1)
    assert quality.passed is True and judge.passed is True


def test_keyword_only_limits_do_not_pass_extreme_bias():
    """리뷰 재현: 100:1, 벤더 100%, 반대 근거 없음 + 한계점에 키워드 3개만 → 통과하면 안 된다."""
    inp = _inp(
        evidence_count={"mla": 100, "pim_cxl": 1},
        asymmetry=True,
        vendor_source_ratio={"mla": 1.0, "pim_cxl": 1.0},
        opposing_missing=[{"tech_id": "mla", "criterion_id": "D2", "reason": "r"}],
    )
    md = _with_limits("비대칭, 반대 근거, 벤더 자료 의존이 있다.")
    for attempts in (0, 1):
        quality, _ = evaluate_report_quality(md, inp, FakeStructuredLLM(), NOW, counter_attempts=attempts)
        assert quality.passed is False and "bias_control" in _failed(quality)
    quality, _ = evaluate_report_quality(md, inp, FakeStructuredLLM(), NOW, counter_attempts=0)
    assert {"asymmetry", "counter_missing", "vendor_heavy"} <= _causes(quality)


def test_bias_opposing_missing_and_vendor_ratio():
    inp = _inp(
        opposing_missing=[{"tech_id": "mla", "criterion_id": "D2", "reason": "r"}],
        vendor_source_ratio={"mla": 0.3, "pim_cxl": 0.7},
    )
    quality, _ = evaluate_report_quality(GOOD_MD, inp, FakeStructuredLLM(), NOW, counter_attempts=1)
    assert _failed(quality) == {"bias_control"}
    detail = quality.checks["bias_control"].detail
    assert "D2" in detail and "70%" in detail

    # 벤더 비율은 수치까지 있어야 공개로 인정
    no_number = _with_limits("DeepSeek-V2 MLA D2는 반대 근거를 찾지 못했다.\n\nPIM/CXL은 벤더 자료 비중이 높다.")
    quality, _ = evaluate_report_quality(no_number, inp, FakeStructuredLLM(), NOW, counter_attempts=1)
    assert _causes(quality) == {"undisclosed"}

    md = _with_limits("DeepSeek-V2 MLA D2는 반대 근거를 찾지 못했다.\n\nPIM/CXL은 벤더 자료 비율이 70%다.")
    quality, _ = evaluate_report_quality(md, inp, FakeStructuredLLM(), NOW, counter_attempts=1)
    assert quality.passed is True


def test_disclosure_instruction_given_before_counter_search():
    """반대 근거 탐색 전 탈락에도 공개 지시를 함께 넘긴다 (탐색 뒤 재작성 기회가 1회뿐)."""
    inp = _inp(vendor_source_ratio={"mla": 0.3, "pim_cxl": 0.8})
    quality, _ = evaluate_report_quality(GOOD_MD, inp, FakeStructuredLLM(), NOW, counter_attempts=0)
    assert "vendor_heavy" in _causes(quality)
    assert any("PIM/CXL" in i and "80.0%" in i for i in quality.revision_instructions)


@pytest.mark.parametrize("source_type", ["not_public", "inference"])
def test_failed_counter_record_is_not_secured(source_type):
    """리뷰 재현: 반대 근거 탐색 실패 기록(not_public/inference COUNTER)만 있으면 확보로 보지 않는다."""
    from techeval.schemas import Evidence

    failed = Evidence(
        evidence_id="mla-COUNTER-01",
        source_type=source_type,
        unit="family",
        quote="",
        locator="not_found",
        search_query="MLA limitations",
        searched_at=NOW,
    )
    inp = _inp(opposing_missing=[{"tech_id": "mla", "criterion_id": "D2", "reason": "r"}])
    inp = inp.model_copy(update={"counter_evidence": [failed]})
    quality, _ = evaluate_report_quality(GOOD_MD, inp, FakeStructuredLLM(), NOW, counter_attempts=1)
    assert quality.checks["bias_control"].passed is False and _causes(quality) == {"undisclosed"}
    assert "실제 확보 기술 없음" in quality.checks["bias_control"].detail

    # 탐색했으나 못 찾은 공백을 구체적으로 공개하면 통과 — 단 '확보'가 아니라 '미확보, 공개'로 기록
    md = _with_limits("DeepSeek-V2 MLA D2는 반대 근거를 찾지 못했다.")
    quality, _ = evaluate_report_quality(md, inp, FakeStructuredLLM(), NOW, counter_attempts=1)
    assert quality.passed is True
    assert "미확보, 한계점 공개됨(확보 아님)" in quality.checks["bias_control"].detail
