import pytest

from techeval.report.lint import lint_report
from techeval.report.sections import join_sections, split_sections
from techeval.report.tables import TABLE_HEADER
from techeval.schemas import Evidence
from tests.report.d_fixtures import load, load_index


@pytest.fixture(scope="module")
def index() -> dict[str, Evidence]:
    return load_index()


@pytest.fixture(scope="module")
def report() -> str:
    return load("report_md.md")


def test_fixture_report_passes(report, index):
    res = lint_report(report, index)
    assert res.passed, res.errors + [res.missing_required]


@pytest.mark.parametrize(
    "term",
    ["더 우수", "추천", "순위", "종합 점수", "우세", "합산", "우월", "평균 레벨"],
)
def test_banned_term(report, index, term):
    bad = report.replace("## 5. 시사점\n", f"## 5. 시사점\n\nMLA가 {term}하다.\n", 1)
    res = lint_report(bad, index)
    assert not res.passed
    assert any(i.kind == "banned_term" and i.section == "5" for i in res.errors)
    assert any(term in ins for ins in res.revision_instructions)


def test_missing_chapter_and_table(report, index):
    no_ch6 = report.replace("## 6. 한계점", "## 여섯. 한계점")
    assert "챕터 누락: 6. 한계점" in lint_report(no_ch6, index).missing_required
    sections = split_sections(report)
    for sec in sections:
        if sec.key == "4.3":
            sec.body = sec.body.replace(TABLE_HEADER, "")
    joined = join_sections(sections)
    assert "요약표 누락: 4.3" in lint_report(joined, index).missing_required


def test_unknown_citation(report, index):
    bad = report.replace("[E: mla-T2-01]", "[E: mla-T2-99]", 1)
    res = lint_report(bad, index)
    assert any(i.kind == "unknown_citation" and "mla-T2-99" in i.message for i in res.errors)


def test_reference_must_match_body(report, index):
    extra = report.rstrip() + " \n99. 가짜 문헌 (근거 ID: mla-T3-02, pim_cxl-M1-01, mla-S4-02, zz-1)\n"
    res = lint_report(extra, index)
    assert any(i.kind == "reference_mismatch" and "미인용" in i.message for i in res.errors)
    body, _ = report.split("## REFERENCE")
    dropped = body + "## REFERENCE\n\n1. 일부만 (근거 ID: mla-T1-01)\n"
    res = lint_report(dropped, index)
    assert any(i.kind == "reference_mismatch" and "REFERENCE에 없음" in i.message for i in res.errors)


def test_number_without_citation_is_warning_only(report, index):
    warn = report.replace("## 5. 시사점\n", "## 5. 시사점\n\n처리량이 3.2x 늘었다.\n", 1)
    res = lint_report(warn, index)
    assert res.passed
    assert any(w.kind == "number_without_citation" and "3.2x" in w.message for w in res.warnings)


def test_split_join_roundtrip(report):
    sections = split_sections(report)
    assert join_sections(sections) == report
    assert [s.key for s in sections] == [
        "title",
        "SUMMARY",
        "1",
        "2",
        "3",
        "3.1",
        "3.2",
        "4",
        "4.1",
        "4.2",
        "4.3",
        "4.4",
        "5",
        "6",
        "REFERENCE",
    ]


def test_negated_aggregate_statement_is_still_banned(report, index):
    """E judge와 같은 기준: "합산하지 않는다" 같은 부정문도 금칙어로 잡는다."""
    bad = report.replace("## 6. 한계점\n", "## 6. 한계점\n\n관점별 레벨은 합산하지 않는다.\n", 1)
    assert any(i.kind == "banned_term" and i.section == "6" for i in lint_report(bad, index).errors)


def test_summary_length_and_intro_are_warnings_not_errors(report, index):
    """분량은 judge 채점 대상이 아니므로(CRITERIA §6) 경고로만 두고, D가 SUMMARY를 스스로 다시 쓰는 데 쓴다."""
    long_intro = "본 보고서는 두 기술을 평가한다. " + "근거 문장이다[E: mla-T1-01]. " * 60
    bad = report.replace("## SUMMARY\n", f"## SUMMARY\n\n{long_intro}\n", 1)
    res = lint_report(bad, index)
    kinds = {w.kind for w in res.warnings if w.section == "SUMMARY"}
    assert {"summary_too_long", "summary_intro"} <= kinds
    assert not any(i.section == "SUMMARY" for i in res.errors)
    assert "SUMMARY" in res.sections_to_fix()
    assert "SUMMARY" not in {w.section for w in lint_report(report, index).warnings}  # 픽스처 SUMMARY는 기준 안


def test_summary_generic_opening_and_uncited_sentences_are_fixable(report, index):
    """실제 LLM 출력에서 본 패턴: 인용 없는 총론 첫 문장, 인용 없는 일반론 마무리."""
    body = (
        "두 기술은 서로 다른 관점에서 상충되는 평가를 받는다. "
        "MLA는 T1과 M2가 엇갈린다[E: mla-T1-01]. "
        "이는 해석에 영향을 준다."
    )
    bad = report.replace("## SUMMARY\n", f"## SUMMARY\n\n{body}\n", 1)
    res = lint_report(bad, index)
    kinds = {w.kind for w in res.warnings if w.section == "SUMMARY"}
    assert {"summary_intro", "summary_uncited"} <= kinds
    assert "SUMMARY" in res.sections_to_fix() and res.passed  # judge가 보는 errors는 아님


def test_uncited_number_in_body_section_is_fixable_but_not_in_ch6(report, index):
    bad = report.replace("## 5. 시사점\n", "## 5. 시사점\n\n처리량이 3.2x 늘었다.\n", 1)
    assert "5" in lint_report(bad, index).sections_to_fix()
    assert "6" not in lint_report(report, index).sections_to_fix()  # 6장 벤더 비율(35.0%)은 제외
