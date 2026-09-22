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


@pytest.mark.parametrize("term", ["더 우수", "추천", "순위", "종합 점수", "우세"])
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
    assert any(
        i.kind == "unknown_citation" and "mla-T2-99" in i.message for i in res.errors
    )


def test_reference_must_match_body(report, index):
    extra = (
        report.rstrip()
        + " \n99. 가짜 문헌 (근거 ID: mla-T3-02, pim_cxl-M1-01, mla-S4-02, zz-1)\n"
    )
    res = lint_report(extra, index)
    assert any(
        i.kind == "reference_mismatch" and "미인용" in i.message for i in res.errors
    )
    body, _ = report.split("## REFERENCE")
    dropped = body + "## REFERENCE\n\n1. 일부만 (근거 ID: mla-T1-01)\n"
    res = lint_report(dropped, index)
    assert any(
        i.kind == "reference_mismatch" and "REFERENCE에 없음" in i.message
        for i in res.errors
    )


def test_number_without_citation_is_warning_only(report, index):
    warn = report.replace(
        "## 5. 시사점\n", "## 5. 시사점\n\n처리량이 3.2x 늘었다.\n", 1
    )
    res = lint_report(warn, index)
    assert res.passed
    assert any(
        w.kind == "number_without_citation" and "3.2x" in w.message
        for w in res.warnings
    )


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
