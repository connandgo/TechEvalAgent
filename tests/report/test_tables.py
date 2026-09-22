from techeval.report.tables import TABLE_HEADER, criterion_table, perspective_table
from techeval.schemas import TECHNOLOGIES, CriterionResult
from tests.report.d_fixtures import load_evals


def _results() -> list[CriterionResult]:
    return [r for rs in load_evals().values() for r in rs]


def test_thirty_results_make_four_tables():
    results = _results()
    assert len(results) == 30
    tables = [
        perspective_table(f"4.{i}", p, TECHNOLOGIES, results)
        for i, p in enumerate(("trl", "market", "stakeholder", "domain"), start=1)
    ]
    assert all(TABLE_HEADER in t for t in tables)
    rows = sum(t.count("\n| ") - 1 for t in tables)  # 헤더 구분선 제외 데이터 행
    assert rows == 30
    trl = tables[0]
    assert "TRL 4-6 (추정: TRL 6 (공개 정보 기반 추정))" in trl
    assert "[E: mla-T1-01][E: mla-T1-02]" in trl  # id마다 괄호 하나 (E judge 형식)


def test_missing_result_row():
    table = criterion_table(["mla", "pim_cxl"], ["T1"], [r for r in _results() if r.tech_id == "mla"])
    assert "| T1 현재 TRL | pim_cxl | 결과 없음 |" in table
