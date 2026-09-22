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
    assert "[E: mla-T1-01] [E: mla-T1-02]" in trl  # id마다 괄호 하나 (E judge 형식)


def test_missing_result_row():
    table = criterion_table(["mla", "pim_cxl"], ["T1"], [r for r in _results() if r.tech_id == "mla"])
    assert "| T1 현재 TRL | pim_cxl | 결과 없음 |" in table


def test_checklist_and_role_levels_show_their_content():
    """D4(checklist)·S1(assigned)은 레벨 문자열만으로 판정이 보이지 않으므로 요지를 함께 적는다."""
    results = _results()
    domain = perspective_table("4.4", "domain", TECHNOLOGIES, results)
    assert "checklist (필요: 서빙 엔진 수정, 메모리 추가; 미확인:" in domain
    stakeholder = perspective_table("4.3", "stakeholder", TECHNOLOGIES, results)
    assert "assigned (모델 개발사 결정권자," in stakeholder
