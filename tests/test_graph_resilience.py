"""하위 에이전트 예외 fallback, not_public 검색 이력 기록 검증."""

from techeval.schemas import MAX_RETRY_PER_PERSPECTIVE
from techeval.state import latest_by_criterion
from tests.graph_harness import Harness
from tests.graph_harness import levels as _levels


def test_agent_exception_falls_back(h: Harness):
    def boom(inp, out, n):
        if inp.tech.tech_id == "pim_cxl":
            raise RuntimeError("market backend down")
        return out

    h.overrides["run_market_eval"] = boom
    final, nodes = h.run()

    assert nodes["render_pdf"] == 1 and final["report_md"]  # 그래프는 끝까지 간다
    assert final["node_status"]["market:pim_cxl"] == "exhausted"
    assert final["node_status"]["market:mla"] == "done"
    assert "market:pim_cxl" in final["last_error"] and "RuntimeError" in final["last_error"]
    assert final["retry_counts"]["market:pim_cxl"] == MAX_RETRY_PER_PERSPECTIVE
    # 첫 실행 1회 + 재작업 MAX회 모두 pim_cxl에 실제로 배정됨
    pim_calls = [i for i in h.inputs["run_market_eval"] if i.tech.tech_id == "pim_cxl"]
    assert len(pim_calls) == 1 + MAX_RETRY_PER_PERSPECTIVE
    lv = _levels(final, "market_eval", "pim_cxl")
    assert {lv[c] for c in ("M1", "M2", "M3")} == {"not_public"}
    assert _levels(final, "market_eval", "mla")["M1"] != "not_public"


# --- not_public 검색 이력 ------------------------------------------------------------


def test_not_public_at_cap_keeps_real_queries(h: Harness):
    """재시도 상한으로 확정한 not_public에 실제 실행한 검색어가 남는다 ('검색어 미기록' 금지)."""

    def empty_m2(inp, out, n):  # mla M2만 계속 결과 누락 → 상한 후 not_public
        return [r for r in out if not (inp.tech.tech_id == "mla" and r.criterion_id == "M2")]

    h.overrides["run_market_eval"] = empty_m2
    final, _ = h.run()
    m2 = next(r for r in latest_by_criterion(final["market_eval"]) if r.tech_id == "mla" and r.criterion_id == "M2")
    ev = m2.evidence[0]
    assert m2.level == "not_public" and ev.locator == "not_found"
    assert "미기록" not in ev.search_query and ev.search_query.strip()
    logged = final["search_log"]["market:mla"]  # 노드가 직접 기록한 검색 이력이 실제로 쓰였는지
    assert logged and any(q in ev.search_query for q in logged)
    assert set(ev.search_query.split(" | ")) <= set(final["search_log"]["market:mla"]) | {
        e.search_query for r in final["market_eval"] for e in r.evidence if e.search_query
    }


def test_run_error_not_public_is_distinguished(h: Harness):
    """실행 오류로 상한에 도달한 not_public은 '검색했으나 공개 근거 없음'과 구분된다."""

    def boom(inp, out, n):
        if inp.tech.tech_id == "pim_cxl":
            raise RuntimeError("market backend down")
        return out

    h.overrides["run_market_eval"] = boom
    final, _ = h.run()
    pim = [r for r in latest_by_criterion(final["market_eval"]) if r.tech_id == "pim_cxl"]
    assert pim
    for r in pim:
        if r.tech_id == "pim_cxl":
            assert r.evidence[0].locator == "run_error" and "실행 오류" in r.evidence[0].search_query
            assert "실행 오류(공개 여부 미확인)" in r.content
