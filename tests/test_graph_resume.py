"""재개 검증: 새 그래프 객체·SQLite(새 프로세스와 같은 조건)에서 체크포인트로 이어서 실행하고 출처 레지스트리를 복원한다."""

import pytest

from techeval.graph import build_graph, invoke_config
from tests.graph_harness import Harness

# --- 재개: 새 그래프 객체 · SQLite(새 프로세스와 동일 조건) -------------------------------


def _run_until(graph, cfg_dict: dict, stop_node: str) -> None:
    for chunk in graph.stream({}, config=cfg_dict, stream_mode="updates"):
        if stop_node in chunk:
            return


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_resume_in_new_graph_restores_state_and_registry(h: Harness, tmp_path, backend):
    from langgraph.checkpoint.memory import MemorySaver

    from techeval.graph import _sqlite_saver

    if backend == "sqlite":  # 선택 의존성 — 없으면 안내와 함께 건너뜀 (README: uv add langgraph-checkpoint-sqlite)
        pytest.importorskip("langgraph.checkpoint.sqlite", reason="langgraph-checkpoint-sqlite 미설치")
    db = str(tmp_path / "ckpt.sqlite")
    saver1 = MemorySaver() if backend == "memory" else _sqlite_saver(db)
    cfg_dict = invoke_config(h.cfg, trace_id="resume-test")

    g1 = build_graph(h.deps, h.agents(), h.cfg, checkpointer=saver1)
    _run_until(g1, cfg_dict, "synthesis")  # 종합까지만 실행하고 중단
    snap = g1.get_state(cfg_dict)
    assert snap.next and snap.values.get("source_urls")
    urls_before = set(snap.values["source_urls"])

    # 새 그래프(새 출처 레지스트리). sqlite는 연결도 새로 열어 프로세스 재시작과 같은 조건
    saver2 = saver1 if backend == "memory" else _sqlite_saver(db)
    calls_before = dict(h.calls)
    cfg2 = h.cfg.model_copy(update={"stub": False})  # 픽스처 URL 미리 등록 없음 → 복원된 URL만 레지스트리에 있다
    g2 = build_graph(h.deps, h.agents(), cfg2, checkpointer=saver2)
    assert not g2.registry.known_urls
    final = g2.invoke(None, config=cfg_dict)

    assert final["report_md"] and final["quality_result"].passed is True
    assert urls_before <= set(g2.registry.known_urls)  # 레지스트리가 State에서 복원됨
    # 이미 끝난 조사는 다시 돌지 않는다
    assert h.calls["run_tech_research"] == calls_before["run_tech_research"]
    assert h.calls["run_market_eval"] == calls_before["run_market_eval"]


def test_resume_right_after_agent_keeps_its_urls(h: Harness):
    """하위 에이전트 직후(supervisor 전) 중단돼도 그 에이전트가 검색한 URL이 State에 남아 재개 시 복원된다."""
    from langgraph.checkpoint.memory import MemorySaver

    saver = MemorySaver()
    cfg_dict = invoke_config(h.cfg, trace_id="resume-agent")
    g1 = build_graph(h.deps, h.agents(), h.cfg, checkpointer=saver)
    g1.invoke({}, config=cfg_dict, interrupt_after=["market_eval"])  # 관점 에이전트 직후, supervisor 전에 중단
    snap = g1.get_state(cfg_dict)
    assert snap.next == ("supervisor",) and snap.values.get("market_eval")
    urls = set(snap.values["source_urls"])
    assert urls >= set(g1.registry.known_urls)  # 에이전트 노드가 기록한 URL 포함

    g2 = build_graph(h.deps, h.agents(), h.cfg.model_copy(update={"stub": False}), checkpointer=saver)
    final = g2.invoke(None, config=cfg_dict)
    assert urls <= set(g2.registry.known_urls) and final["quality_result"].passed is True
