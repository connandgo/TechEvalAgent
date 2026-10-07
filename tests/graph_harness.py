"""그래프 테스트 공용 하네스: stub_agents를 감싼 카운팅 래퍼 + `h` 픽스처 (tests/conftest.py에서 공유)."""

from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest

from techeval import stub_agents
from techeval.agents._deps import Deps
from techeval.graph import Agents, GraphConfig, build_graph, invoke_config
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
        self.order: list[str] = []  # 노드 실행 순서 (updates 스트림 기준)

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
                    self.order.append(node)
            else:
                final = chunk
        return final, node_counts


@pytest.fixture
def h(tmp_path) -> Harness:
    # stub_agents가 B/C/D 픽스처를 읽지 않도록 (있으면 픽스처 chunk_id가 MemRetriever와 안 맞는다)
    stub_agents.FIXTURES_DIR = tmp_path / "no_fixtures"
    yield Harness(tmp_path)
    stub_agents.FIXTURES_DIR = stub_agents.DEFAULT_FIXTURES_DIR


def levels(state: dict, key: str, tech: str) -> dict[str, str]:
    return {r.criterion_id: r.level for r in latest_by_criterion(state[key]) if r.tech_id == tech}
