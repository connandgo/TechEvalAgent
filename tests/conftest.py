"""공용 pytest 픽스처 (E).

- `fake_llm`: 픽스처 기반 `FakeStructuredLLM` (구현은 `techeval.stub_llm`)
- `deps_stub`: `Deps(retriever=StubRetriever(), web_search=stub_web_search, llm=FakeStructuredLLM(), ...)`
  A/C의 스텁이 아직 없으면 빈 결과를 돌려주는 대체물을 쓰고 경고를 남긴다.
- `fixed_now`: 고정 시각 문자열
"""

import logging
from pathlib import Path

import pytest

from techeval.agents._deps import Deps
from techeval.stub_llm import FakeStructuredLLM

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXED_NOW = "2026-01-01T00:00:00"


class _FallbackRetriever:
    """A의 `StubRetriever`가 머지되기 전까지 쓰는 빈 검색기."""

    def search(self, query: str, *, top_k: int = 5, doc_ids=None, mode: str = "hybrid") -> list:
        return []

    def get_chunk(self, chunk_id: str):
        return None


def _fallback_web_search(query: str, **kwargs) -> list:
    """C의 `stub_web_search`가 머지되기 전까지 쓰는 빈 검색 함수."""
    return []


def make_stub_retriever():
    try:
        from techeval.retrieval.stub import StubRetriever

        return StubRetriever()
    except ImportError:
        logger.warning("techeval.retrieval.stub.StubRetriever 없음 (역할 A) — 빈 검색기로 대체")
        return _FallbackRetriever()


def make_stub_web_search():
    try:
        from techeval.tools.stub import stub_web_search

        return stub_web_search
    except ImportError:
        logger.warning("techeval.tools.stub.stub_web_search 없음 (역할 C) — 빈 검색 함수로 대체")
        return _fallback_web_search


@pytest.fixture
def fixed_now() -> str:
    return FIXED_NOW


@pytest.fixture
def fake_llm() -> FakeStructuredLLM:
    return FakeStructuredLLM(fixtures_dir=FIXTURES_DIR)


@pytest.fixture
def deps_stub(fake_llm: FakeStructuredLLM) -> Deps:
    return Deps(
        retriever=make_stub_retriever(),
        web_search=make_stub_web_search(),
        llm=fake_llm,
        judge_llm=FakeStructuredLLM(fixtures_dir=FIXTURES_DIR),
        now=lambda: FIXED_NOW,
    )
