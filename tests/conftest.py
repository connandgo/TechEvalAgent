"""공용 pytest 픽스처 (E).

- `fake_llm`: 픽스처 기반 `FakeStructuredLLM` (구현은 `techeval.stub_llm`)
- `deps_stub`: `Deps(retriever=StubRetriever(), web_search=stub_web_search, llm=FakeStructuredLLM(), ...)`
  A/C의 스텁이 아직 없으면 빈 결과를 돌려주는 대체물을 쓰고 경고를 남긴다 (`techeval.config` 참조).
- `fixed_now`: 고정 시각 문자열
"""

from pathlib import Path

import pytest

from techeval.agents._deps import Deps
from techeval.config import make_stub_retriever, make_stub_web_search
from techeval.stub_llm import FakeStructuredLLM

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXED_NOW = "2026-01-01T00:00:00"


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
