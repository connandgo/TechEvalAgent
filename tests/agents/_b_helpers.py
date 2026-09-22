"""B(기술 조사·도메인) 테스트 전용 글루.

E의 `FakeStructuredLLM`(techeval.stub_llm)에 B 드래프트 모델 응답을 `overrides`로 꽂고,
A의 `StubRetriever`·C의 `stub_web_search`를 명시 경로로 주입한 `Deps`를 만든다.
C의 스텁/픽스처가 main에 없으면 B 테스트는 통째로 skip 된다.
"""

import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

from techeval.agents._deps import Deps
from techeval.agents.tech_research import ArtifactDraft, CriterionDraft, ProfileDraft
from techeval.retrieval.stub import StubRetriever
from techeval.stub_llm import FakeStructuredLLM
from tests.conftest import FIXED_NOW

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CHUNKS_PATH = FIXTURES / "chunks.json"
WEB_PATH = FIXTURES / "web_results.json"

stub_web_search = pytest.importorskip(
    "techeval.tools.stub", reason="C의 stub_web_search 미제공"
).stub_web_search
if not WEB_PATH.exists():
    pytest.skip("C의 web_results.json 미제공", allow_module_level=True)

_HEADING = re.compile(r"^# (T\d|D\d|기술 개요)", re.MULTILINE)


class RecordingRetriever:
    """A의 StubRetriever를 감싸 search 호출 인자를 기록한다 (doc_ids 범위 검증용)."""

    def __init__(self, inner: StubRetriever):
        self._inner = inner
        self.calls: list[dict] = []

    def search(self, query, *, top_k=5, doc_ids=None, mode="hybrid"):
        self.calls.append(
            {"query": query, "top_k": top_k, "doc_ids": doc_ids, "mode": mode}
        )
        return self._inner.search(query, top_k=top_k, doc_ids=doc_ids, mode=mode)

    def get_chunk(self, chunk_id):
        return self._inner.get_chunk(chunk_id)


def prompt_key(text: str) -> str:
    m = _HEADING.search(text)
    if not m:
        raise AssertionError("프롬프트 제목을 찾을 수 없음")
    return "PROFILE" if m.group(1) == "기술 개요" else m.group(1)


def make_llm(table: dict[str, dict]) -> FakeStructuredLLM:
    """{'PROFILE': {...}, 'T1': {...}, ...} 테이블을 B 드래프트 모델 3종의 override 로 등록한다."""

    def route(text: str) -> dict:
        key = prompt_key(text)
        if key not in table:
            raise AssertionError(f"테이블에 {key} 응답이 없음")
        return table[key]

    fn: Callable[[str], dict] = route
    return FakeStructuredLLM(
        overrides={ProfileDraft: fn, CriterionDraft: fn, ArtifactDraft: fn}
    )


def prompts(llm: FakeStructuredLLM) -> list[str]:
    return [c["prompt"] for c in llm.calls]


def make_deps(llm) -> Deps:
    return Deps(
        retriever=RecordingRetriever(StubRetriever(str(CHUNKS_PATH))),
        web_search=partial(stub_web_search, fixture_path=str(WEB_PATH)),
        llm=llm,
        now=lambda: FIXED_NOW,
    )
