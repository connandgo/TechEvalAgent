"""C(시장·이해관계자) 테스트 전용 글루.

E의 `FakeStructuredLLM`은 `CriterionDraft`를 자동으로 채우지 못하므로(필수 필드 `level`),
C의 커밋된 픽스처에서 draft를 역산해 `register`로 꽂는다. 이렇게 하면 인용문이 픽스처와
자동으로 같아져서, `web_results.json`이 바뀌어도 테스트 쪽 quote를 따로 고칠 일이 없다.

- `M2`·`M3`의 `level`은 빈 문자열로 둔다. 코드가 계산하는지 보려는 것이다.
- `S1`은 `assigned`, `S2`~`S4`는 `narrative`를 에이전트가 덮어쓴다.
"""

import json
from pathlib import Path

import pytest

from techeval.agents._deps import Deps
from techeval.agents.tech_research import CriterionDraft
from techeval.retrieval.stub import StubRetriever
from techeval.stub_llm import FakeStructuredLLM, extract_criterion_ids, extract_tech_id

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MARKET_PATH = FIXTURES / "market_eval.json"
STAKEHOLDER_PATH = FIXTURES / "stakeholder_eval.json"

stub_web_search = pytest.importorskip(
    "techeval.tools.stub", reason="C의 stub_web_search 미제공"
).stub_web_search

FIXED_NOW = "2026-01-01T00:00:00"

#: 코드가 계산해야 하는 기준 — draft의 level을 비워 둔다.
CODE_COMPUTED_LEVEL: frozenset[str] = frozenset({"M2", "M3"})


def _load(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():  # pragma: no cover - 픽스처 누락 시 테스트가 skip 된다
        return {}
    return {
        (r["tech_id"], r["criterion_id"]): r
        for r in json.loads(path.read_text("utf-8"))
    }


RESULTS: dict[tuple[str, str], dict] = {**_load(MARKET_PATH), **_load(STAKEHOLDER_PATH)}


def draft_for(tech_id: str, criterion_id: str) -> CriterionDraft:
    """픽스처 1건을 LLM이 냈을 법한 CriterionDraft로 되돌린다."""
    r = RESULTS[(tech_id, criterion_id)]
    citations = [
        {
            "source": "chunk" if e.get("chunk_id") else "web",
            "ref": e.get("chunk_id") or e.get("url") or e["locator"],
            "quote": e["quote"],
        }
        for e in r["evidence"]
        if e["source_type"] != "inference"  # 추론 근거는 에이전트가 직접 만든다
    ]
    return CriterionDraft(
        level="" if criterion_id in CODE_COMPUTED_LEVEL else r["level"],
        content=r["content"],
        details=json.loads(json.dumps(r["details"])),  # 원본 훼손 방지
        measurements=[],
        citations=citations,
    )


def make_llm(
    overrides: dict[tuple[str, str], CriterionDraft] | None = None,
) -> FakeStructuredLLM:
    """프롬프트에서 tech_id·criterion_id를 읽어 해당 draft를 돌려주는 가짜 LLM.

    `overrides`로 특정 (tech_id, criterion_id)의 draft를 갈아끼울 수 있다.
    """
    overrides = overrides or {}

    def _draft(text: str) -> CriterionDraft:
        tech_id = extract_tech_id(text)
        ids = extract_criterion_ids(text)
        if not tech_id or not ids:
            raise AssertionError(
                f"프롬프트에서 tech_id/criterion_id를 못 읽었다: {text[:200]!r}"
            )
        criterion_id = ids.most_common(1)[0][0]
        if (tech_id, criterion_id) in overrides:
            return overrides[(tech_id, criterion_id)]
        return draft_for(tech_id, criterion_id)

    llm = FakeStructuredLLM()
    llm.register(CriterionDraft, _draft)
    return llm


def make_deps(llm: FakeStructuredLLM | None = None, web_search=None) -> Deps:
    return Deps(
        retriever=StubRetriever(),
        web_search=web_search or stub_web_search,
        llm=llm or make_llm(),
        now=lambda: FIXED_NOW,
    )
