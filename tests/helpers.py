"""테스트 공용 헬퍼 (E): 메모리 검색기·웹검색·근거/결과 팩토리."""

from collections import Counter
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from techeval.schemas import DOC_IDS, CriterionResult, Evidence, Measurement, TechProfile, TechRef, get_tech

MLA: TechRef = get_tech("mla")
PIM: TechRef = get_tech("pim_cxl")
NOW = "2026-01-01T00:00:00"


# --- 메모리 검색기 / 웹검색 ---------------------------------------------------------


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    section: str | None
    text: str
    score: float = 1.0
    chunk_type: str = "text"


class MemRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.calls: list[str] = []

    def search(self, query: str, *, top_k: int = 5, doc_ids=None, mode: str = "hybrid") -> list[Chunk]:
        self.calls.append(query)
        out = [c for c in self.chunks if doc_ids is None or c.doc_id in doc_ids]
        return out[:top_k]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)


class WebResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: str | None = None
    publisher: str | None = None
    published_date: str | None = None
    fetched_at: str = "2026-01-01T00:00:00"
    source_kind: str = "other"
    query: str = ""


def make_chunks(per_doc: int = 4) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc}:{p:03d}:01",
            doc_id=doc,
            doc_title=doc,
            page=p,
            section=f"{p}.1",
            text=f"Text of {doc} page {p}. " * 5,
        )
        for doc in DOC_IDS
        for p in range(1, per_doc + 1)
    ]


def make_web_search(n: int = 2) -> Callable[..., list[WebResult]]:
    counter = Counter()

    def _search(query: str, **kwargs: Any) -> list[WebResult]:
        counter[query] += 1
        k = sum(counter.values())
        return [
            WebResult(
                title=f"result {k}-{i}",
                url=f"https://site{i}.example.com/{k}",
                snippet=f"snippet for {query} #{i}",
                publisher=f"site{i}",
                query=query,
            )
            for i in range(n)
        ]

    _search.counter = counter  # type: ignore[attr-defined]
    return _search


# --- 근거/결과 팩토리 -------------------------------------------------------------


def paper_evidence(evidence_id: str, chunk: Chunk, *, quote: str | None = None, unit: str = "paper") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type="paper",
        unit=unit,
        quote=quote if quote is not None else chunk.text[:40],
        locator=f"{chunk.doc_id} p.{chunk.page}",
        doc_id=chunk.doc_id,
        page=chunk.page,
        chunk_id=chunk.chunk_id,
    )


def web_evidence(evidence_id: str, url: str, *, source_type: str = "web") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        unit="family",
        quote="web quote",
        locator=url,
        url=url,
        accessed_date="2026-01-01",
    )


def not_public_evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type="not_public",
        unit="paper",
        quote="",
        locator="not_found",
        search_query="q",
        searched_at=NOW,
    )


def result(
    tech: TechRef,
    cid: str,
    perspective: str,
    evidence: list[Evidence],
    *,
    level: str = "L2",
    details: dict | None = None,
    measurements: list[Measurement] | None = None,
    confidence: str | None = None,
    generated_at: str = NOW,
) -> CriterionResult:
    from techeval.schemas import compute_confidence

    if measurements is None and cid in ("D1", "D2", "D3") and level != "not_public":
        measurements = [Measurement(metric="m", value="1", evidence_id=evidence[0].evidence_id)]
    return CriterionResult(
        tech_id=tech.tech_id,
        perspective=perspective,
        criterion_id=cid,
        level=level,
        content="c",
        evidence=evidence,
        confidence=confidence or compute_confidence(evidence),
        evidence_unit="paper" if perspective in ("trl", "domain") else "family",
        measurements=measurements or [],
        details=details or {},
        generated_at=generated_at,
    )


def profile(tech: TechRef, evidence: list[Evidence], **overrides: Any) -> TechProfile:
    base = dict(
        tech_id=tech.tech_id,
        principle="p",
        scope="s",
        limitations=["l"],
        measurements=[Measurement(metric="m", value="1", evidence_id=evidence[0].evidence_id)],
        validation_env="v",
        public_artifacts={},
        evidence=evidence,
        search_queries_used=["q0"],
        generated_at=NOW,
    )
    base.update(overrides)
    return TechProfile(**base)
