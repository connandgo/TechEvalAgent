"""Dense + BM25 + RRF 하이브리드 검색기."""

import logging
import re
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from techeval.schemas import DocId, Evidence, EvidenceUnit

from .bm25 import BM25Index
from .embedder import BGEEmbedder
from .store import ChromaStore

logger = logging.getLogger(__name__)

DOC_META = {
    "deepseek_v2": {
        "authors": "DeepSeek-AI",
        "published_date": "2024-06-19",
        "url": "https://arxiv.org/abs/2405.04434",
    },
    "pim_cxl_1m": {
        "authors": "Dowon Kim et al.",
        "published_date": "2025-10-31",
        "url": "https://arxiv.org/abs/2511.00321",
    },
    "io_survey": {
        "authors": "Rajarshi Chowdhury",
        "published_date": "2026-07-09",
        "url": "https://doi.org/10.1007/s10462-026-11651-1",
    },
    "kv_survey": {
        "authors": "Haoyang Li et al.",
        "published_date": "2025-07-30",
        "url": "https://arxiv.org/abs/2412.19442",
    },
}


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: DocId
    doc_title: str
    page: int
    section: str | None = None
    text: str
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    chunk_type: Literal["text", "table", "figure_caption"] = "text"

    def to_evidence(
        self, *, evidence_id: str, quote: str, unit: EvidenceUnit = "paper"
    ) -> Evidence:
        """공용 Evidence 모델로 변환하고, LLM 인용문이 원문에 있는지 확인한다."""
        normalized_quote = re.sub(r"\s+", " ", quote).strip()
        normalized_text = re.sub(r"\s+", " ", self.text).strip()
        if not normalized_quote or normalized_quote not in normalized_text:
            raise ValueError(
                f"quote must be a substring of retrieved chunk {self.chunk_id}"
            )
        meta = DOC_META.get(self.doc_id, {})
        section = f" §{self.section}" if self.section else ""
        return Evidence(
            evidence_id=evidence_id,
            source_type="paper",
            unit=unit,
            quote=quote,
            locator=f"{self.doc_id} p.{self.page}{section}",
            title=self.doc_title,
            authors=meta.get("authors"),
            publisher="arXiv"
            if "arxiv.org" in meta.get("url", "")
            else "Artificial Intelligence Review",
            published_date=meta.get("published_date"),
            url=meta.get("url"),
            doc_id=self.doc_id,
            page=self.page,
            section=self.section,
            chunk_id=self.chunk_id,
        )


class BaseRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    ) -> list[RetrievedChunk]: ...

    def get_chunk(self, chunk_id: str) -> RetrievedChunk | None: ...


def rrf_fuse(
    dense_ranked: list[RetrievedChunk],
    bm25_ranked: list[RetrievedChunk],
    *,
    k: int = 60,
) -> list[RetrievedChunk]:
    """두 순위 목록을 reciprocal-rank fusion으로 결합한다."""
    if k < 1:
        raise ValueError("k must be positive")
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for ranked, score_name in (
        (dense_ranked, "dense_score"),
        (bm25_ranked, "bm25_score"),
    ):
        for rank, chunk in enumerate(ranked, start=1):
            current = fused.get(chunk.chunk_id)
            if current is None:
                current = chunk.model_copy(deep=True)
                fused[chunk.chunk_id] = current
            setattr(current, score_name, chunk.score)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
    for chunk_id, chunk in fused.items():
        chunk.score = scores[chunk_id]
    return sorted(fused.values(), key=lambda chunk: chunk.score, reverse=True)


class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        chroma_dir: str,
        collection: str = "papers",
        embedding_model: str = "BAAI/bge-m3",
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self._store = ChromaStore(self.chroma_dir, collection=collection)
        self._embedder = BGEEmbedder(embedding_model)
        self._bm25 = BM25Index.load(self.chroma_dir / "bm25.pkl")

    @staticmethod
    def _from_row(
        chunk_id: str, document: str, metadata: dict, score: float
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=metadata["doc_id"],
            doc_title=metadata["doc_title"],
            page=int(metadata["page"]),
            section=metadata.get("section") or None,
            text=document,
            score=score,
            chunk_type=metadata.get("chunk_type", "text"),
        )

    def _dense_search(
        self, query: str, *, top_k: int, doc_ids: list[str] | None
    ) -> list[RetrievedChunk]:
        result = self._store.query(
            self._embedder.embed_query(query), top_k=top_k, doc_ids=doc_ids
        )
        return [
            self._from_row(chunk_id, document, metadata, 1.0 / (1.0 + float(distance)))
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def _bm25_search(
        self, query: str, *, top_k: int, doc_ids: list[str] | None
    ) -> list[RetrievedChunk]:
        allowed_ids = self._store.all_ids(doc_ids=doc_ids)
        chunks: list[RetrievedChunk] = []
        for chunk_id, score in self._bm25.search(
            query, top_k=top_k, allowed_ids=allowed_ids
        ):
            chunk = self.get_chunk(chunk_id)
            if chunk is not None:
                chunk.score = score
                chunks.append(chunk)
        return chunks

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if mode not in {"hybrid", "dense", "bm25"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        candidate_count = max(top_k * 4, 20)
        dense = (
            self._dense_search(query, top_k=candidate_count, doc_ids=doc_ids)
            if mode != "bm25"
            else []
        )
        bm25 = (
            self._bm25_search(query, top_k=candidate_count, doc_ids=doc_ids)
            if mode != "dense"
            else []
        )
        ranked = rrf_fuse(dense, bm25) if mode == "hybrid" else (dense or bm25)
        result = ranked[:top_k]
        logger.debug(
            "retrieval query=%r mode=%s doc_ids=%s results=%s",
            query,
            mode,
            doc_ids,
            [item.chunk_id for item in result],
        )
        return result

    def get_chunk(self, chunk_id: str) -> RetrievedChunk | None:
        result = self._store.get(chunk_id)
        if not result["ids"]:
            return None
        return self._from_row(
            result["ids"][0], result["documents"][0], result["metadatas"][0], 0.0
        )
