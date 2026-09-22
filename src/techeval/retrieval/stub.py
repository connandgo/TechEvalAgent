"""실제 모델과 인덱스가 필요 없는 fixture 기반 retriever."""

import json
from pathlib import Path
from typing import Literal

from .retriever import RetrievedChunk


class StubRetriever:
    def __init__(self, fixture_path: str = "tests/fixtures/chunks.json") -> None:
        path = Path(fixture_path)
        if not path.exists():
            raise FileNotFoundError(f"StubRetriever fixture not found: {path}")
        raw_chunks = json.loads(path.read_text(encoding="utf-8"))
        self._chunks = [RetrievedChunk.model_validate(chunk) for chunk in raw_chunks]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    ) -> list[RetrievedChunk]:
        del query, mode
        results = self._chunks
        if doc_ids is not None:
            results = [chunk for chunk in results if chunk.doc_id in doc_ids]
        return [chunk.model_copy(deep=True) for chunk in results[:top_k]]

    def get_chunk(self, chunk_id: str) -> RetrievedChunk | None:
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                return chunk.model_copy(deep=True)
        return None
