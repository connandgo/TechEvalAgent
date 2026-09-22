"""Chroma 영구 컬렉션 접근 계층."""

from collections.abc import Iterable
from pathlib import Path


class ChromaStore:
    def __init__(self, chroma_dir: str | Path, *, collection: str = "papers") -> None:
        import chromadb

        self.path = Path(chroma_dir)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def replace(self, chunks: Iterable[object], embeddings: list[list[float]]) -> None:
        chunk_list = list(chunks)
        if len(chunk_list) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        existing = self._collection.get(include=[])
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])

        self._collection.add(
            ids=[str(chunk.chunk_id) for chunk in chunk_list],
            documents=[str(chunk.text) for chunk in chunk_list],
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": str(chunk.doc_id),
                    "doc_title": str(chunk.doc_title),
                    "page": int(chunk.page),
                    "section": chunk.section or "",
                    "chunk_type": str(chunk.chunk_type),
                }
                for chunk in chunk_list
            ],
        )

    def query(self, embedding: list[float], *, top_k: int, doc_ids: list[str] | None = None) -> dict:
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        return self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def get(self, chunk_id: str) -> dict:
        return self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )

    def all_ids(self, *, doc_ids: list[str] | None = None) -> set[str]:
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        result = self._collection.get(where=where, include=[])
        return set(result["ids"])
