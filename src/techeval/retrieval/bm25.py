"""약어 보존형 BM25 인덱스."""

import pickle
import re
from collections.abc import Iterable
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """MLA, PIM, CXL, TTFT 같은 기술 약어를 하나의 토큰으로 보존한다."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BM25Index:
    def __init__(self, *, chunk_ids: list[str], corpus: list[list[str]]) -> None:
        if len(chunk_ids) != len(corpus):
            raise ValueError("chunk_ids and corpus must have the same length")
        self.chunk_ids = chunk_ids
        self._bm25 = BM25Okapi(corpus)

    @classmethod
    def build(cls, chunks: Iterable[object]) -> "BM25Index":
        chunk_list = list(chunks)
        return cls(
            chunk_ids=[str(chunk.chunk_id) for chunk in chunk_list],
            corpus=[tokenize(str(chunk.text)) for chunk in chunk_list],
        )

    def search(self, query: str, *, top_k: int = 20, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            ((self.chunk_ids[index], float(score)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        if allowed_ids is not None:
            ranked = [item for item in ranked if item[0] in allowed_ids]
        return ranked[:top_k]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as file:
            value = pickle.load(file)
        if not isinstance(value, cls):
            raise TypeError(f"{path} does not contain a BM25Index")
        return value
