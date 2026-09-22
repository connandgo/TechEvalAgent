"""BGE-M3 dense embedding 래퍼.

모델은 생성 시점이 아니라 첫 embedding 요청 시점에만 불러온다. 따라서
StubRetriever를 쓰는 테스트나 그래프 초기화에서는 모델 다운로드가 발생하지 않는다.
"""

from collections.abc import Sequence


class BGEEmbedder:
    """동일한 BGE-M3 모델로 문서와 질의를 임베딩한다."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def embed_documents(self, texts: Sequence[str], *, batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query], batch_size=1)[0]
