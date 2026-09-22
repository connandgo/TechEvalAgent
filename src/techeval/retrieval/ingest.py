"""PDF 원문을 절·표 메타데이터를 보존한 검색 청크로 변환한다."""

import re
from collections.abc import Iterable
from pathlib import Path

import pymupdf
from pydantic import BaseModel

from .bm25 import BM25Index
from .embedder import BGEEmbedder
from .store import ChromaStore

HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+){0,4}|[IVXLC]+)\.?\s+[A-Z][A-Za-z0-9 ,:/()\-]{3,}$"
)
TABLE_PATTERN = re.compile(r"^\s*(?:Table|TABLE)\s+\d+", re.IGNORECASE)


class ParsedPage(BaseModel):
    page: int
    text: str


class IngestChunk(BaseModel):
    chunk_id: str
    seq: int
    doc_id: str
    doc_title: str
    page: int
    section: str | None = None
    text: str
    chunk_type: str = "text"


def _normalize_pdf_text(text: str) -> str:
    """PDF 줄바꿈으로 생긴 단어 중간 하이픈을 복원한다.

    예를 들어 ``inter-\nface``를 ``interface``로 바꿔, LLM이 사용하는
    자연스러운 인용문도 원문 청크의 공백 정규화 비교를 통과하게 한다.
    """
    return re.sub(r"(?<=\w)-\n(?=\w)", "", text)


def parse_pdf(pdf_path: str | Path) -> list[ParsedPage]:
    """PDF를 페이지 단위 텍스트로 추출한다. 스캔본은 명시적으로 실패시킨다."""
    document = pymupdf.open(pdf_path)
    try:
        pages = [
            ParsedPage(
                page=page_number,
                text=_normalize_pdf_text(page.get_text("text")).strip(),
            )
            for page_number, page in enumerate(document, start=1)
        ]
    finally:
        document.close()
    if not any(page.text for page in pages):
        raise ValueError(f"No extractable text found in {pdf_path}; OCR is required")
    return pages


def _split_regions(
    page: ParsedPage, current_section: str | None
) -> tuple[list[tuple[str, str, str | None]], str | None]:
    """페이지를 일반 본문과 표 캡션 이후 영역으로 나누고 현재 절 제목을 유지한다."""
    lines = [line.strip() for line in page.text.splitlines() if line.strip()]
    regions: list[tuple[str, str, str | None]] = []
    buffer: list[str] = []
    chunk_type = "text"
    section = current_section

    def flush() -> None:
        if buffer:
            regions.append((chunk_type, "\n".join(buffer), section))
            buffer.clear()

    for line in lines:
        if HEADING_PATTERN.match(line):
            flush()
            section = line
            chunk_type = "text"
            continue
        if TABLE_PATTERN.match(line):
            flush()
            chunk_type = "table"
        buffer.append(line)
    flush()
    return regions, section


def _token_length(text: str, tokenizer: object | None) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def _tail_tokens(text: str, tokenizer: object | None, count: int) -> str:
    if tokenizer is None:
        return " ".join(text.split()[-count:])
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[-count:], skip_special_tokens=True)


def _split_oversized_unit(
    text: str, tokenizer: object | None, *, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """문장 하나가 토큰 예산을 넘을 때 공백 경계에서 안전하게 분할한다."""
    words = re.findall(r"\S+(?:\s+|$)", text)
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and _token_length(candidate, tokenizer) > max_tokens:
            chunks.append(current)
            current = (
                f"{_tail_tokens(current, tokenizer, overlap_tokens)} {word}".strip()
            )
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_to_token_budget(
    text: str, tokenizer: object | None, *, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """절 내부도 BGE-M3 토크나이저 기준 최대 길이를 넘지 않게 분할한다."""
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        # PDF는 줄바꿈만 있고 빈 줄이 없는 경우가 많으므로 문장 경계도 fallback으로 쓴다.
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", paragraph)
            if part.strip()
        ]
        units.extend(sentences or [paragraph])
    chunks: list[str] = []
    current = ""
    for unit in units:
        if _token_length(unit, tokenizer) > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                _split_oversized_unit(
                    unit,
                    tokenizer,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            continue
        for part in [unit]:
            candidate = f"{current}\n\n{part}".strip()
            if current and _token_length(candidate, tokenizer) > max_tokens:
                chunks.append(current)
                current = f"{_tail_tokens(current, tokenizer, overlap_tokens)}\n\n{part}".strip()
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_document(
    pages: Iterable[ParsedPage],
    *,
    doc_id: str,
    doc_title: str,
    tokenizer: object | None = None,
    max_tokens: int = 1_500,
    overlap_tokens: int = 120,
) -> list[IngestChunk]:
    """절 경계를 우선하며, 긴 영역만 토큰 예산 내로 쪼갠다.

    짧은 절·표는 문맥 보존을 위해 1,000 토큰보다 작아도 분리하지 않는다.
    """
    if max_tokens <= overlap_tokens:
        raise ValueError("max_tokens must be greater than overlap_tokens")
    result: list[IngestChunk] = []
    current_section: str | None = None
    sequence = 1
    for page in pages:
        regions, current_section = _split_regions(page, current_section)
        for chunk_type, text, section in regions:
            for part in _split_to_token_budget(
                text,
                tokenizer,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            ):
                result.append(
                    IngestChunk(
                        chunk_id=f"{doc_id}:{page.page:03d}:{sequence:02d}",
                        seq=sequence,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        page=page.page,
                        section=section,
                        text=part,
                        chunk_type=chunk_type,
                    )
                )
                sequence += 1
    return result


def build_index(
    documents: Iterable[tuple[str, str, str | Path]],
    *,
    chroma_dir: str | Path,
    embedding_model: str = "BAAI/bge-m3",
    rebuild: bool = False,
) -> dict[str, int]:
    """문서 전체를 Chroma와 BM25에 동기화하고 문서별 청크 수를 반환한다."""
    embedder = BGEEmbedder(embedding_model)
    all_chunks: list[IngestChunk] = []
    counts: dict[str, int] = {}
    for doc_id, title, pdf_path in documents:
        chunks = chunk_document(
            parse_pdf(pdf_path),
            doc_id=doc_id,
            doc_title=title,
            tokenizer=embedder.tokenizer,
        )
        if not chunks:
            raise ValueError(f"No chunks produced for {pdf_path}")
        all_chunks.extend(chunks)
        counts[doc_id] = len(chunks)

    embeddings = embedder.embed_documents([chunk.text for chunk in all_chunks])
    ChromaStore(chroma_dir).write(all_chunks, embeddings, rebuild=rebuild)
    BM25Index.build(all_chunks).save(Path(chroma_dir) / "bm25.pkl")
    return counts
