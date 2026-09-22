import json
import re
from pathlib import Path

import pytest

from techeval.retrieval import RetrievedChunk, StubRetriever, rrf_fuse
from techeval.retrieval.ingest import ParsedPage, chunk_document, parse_pdf


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="deepseek_v2",
        doc_title="DeepSeek-V2",
        page=1,
        text="MLA compresses the KV cache into a latent vector.",
        score=score,
    )


def test_rrf_fuse_deduplicates_and_preserves_source_scores() -> None:
    first = _chunk("deepseek_v2:001:01", 0.9)
    second = _chunk("deepseek_v2:001:02", 0.8)
    result = rrf_fuse([first, second], [second, first])

    assert [chunk.chunk_id for chunk in result] == [
        "deepseek_v2:001:01",
        "deepseek_v2:001:02",
    ]
    assert result[0].dense_score == 0.9
    assert result[0].bm25_score == 0.9


def test_chunk_to_evidence_rejects_non_verbatim_quote() -> None:
    chunk = _chunk("deepseek_v2:001:01", 0.9)
    with pytest.raises(ValueError, match="substring"):
        chunk.to_evidence(evidence_id="mla-T1-01", quote="invented claim")


def test_chunk_to_evidence_builds_contract_evidence() -> None:
    chunk = _chunk("deepseek_v2:001:01", 0.9)
    evidence = chunk.to_evidence(
        evidence_id="mla-T1-01",
        quote="MLA compresses the KV cache into a latent vector.",
    )

    assert evidence.doc_id == "deepseek_v2"
    assert evidence.locator == "deepseek_v2 p.1"
    assert evidence.chunk_id == chunk.chunk_id


def test_stub_retriever_applies_doc_filter() -> None:
    retriever = StubRetriever("tests/fixtures/chunks.json")
    chunks = retriever.search("CXL", doc_ids=["pim_cxl_1m"], top_k=10)

    assert len(chunks) == 5
    assert {chunk.doc_id for chunk in chunks} == {"pim_cxl_1m"}


def test_chunk_document_enforces_token_budget_and_records_sequence() -> None:
    pages = [ParsedPage(page=1, text="1. Introduction\n" + "word " * 40)]
    chunks = chunk_document(
        pages,
        doc_id="deepseek_v2",
        doc_title="DeepSeek-V2",
        max_tokens=10,
        overlap_tokens=2,
    )

    assert len(chunks) > 1
    assert [chunk.seq for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(len(chunk.text.split()) <= 10 for chunk in chunks)


@pytest.mark.integration
def test_all_papers_parse_and_include_table_chunks() -> None:
    papers = (
        ("deepseek_v2", "DeepSeek-V2", "data/papers/Deepseek_v2.pdf"),
        ("pim_cxl_1m", "PIM/CXL", "data/papers/PIM:CXL_KVcache.pdf"),
        ("io_survey", "I/O survey", "data/papers/LLM_storage_HW_survey.pdf"),
        ("kv_survey", "KV survey", "data/papers/KV_manage_survey.pdf"),
    )
    for doc_id, title, filename in papers:
        chunks = chunk_document(
            parse_pdf(Path(filename)), doc_id=doc_id, doc_title=title
        )
        assert len(chunks) >= 5
        assert any(chunk.chunk_type == "table" for chunk in chunks)


@pytest.mark.integration
def test_fixture_chunks_are_verbatim_substrings_of_the_papers() -> None:
    papers = (
        ("deepseek_v2", "DeepSeek-V2", "data/papers/Deepseek_v2.pdf"),
        ("pim_cxl_1m", "PIM/CXL", "data/papers/PIM:CXL_KVcache.pdf"),
        ("io_survey", "I/O survey", "data/papers/LLM_storage_HW_survey.pdf"),
        ("kv_survey", "KV survey", "data/papers/KV_manage_survey.pdf"),
    )
    original = {}
    for doc_id, title, filename in papers:
        original.update(
            {
                chunk.chunk_id: chunk
                for chunk in chunk_document(
                    parse_pdf(Path(filename)), doc_id=doc_id, doc_title=title
                )
            }
        )

    fixture = json.loads(Path("tests/fixtures/chunks.json").read_text(encoding="utf-8"))
    assert {item["doc_id"] for item in fixture} == {item[0] for item in papers}
    for doc_id, _, _ in papers:
        assert sum(item["doc_id"] == doc_id for item in fixture) >= 5

    normalize = lambda text: re.sub(r"\s+", " ", text).strip()
    for item in fixture:
        assert item["chunk_id"] in original
        assert normalize(item["text"]) in normalize(original[item["chunk_id"]].text)


@pytest.mark.integration
def test_vector_retriever_cross_language_and_bm25_contract() -> None:
    chroma_dir = Path("data/chroma")
    if not (chroma_dir / "bm25.pkl").exists():
        pytest.skip("Run scripts/ingest.py before vector integration tests")

    from techeval.retrieval import VectorRetriever

    retriever = VectorRetriever(str(chroma_dir))
    mla = retriever.search("MLA의 KV cache 감소율", top_k=3)
    cxl = retriever.search("CXL 메모리로 KV cache 확장 실험 하드웨어", top_k=3)
    pnm = retriever.search("PNM", top_k=1, mode="bm25")
    survey = retriever.search("KV cache taxonomy", top_k=5, doc_ids=["kv_survey"])

    assert any(chunk.doc_id == "deepseek_v2" for chunk in mla)
    assert any(chunk.doc_id == "pim_cxl_1m" for chunk in cxl)
    assert pnm[0].doc_id == "pim_cxl_1m"
    assert {chunk.doc_id for chunk in survey} == {"kv_survey"}
