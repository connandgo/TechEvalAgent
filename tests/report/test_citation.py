import pytest

from techeval.report.citation import (
    build_reference_section,
    collect_cited_ids,
    format_citations,
    format_reference,
    normalize_citations,
    reference_ids,
)
from techeval.schemas import Evidence


def _paper(eid: str, page: int) -> Evidence:
    return Evidence(
        evidence_id=eid,
        source_type="paper",
        unit="paper",
        quote="q",
        locator=f"deepseek_v2 p.{page}",
        title="DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
        authors="DeepSeek-AI",
        publisher="arXiv:2405.04434",
        published_date="2024-06-19",
        doc_id="deepseek_v2",
        page=page,
    )


def _web(eid: str, **kw) -> Evidence:
    base = {
        "evidence_id": eid,
        "source_type": "official",
        "unit": "family",
        "quote": "q",
        "locator": "https://example.com/cmm-d",
        "url": "https://example.com/cmm-d",
        "title": "CXL Memory Module",
        "publisher": "Samsung Semiconductor",
        "published_date": "2024-05-02",
        "accessed_date": "2026-09-20",
    }
    return Evidence(**(base | kw))


def test_format_paper():
    assert format_reference(_paper("mla-T1-01", 4)) == (
        "DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. "
        "arXiv:2405.04434, p. 4."
    )


def test_format_patent():
    e = Evidence(
        evidence_id="pim_cxl-M3-09",
        source_type="patent",
        unit="family",
        quote="q",
        locator="US 11,000,000 B2",
        url="https://patents.example/US11000000",
        title="Near-memory processing for KV cache",
        authors="Samsung Electronics",
        published_date="2025-03-11",
    )
    assert format_reference(e) == (
        "Samsung Electronics(2025-03). Near-memory processing for KV cache, US 11,000,000 B2, "
        "https://patents.example/US11000000"
    )


def test_format_web_with_and_without_date():
    assert format_reference(_web("pim_cxl-M2-01")) == (
        "Samsung Semiconductor(2024-05-02). CXL Memory Module. Samsung Semiconductor, https://example.com/cmm-d"
    )
    undated = format_reference(_web("pim_cxl-M2-01", published_date=None, source_type="web"))
    assert undated.startswith("Samsung Semiconductor(n.d., 확인일 2026-09-20). ")


def test_inference_and_not_public_are_not_references():
    inf = Evidence(
        evidence_id="mla-S2-03",
        source_type="inference",
        unit="family",
        quote="추론",
        locator="inference",
    )
    with pytest.raises(ValueError):
        format_reference(inf)


def test_collect_cited_ids_order_and_multi():
    text = "a[E: mla-T1-01] b[E: mla-T2-01, mla-T1-01] c[E:pim_cxl-T1-01]"
    assert collect_cited_ids(text) == ["mla-T1-01", "mla-T2-01", "pim_cxl-T1-01"]


def test_reference_merges_same_paper_and_skips_uncited_and_inference():
    np = Evidence(
        evidence_id="mla-M1-01",
        source_type="not_public",
        unit="family",
        quote="",
        locator="not_found",
        search_query="q",
        searched_at="2026-09-20",
    )
    index = {
        e.evidence_id: e
        for e in [
            _paper("mla-T1-01", 13),
            _paper("mla-D1-01", 1),
            _web("pim_cxl-M2-01"),
            _paper("mla-UNUSED-01", 7),
            np,
        ]
    }
    body = "x[E: mla-T1-01] y[E: pim_cxl-M2-01] z[E: mla-D1-01] w[E: mla-M1-01]"
    section = build_reference_section(body, index)
    lines = [ln for ln in section.splitlines() if ln[:1].isdigit()]
    assert len(lines) == 2  # 같은 논문 2건 병합 + 웹 1건, not_public 제외
    assert "pp. 1, 13" in lines[0]
    assert set(reference_ids(section)) == {"mla-T1-01", "mla-D1-01", "pim_cxl-M2-01"}
    assert "mla-UNUSED-01" not in section


def test_citations_use_one_id_per_bracket():
    """E의 judge는 `[E: id]` 한 괄호 한 id만 읽는다. 출력은 항상 이 형식으로 맞춘다."""
    assert format_citations(["mla-T1-01", "mla-T1-02", "mla-T1-01"]) == "[E: mla-T1-01][E: mla-T1-02]"
    assert normalize_citations("근거[E: mla-T1-01, mla-T2-01].") == "근거[E: mla-T1-01][E: mla-T2-01]."
