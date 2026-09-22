"""반대 근거 탐색 노드 (E). CONTRACTS §6.

`evidence_gap.opposing_missing` 항목별로 `web_search`(기술 계열 단위) + `retriever`(서베이 2편 우선) 각 1회.
결과는 `Evidence`(`evidence_id="{tech}-COUNTER-{NN}"`)로만 반환하고 의견·판정을 붙이지 않는다. 그래프에서 1회 상한.
"""

import logging
from typing import Any

from techeval.agents._deps import Deps
from techeval.control.query_rewrite import ITEM_KEYWORDS
from techeval.schemas import SURVEY_DOC_IDS, Evidence, EvidenceGap, TechRef, get_tech

logger = logging.getLogger(__name__)

QUOTE_MAX_CHARS = 400
COUNTER_SUFFIX = "limitations drawbacks overhead criticism"


def _counter_query(tech: TechRef, criterion_id: str) -> str:
    kw = ITEM_KEYWORDS.get(criterion_id, [criterion_id])[0]
    return f"{tech.family} {kw} {COUNTER_SUFFIX}"


def _quote_from_text(text: str) -> str:
    """청크 text의 앞부분을 그대로 잘라 인용한다 (V6: 부분 문자열 유지)."""
    return text[:QUOTE_MAX_CHARS].strip()


def chunk_to_evidence(chunk: Any, *, evidence_id: str, accessed: str) -> Evidence:
    section = getattr(chunk, "section", None)
    loc = f"{chunk.doc_id} p.{chunk.page}" + (f" §{section}" if section else "")
    return Evidence(
        evidence_id=evidence_id,
        source_type="paper",
        unit="family",
        quote=_quote_from_text(chunk.text),
        locator=loc,
        title=getattr(chunk, "doc_title", None),
        doc_id=chunk.doc_id,
        page=chunk.page,
        section=section,
        chunk_id=chunk.chunk_id,
        accessed_date=accessed[:10],
    )


def web_to_evidence(r: Any, *, evidence_id: str, accessed: str) -> Evidence:
    kind = getattr(r, "source_kind", "other")
    source_type = "official" if kind == "official" else "paper" if kind == "paper" else "web"
    text = getattr(r, "content", None) or r.snippet
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        unit="family",
        quote=_quote_from_text(text),
        locator=r.url,
        title=r.title,
        publisher=getattr(r, "publisher", None),
        published_date=getattr(r, "published_date", None),
        url=r.url,
        accessed_date=accessed[:10],
    )


def search_counter_evidence(
    gap: EvidenceGap,
    deps: Deps,
    *,
    technologies: list[TechRef] | None = None,
    web_results_per_item: int = 2,
    chunks_per_item: int = 2,
) -> list[Evidence]:
    """opposing_missing 항목별 web 1회 + retriever 1회.

    검색 실패는 로그 후 건너뛴다 (빈 항목은 not_public이 아니라 '탐색 없음').
    """
    out: list[Evidence] = []
    seq: dict[str, int] = {}
    now = deps.now()

    def next_id(tech_id: str) -> str:
        seq[tech_id] = seq.get(tech_id, 0) + 1
        return f"{tech_id}-COUNTER-{seq[tech_id]:02d}"

    for item in gap.opposing_missing:
        tech = (
            get_tech(item["tech_id"])
            if technologies is None
            else next(t for t in technologies if t.tech_id == item["tech_id"])
        )
        cid = item["criterion_id"]
        query = _counter_query(tech, cid)
        logger.info("counter_evidence_search %s/%s: %r", tech.tech_id, cid, query)

        web_results = deps.web_search(query, max_results=web_results_per_item)
        for r in web_results[:web_results_per_item]:
            out.append(web_to_evidence(r, evidence_id=next_id(tech.tech_id), accessed=now))

        chunks = deps.retriever.search(query, top_k=chunks_per_item, doc_ids=list(SURVEY_DOC_IDS))
        for c in chunks[:chunks_per_item]:
            out.append(chunk_to_evidence(c, evidence_id=next_id(tech.tech_id), accessed=now))

        if not web_results and not chunks:
            logger.warning("counter_evidence_search %s/%s: 결과 없음 (%r)", tech.tech_id, cid, query)

    logger.info("counter_evidence_search: %d evidence", len(out))
    return out
