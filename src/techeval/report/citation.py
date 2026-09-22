"""본문 인용([E: id]) 수집과 REFERENCE 포맷 (docs/CRITERIA.md §5)."""

import logging
import re
from collections.abc import Iterable

from techeval.schemas import CriterionResult, Evidence, TechProfile

logger = logging.getLogger(__name__)

REFERENCE_HEADING = "## REFERENCE"
# 본문 인용 표기. "[E: a, b]"처럼 한 괄호에 여러 id를 쉼표로 적는 것도 허용한다.
CITATION_RE = re.compile(r"\[E:\s*([^\]]+?)\s*\]")
# REFERENCE 항목 끝에 붙는 근거 id 목록 표기. lint가 본문 인용 집합과 대조할 때 쓴다.
REF_IDS_RE = re.compile(r"\(근거 ID:\s*([^)]+)\)\s*$")
NON_REFERENCE_TYPES = ("inference", "not_public")


def build_evidence_index(
    *,
    results: Iterable[CriterionResult] = (),
    profiles: Iterable[TechProfile] = (),
    counter_evidence: Iterable[Evidence] = (),
) -> dict[str, Evidence]:
    """State 안의 모든 Evidence를 evidence_id로 색인한다. 같은 id가 두 번 나오면 먼저 나온 것을 쓴다."""
    index: dict[str, Evidence] = {}
    groups: list[Iterable[Evidence]] = [p.evidence for p in profiles]
    groups += [r.evidence for r in results]
    groups.append(counter_evidence)
    for group in groups:
        for e in group:
            if e.evidence_id in index and index[e.evidence_id] != e:
                logger.warning(
                    "evidence_id 중복(내용 다름): %s — 첫 항목 사용", e.evidence_id
                )
            index.setdefault(e.evidence_id, e)
    return index


def collect_cited_ids(text: str) -> list[str]:
    """본문에 등장한 evidence_id를 첫 등장 순서대로 중복 없이 반환한다."""
    seen: dict[str, None] = {}
    for m in CITATION_RE.finditer(text):
        for raw in m.group(1).split(","):
            eid = raw.strip()
            if eid:
                seen.setdefault(eid, None)
    return list(seen)


def _year(e: Evidence) -> str:
    return e.published_date[:4] if e.published_date else "n.d."


def _url(e: Evidence) -> str | None:
    if e.url:
        return e.url
    return e.locator if e.locator.startswith("http") else None


def _pages(pages: list[int]) -> str | None:
    uniq = sorted(set(pages))
    if not uniq:
        return None
    return (
        f"p. {uniq[0]}" if len(uniq) == 1 else "pp. " + ", ".join(str(p) for p in uniq)
    )


def _format_paper(e: Evidence, pages: list[int]) -> str:
    # 저자(YYYY). 제목. 학술지/arXiv, 권(호), 페이지.  — 권(호)는 Evidence에 필드가 없어 생략
    head = f"{e.authors or e.publisher or '저자 미상'}({_year(e)})."
    tail = [x for x in (e.publisher, _pages(pages), _url(e)) if x]
    parts = [head]
    if e.title:
        parts.append(f"{e.title}.")
    if tail:
        parts.append(", ".join(tail) + ".")
    return " ".join(parts)


def _format_patent(e: Evidence) -> str:
    # 출원인(YYYY-MM). 특허명, 번호, URL
    date = e.published_date[:7] if e.published_date else "n.d."
    number = e.locator if not e.locator.startswith("http") else None
    body = ", ".join(x for x in (e.title, number, _url(e)) if x)
    return f"{e.authors or e.publisher or '출원인 미상'}({date}). {body}"


def _format_web(e: Evidence) -> str:
    # 기관(YYYY-MM-DD). 제목. 사이트명, URL
    org = e.authors or e.publisher or "기관 미상"
    if e.published_date:
        date = e.published_date
    else:
        date = f"n.d., 확인일 {e.accessed_date}" if e.accessed_date else "n.d."
    parts = [f"{org}({date})."]
    if e.title:
        parts.append(f"{e.title}.")
    tail = ", ".join(x for x in (e.publisher, _url(e)) if x)
    if tail:
        parts.append(tail)
    return " ".join(parts)


def format_reference(e: Evidence, *, pages: list[int] | None = None) -> str:
    """참고문헌 1줄. pages는 같은 논문의 여러 evidence를 병합할 때 넘긴다(기본은 e.page)."""
    if e.source_type in NON_REFERENCE_TYPES:
        raise ValueError(
            f"{e.source_type} evidence는 참고문헌으로 포맷하지 않는다: {e.evidence_id}"
        )
    if e.source_type == "paper":
        return _format_paper(
            e, pages if pages is not None else ([e.page] if e.page else [])
        )
    if e.source_type == "patent":
        return _format_patent(e)
    return _format_web(e)


def _merge_key(e: Evidence) -> str:
    if e.source_type == "paper":
        if e.doc_id:
            return f"paper:{e.doc_id}"
        return f"paper:{e.title}|{e.authors}"
    return f"{e.source_type}:{_url(e) or e.locator}|{e.title}"


def reference_entries(
    cited_ids: list[str], evidence_index: dict[str, Evidence]
) -> list[tuple[str, list[str]]]:
    """(참고문헌 문자열, 병합된 evidence_id 목록) 리스트. 인용 순서 유지, inference/not_public 제외."""
    groups: dict[str, list[Evidence]] = {}
    for eid in cited_ids:
        e = evidence_index.get(eid)
        if e is None:
            logger.warning("REFERENCE: 본문 인용 id가 evidence 인덱스에 없음: %s", eid)
            continue
        if e.source_type in NON_REFERENCE_TYPES:
            continue
        groups.setdefault(_merge_key(e), []).append(e)
    entries = []
    for evs in groups.values():
        pages = [e.page for e in evs if e.page]
        entries.append(
            (format_reference(evs[0], pages=pages), [e.evidence_id for e in evs])
        )
    return entries


def build_reference_section(
    report_body: str, evidence_index: dict[str, Evidence]
) -> str:
    """본문의 [E: id] 인용만 모아 REFERENCE 절 마크다운을 만든다."""
    entries = reference_entries(collect_cited_ids(report_body), evidence_index)
    lines = [REFERENCE_HEADING, ""]
    if not entries:
        lines.append("본문에서 인용한 문헌·웹 자료가 없다.")
    for i, (text, ids) in enumerate(entries, start=1):
        lines.append(f"{i}. {text} (근거 ID: {', '.join(ids)})")
    return "\n".join(lines) + "\n"


def reference_ids(reference_section: str) -> list[str]:
    """REFERENCE 절에 적힌 근거 id 목록을 모은다(lint용)."""
    ids: list[str] = []
    for line in reference_section.splitlines():
        m = REF_IDS_RE.search(line.strip())
        if m:
            ids += [x.strip() for x in m.group(1).split(",") if x.strip()]
    return ids
