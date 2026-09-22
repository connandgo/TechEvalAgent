"""본문 인용([E: id]) 수집과 REFERENCE 포맷 (docs/CRITERIA.md §5)."""

import logging
import re
from collections.abc import Iterable

from techeval.schemas import CriterionResult, Evidence, TechProfile

logger = logging.getLogger(__name__)

REFERENCE_HEADING = "## REFERENCE"
# 본문 인용 표기. 출력은 항상 `[E: a][E: b]`(format_citations)이고, 읽을 때는 "[E: a, b]"도 허용한다.
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
                logger.warning("evidence_id 중복(내용 다름): %s — 첫 항목 사용", e.evidence_id)
            index.setdefault(e.evidence_id, e)
    return index


def format_citations(ids: list[str], sep: str = "") -> str:
    """인용 표기. CRITERIA §5 형식대로 id마다 괄호 하나: `[E: a][E: b]` (E의 judge도 이 형식만 읽는다).
    표 셀에서는 sep=" "로 괄호 사이에 공백을 둬 PDF 표가 줄바꿈할 수 있게 한다."""
    return sep.join(f"[E: {i}]" for i in dict.fromkeys(ids))


def normalize_citations(text: str) -> str:
    """`[E: a, b]`처럼 한 괄호에 여러 id를 쓴 표기를 `[E: a][E: b]`로 펼친다."""
    return CITATION_RE.sub(
        lambda m: format_citations([i.strip() for i in m.group(1).split(",") if i.strip()]),
        text,
    )


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
    return f"p. {uniq[0]}" if len(uniq) == 1 else "pp. " + ", ".join(str(p) for p in uniq)


def _format_paper(e: Evidence, pages: list[int]) -> str:
    # 저자(YYYY). 제목. 학술지/arXiv, 권(호), 페이지.  — 권(호)는 Evidence에 필드가 없어 생략
    head = f"{e.authors or '저자 미상'}({_year(e)})."  # 발행처(arXiv 등)를 저자 자리에 쓰지 않는다
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
        raise ValueError(f"{e.source_type} evidence는 참고문헌으로 포맷하지 않는다: {e.evidence_id}")
    if e.source_type == "paper":
        return _format_paper(e, pages if pages is not None else ([e.page] if e.page else []))
    if e.source_type == "patent":
        return _format_patent(e)
    return _format_web(e)


def _norm_url(url: str | None) -> str | None:
    """비교용 URL: 스킴·www·끝 슬래시·arXiv 버전(v2)·abs/pdf 차이를 없앤다."""
    if not url or not url.startswith("http"):
        return None
    u = re.sub(r"^https?://(www\.)?", "", url.strip().lower()).rstrip("/")
    u = re.sub(r"arxiv\.org/(abs|pdf)/([\d.]+)(v\d+)?(\.pdf)?", r"arxiv.org/\2", u)
    return u


def _norm_title(title: str | None) -> str | None:
    t = re.sub(r"[^0-9a-z가-힣]", "", (title or "").lower())
    return t or None


def _merge_keys(e: Evidence) -> set[str]:
    """이 evidence가 가리키는 문헌을 식별하는 키들. 키가 하나라도 겹치면 같은 문헌이다."""
    kind = "paper" if e.source_type == "paper" else e.source_type
    keys = {f"{kind}:url:{u}" for u in (_norm_url(e.url), _norm_url(e.locator)) if u}
    if e.source_type == "paper":
        if e.doc_id:
            keys.add(f"paper:doc:{e.doc_id}")
        if t := _norm_title(e.title):
            keys.add(f"paper:title:{t}")
    if not keys:
        keys.add(f"{kind}:loc:{e.locator}|{e.title}")
    return keys


def _group_same_source(evs: list[Evidence]) -> list[list[Evidence]]:
    """키를 공유하는 evidence를 한 문헌으로 묶는다(연결 요소). 첫 인용 순서를 유지한다."""
    parent = list(range(len(evs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, e in enumerate(evs):
        for k in _merge_keys(e):
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i
    groups: dict[int, list[Evidence]] = {}
    for i, e in enumerate(evs):
        groups.setdefault(find(i), []).append(e)
    return sorted(groups.values(), key=lambda g: evs.index(g[0]))


def _representative(group: list[Evidence]) -> Evidence:
    """병합된 문헌의 대표 메타데이터: 저자·코퍼스 doc_id·날짜·제목이 있는 쪽을 우선한다."""
    return max(
        group,
        key=lambda e: (bool(e.authors), bool(e.doc_id), bool(e.published_date), bool(e.url), len(e.title or "")),
    )


def reference_entries(cited_ids: list[str], evidence_index: dict[str, Evidence]) -> list[tuple[str, list[str]]]:
    """(참고문헌 문자열, 병합된 evidence_id 목록) 리스트. 인용 순서 유지, inference/not_public 제외.

    같은 논문을 코퍼스 PDF(doc_id)와 웹(arXiv URL)으로 따로 인용해도, doc_id·정규화 URL·정규화 제목 중
    하나라도 겹치면 1항목으로 합치고 페이지를 모두 나열한다.
    """
    evs: list[Evidence] = []
    for eid in cited_ids:
        e = evidence_index.get(eid)
        if e is None:
            logger.warning("REFERENCE: 본문 인용 id가 evidence 인덱스에 없음: %s", eid)
            continue
        if e.source_type not in NON_REFERENCE_TYPES:
            evs.append(e)
    entries = []
    for group in _group_same_source(evs):
        rep = _representative(group)
        pages = [e.page for e in group if e.page]
        if not rep.url:
            url = next((e.url for e in group if e.url), None)
            rep = rep.model_copy(update={"url": url}) if url else rep
        entries.append((format_reference(rep, pages=pages), [e.evidence_id for e in group]))
    return entries


def build_reference_section(report_body: str, evidence_index: dict[str, Evidence]) -> str:
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
