"""(C) 공통 웹 검색 도구 — WebResult 통일 형식, 출처 분류, 날짜 파싱, not_public 기록.

CONTRACTS.md §4의 계약을 구현한다. B(T3·D4)와 E(counter_evidence)도 이 모듈을 쓴다.

구현 범위 (역할 C 문서 §4의 1단계):
  WebResult / WebResult.to_evidence / not_public_evidence / 출처 분류 / 날짜 파싱
실제 provider 호출(`web_search`)과 캐시는 2·3단계에서 이 파일에 추가한다.
"""

import logging
import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator

from techeval.schemas import Evidence, EvidenceUnit, SourceType

logger = logging.getLogger(__name__)

SourceKind = Literal["official", "news", "blog", "paper", "forum", "other"]

#: 본문 추출 상한 (역할 C 문서 §4-2)
MAX_CONTENT_CHARS = 4000

# --- 출처 분류 표 -------------------------------------------------------------
# AGENTS.md 8: 벤더 발표 자료를 news/paper로 표시하면 E의 근거 비대칭 계산이 틀어진다.

#: 도메인 → 발행 주체 표기. 역할 C 문서 §5.1의 PUBLISHER_MAP.
PUBLISHER_MAP: dict[str, str] = {
    "arxiv.org": "arXiv",
    "acm.org": "ACM",
    "dl.acm.org": "ACM",
    "ieee.org": "IEEE",
    "ieeexplore.ieee.org": "IEEE",
    "openreview.net": "OpenReview",
    "usenix.org": "USENIX",
    "nvidia.com": "NVIDIA",
    "developer.nvidia.com": "NVIDIA",
    "blogs.nvidia.com": "NVIDIA",
    "samsung.com": "Samsung",
    "news.samsung.com": "Samsung",
    "semiconductor.samsung.com": "Samsung",
    "skhynix.com": "SK hynix",
    "news.skhynix.com": "SK hynix",
    "micron.com": "Micron",
    "intel.com": "Intel",
    "amd.com": "AMD",
    "marvell.com": "Marvell",
    "astera-labs.com": "Astera Labs",
    "computeexpresslink.org": "CXL Consortium",
    "jedec.org": "JEDEC",
    "deepseek.com": "DeepSeek",
    "api-docs.deepseek.com": "DeepSeek",
    "huggingface.co": "Hugging Face",
    "github.com": "GitHub",
    "docs.vllm.ai": "vLLM",
    "blog.vllm.ai": "vLLM",
    "lmsys.org": "LMSYS",
    "docs.sglang.ai": "SGLang",
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "tomshardware.com": "Tom's Hardware",
    "servethehome.com": "ServeTheHome",
    "theregister.com": "The Register",
    "nextplatform.com": "The Next Platform",
    "eetimes.com": "EE Times",
    "trendforce.com": "TrendForce",
    "semianalysis.com": "SemiAnalysis",
}

#: 벤더·제안사·표준화 기구 공식 도메인 → source_kind="official"
OFFICIAL_DOMAINS: frozenset[str] = frozenset(
    {
        "nvidia.com",
        "developer.nvidia.com",
        "blogs.nvidia.com",
        "samsung.com",
        "news.samsung.com",
        "semiconductor.samsung.com",
        "skhynix.com",
        "news.skhynix.com",
        "micron.com",
        "intel.com",
        "amd.com",
        "marvell.com",
        "astera-labs.com",
        "computeexpresslink.org",
        "jedec.org",
        "deepseek.com",
        "api-docs.deepseek.com",
        "docs.vllm.ai",
        "blog.vllm.ai",
        "docs.sglang.ai",
    }
)

#: 학술 출판 도메인 → source_kind="paper"
PAPER_DOMAINS: frozenset[str] = frozenset(
    {
        "arxiv.org",
        "acm.org",
        "dl.acm.org",
        "ieee.org",
        "ieeexplore.ieee.org",
        "openreview.net",
        "usenix.org",
        "sciencedirect.com",
        "springer.com",
        "link.springer.com",
        "mlsys.org",
        "proceedings.neurips.cc",
        "aclanthology.org",
    }
)

#: 언론 도메인 → source_kind="news"
NEWS_DOMAINS: frozenset[str] = frozenset(
    {
        "reuters.com",
        "bloomberg.com",
        "tomshardware.com",
        "servethehome.com",
        "theregister.com",
        "nextplatform.com",
        "eetimes.com",
        "trendforce.com",
        "anandtech.com",
        "datacenterdynamics.com",
        "hpcwire.com",
        "zdnet.com",
        "techcrunch.com",
        "cnbc.com",
        "wsj.com",
        "ft.com",
        "etnews.com",
        "thelec.kr",
        "businesskorea.co.kr",
    }
)

#: 개인 블로그·미디어 플랫폼 → source_kind="blog"
BLOG_DOMAINS: frozenset[str] = frozenset(
    {
        "medium.com",
        "substack.com",
        "blogspot.com",
        "wordpress.com",
        "dev.to",
        "velog.io",
        "tistory.com",
        "semianalysis.com",
        "hashnode.dev",
    }
)

#: 포럼·커뮤니티 → source_kind="forum"
FORUM_DOMAINS: frozenset[str] = frozenset(
    {
        "reddit.com",
        "news.ycombinator.com",
        "ycombinator.com",
        "stackoverflow.com",
        "stackexchange.com",
        "quora.com",
        "discuss.pytorch.org",
        "forums.developer.nvidia.com",
    }
)

#: source_kind → Evidence.source_type 매핑 (CONTRACTS.md §4 to_evidence 주석)
_SOURCE_TYPE_BY_KIND: dict[str, SourceType] = {
    "official": "official",
    "paper": "paper",
    "news": "web",
    "blog": "web",
    "forum": "web",
    "other": "web",
}


def _now_iso() -> str:
    """현재 시각 ISO datetime (UTC)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_host(url: str) -> str:
    """URL에서 소문자 호스트를 뽑고 선행 www.를 제거한다."""
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def _host_candidates(host: str) -> list[str]:
    """호스트와 그 상위 도메인들을 긴 것부터 반환한다.

    news.samsung.com -> ["news.samsung.com", "samsung.com", "com"]
    서브도메인이 표에 없어도 등록 도메인으로 분류되게 한다.
    """
    parts = host.split(".")
    return [".".join(parts[i:]) for i in range(len(parts))]


def classify_source_kind(url: str) -> SourceKind:
    """URL의 도메인으로 출처 종류를 판정한다 (역할 C 문서 §5.1).

    official 비율은 E의 근거 비대칭 검사와 보고서 한계점에서 계산되므로 정확히 표기한다.
    판정 우선순위: paper > official > forum > blog > news. 표에 없으면 "other".
    """
    host = normalize_host(url)
    if not host:
        return "other"
    for candidate in _host_candidates(host):
        if candidate in PAPER_DOMAINS:
            return "paper"
        if candidate in OFFICIAL_DOMAINS:
            return "official"
        if candidate in FORUM_DOMAINS:
            return "forum"
        if candidate in BLOG_DOMAINS:
            return "blog"
        if candidate in NEWS_DOMAINS:
            return "news"
    return "other"


def extract_publisher(url: str) -> str | None:
    """PUBLISHER_MAP에 있으면 기관명, 없으면 등록 도메인을 반환한다."""
    host = normalize_host(url)
    if not host:
        return None
    for candidate in _host_candidates(host):
        if candidate in PUBLISHER_MAP:
            return PUBLISHER_MAP[candidate]
    parts = host.split(".")
    # co.kr / co.uk 같은 2단 접미사는 3단까지 남긴다.
    if len(parts) >= 3 and parts[-2] in ("co", "com", "or", "ac", "go", "net"):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


# --- published_date 파싱 ------------------------------------------------------
# 역할 C 문서 §5.1: 메타태그 → URL 패턴 → 본문 첫 날짜 순으로 시도. 실패 시 None(추정 금지).

_META_DATE_KEYS = (
    "article:published_time",
    "article:published",
    "citation_publication_date",
    "datepublished",
    "pubdate",
    "publish-date",
    "date",
    "dc.date",
    "dc.date.issued",
    "og:published_time",
)

_META_RE = re.compile(
    r"""<meta[^>]+?
        (?:property|name|itemprop)\s*=\s*["']([^"']+)["']
        [^>]*?content\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE | re.VERBOSE,
)
_META_REVERSED_RE = re.compile(
    r"""<meta[^>]+?content\s*=\s*["']([^"']+)["']
        [^>]*?(?:property|name|itemprop)\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE | re.VERBOSE,
)
_TIME_TAG_RE = re.compile(r"""<time[^>]+datetime\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

_URL_DATE_RES = (
    re.compile(r"/(20\d{2})[/\-](0[1-9]|1[0-2])[/\-](0[1-9]|[12]\d|3[01])(?:[/\-_.]|$)"),
    re.compile(r"[?&](?:date|d)=(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"),
)
#: arXiv ID는 발행 연월을 담는다: 2405.04434 -> 2024-05
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{2})(0[1-9]|1[0-2])\.\d{4,5}")

# 뒤가 T/시간으로 이어지는 ISO datetime도 잡되, 숫자나 -로 이어지면 날짜가 아니다.
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?![\d-])")
_MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_TEXT_DATE_RE = re.compile(r"\b([A-Z][a-z]{2,8})\.?\s+(\d{1,2}),?\s+(20\d{2})\b")
_KO_DATE_RE = re.compile(r"\b(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")


def _valid_ymd(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _normalize_date_string(raw: str) -> str | None:
    """메타태그 값 등 날짜 문자열을 "YYYY-MM-DD"로 정규화한다. 실패 시 None."""
    text = raw.strip()
    if not text:
        return None
    iso = _ISO_DATE_RE.search(text)
    if iso:
        return _valid_ymd(*(int(g) for g in iso.groups()))
    slash = re.search(r"\b(20\d{2})/(\d{1,2})/(\d{1,2})\b", text)
    if slash:
        return _valid_ymd(*(int(g) for g in slash.groups()))
    named = _TEXT_DATE_RE.search(text)
    if named:
        month = _MONTH_NAMES.get(named.group(1).lower())
        if month:
            return _valid_ymd(int(named.group(3)), month, int(named.group(2)))
    ko = _KO_DATE_RE.search(text)
    if ko:
        return _valid_ymd(*(int(g) for g in ko.groups()))
    return None


def parse_published_date(*, html: str | None = None, url: str | None = None, text: str | None = None) -> str | None:
    """발행일을 "YYYY-MM-DD"로 파싱한다. 못 찾으면 None — 추정하지 않는다.

    시도 순서 (역할 C 문서 §5.1): 메타태그 → <time datetime> → URL 패턴 → 본문 첫 날짜.
    """
    if html:
        found: dict[str, str] = {}
        for pattern, key_idx in ((_META_RE, 0), (_META_REVERSED_RE, 1)):
            for groups in pattern.findall(html):
                key = groups[key_idx].strip().lower()
                value = groups[1 - key_idx]
                found.setdefault(key, value)
        for key in _META_DATE_KEYS:
            if key in found:
                parsed = _normalize_date_string(found[key])
                if parsed:
                    return parsed
        time_tag = _TIME_TAG_RE.search(html)
        if time_tag:
            parsed = _normalize_date_string(time_tag.group(1))
            if parsed:
                return parsed

    if url:
        for pattern in _URL_DATE_RES:
            match = pattern.search(url)
            if match:
                parsed = _valid_ymd(*(int(g) for g in match.groups()))
                if parsed:
                    return parsed
        arxiv = _ARXIV_ID_RE.search(url)
        if arxiv:
            # arXiv ID는 일(day)을 담지 않으므로 연-월까지만 반환한다.
            return f"20{arxiv.group(1)}-{arxiv.group(2)}"

    if text:
        parsed = _normalize_date_string(text)
        if parsed:
            return parsed

    return None


# --- WebResult ----------------------------------------------------------------


class WebResult(BaseModel):
    """웹 검색 결과 1건. CONTRACTS.md §4의 형식을 그대로 따른다."""

    title: str
    url: str
    snippet: str
    content: str | None = None
    publisher: str | None = None
    published_date: str | None = None
    fetched_at: str
    source_kind: SourceKind = "other"
    query: str

    @field_validator("content")
    @classmethod
    def _cap_content(cls, v: str | None) -> str | None:
        """본문은 MAX_CONTENT_CHARS까지만 보관한다."""
        if v is None:
            return None
        return v[:MAX_CONTENT_CHARS]

    @property
    def accessed_date(self) -> str:
        """Evidence.accessed_date — fetched_at의 날짜 부분 (역할 C 문서 §5.4)."""
        return self.fetched_at[:10]

    @property
    def source_type(self) -> SourceType:
        """source_kind에서 파생되는 Evidence.source_type."""
        return _SOURCE_TYPE_BY_KIND[self.source_kind]

    def to_evidence(self, *, evidence_id: str, quote: str, unit: EvidenceUnit) -> Evidence:
        """이 결과를 Evidence로 변환한다.

        `quote`는 snippet 또는 content의 부분 문자열이어야 한다 (역할 C 문서 §5.4).
        지어낸 인용을 막기 위해 코드에서 검증한다 (AGENTS.md 규칙 1).
        """
        if not quote or not quote.strip():
            raise ValueError(f"{evidence_id}: quote is required for a web evidence")
        if not self._contains_quote(quote):
            raise ValueError(f"{evidence_id}: quote is not a substring of snippet/content of {self.url}")
        return Evidence(
            evidence_id=evidence_id,
            source_type=self.source_type,
            unit=unit,
            quote=quote.strip(),
            locator=self.url,
            title=self.title,
            publisher=self.publisher,
            published_date=self.published_date,
            url=self.url,
            accessed_date=self.accessed_date,
        )

    def _contains_quote(self, quote: str) -> bool:
        """공백 정규화 후 snippet/content에 quote가 있는지 확인한다 (V6과 같은 방식)."""
        needle = _squash_ws(quote)
        haystacks = (_squash_ws(self.snippet), _squash_ws(self.content or ""))
        return any(needle in h for h in haystacks if h)


def _squash_ws(text: str) -> str:
    """연속 공백을 하나로 줄인다."""
    return re.sub(r"\s+", " ", text).strip()


WebSearchFn = Callable[..., list[WebResult]]


def build_web_result(
    *,
    title: str,
    url: str,
    snippet: str,
    query: str,
    content: str | None = None,
    html: str | None = None,
    fetched_at: str | None = None,
    published_date: str | None = None,
    publisher: str | None = None,
) -> WebResult:
    """provider 응답 1건을 WebResult로 만든다.

    publisher / source_kind / published_date는 provider가 주지 않으면 URL·HTML에서 뽑는다.
    published_date는 파싱 실패 시 None으로 남긴다 — 추정하지 않는다.
    """
    return WebResult(
        title=title,
        url=url,
        snippet=snippet,
        content=content,
        publisher=publisher or extract_publisher(url),
        published_date=published_date or parse_published_date(html=html, url=url, text=content or snippet),
        fetched_at=fetched_at or _now_iso(),
        source_kind=classify_source_kind(url),
        query=query,
    )


def not_public_evidence(*, evidence_id: str, queries: list[str], scope: str, unit: EvidenceUnit) -> Evidence:
    """검색 실패 기록 헬퍼 (CONTRACTS.md §4).

    2회 재검색 후에도 못 찾은 정보는 추측하지 않고 이 Evidence로 남긴다
    (AGENTS.md 규칙 2). searched_at은 현재 시각으로 채운다.
    """
    if not queries or not any(q.strip() for q in queries):
        raise ValueError(f"{evidence_id}: not_public evidence requires at least one search query")
    if not scope.strip():
        raise ValueError(f"{evidence_id}: not_public evidence requires a search scope")
    cleaned = [q.strip() for q in queries if q.strip()]
    return Evidence(
        evidence_id=evidence_id,
        source_type="not_public",
        unit=unit,
        quote="",
        locator="not_found",
        search_query=" | ".join(cleaned),
        searched_at=_now_iso(),
        search_scope=scope.strip(),
    )
