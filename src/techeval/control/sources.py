"""검색 결과 출처 기록기 (V5 근거 실존 검사의 기준 집합).

에이전트가 `deps.web_search`를 호출할 때마다 돌려받은 URL을 기록해 두고, 검사 노드가 evidence의 `url`/`locator`가
실제 검색 결과에 있었는지 대조한다. 논문 청크는 `retriever.get_chunk(chunk_id)`로 직접 확인하므로 여기서 다루지 않는다.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """공백 정규화 (V6 quote 부분 문자열 비교용)."""
    return _WS.sub(" ", text).strip().lower()


def normalize_url(url: str) -> str:
    u = url.strip()
    u = re.sub(r"^https?://", "", u, flags=re.IGNORECASE)
    u = u.removeprefix("www.")
    u = u.split("#", 1)[0]
    return u.rstrip("/").lower()


class SourceRegistry:
    """`web_search` 호출을 감싸 결과 URL을 누적한다. `wrap(fn)`이 돌려주는 함수를 Deps.web_search로 주입한다."""

    def __init__(self, preload_urls: list[str] | None = None):
        self._urls: set[str] = {normalize_url(u) for u in (preload_urls or [])}
        self.calls: list[dict[str, Any]] = []  # {"query": str, "n": int}

    @property
    def known_urls(self) -> frozenset[str]:
        return frozenset(self._urls)

    def add(self, url: str) -> None:
        if url:
            self._urls.add(normalize_url(url))

    def has(self, url: str | None) -> bool:
        return bool(url) and normalize_url(url) in self._urls

    def wrap(self, fn: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
        def _wrapped(query: str, **kwargs: Any) -> list[Any]:
            results = fn(query, **kwargs)
            for r in results:
                url = getattr(r, "url", None) or (r.get("url") if isinstance(r, dict) else None)
                if url:
                    self.add(url)
            self.calls.append({"query": query, "n": len(results)})
            logger.debug("web_search(%r) -> %d results", query, len(results))
            return results

        _wrapped.__name__ = getattr(fn, "__name__", "web_search")
        return _wrapped


def chunk_exists(retriever: Any, chunk_id: str | None) -> Any | None:
    """`retriever.get_chunk(chunk_id)`. chunk_id가 없거나 검색기가 모르면 None."""
    if not chunk_id:
        return None
    try:
        return retriever.get_chunk(chunk_id)
    except Exception:  # 검색기 오류는 "실존 확인 실패"로 취급하고 로그를 남긴다
        logger.exception("retriever.get_chunk(%r) failed", chunk_id)
        return None


def quote_in_chunk(quote: str, chunk: Any) -> bool:
    """V6: quote가 chunk.text의 부분 문자열인지 (공백 정규화 후)."""
    text = getattr(chunk, "text", None) or (chunk.get("text") if isinstance(chunk, dict) else "")
    if not quote or not text:
        return False
    return normalize_ws(quote) in normalize_ws(text)


def evidence_problems(e: Any, retriever: Any, registry: SourceRegistry | None) -> list[str]:
    """Evidence 1건의 V5/V6 위반 목록. 비어 있으면 통과.

    - paper(코퍼스): chunk_id가 있으면 retriever에 실존 + quote가 chunk.text의 부분 문자열
    - paper(웹 검색으로 얻은 논문, chunk_id 없음)·web/official/patent: url(또는 locator)이 registry에 기록된
      검색 결과여야 함 (registry가 None이면 생략)
    - inference/not_public: 검사 대상 아님 (스키마 validator가 검색어·검색일을 강제)
    """
    problems: list[str] = []
    st = e.source_type
    if st == "paper" and not e.chunk_id:
        if not (e.url or "://" in e.locator):
            problems.append(f"{e.evidence_id}: paper evidence has neither chunk_id nor url (V5)")
            return problems
        st = "web"  # 웹에서 얻은 논문은 URL 경로로 검증
    if st == "paper":
        chunk = chunk_exists(retriever, e.chunk_id)
        if chunk is None:
            problems.append(f"{e.evidence_id}: chunk_id {e.chunk_id!r} not found in retriever (V5)")
        elif not quote_in_chunk(e.quote, chunk):
            problems.append(f"{e.evidence_id}: quote is not a substring of chunk {e.chunk_id!r} (V6)")
    elif st in ("web", "official", "patent") and registry is not None:
        url = e.url or e.locator
        if not registry.has(url):
            problems.append(f"{e.evidence_id}: url {url!r} was not returned by web_search (V5)")
    return problems
