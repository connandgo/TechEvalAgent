"""(C) 웹 검색 스텁 — 네트워크 없이 B·C·E가 개발·테스트할 때 쓴다.

CONTRACTS.md §4: `stub_web_search(query, **kwargs) -> list[WebResult]`.
`tests/fixtures/web_results.json`에서 query 키워드를 매칭하고, 없으면 빈 리스트를 돌려준다.
"""

import functools
import json
import logging
import re
from pathlib import Path

from techeval.tools.web_search import WebResult, normalize_host

logger = logging.getLogger(__name__)

#: src/techeval/tools/stub.py -> 저장소 루트
_REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "web_results.json"

#: 키워드 매칭에서 무시하는 흔한 단어
_STOPWORDS = frozenset({"the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "with", "is"})


def _tokenize(text: str) -> set[str]:
    """소문자 토큰 집합. 하이픈·점은 유지해 "deepseek-v2", "cxl3.1"을 한 토큰으로 본다."""
    tokens = re.findall(r"[a-z0-9][a-z0-9.\-]*", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


@functools.lru_cache(maxsize=1)
def load_fixture(path: str | None = None) -> dict[str, list[WebResult]]:
    """픽스처를 읽어 WebResult로 검증한다. 스키마가 깨지면 여기서 예외가 난다."""
    fixture_path = Path(path) if path else FIXTURE_PATH
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    return {key: [WebResult.model_validate(item) for item in items] for key, items in raw.items()}


def _match_key(query: str, keys: list[str]) -> str | None:
    """검색어와 가장 많이 겹치는 픽스처 키를 고른다.

    키 토큰의 절반 이상이 검색어에 나타나야 매칭으로 인정한다(오탐 방지).
    동점이면 키 토큰이 많은 쪽 — 더 구체적인 키 — 을 택한다.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return None
    best: tuple[float, int, str] | None = None
    for key in keys:
        key_tokens = _tokenize(key)
        if not key_tokens:
            continue
        overlap = len(key_tokens & query_tokens)
        ratio = overlap / len(key_tokens)
        if ratio < 0.5:
            continue
        candidate = (ratio, len(key_tokens), key)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best else None


def stub_web_search(
    query: str,
    *,
    max_results: int = 5,
    recency_days: int | None = None,
    site_filter: list[str] | None = None,
    fetch_content: bool = False,
    **_ignored,
) -> list[WebResult]:
    """픽스처 기반 web_search. 매칭되는 키가 없으면 빈 리스트.

    실제 `web_search`와 달리 예외를 올리지 않는다. 호출자는 빈 리스트를
    `not_public` 후보로 다루면 된다.

    - `query`는 호출자가 실제로 쓴 검색어로 각 결과에 다시 기록한다.
    - `fetch_content=False`면 `content`를 떼어내 실제 provider 동작에 맞춘다.
    - `site_filter`는 호스트 접미사로 필터링한다. `recency_days`는 스텁에서 무시한다.
    """
    fixture = load_fixture()
    key = _match_key(query, list(fixture))
    if key is None:
        logger.debug("stub_web_search: no fixture key matched query=%r", query)
        return []

    results = fixture[key]
    if site_filter:
        allowed = tuple(normalize_host(f"https://{s}") or s.lower() for s in site_filter)
        results = [
            r
            for r in results
            if any(normalize_host(r.url) == a or normalize_host(r.url).endswith("." + a) for a in allowed)
        ]

    updates: dict = {"query": query}
    if not fetch_content:
        updates["content"] = None
    return [r.model_copy(update=updates) for r in results[:max_results]]
