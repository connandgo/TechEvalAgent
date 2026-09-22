"""(C) web_search 도구 단위 테스트 — 네트워크·모델 불필요.

주의: `tests/fixtures/web_results.json`의 URL·snippet은 **픽스처 전용 합성 데이터**다.
도메인과 형식은 실제 출처를 닮게 만들었지만 개별 URL이 실재한다고 보장하지 않는다.
보고서에 들어갈 근거는 실제 `web_search()` 결과만 쓴다(V5가 대조한다).
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from techeval.schemas import Evidence, compute_confidence
from techeval.tools import web_search as ws
from techeval.tools.stub import FIXTURE_PATH, _match_key, load_fixture, stub_web_search
from techeval.tools.web_search import (
    MAX_CONTENT_CHARS,
    WebResult,
    build_web_result,
    classify_source_kind,
    extract_publisher,
    not_public_evidence,
    parse_published_date,
)


def _result(**overrides) -> WebResult:
    base = {
        "title": "Leo CXL Smart Memory Controllers",
        "url": "https://www.astera-labs.com/products/leo-cxl-smart-memory-controllers/",
        "snippet": "Leo CXL memory controllers are shipping in production platforms.",
        "fetched_at": "2026-09-20T09:00:00+00:00",
        "source_kind": "official",
        "query": "CXL memory expander KV cache",
    }
    return WebResult.model_validate({**base, **overrides})


# --- source_kind 분류 ---------------------------------------------------------
# AGENTS.md 8: 벤더 발표 자료를 news/paper로 표시하면 E의 근거 비대칭 계산이 틀어진다.


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://semiconductor.samsung.com/dram/hbm/hbm-pim/", "official"),
        ("https://news.skhynix.com/sk-hynix-aim-accelerator-card/", "official"),
        ("https://computeexpresslink.org/cxl-specification/", "official"),
        ("https://docs.vllm.ai/en/latest/models/supported_models.html", "official"),
        ("https://arxiv.org/abs/2405.04434", "paper"),
        ("https://dl.acm.org/doi/10.1145/3620666", "paper"),
        ("https://www.eetimes.com/processing-in-memory-parts/", "news"),
        ("https://www.reuters.com/technology/memory-makers-raise-spending/", "news"),
        ("https://semianalysis.com/2025/05/12/long-context/", "blog"),
        ("https://medium.com/@someone/kv-cache-notes", "blog"),
        ("https://www.reddit.com/r/LocalLLaMA/comments/abc/", "forum"),
        ("https://news.ycombinator.com/item?id=40000000", "forum"),
        ("https://huggingface.co/deepseek-ai/DeepSeek-V2", "other"),
        ("not a url", "other"),
    ],
)
def test_classify_source_kind(url, expected):
    assert classify_source_kind(url) == expected


def test_classify_source_kind_uses_registered_domain_for_unknown_subdomain():
    """표에 없는 서브도메인도 등록 도메인으로 분류된다."""
    assert classify_source_kind("https://research.samsung.com/whatever") == "official"


def test_paper_domain_wins_over_vendor():
    """학술 도메인이 먼저 판정된다 (arXiv는 벤더 자료가 아니다)."""
    assert classify_source_kind("https://arxiv.org/abs/2505.11266") == "paper"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/abs/2405.04434", "arXiv"),
        ("https://news.samsung.com/global/whatever", "Samsung"),
        ("https://www.astera-labs.com/products/leo/", "Astera Labs"),
        (
            "https://www.marketsandmarkets.com/Market-Reports/x.asp",
            "marketsandmarkets.com",
        ),
        ("https://www.thelec.kr/news/articleView.html?idxno=1", "thelec.kr"),
        ("https://www.businesskorea.co.kr/news/article/1", "businesskorea.co.kr"),
    ],
)
def test_extract_publisher(url, expected):
    assert extract_publisher(url) == expected


# --- published_date 파싱 ------------------------------------------------------
# 역할 C 문서 §5.1: 메타태그 -> URL 패턴 -> 본문 첫 날짜. 실패 시 None (추정 금지).


def test_parse_date_from_meta_tag():
    html = '<meta property="article:published_time" content="2025-03-11T08:00:00Z">'
    assert parse_published_date(html=html) == "2025-03-11"


def test_parse_date_from_meta_tag_reversed_attribute_order():
    html = '<meta content="2024-06-25" name="citation_publication_date">'
    assert parse_published_date(html=html) == "2024-06-25"


def test_parse_date_from_time_tag():
    html = '<article><time datetime="2026-01-19T11:30:00+09:00">Jan 19</time></article>'
    assert parse_published_date(html=html) == "2026-01-19"


def test_meta_tag_beats_url_pattern():
    html = '<meta name="date" content="2025-07-30">'
    url = "https://www.eetimes.com/2024/01/02/older-path/"
    assert parse_published_date(html=html, url=url) == "2025-07-30"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.nextplatform.com/2025/04/08/open-weight-moe/", "2025-04-08"),
        ("https://www.reuters.com/technology/cloud-2026-02-11/", None),
        ("https://arxiv.org/abs/2405.04434", "2024-05"),
        ("https://arxiv.org/pdf/2505.11266", "2025-05"),
    ],
)
def test_parse_date_from_url(url, expected):
    assert parse_published_date(url=url) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Published 2025-12-03 by the newsroom", "2025-12-03"),
        ("Posted on March 18, 2025 in memory", "2025-03-18"),
        ("Sept. 9, 2025", "2025-09-09"),
        ("2024년 11월 5일 공개", "2024-11-05"),
        ("2025/08/01 update", "2025-08-01"),
    ],
)
def test_parse_date_from_text(text, expected):
    assert parse_published_date(text=text) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "last updated recently",
        "2025-13-45",
        "sometime in 2025",
        "February 2025",
    ],
)
def test_parse_date_returns_none_when_unparseable(bad):
    """못 찾으면 None. 연도만 있다고 1월 1일로 추정하지 않는다."""
    assert parse_published_date(html=bad, url="https://example.com/x", text=bad) is None


def test_parse_date_with_no_input():
    assert parse_published_date() is None


# --- WebResult ----------------------------------------------------------------


def test_content_is_capped():
    r = _result(content="x" * (MAX_CONTENT_CHARS + 500))
    assert len(r.content) == MAX_CONTENT_CHARS


def test_accessed_date_is_fetched_at_date():
    assert _result().accessed_date == "2026-09-20"


@pytest.mark.parametrize(
    ("source_kind", "source_type"),
    [
        ("official", "official"),
        ("paper", "paper"),
        ("news", "web"),
        ("blog", "web"),
        ("forum", "web"),
        ("other", "web"),
    ],
)
def test_source_type_mapping(source_kind, source_type):
    assert _result(source_kind=source_kind).source_type == source_type


def test_to_evidence_maps_all_fields():
    r = _result(publisher="Astera Labs", published_date="2024-09-17")
    e = r.to_evidence(
        evidence_id="pim_cxl-M3-01",
        quote="shipping in production platforms",
        unit="family",
    )
    assert isinstance(e, Evidence)
    assert e.evidence_id == "pim_cxl-M3-01"
    assert e.source_type == "official"
    assert e.unit == "family"
    assert e.locator == r.url == e.url
    assert e.publisher == "Astera Labs"
    assert e.published_date == "2024-09-17"
    assert e.accessed_date == "2026-09-20"


def test_to_evidence_accepts_quote_from_content():
    r = _result(content="Leo supports memory pooling for server hosts.")
    e = r.to_evidence(
        evidence_id="pim_cxl-M3-02", quote="memory pooling", unit="family"
    )
    assert e.quote == "memory pooling"


def test_to_evidence_normalizes_whitespace_when_matching():
    r = _result(snippet="Leo CXL memory controllers\n   are shipping in production.")
    e = r.to_evidence(
        evidence_id="pim_cxl-M3-03", quote="controllers are shipping", unit="family"
    )
    assert e.quote == "controllers are shipping"


def test_to_evidence_rejects_invented_quote():
    """AGENTS.md 1 — 검색 결과에 없는 인용은 코드에서 막는다."""
    with pytest.raises(ValueError, match="not a substring"):
        _result().to_evidence(
            evidence_id="pim_cxl-M3-04",
            quote="Astera Labs shipped 2 million units in 2025",
            unit="family",
        )


def test_to_evidence_rejects_empty_quote():
    with pytest.raises(ValueError, match="quote is required"):
        _result().to_evidence(evidence_id="pim_cxl-M3-05", quote="  ", unit="family")


def test_build_web_result_fills_publisher_kind_and_date():
    r = build_web_result(
        title="Processing-in-memory parts remain sampling-stage",
        url="https://www.eetimes.com/processing-in-memory-parts/",
        snippet="Analysts note that PIM devices are sampled to selected customers.",
        query="Samsung PIM HBM",
        html='<meta property="article:published_time" content="2025-07-30">',
    )
    assert r.publisher == "EE Times"
    assert r.source_kind == "news"
    assert r.published_date == "2025-07-30"
    assert r.query == "Samsung PIM HBM"
    assert r.fetched_at


def test_build_web_result_leaves_date_none_when_unparseable():
    r = build_web_result(
        title="Tracking issue",
        url="https://github.com/vllm-project/vllm/issues/mla-backend-tracking",
        snippet="Tracking issue for the MLA attention backend.",
        query="MLA vLLM support",
    )
    assert r.published_date is None
    assert r.source_kind == "other"


# --- not_public_evidence ------------------------------------------------------
# AGENTS.md 2: 못 찾은 정보는 추측하지 않고 검색어·검색일·범위를 남긴다.


def test_not_public_evidence_records_queries_and_scope():
    e = not_public_evidence(
        evidence_id="pim_cxl-M2-03",
        queries=["PIM production deployment LLM", "PNM 상용 운영 사례"],
        scope="vendor newsrooms, arxiv, news 2024-2026",
        unit="family",
    )
    assert e.source_type == "not_public"
    assert e.locator == "not_found"
    assert e.quote == ""
    assert "PIM production deployment LLM" in e.search_query
    assert "PNM 상용 운영 사례" in e.search_query
    assert e.searched_at
    assert e.search_scope == "vendor newsrooms, arxiv, news 2024-2026"


@pytest.mark.parametrize(
    ("queries", "scope"),
    [([], "arxiv"), (["   "], "arxiv"), (["q"], "  ")],
)
def test_not_public_evidence_requires_query_and_scope(queries, scope):
    with pytest.raises(ValueError):
        not_public_evidence(
            evidence_id="pim_cxl-M2-04", queries=queries, scope=scope, unit="family"
        )


def test_not_public_evidence_makes_confidence_low():
    """신뢰도 규칙 — not_public이 섞이면 low (AGENTS.md 3)."""
    paper = _result(
        url="https://arxiv.org/abs/2405.04434", source_kind="paper"
    ).to_evidence(
        evidence_id="mla-M2-01", quote="shipping in production", unit="family"
    )
    gap = not_public_evidence(
        evidence_id="mla-M2-02", queries=["q"], scope="news", unit="family"
    )
    assert compute_confidence([paper]) == "medium"
    assert compute_confidence([paper, gap]) == "low"


def test_confidence_high_requires_two_independent_sources_with_primary():
    paper = _result(
        url="https://arxiv.org/abs/2405.04434", source_kind="paper"
    ).to_evidence(
        evidence_id="mla-M2-01", quote="shipping in production", unit="family"
    )
    news = _result(
        url="https://www.nextplatform.com/2025/04/08/moe/", source_kind="news"
    ).to_evidence(
        evidence_id="mla-M2-02", quote="shipping in production", unit="family"
    )
    other_paper = _result(
        url="https://arxiv.org/abs/2412.19442", source_kind="paper"
    ).to_evidence(
        evidence_id="mla-M2-03", quote="shipping in production", unit="family"
    )
    same_host_news = _result(
        url="https://www.nextplatform.com/2025/06/02/cxl/", source_kind="news"
    ).to_evidence(
        evidence_id="mla-M2-04", quote="shipping in production", unit="family"
    )
    assert compute_confidence([paper, news]) == "high"
    # 같은 호스트의 논문 두 편은 서로 다른 문헌이므로 독립 출처 2개로 센다.
    assert compute_confidence([paper, other_paper]) == "high"
    # 반면 같은 매체의 기사 두 건은 독립 출처 1개다 (1차 자료도 없어 medium).
    assert compute_confidence([news, same_host_news]) == "medium"


# --- 픽스처 -------------------------------------------------------------------


def test_fixture_path_exists():
    assert FIXTURE_PATH.exists(), f"fixture missing: {FIXTURE_PATH}"


def test_fixture_validates_as_web_results():
    """dict[str, list[WebResult]] (CONTRACTS.md §9)."""
    fixture = load_fixture()
    assert fixture
    for key, results in fixture.items():
        assert results, f"{key}: empty result list"
        assert len(results) <= 5, f"{key}: 결과는 5개 이내"
        for r in results:
            assert isinstance(r, WebResult)


def test_fixture_metadata_is_filled():
    """publisher·source_kind는 모두 채운다. published_date는 파싱 실패 시 None 허용."""
    fixture = load_fixture()
    dated = 0
    total = 0
    for key, results in fixture.items():
        for r in results:
            total += 1
            assert r.publisher, f"{key}: {r.url} publisher 누락"
            assert r.query == key
            assert r.snippet.strip(), f"{key}: {r.url} snippet 누락"
            if r.published_date:
                dated += 1
    assert dated / total >= 0.6, (
        "픽스처 날짜 채움 비율이 통합 테스트 기준(60%)보다 낮다"
    )


def test_fixture_source_kind_matches_url_classification():
    """벤더 자료가 news/paper로 잘못 표기되지 않았는지 확인 (근거 비대칭 계산 보호)."""
    fixture = load_fixture()
    for key, results in fixture.items():
        for r in results:
            assert r.source_kind == classify_source_kind(r.url), f"{key}: {r.url}"


def test_fixture_covers_both_technology_families():
    """M·S 평가는 계열 단위 — 두 계열 모두 검색 가능해야 한다."""
    keys = " ".join(load_fixture()).lower()
    assert "mla" in keys or "deepseek" in keys
    assert "cxl" in keys or "pim" in keys


def test_fixture_is_sorted_json_loadable_raw():
    raw = json.loads(Path(FIXTURE_PATH).read_text(encoding="utf-8"))
    assert set(raw) == set(load_fixture())


# --- stub_web_search ----------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_key"),
    [
        ("MLA vLLM support", "MLA vLLM support"),
        ("vLLM support for MLA", "MLA vLLM support"),
        ("Does vLLM support MLA?", "MLA vLLM support"),
        ("CXL memory expander KV cache offload", "CXL memory expander KV cache"),
        ("Samsung PIM HBM roadmap", "Samsung PIM HBM"),
        ("DeepSeek-V2 adoption in production", "DeepSeek-V2 adoption"),
        ("CXL standardization JEDEC liaison", "CXL standardization JEDEC"),
    ],
)
def test_stub_matches_keyword(query, expected_key):
    assert _match_key(query, list(load_fixture())) == expected_key
    results = stub_web_search(query)
    assert results
    assert all(r.query == query for r in results), "결과에는 호출자의 검색어를 기록한다"


@pytest.mark.parametrize(
    "query",
    ["quantum annealing for protein folding", "", "   ", "xyzzy"],
)
def test_stub_returns_empty_for_unmatched_query(query):
    assert stub_web_search(query) == []


def test_stub_respects_max_results():
    assert len(stub_web_search("MLA vLLM support", max_results=2)) == 2


def test_stub_omits_content_unless_requested():
    without = stub_web_search("DeepSeek-V2 adoption")
    assert all(r.content is None for r in without)
    with_content = stub_web_search("DeepSeek-V2 adoption", fetch_content=True)
    assert any(r.content for r in with_content)


def test_stub_applies_site_filter():
    results = stub_web_search("MLA vLLM support", site_filter=["docs.vllm.ai"])
    assert [r.publisher for r in results] == ["vLLM"]


def test_stub_site_filter_matches_subdomains():
    results = stub_web_search("Samsung PIM HBM", site_filter=["samsung.com"])
    assert results and all("samsung.com" in r.url for r in results)


def test_stub_does_not_mutate_cached_fixture():
    """model_copy를 쓰므로 캐시된 픽스처는 그대로 남는다."""
    stub_web_search("DeepSeek-V2 adoption", fetch_content=False)
    cached = load_fixture()["DeepSeek-V2 adoption"]
    assert any(r.content for r in cached)
    assert all(r.query == "DeepSeek-V2 adoption" for r in cached)


def test_stub_ignores_unknown_kwargs():
    assert stub_web_search("MLA vLLM support", recency_days=30, something_new=1)


def test_stub_results_convert_to_evidence():
    """스텁 결과만으로 하류(B·E)가 Evidence를 만들 수 있어야 한다."""
    results = stub_web_search("CXL memory expander KV cache")
    evidence = [
        r.to_evidence(
            evidence_id=f"pim_cxl-M3-{i:02d}",
            quote=r.snippet[:40],
            unit="family",
        )
        for i, r in enumerate(results, start=1)
    ]
    assert len(evidence) == len(results)
    assert compute_confidence(evidence) == "high"


def test_fixture_rejects_broken_schema(tmp_path: Path):
    """픽스처가 스키마를 깨면 로드에서 바로 예외 (main 머지 차단)."""
    bad = tmp_path / "web_results.json"
    bad.write_text(json.dumps({"k": [{"title": "no url"}]}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_fixture(str(bad))


# --- web_search 본체 (provider·캐시) ---------------------------------------------
# 네트워크는 타지 않는다. PROVIDERS를 가짜 함수로 바꿔 호출 횟수만 센다.


@pytest.fixture
def fake_provider(monkeypatch, tmp_path):
    calls: list[str] = []

    def _fake(query, *, api_key, max_results, recency_days):
        calls.append(query)
        return [
            {
                "title": "Leo CXL Smart Memory Controllers",
                "url": "https://www.astera-labs.com/products/leo/",
                "snippet": "Leo controllers are shipping in production platforms.",
                "content": None,
                "published_date": None,
            }
        ]

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.delenv("WEB_SEARCH_NO_CACHE", raising=False)
    monkeypatch.setitem(ws.PROVIDERS, "tavily", _fake)
    return calls


def test_web_search_records_query_and_classifies_source(fake_provider):
    results = ws.web_search("CXL memory expander KV cache")
    assert [r.query for r in results] == ["CXL memory expander KV cache"]
    assert results[0].source_kind == "official"
    assert results[0].publisher == "Astera Labs"


def test_web_search_uses_cache_on_second_call(fake_provider):
    first = ws.web_search("CXL memory expander KV cache")
    second = ws.web_search("CXL memory expander KV cache")
    assert len(fake_provider) == 1, "두 번째 호출은 캐시에서 와야 한다"
    assert first == second


def test_web_search_cache_key_separates_parameters(fake_provider):
    ws.web_search("CXL memory expander KV cache")
    ws.web_search("CXL memory expander KV cache", max_results=3)
    assert len(fake_provider) == 2, "파라미터가 다르면 다른 캐시여야 한다"


def test_no_cache_env_bypasses_cache(fake_provider, monkeypatch):
    """config.build_deps(use_cache=False)가 이 환경변수로 알린다."""
    ws.web_search("CXL memory expander KV cache")
    monkeypatch.setenv("WEB_SEARCH_NO_CACHE", "1")
    ws.web_search("CXL memory expander KV cache")
    assert len(fake_provider) == 2


def test_web_search_raises_when_provider_unset(monkeypatch, tmp_path):
    """검색 실패는 빈 리스트가 아니라 예외 — not_public 판단은 호출자 몫이다."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "")
    with pytest.raises(RuntimeError, match="WEB_SEARCH_PROVIDER"):
        ws.web_search("whatever")


def test_web_search_raises_when_api_key_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "")
    with pytest.raises(RuntimeError, match="WEB_SEARCH_API_KEY"):
        ws.web_search("whatever")


def test_build_query_applies_site_filter():
    assert ws._build_query("CXL", ["vllm.ai"]) == "CXL (site:vllm.ai)"
    assert ws._build_query("CXL", None) == "CXL"


def test_same_url_has_identical_metadata_across_keys():
    """같은 URL이 여러 검색어에 걸려도 메타데이터는 동일해야 한다.

    실제 provider는 검색어가 달라도 같은 문서면 같은 title·snippet을 돌려준다.
    픽스처가 이를 어기면, 소비자가 여러 결과를 합쳐 quote를 찾은 뒤 그중 하나로만
    Evidence를 만들 때 `to_evidence`의 부분 문자열 검사에서 터진다.
    """
    seen: dict[str, WebResult] = {}
    for key, results in load_fixture().items():
        for r in results:
            first = seen.setdefault(r.url, r)
            for field in (
                "title",
                "snippet",
                "content",
                "publisher",
                "published_date",
                "source_kind",
            ):
                assert getattr(r, field) == getattr(first, field), (
                    f"{key}: {r.url} 의 {field} 가 다른 키의 값과 다르다"
                )
