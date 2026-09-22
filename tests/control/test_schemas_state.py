"""schemas.compute_confidence / state.latest_by_* / validator 규칙 (V1~V3) 단위 테스트."""

import pytest
from pydantic import ValidationError

from techeval.schemas import (
    TECHNOLOGIES,
    CriterionResult,
    Evidence,
    Measurement,
    TechProfile,
    compute_confidence,
    get_tech,
)
from techeval.state import build_initial_state, latest_by_criterion, latest_by_tech


def paper(evidence_id="mla-T1-01", doc_id="deepseek_v2", quote="KV cache reduced by 93.3%") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type="paper",
        unit="paper",
        quote=quote,
        locator=f"{doc_id} p.4 §2.1",
        doc_id=doc_id,
        page=4,
        chunk_id=f"{doc_id}:004:01",
    )


def web(evidence_id="mla-M1-01", url="https://example.com/a", source_type="web") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        unit="family",
        quote="some quote",
        locator=url,
        url=url,
        accessed_date="2026-01-01",
    )


def not_public(evidence_id="mla-T3-01") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type="not_public",
        unit="paper",
        quote="",
        locator="not_found",
        search_query="third-party reproduction MLA",
        searched_at="2026-01-01",
    )


def result(criterion_id="T1", perspective="trl", tech_id="mla", generated_at="2026-01-01T00:00:00", **kw):
    base = dict(
        tech_id=tech_id,
        perspective=perspective,
        criterion_id=criterion_id,
        level="TRL 4-6",
        content="평가 내용",
        evidence=[paper(evidence_id=f"{tech_id}-{criterion_id}-01")],
        confidence="medium",
        evidence_unit="paper",
        generated_at=generated_at,
    )
    base.update(kw)
    return CriterionResult(**base)


# --- compute_confidence ------------------------------------------------------


class TestComputeConfidence:
    def test_empty_is_low(self):
        assert compute_confidence([]) == "low"

    def test_inference_or_not_public_is_low(self):
        assert compute_confidence([paper(), not_public()]) == "low"
        inf = Evidence(evidence_id="x", source_type="inference", unit="family", quote="추론", locator="inference")
        assert compute_confidence([paper(), web(), inf]) == "low"

    def test_single_paper_is_medium(self):
        assert compute_confidence([paper()]) == "medium"

    def test_two_chunks_same_doc_is_medium(self):
        assert compute_confidence([paper(), paper(evidence_id="mla-T1-02", quote="other")]) == "medium"

    def test_two_web_only_is_medium(self):
        assert compute_confidence([web(), web(url="https://other.org/b")]) == "medium"

    def test_paper_plus_web_is_high(self):
        assert compute_confidence([paper(), web()]) == "high"

    def test_two_papers_is_high(self):
        assert compute_confidence([paper(), paper(doc_id="kv_survey")]) == "high"

    def test_official_plus_other_domain_is_high(self):
        assert compute_confidence([web(source_type="official"), web(url="https://news.example.org/x")]) == "high"

    def test_same_domain_two_pages_is_medium(self):
        evs = [web(source_type="official", url="https://a.com/p1"), web(url="https://www.a.com/p2")]
        assert compute_confidence(evs) == "medium"


# --- validators (V1~V3) ------------------------------------------------------


class TestValidators:
    def test_evidence_required(self):
        with pytest.raises(ValidationError):
            result(evidence=[])

    def test_criterion_must_match_perspective(self):
        with pytest.raises(ValidationError):
            result(criterion_id="M1", perspective="trl")

    def test_not_public_level_requires_not_public_evidence(self):
        with pytest.raises(ValidationError):
            result(level="not_public")
        r = result(level="not_public", evidence=[not_public()], confidence="low")
        assert r.level == "not_public"

    def test_d1_requires_measurements(self):
        with pytest.raises(ValidationError):
            result(criterion_id="D1", perspective="domain", level="L3")
        m = Measurement(metric="KV cache reduction", value="93.3%", evidence_id="mla-D1-01")
        r = result(criterion_id="D1", perspective="domain", level="L3", measurements=[m])
        assert r.measurements[0].value == "93.3%"

    def test_not_public_evidence_requires_query_and_date(self):
        with pytest.raises(ValidationError):
            Evidence(evidence_id="x", source_type="not_public", unit="paper", quote="", locator="not_found")

    def test_quoted_evidence_requires_quote(self):
        with pytest.raises(ValidationError):
            paper(quote="")


# --- state helpers ------------------------------------------------------------


class TestStateHelpers:
    def test_latest_by_criterion_keeps_newest(self):
        old = result(generated_at="2026-01-01T00:00:00", content="old")
        new = result(generated_at="2026-01-02T00:00:00", content="new", retry_count=1)
        other = result(criterion_id="T2", tech_id="pim_cxl")
        out = latest_by_criterion([old, other, new])
        assert [(r.tech_id, r.criterion_id, r.content) for r in out] == [
            ("mla", "T1", "new"),
            ("pim_cxl", "T2", "평가 내용"),
        ]

    def test_latest_by_criterion_same_timestamp_prefers_later(self):
        a = result(content="a")
        b = result(content="b")
        assert latest_by_criterion([a, b])[0].content == "b"

    def test_latest_by_tech(self):
        def profile(gen, principle):
            return TechProfile(
                tech_id="mla",
                principle=principle,
                scope="s",
                limitations=["l"],
                measurements=[],
                validation_env="v",
                public_artifacts={},
                evidence=[paper()],
                generated_at=gen,
            )

        out = latest_by_tech([profile("2026-01-01T00:00:00", "old"), profile("2026-01-03T00:00:00", "new")])
        assert len(out) == 1 and out[0].principle == "new"

    def test_build_initial_state(self):
        s = build_initial_state()
        assert [t.tech_id for t in s["technologies"]] == ["mla", "pim_cxl"]
        assert s["retry_counts"]["trl:mla"] == 0
        assert s["retry_counts"]["counter"] == 0 and s["retry_counts"]["report"] == 0
        assert s["missing_criteria"] == {"trl": [], "market": [], "stakeholder": [], "domain": []}
        assert s["trl_eval"] == [] and s["counter_evidence"] == []

    def test_technologies_constant(self):
        assert get_tech("mla").primary_doc_id == "deepseek_v2"
        assert get_tech("pim_cxl").primary_doc_id == "pim_cxl_1m"
        assert len(TECHNOLOGIES) == 2
        with pytest.raises(KeyError):
            get_tech("nope")
