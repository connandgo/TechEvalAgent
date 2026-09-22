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


# --- 에이전트별 모델 설정 --------------------------------------------------------


class TestAgentModelOverride:
    def test_model_for_falls_back_to_default(self):
        from techeval.config import Settings

        s = Settings(llm_provider="p", llm_model="base", judge_model="judge", llm_model_report="strong")
        assert s.model_for(None) == "base"
        assert s.model_for("domain") == "base"
        assert s.model_for("report") == "strong"
        assert s.agent_overrides() == {"report": "strong"}
        with pytest.raises(KeyError):
            s.model_for("nope")

    def test_graph_uses_agent_specific_deps(self, tmp_path):
        """agent_deps 에 준 Deps 가 해당 노드에만 들어가고 나머지는 공용 deps 를 쓴다."""
        from techeval import stub_agents
        from techeval.agents._deps import Deps
        from techeval.graph import Agents, GraphConfig, build_graph, invoke_config
        from techeval.stub_llm import FakeStructuredLLM
        from tests.helpers import MemRetriever, make_chunks, make_web_search

        seen: dict[str, list[str]] = {}

        def spy(name, base):
            def fn(inp, deps):
                seen.setdefault(name, []).append(getattr(deps.llm, "tag", "shared"))
                return base(inp, deps)

            return fn

        shared = FakeStructuredLLM()
        strong = FakeStructuredLLM()
        strong.tag = "strong"
        deps = Deps(
            retriever=MemRetriever(make_chunks()),
            web_search=make_web_search(),
            llm=shared,
            judge_llm=FakeStructuredLLM(),
            now=lambda: "2026-01-01T00:00:00",
        )
        agents = Agents(
            run_tech_research=spy("tech_research", stub_agents.run_tech_research),
            run_domain_eval=spy("domain", stub_agents.run_domain_eval),
            run_market_eval=spy("market", stub_agents.run_market_eval),
            run_stakeholder_eval=spy("stakeholder", stub_agents.run_stakeholder_eval),
            run_synthesis=spy("synthesis", stub_agents.run_synthesis),
            run_report=spy("report", stub_agents.run_report),
            render_pdf=stub_agents.render_pdf,
        )
        stub_agents.FIXTURES_DIR = tmp_path / "none"
        try:
            cfg = GraphConfig(stub=True, output_dir=str(tmp_path), skip_pdf=True)
            g = build_graph(deps, agents, cfg, agent_deps={"report": deps.model_copy(update={"llm": strong})})
            g.invoke({}, config=invoke_config(cfg))
        finally:
            stub_agents.FIXTURES_DIR = stub_agents.DEFAULT_FIXTURES_DIR
        assert set(seen["report"]) == {"strong"}
        assert all(set(v) == {"shared"} for k, v in seen.items() if k != "report")


class TestV3Relaxed:
    """V3: D1~D3는 L2/L3일 때만 measurements 필수. L1(근거 없음)은 evidence만 있으면 된다."""

    def test_l2_l3_require_measurements(self):
        for level in ("L2", "L3"):
            with pytest.raises(ValidationError):
                result(criterion_id="D3", perspective="domain", level=level)

    def test_l1_without_measurements_is_valid(self):
        r = result(criterion_id="D3", perspective="domain", level="L1")
        assert r.measurements == [] and len(r.evidence) == 1

    def test_l1_still_requires_evidence(self):
        with pytest.raises(ValidationError):
            result(criterion_id="D3", perspective="domain", level="L1", evidence=[])

    def test_perspective_check_accepts_l1_without_measurements(self):
        from techeval.control.perspective_check import _v4_problems

        d = {"directness": "L1"}
        assert _v4_problems(result(criterion_id="D1", perspective="domain", level="L1", details=d)) == []
        m = Measurement(metric="m", value="1", evidence_id="mla-D1-01")
        d3 = {"directness": "L3"}
        assert (
            _v4_problems(result(criterion_id="D1", perspective="domain", level="L3", measurements=[m], details=d3))
            == []
        )
        # validator를 우회해 만든 L2·수치 없음 결과는 검사 노드가 V3로 잡는다
        broken = CriterionResult.model_construct(
            tech_id="mla",
            perspective="domain",
            criterion_id="D1",
            level="L2",
            content="c",
            evidence=[paper()],
            confidence="medium",
            evidence_unit="paper",
            measurements=[],
            details={"directness": "L2", "extrapolation_logic": "x"},
            generated_at="2026-01-01T00:00:00",
            retry_count=0,
        )
        assert any("V3" in p for p in _v4_problems(broken))
