import pytest

from techeval.agents._deps import Deps, SynthesisInput
from techeval.agents.synthesis import SynthesisDraft, detect_gaps, run_synthesis
from techeval.schemas import (
    TECHNOLOGIES,
    CriterionResult,
    Evidence,
    SynthesisResult,
    TechProfile,
)
from tests.report.d_fixtures import load


def _draft_from_fixture() -> dict:
    """synthesis.json은 SynthesisDraft 필드를 모두 가진다(추가 필드는 무시)."""
    data = load("synthesis.json")
    data["gaps"] = [g for g in data["gaps"] if g["gap_type"] == "opposing_missing"]
    data["evidence_asymmetry_note"] = "PIM/CXL은 벤더 자료 비중이 높다."
    return data


class StructuredStub:
    def __init__(self, drafts: list[dict]):
        self.drafts = drafts
        self.calls: list[list] = []

    def with_structured_output(self, model):
        assert model is SynthesisDraft
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return SynthesisDraft.model_validate(
            self.drafts[min(len(self.calls), len(self.drafts)) - 1]
        )


class SearchSpy:
    def __init__(self):
        self.called = False

    def __call__(self, *a, **k):
        self.called = True
        raise AssertionError("synthesis는 web_search를 호출하면 안 된다")

    def search(self, *a, **k):
        return self(*a, **k)

    def get_chunk(self, *a, **k):
        return self(*a, **k)


@pytest.fixture
def inp() -> SynthesisInput:
    return SynthesisInput(
        technologies=TECHNOLOGIES,
        tech_profiles=load("tech_profiles.json", TechProfile),
        trl_eval=load("trl_eval.json", CriterionResult),
        market_eval=load("market_eval.json", CriterionResult),
        stakeholder_eval=load("stakeholder_eval.json", CriterionResult),
        domain_eval=load("domain_eval.json", CriterionResult),
        counter_evidence=load("counter_evidence.json", Evidence),
    )


def _deps(llm) -> tuple[Deps, SearchSpy, SearchSpy]:
    retriever, web = SearchSpy(), SearchSpy()
    return (
        Deps(
            retriever=retriever,
            web_search=web,
            llm=llm,
            now=lambda: "2026-01-01T00:00:00",
        ),
        retriever,
        web,
    )


def test_fixture_is_valid_synthesis_result():
    syn = load("synthesis.json", SynthesisResult)
    for tid in ("mla", "pim_cxl"):
        assert any(
            c.tech_id == tid and "S4" in (c.criterion_a, c.criterion_b)
            for c in syn.conflicts
        )


def test_run_synthesis_with_shared_fake_llm(inp, fake_llm):
    """E의 FakeStructuredLLM: SynthesisDraft가 SynthesisResult의 부분집합이라 synthesis.json이 그대로 매핑된다."""
    deps, retriever, web = _deps(fake_llm)
    result = run_synthesis(inp, deps)
    assert not retriever.called and not web.called
    assert fake_llm.prompts_for(SynthesisDraft)  # 구조화 호출이 SynthesisDraft로 나갔다
    for tid in ("mla", "pim_cxl"):
        assert any(
            c.tech_id == tid and "S4" in (c.criterion_a, c.criterion_b)
            for c in result.conflicts
        )


def test_run_synthesis_no_search_and_s4_conflicts(inp):
    llm = StructuredStub([_draft_from_fixture()])
    deps, retriever, web = _deps(llm)
    result = run_synthesis(inp, deps)
    assert not retriever.called and not web.called
    assert len(llm.calls) == 1
    for tid in ("mla", "pim_cxl"):
        assert any(
            c.tech_id == tid and "S4" in (c.criterion_a, c.criterion_b)
            for c in result.conflicts
        )
    assert result.generated_at == "2026-01-01T00:00:00"
    assert result.evidence_asymmetry_note.startswith("근거 수: ")
    # 프롬프트에 매트릭스·S4 tradeoffs·반대 근거가 들어간다
    human = llm.calls[0][-1][1]
    assert (
        "| mla | stakeholder | S4 |" in human
        and "S4 tradeoffs" in human
        and "mla-COUNTER-01" in human
    )


def test_unknown_evidence_ids_are_removed(inp):
    draft = _draft_from_fixture()
    draft["conflicts"][0]["evidence_ids"].append("mla-FAKE-99")
    draft["agreements"][0]["evidence_ids"].append("nope-1")
    result = run_synthesis(inp, _deps(StructuredStub([draft]))[0])
    all_ids = {i for c in result.conflicts for i in c.evidence_ids} | {
        i for a in result.agreements for i in a.evidence_ids
    }
    assert "mla-FAKE-99" not in all_ids and "nope-1" not in all_ids


def test_missing_s4_conflict_retries_then_raises(inp):
    draft = _draft_from_fixture()
    no_s4 = dict(
        draft,
        conflicts=[
            c
            for c in draft["conflicts"]
            if "S4" not in (c["criterion_a"], c["criterion_b"])
        ],
    )
    llm = StructuredStub([no_s4, draft])
    result = run_synthesis(inp, _deps(llm)[0])  # 2차 시도에서 통과
    assert len(llm.calls) == 2
    assert (
        "S4(details.tradeoffs)" in llm.calls[1][-1][1]
    )  # 피드백이 재시도 프롬프트에 들어감
    assert result.conflicts

    with pytest.raises(ValueError, match="S4"):
        run_synthesis(inp, _deps(StructuredStub([no_s4, no_s4]))[0])


def test_conflict_with_all_fake_ids_fails(inp):
    draft = _draft_from_fixture()
    for c in draft["conflicts"]:
        c["evidence_ids"] = ["x-1", "x-2"]
    with pytest.raises(ValueError):
        run_synthesis(inp, _deps(StructuredStub([draft]))[0])


def test_banned_terms_in_cause_rejected(inp):
    draft = _draft_from_fixture()
    draft["conflicts"][0]["cause"] = "MLA가 시장에서 우세하다"
    llm = StructuredStub([draft, _draft_from_fixture()])
    run_synthesis(inp, _deps(llm)[0])
    assert len(llm.calls) == 2


def test_detect_gaps_rules(inp):
    results = [*inp.trl_eval, *inp.market_eval, *inp.stakeholder_eval, *inp.domain_eval]
    gaps = {(g.tech_id, g.criterion_id, g.gap_type) for g in detect_gaps(results)}
    assert ("mla", "M1", "not_public") in gaps  # level=not_public
    assert ("mla", "T4", "not_public") in gaps  # 일부 not_public evidence
    assert ("pim_cxl", "M1", "single_source") in gaps  # 출처 1개, medium
    assert ("mla", "T4", "single_source") not in gaps  # not_public과 중복 표시하지 않음
    assert not any(t == "unit_mismatch" for *_, t in gaps)

    m2 = next(
        r for r in inp.market_eval if r.tech_id == "mla" and r.criterion_id == "M2"
    )
    shifted = m2.model_copy(
        update={"evidence_unit": "paper"}
    )  # 시장 판정을 논문 단위로 → 기대 단위(family)와 불일치
    assert any(g.gap_type == "unit_mismatch" for g in detect_gaps([shifted]))
