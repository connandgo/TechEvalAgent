"""FakeStructuredLLM: 프롬프트 → 픽스처 매칭, 호출 기록, 오버라이드."""

import json

import pytest
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from techeval.agents._deps import TechResearchOutput
from techeval.schemas import CriterionResult, JudgeResult, TechProfile
from techeval.stub_llm import FakeStructuredLLM, FixtureLookupError, extract_tech_id

EV = {
    "evidence_id": "{tech}-{cid}-01",
    "source_type": "paper",
    "unit": "paper",
    "quote": "quote",
    "locator": "deepseek_v2 p.1 §1",
    "doc_id": "deepseek_v2",
    "chunk_id": "deepseek_v2:001:01",
}


def _cr(tech, cid, perspective):
    ev = {k: v.format(tech=tech, cid=cid) if isinstance(v, str) else v for k, v in EV.items()}
    return {
        "tech_id": tech,
        "perspective": perspective,
        "criterion_id": cid,
        "level": "L2",
        "content": f"{tech} {cid}",
        "evidence": [ev],
        "confidence": "medium",
        "evidence_unit": "paper",
        "generated_at": "2026-01-01T00:00:00",
        "details": {},
    }


@pytest.fixture
def fixtures_dir(tmp_path):
    trl = [_cr(t, c, "trl") for t in ("mla", "pim_cxl") for c in ("T1", "T2", "T3", "T4")]
    market = [_cr(t, c, "market") for t in ("mla", "pim_cxl") for c in ("M1", "M2", "M3")]
    profiles = [
        {
            "tech_id": t,
            "principle": f"{t} principle",
            "scope": "s",
            "limitations": ["l"],
            "measurements": [],
            "validation_env": "v",
            "public_artifacts": {},
            "evidence": [dict(EV, evidence_id=f"{t}-PROFILE-01")],
            "generated_at": "2026-01-01T00:00:00",
        }
        for t in ("mla", "pim_cxl")
    ]
    (tmp_path / "trl_eval.json").write_text(json.dumps(trl))
    (tmp_path / "market_eval.json").write_text(json.dumps(market))
    (tmp_path / "tech_profiles.json").write_text(json.dumps(profiles))
    return tmp_path


@pytest.fixture
def llm(fixtures_dir):
    return FakeStructuredLLM(fixtures_dir=fixtures_dir)


def test_single_criterion_from_prompt(llm):
    out = llm.with_structured_output(CriterionResult).invoke("tech_id: pim_cxl\n기준 T2 검증 환경을 판정하라")
    assert isinstance(out, CriterionResult)
    assert (out.tech_id, out.criterion_id) == ("pim_cxl", "T2")
    assert llm.call_count == 1 and "T2" in llm.prompts_for(CriterionResult)[0]


def test_list_of_criteria_by_mentioned_ids(llm):
    out = llm.with_structured_output(list[CriterionResult]).invoke("tech_id: mla — M1, M3만 다시 생성")
    assert [r.criterion_id for r in out] == ["M1", "M3"]


def test_list_of_criteria_by_perspective_keyword(llm):
    out = llm.with_structured_output(list[CriterionResult]).invoke("tech_id: mla 시장성 평가를 수행하라")
    assert [r.criterion_id for r in out] == ["M1", "M2", "M3"]


def test_tech_research_output_composed(llm):
    out = llm.with_structured_output(TechResearchOutput).invoke("tech_id: mla T1 T2 T3 T4")
    assert out.tech_profile.tech_id == "mla"
    assert [r.criterion_id for r in out.trl_eval] == ["T1", "T2", "T3", "T4"]


def test_generic_wrapper_model(llm):
    class MarketOutput(BaseModel):
        results: list[CriterionResult]
        profile: TechProfile

    out = llm.with_structured_output(MarketOutput).invoke("tech_id: pim_cxl M1 M2 M3")
    assert len(out.results) == 3 and out.profile.tech_id == "pim_cxl"


def test_lcel_chain_with_prompt_template(llm):
    prompt = ChatPromptTemplate.from_messages([("system", "평가자"), ("human", "tech_id: {tech} 기준 {cid}")])
    chain = prompt | llm.with_structured_output(CriterionResult)
    out = chain.invoke({"tech": "mla", "cid": "T3"})
    assert (out.tech_id, out.criterion_id) == ("mla", "T3")


def test_missing_fixture_names_owner(llm):
    with pytest.raises(FixtureLookupError, match="owner: E"):
        llm.with_structured_output(JudgeResult).invoke("검수")


def test_ambiguous_tech_raises(llm):
    with pytest.raises(FixtureLookupError, match="tech_id"):
        llm.with_structured_output(CriterionResult).invoke("T1 without tech")


def test_register_override(llm):
    llm.register(
        JudgeResult,
        lambda text: {
            "scores": {"evidence": 1, "criteria_compliance": 3, "neutrality": 3, "specificity": 3, "format": 3},
            "reasons": {},
            "missing_required": ["4.2"],
            "revision_instructions": ["fix"],
            "passed": False,
            "judged_at": "2026-01-01T00:00:00",
        },
    )
    out = llm.with_structured_output(JudgeResult).invoke("검수")
    assert out.passed is False and out.missing_required == ["4.2"]


def test_plain_invoke_records(llm):
    msg = llm.invoke("hello")
    assert msg.content == "" and llm.calls[-1]["schema"] == "str"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("tech_id: mla", "mla"),
        ('"tech_id": "pim_cxl"', "pim_cxl"),
        ("MLA compared with MLA and PIM", "mla"),
        ("PIM CXL and mla", "pim_cxl"),
        ("MLA vs PIM", None),
        ("nothing", None),
    ],
)
def test_extract_tech_id(text, expected):
    assert extract_tech_id(text) == expected


def test_draft_model_maps_to_object_fixture(fixtures_dir):
    """D의 SynthesisDraft처럼 계약 모델의 부분집합 필드를 가진 모델은 해당 픽스처로 채운다."""
    from techeval.schemas import Agreement, Conflict, Gap

    synthesis = {
        "agreements": [
            {
                "tech_id": "mla",
                "perspectives": ["trl", "domain"],
                "criterion_ids": ["T2", "D1"],
                "statement": "s",
                "evidence_ids": ["mla-T2-01"],
            }
        ],
        "conflicts": [
            {
                "tech_id": "mla",
                "perspective_a": "market",
                "criterion_a": "M2",
                "perspective_b": "domain",
                "criterion_b": "D2",
                "statement": "s",
                "cause": "c",
                "evidence_ids": ["mla-M2-01", "mla-D2-01"],
            }
        ],
        "gaps": [{"tech_id": "pim_cxl", "criterion_id": "M2", "gap_type": "single_source", "description": "d"}],
        "unit_notes": "u",
        "evidence_asymmetry_note": "a",
        "generated_at": "2026-01-01T00:00:00",
    }
    (fixtures_dir / "synthesis.json").write_text(json.dumps(synthesis))

    class SynthesisDraft(BaseModel):  # generated_at 없음 — D가 deps.now()로 채움
        agreements: list[Agreement]
        conflicts: list[Conflict]
        gaps: list[Gap]
        unit_notes: str
        evidence_asymmetry_note: str

    llm = FakeStructuredLLM(fixtures_dir=fixtures_dir)
    out = llm.with_structured_output(SynthesisDraft).invoke("종합 평가")
    assert isinstance(out, SynthesisDraft)
    assert out.conflicts[0].criterion_a == "M2" and not hasattr(out, "generated_at")


def test_draft_profile_model_picks_tech(fixtures_dir):
    class ProfileDraft(BaseModel):
        principle: str
        limitations: list[str]

    llm = FakeStructuredLLM(fixtures_dir=fixtures_dir)
    out = llm.with_structured_output(ProfileDraft).invoke("tech_id: pim_cxl 기술 개요")
    assert out.principle == "pim_cxl principle"
