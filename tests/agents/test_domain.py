"""run_domain_eval 스텁 테스트 (docs/roles/B-tech-domain.md §6)."""

from pathlib import Path

import pytest

from techeval.agents._deps import AgentInput
from techeval.agents.domain import D4_KEYS, DOMAIN_CRITERIA, run_domain_eval
from techeval.agents.tech_research import run_tech_research
from techeval.schemas import TECHNOLOGIES, CriterionResult, compute_confidence
from tests.agents._b_drafts import drafts
from tests.agents._b_helpers import make_deps, make_llm, prompts
from tests.conftest import FIXED_NOW

TECH = {t.tech_id: t for t in TECHNOLOGIES}
PROMPT_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "techeval" / "prompts" / "domain"
)


def profile_for(tech_id: str):
    llm = make_llm(drafts(tech_id))
    return run_tech_research(
        AgentInput(tech=TECH[tech_id]), make_deps(llm)
    ).tech_profile


def run(tech_id: str, table: dict | None = None, **inp_kwargs):
    table = table or drafts(tech_id)
    llm = make_llm(table)
    deps = make_deps(llm)
    inp_kwargs.setdefault("tech_profile", profile_for(tech_id))
    results = run_domain_eval(AgentInput(tech=TECH[tech_id], **inp_kwargs), deps)
    return results, deps, llm


# ------------------------------------------------------------------ 흐름


@pytest.mark.parametrize("tech_id", ["mla", "pim_cxl"])
def test_full_run_returns_four_domain_results(tech_id):
    results, _, _ = run(tech_id)
    assert [r.criterion_id for r in results] == list(DOMAIN_CRITERIA)
    for r in results:
        CriterionResult.model_validate(r.model_dump())
        assert r.tech_id == tech_id and r.perspective == "domain"
        assert r.generated_at == FIXED_NOW
        assert r.confidence == compute_confidence(r.evidence)
    for r in results[:3]:
        if r.level in ("L2", "L3"):
            assert r.measurements, (
                f"{r.criterion_id}: L2/L3 는 Measurement 필수 (V3)"
            )  # V3
        assert r.details["directness"] == r.level


def test_profile_evidence_chunks_are_restored_via_get_chunk():
    _, _, llm = run("mla")
    ctx_text = prompts(llm)[0]
    for e in profile_for("mla").evidence:
        if e.chunk_id:
            assert e.chunk_id in ctx_text, (
                "tech_profile 근거 청크가 도메인 프롬프트에 없음"
            )


def test_search_is_scoped_to_primary_doc_and_surveys():
    _, deps, _ = run("pim_cxl")
    doc_id_sets = {tuple(c["doc_ids"]) for c in deps.retriever.calls}
    assert doc_id_sets == {("pim_cxl_1m",), ("io_survey", "kv_survey")}


def test_missing_criteria_returns_only_that_criterion():
    results, _, llm = run("mla", missing_criteria=["D2"], retry_count=2)
    assert [r.criterion_id for r in results] == ["D2"] and results[0].retry_count == 2
    assert not any(
        "# D1 —" in p or "# D3 —" in p or "# D4 —" in p for p in prompts(llm)
    )


# ------------------------------------------------------------------ D1~D3 직접성


def test_l2_without_extrapolation_logic_is_omitted():
    table = drafts("mla")
    table["D1"]["details"].pop("extrapolation_logic")
    results, _, _ = run("mla", table)
    assert "D1" not in [r.criterion_id for r in results]


def test_d1_without_any_valid_measurement_becomes_not_public():
    table = drafts("mla")
    table["D1"]["measurements"] = []
    results, _, _ = run("mla", table)
    d1 = next(r for r in results if r.criterion_id == "D1")
    assert d1.level == "not_public" and d1.evidence[0].source_type == "not_public"
    assert d1.confidence == "low"


def test_measurement_conditions_missing_in_paper_are_marked_not_stated():
    results, _, _ = run("mla")
    d2 = next(r for r in results if r.criterion_id == "D2")
    m = d2.measurements[0]
    assert (
        m.batch_size is None
        and "원문 미기재" in m.condition_note
        and "batch_size" in m.condition_note
    )


# ------------------------------------------------------------------ D4 체크리스트


def test_d4_checklist_has_exactly_six_keys_with_allowed_values():
    for tech_id in ("mla", "pim_cxl"):
        results, _, _ = run(tech_id)
        d4 = next(r for r in results if r.criterion_id == "D4")
        assert d4.level == "checklist" and d4.measurements == []
        assert tuple(d4.details["checklist"]) == D4_KEYS
        assert set(d4.details["checklist"].values()) <= {"Y", "N", "unknown"}


def test_d4_with_missing_key_is_omitted_not_defaulted():
    table = drafts("mla")
    table["D4"]["details"]["checklist"].pop("memory_add")
    results, _, _ = run("mla", table)
    assert "D4" not in [r.criterion_id for r in results]


def test_d4_with_bad_value_is_omitted():
    table = drafts("pim_cxl")
    table["D4"]["details"]["checklist"]["hw_replace"] = "maybe"
    results, _, _ = run("pim_cxl", table)
    assert "D4" not in [r.criterion_id for r in results]


def test_d4_web_evidence_comes_from_returned_results():
    results, _, _ = run("mla")
    d4 = next(r for r in results if r.criterion_id == "D4")
    web = [e for e in d4.evidence if e.url]
    assert web and web[0].source_type == "official" and web[0].accessed_date


# ------------------------------------------------------------------ 편향 방지 (§5.2: 예상 답 미주입, 동일 템플릿)


@pytest.mark.parametrize("name", ["system", "D1", "D2", "D3", "D4"])
def test_domain_prompt_files_contain_no_tech_specific_hints(name):
    text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    for banned in (
        "MLA",
        "DeepSeek",
        "PIM",
        "CXL",
        "재학습이 필요",
        "재학습 필요",
        "하드웨어 교체가 필요",
    ):
        assert banned not in text, f"{name}.md 에 힌트/기술명 '{banned}' 포함"
