"""(C) run_stakeholder_eval 단위 테스트 — 네트워크·모델 불필요."""

import pytest

from techeval.agents._deps import AgentInput, Deps
from techeval.agents.stakeholder import STAKEHOLDER_CRITERIA, run_stakeholder_eval
from techeval.agents.tech_research import CriterionDraft
from techeval.schemas import STAKEHOLDERS, CriterionResult, compute_confidence, get_tech
from tests.agents._c_drafts import FIXED_NOW, draft_for, make_deps, make_llm

TECHS = ("mla", "pim_cxl")
ROLES = {"decision_maker", "affected", "supplier", "unrelated"}


def run(tech_id: str, **kw):
    deps = make_deps(llm=kw.pop("llm", None))
    return run_stakeholder_eval(AgentInput(tech=get_tech(tech_id), **kw), deps)


# --- 전체 실행 ----------------------------------------------------------------


@pytest.mark.parametrize("tech_id", TECHS)
def test_full_run_returns_four_stakeholder_results(tech_id):
    results = run(tech_id)
    assert [r.criterion_id for r in results] == list(STAKEHOLDER_CRITERIA)
    for r in results:
        CriterionResult.model_validate(r.model_dump())
        assert r.tech_id == tech_id and r.perspective == "stakeholder"
        assert r.generated_at == FIXED_NOW
        assert r.confidence == compute_confidence(r.evidence)  # V7
        assert r.evidence, f"{r.criterion_id}: 근거 없음 (V1)"
        assert r.evidence_unit == "family"
        assert not r.measurements


@pytest.mark.parametrize("tech_id", TECHS)
def test_levels_are_fixed_strings(tech_id):
    by_id = {r.criterion_id: r for r in run(tech_id)}
    assert by_id["S1"].level == "assigned"
    for cid in ("S2", "S3", "S4"):
        assert by_id[cid].level == "narrative"


def test_model_supplied_level_is_overridden():
    """모델이 엉뚱한 레벨을 내도 에이전트가 고정값으로 덮어쓴다."""
    draft = draft_for("mla", "S1")
    draft.level = "L3"
    results = run("mla", llm=make_llm({("mla", "S1"): draft}), missing_criteria=["S1"])
    assert results[0].level == "assigned"


# --- S1 역할 배정 -------------------------------------------------------------


@pytest.mark.parametrize("tech_id", TECHS)
def test_s1_assigns_every_stakeholder_a_known_role(tech_id):
    s1 = next(r for r in run(tech_id) if r.criterion_id == "S1")
    roles = s1.details["roles"]
    assert list(roles) == list(STAKEHOLDERS), "4주체 순서를 고정한다"
    assert set(roles.values()) <= ROLES


def test_s1_missing_stakeholder_is_discarded():
    draft = draft_for("mla", "S1")
    draft.details["roles"].pop("investor")
    results = run("mla", llm=make_llm({("mla", "S1"): draft}), missing_criteria=["S1"])
    assert results == []


def test_s1_unknown_role_value_is_discarded():
    draft = draft_for("mla", "S1")
    draft.details["roles"]["investor"] = "관망"
    results = run("mla", llm=make_llm({("mla", "S1"): draft}), missing_criteria=["S1"])
    assert results == []


# --- S2/S3 4주체 보장 (V4) -----------------------------------------------------


@pytest.mark.parametrize("tech_id", TECHS)
@pytest.mark.parametrize(("cid", "key"), [("S2", "benefits"), ("S3", "burdens")])
def test_s2_s3_cover_all_four_stakeholders(tech_id, cid, key):
    r = next(x for x in run(tech_id) if x.criterion_id == cid)
    entries = r.details[key]
    for s in STAKEHOLDERS:
        assert entries.get(s), f"{cid}: {s} 가 비어 있다"
        for item in entries[s]:
            assert item["text"].strip()
            assert "is_inference" in item


@pytest.mark.parametrize(("cid", "key"), [("S2", "benefits"), ("S3", "burdens")])
def test_s2_s3_missing_one_stakeholder_is_discarded(cid, key):
    draft = draft_for("mla", cid)
    draft.details[key]["memory_semiconductor_vendor"] = []
    results = run("mla", llm=make_llm({("mla", cid): draft}), missing_criteria=[cid])
    assert results == [], "4주체 중 하나라도 비면 폐기돼야 한다 (V4)"


# --- 추론 표시와 신뢰도 -------------------------------------------------------


@pytest.mark.parametrize(("cid", "key"), [("S2", "benefits"), ("S3", "burdens")])
def test_inference_entries_get_their_own_evidence_and_lower_confidence(cid, key):
    """직접 자료가 없는 항목은 inference Evidence를 얻고 신뢰도가 low로 내려간다."""
    r = next(x for x in run("mla") if x.criterion_id == cid)
    inferred = [
        item
        for items in r.details[key].values()
        for item in items
        if item["is_inference"]
    ]
    assert inferred, "픽스처에 추론 항목이 있어야 이 테스트가 의미 있다"
    inference_ids = {e.evidence_id for e in r.evidence if e.source_type == "inference"}
    assert inference_ids
    for item in inferred:
        assert item["evidence_id"] in inference_ids, (
            "추론 항목은 inference 근거를 가리켜야 한다"
        )
    assert r.confidence == "low", "inference가 섞이면 low (AGENTS.md 3)"


def test_inference_evidence_records_its_starting_point():
    r = next(x for x in run("mla") if x.criterion_id == "S2")
    for e in r.evidence:
        if e.source_type == "inference":
            assert e.locator.startswith("inference"), e.locator
            assert e.quote, "추론 근거도 무엇을 추론했는지 남긴다"


def test_all_direct_sources_keeps_confidence_above_low():
    """추론이 없으면 신뢰도가 low로 떨어지지 않는다 — low가 상수가 아님을 확인."""
    draft = draft_for("mla", "S2")
    for items in draft.details["benefits"].values():
        for item in items:
            item["is_inference"] = False
            item["evidence_id"] = "mla-S2-01"
    results = run("mla", llm=make_llm({("mla", "S2"): draft}), missing_criteria=["S2"])
    assert results[0].confidence != "low"


# --- S4 상충 ------------------------------------------------------------------


@pytest.mark.parametrize("tech_id", TECHS)
def test_s4_has_at_least_one_pair_with_distinct_sides(tech_id):
    s4 = next(r for r in run(tech_id) if r.criterion_id == "S4")
    pairs = s4.details["tradeoffs"]
    assert pairs, "기술당 최소 1쌍 (V4)"
    for t in pairs:
        assert t["beneficiary"] in STAKEHOLDERS
        assert t["burdened"] in STAKEHOLDERS
        assert t["beneficiary"] != t["burdened"]
        assert "편익" in t["text"] and "부담" in t["text"], (
            "'A에게 편익인 것이 B에게 부담' 형식을 지킨다"
        )


def test_s4_empty_tradeoffs_is_discarded():
    draft = draft_for("mla", "S4")
    draft.details["tradeoffs"] = []
    results = run("mla", llm=make_llm({("mla", "S4"): draft}), missing_criteria=["S4"])
    assert results == []


def test_s4_same_side_on_both_ends_is_discarded():
    draft = draft_for("mla", "S4")
    draft.details["tradeoffs"][0]["burdened"] = draft.details["tradeoffs"][0][
        "beneficiary"
    ]
    results = run("mla", llm=make_llm({("mla", "S4"): draft}), missing_criteria=["S4"])
    assert results == []


def test_s4_unknown_stakeholder_is_discarded():
    draft = draft_for("mla", "S4")
    draft.details["tradeoffs"][0]["burdened"] = "정부"
    results = run("mla", llm=make_llm({("mla", "S4"): draft}), missing_criteria=["S4"])
    assert results == []


# --- RAG 미사용 ---------------------------------------------------------------


def test_stakeholder_never_calls_the_retriever():
    """이해관계자 평가는 웹만 쓴다 (역할 C 문서 §5.3)."""
    calls: list[str] = []

    class Recording:
        def search(self, query, **kw):
            calls.append(query)
            return []

        def get_chunk(self, chunk_id):
            return None

    deps = Deps(
        retriever=Recording(),
        web_search=make_deps().web_search,
        llm=make_llm(),
        now=lambda: FIXED_NOW,
    )
    run_stakeholder_eval(AgentInput(tech=get_tech("mla")), deps)
    assert calls == [], "retriever.search 를 부르지 않아야 한다"


# --- 부분 재실행·프롬프트 -----------------------------------------------------


def test_missing_criteria_runs_only_those():
    results = run("mla", missing_criteria=["S4"], retry_count=2)
    assert [r.criterion_id for r in results] == ["S4"]
    assert results[0].retry_count == 2


def test_missing_criteria_outside_this_agent_raises():
    with pytest.raises(ValueError, match="소관이 아닌"):
        run("mla", missing_criteria=["M1"])


def test_prompts_carry_tech_id_and_the_four_stakeholders():
    llm = make_llm()
    run("mla", llm=llm)
    prompts = llm.prompts_for(CriterionDraft)
    assert prompts
    for p in prompts:
        assert "tech_id: mla" in p
        assert "pim_cxl" not in p, "다른 기술을 언급하지 않는다 (AGENTS.md 6)"
    system_covered = any(all(s in p for s in STAKEHOLDERS) for p in prompts)
    assert system_covered, "4주체 id가 프롬프트에 그대로 들어가야 한다"


def test_both_technologies_use_the_same_prompt_template():
    """기술별 분기 프롬프트를 두지 않는다 (역할 C 문서 §5.3)."""
    llm_a, llm_b = make_llm(), make_llm()
    run("mla", llm=llm_a, missing_criteria=["S4"])
    run("pim_cxl", llm=llm_b, missing_criteria=["S4"])
    a = llm_a.prompts_for(CriterionDraft)[0].replace("mla", "<T>")
    b = llm_b.prompts_for(CriterionDraft)[0].replace("pim_cxl", "<T>")
    # 검색 결과 블록만 다르고, 지시문은 같아야 한다.
    assert a.split("## 검색 결과")[0] == b.split("## 검색 결과")[0]
