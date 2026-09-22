"""(C) run_market_eval 단위 테스트 — 네트워크·모델 불필요."""

import pytest

from techeval.agents._deps import AgentInput
from techeval.agents.market import (
    MARKET_CRITERIA,
    m2_level,
    m3_level,
    run_market_eval,
)
from techeval.agents.tech_research import CriterionDraft
from techeval.schemas import CriterionResult, compute_confidence, get_tech
from tests.agents._c_drafts import FIXED_NOW, draft_for, make_deps, make_llm

TECHS = ("mla", "pim_cxl")


def run(tech_id: str, **kw):
    deps = make_deps(llm=kw.pop("llm", None))
    inp = AgentInput(tech=get_tech(tech_id), **kw)
    return run_market_eval(inp, deps), deps


# --- M2 레벨 코드 계산 ---------------------------------------------------------
# 역할 C 문서 §5.2: 레벨은 확인된 최고 단계로 코드가 계산한다. LLM이 정하지 않는다.


@pytest.mark.parametrize(
    ("adopters", "expected"),
    [
        ([], "L0"),
        ([{"name": "A대학", "stage": "research"}], "L1"),
        ([{"name": "Samsung", "stage": "poc"}], "L2"),
        ([{"name": "Astera Labs", "stage": "product"}], "L3"),
        ([{"name": "DeepSeek", "stage": "production"}], "L4"),
        # 여러 사례가 있으면 가장 높은 단계
        (
            [
                {"name": "A", "stage": "poc"},
                {"name": "B", "stage": "production"},
                {"name": "C", "stage": "research"},
            ],
            "L4",
        ),
    ],
)
def test_m2_level_takes_highest_confirmed_stage(adopters, expected):
    assert m2_level(adopters) == expected


def test_m2_ignores_adopters_without_a_name():
    """채택 주체 없는 사례는 인정하지 않는다 (역할 C 문서 §5.2)."""
    assert m2_level([{"name": "", "stage": "production"}]) == "L0"
    assert m2_level([{"name": "   ", "stage": "product"}]) == "L0"


def test_m2_unknown_stage_does_not_raise_level():
    assert m2_level([{"name": "X", "stage": "널리 쓰임"}]) == "L0"


# --- M3 레벨 코드 계산 ---------------------------------------------------------


@pytest.mark.parametrize(
    ("yes_keys", "expected"),
    [
        ([], "L1"),
        (["framework"], "L2"),
        (["framework", "vendor_product"], "L2"),
        (["framework", "vendor_product", "standardization"], "L3"),
        (
            [
                "framework",
                "vendor_product",
                "standardization",
                "third_party_research_tools",
            ],
            "L3",
        ),
    ],
)
def test_m3_level_counts_yes(yes_keys, expected):
    checklist = {
        k: ("Y" if k in yes_keys else "N")
        for k in (
            "framework",
            "vendor_product",
            "standardization",
            "third_party_research_tools",
        )
    }
    assert m3_level(checklist) == expected


# --- 전체 실행 ----------------------------------------------------------------


@pytest.mark.parametrize("tech_id", TECHS)
def test_full_run_returns_three_market_results(tech_id):
    results, _ = run(tech_id)
    assert [r.criterion_id for r in results] == list(MARKET_CRITERIA)
    for r in results:
        CriterionResult.model_validate(r.model_dump())
        assert r.tech_id == tech_id and r.perspective == "market"
        assert r.generated_at == FIXED_NOW
        assert r.confidence == compute_confidence(r.evidence)  # V7
        assert r.evidence, f"{r.criterion_id}: 근거 없음 (V1)"
        assert not r.measurements, "시장 기준은 Measurement 대상이 아니다"


@pytest.mark.parametrize("tech_id", TECHS)
def test_market_evidence_is_family_unit(tech_id):
    """시장은 기술 계열 단위 조사 (AGENTS.md 4)."""
    results, _ = run(tech_id)
    for r in results:
        assert r.evidence_unit == "family"


@pytest.mark.parametrize("tech_id", TECHS)
def test_evidence_ids_follow_the_naming_rule(tech_id):
    results, _ = run(tech_id)
    for r in results:
        for e in r.evidence:
            assert e.evidence_id.startswith(f"{tech_id}-{r.criterion_id}-"), (
                e.evidence_id
            )


@pytest.mark.parametrize("tech_id", TECHS)
def test_level_is_computed_by_code_not_by_the_model(tech_id):
    """draft.level이 비어 있어도 M2·M3 레벨이 채워진다."""
    results, _ = run(tech_id)
    by_id = {r.criterion_id: r for r in results}
    assert by_id["M2"].level == m2_level(by_id["M2"].details["adopters"])
    assert by_id["M3"].level == m3_level(by_id["M3"].details["checklist"])


def test_model_supplied_level_is_overridden_for_m2():
    """모델이 레벨을 적어 보내도 코드 계산값이 이긴다."""
    draft = draft_for("mla", "M2")
    draft.level = "L0"  # 모델이 엉뚱한 값을 냈다고 가정
    results, _ = run("mla", llm=make_llm({("mla", "M2"): draft}))
    m2 = next(r for r in results if r.criterion_id == "M2")
    assert m2.level == "L4"


# --- details 형식 강제 ---------------------------------------------------------


def test_m1_requires_publisher_on_every_figure():
    """시장 수치는 발행 주체 없이 기록하지 않는다 (AGENTS.md 5, 역할 C §5.2)."""
    draft = draft_for("mla", "M1")
    draft.details["market_figures"][0].pop("publisher")
    results, _ = run("mla", llm=make_llm({("mla", "M1"): draft}))
    assert "M1" not in [r.criterion_id for r in results], (
        "발행 주체 없는 수치는 폐기돼야 한다"
    )


def test_m3_checklist_missing_key_is_discarded():
    """4항목 중 하나가 빠지면 기본값으로 채우지 않고 폐기한다."""
    draft = draft_for("mla", "M3")
    draft.details["checklist"].pop("standardization")
    results, _ = run("mla", llm=make_llm({("mla", "M3"): draft}))
    assert "M3" not in [r.criterion_id for r in results]


def test_m2_adopter_with_unknown_stage_is_discarded():
    draft = draft_for("mla", "M2")
    draft.details["adopters"][0]["stage"] = "출시 임박"
    results, _ = run("mla", llm=make_llm({("mla", "M2"): draft}))
    assert "M2" not in [r.criterion_id for r in results]


# --- 부분 재실행 --------------------------------------------------------------


def test_missing_criteria_runs_only_those():
    """재실행 시 지정된 기준만 생성한다 (AGENTS.md §6.2)."""
    results, _ = run("mla", missing_criteria=["M2"], retry_count=1)
    assert [r.criterion_id for r in results] == ["M2"]
    assert results[0].retry_count == 1


def test_missing_criteria_outside_this_agent_raises():
    with pytest.raises(ValueError, match="소관이 아닌"):
        run("mla", missing_criteria=["T1"])


# --- 검색 동작 ----------------------------------------------------------------


def test_m3_cross_checks_the_paper_with_the_retriever():
    """M3 프레임워크 지원은 논문 원문과 교차 확인한다 (역할 C §5.2)."""
    results, _ = run("mla", missing_criteria=["M3"])
    m3 = results[0]
    assert any(e.source_type == "paper" and e.chunk_id for e in m3.evidence), (
        "M3 근거에 논문 청크가 포함돼야 한다"
    )


def test_stakeholder_free_criteria_do_not_touch_the_retriever():
    """M1·M2는 논문 교차 확인이 필요 없다 — retriever를 부르지 않는다."""
    calls: list[str] = []

    class Recording:
        def search(self, query, **kw):
            calls.append(query)
            return []

        def get_chunk(self, chunk_id):
            return None

    from techeval.agents._deps import Deps

    deps = Deps(
        retriever=Recording(),
        web_search=make_deps().web_search,
        llm=make_llm(),
        now=lambda: FIXED_NOW,
    )
    run_market_eval(
        AgentInput(tech=get_tech("mla"), missing_criteria=["M1", "M2"]), deps
    )
    assert calls == []


def test_search_failure_does_not_abort_the_run():
    """질의 하나가 예외를 내도 나머지 기준은 계속 생성한다."""

    def boom(query, **kw):
        raise RuntimeError("provider down")

    deps = make_deps(llm=make_llm(), web_search=boom)
    results = run_market_eval(AgentInput(tech=get_tech("mla")), deps)
    by_id = {r.criterion_id: r for r in results}
    # 웹만 쓰는 M1·M2는 근거가 없으니 not_public. 예외로 죽지는 않는다.
    assert by_id["M1"].level == "not_public"
    assert by_id["M2"].level == "not_public"
    # M3은 논문 청크로 근거를 얻으므로 판정이 남는다.
    assert by_id["M3"].level != "not_public"
    assert all(e.source_type == "paper" for e in by_id["M3"].evidence)


def test_prompts_carry_tech_id_and_criterion_id():
    """FakeStructuredLLM이 프롬프트에서 둘을 읽어야 픽스처를 고를 수 있다 (CONTRACTS §9)."""
    llm = make_llm()
    run("mla", llm=llm)
    prompts = llm.prompts_for(CriterionDraft)
    assert prompts
    for p in prompts:
        assert "tech_id: mla" in p
    assert any("criterion_id: M1" in p for p in prompts)


def test_prompts_never_mention_the_other_technology():
    """각 기술은 독립 평가 — 프롬프트에 다른 기술이 섞이지 않는다 (AGENTS.md 6)."""
    llm = make_llm()
    run("mla", llm=llm)
    for p in llm.prompts_for(CriterionDraft):
        assert "pim_cxl" not in p
