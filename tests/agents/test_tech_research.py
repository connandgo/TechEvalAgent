"""run_tech_research 스텁 테스트 (docs/roles/B-tech-domain.md §6)."""

import pytest

from techeval.agents._deps import AgentInput
from techeval.agents.tech_research import (
    ARTIFACT_KEYS,
    SURVEY_DOC_IDS,
    TRL_CRITERIA,
    compute_t3_level,
    run_tech_research,
)
from techeval.schemas import (
    TECHNOLOGIES,
    CriterionResult,
    TechProfile,
    compute_confidence,
)
from tests.agents._b_drafts import drafts
from tests.agents._b_helpers import CHUNKS_PATH, make_deps, make_llm, prompts
from tests.conftest import FIXED_NOW

TECH = {t.tech_id: t for t in TECHNOLOGIES}


@pytest.fixture
def stub_retriever():
    from techeval.retrieval.stub import StubRetriever

    return StubRetriever(str(CHUNKS_PATH))


def run(tech_id: str, table: dict | None = None, **inp_kwargs):
    table = table or drafts(tech_id)
    llm = make_llm(table)
    deps = make_deps(llm)
    out = run_tech_research(AgentInput(tech=TECH[tech_id], **inp_kwargs), deps)
    return out, deps, llm


# ------------------------------------------------------------------ 흐름


@pytest.mark.parametrize("tech_id", ["mla", "pim_cxl"])
def test_full_run_returns_profile_and_four_trl_results(tech_id):
    out, _, _ = run(tech_id)
    assert isinstance(out.tech_profile, TechProfile)
    assert [r.criterion_id for r in out.trl_eval] == list(TRL_CRITERIA)
    for r in out.trl_eval:
        CriterionResult.model_validate(r.model_dump())  # 스키마 재검증
        assert r.tech_id == tech_id and r.perspective == "trl"
        assert r.generated_at == FIXED_NOW and r.evidence_unit == "paper"
        assert r.confidence == compute_confidence(r.evidence)  # V7


def test_search_passes_primary_doc_and_survey_doc_ids():
    _, deps, _ = run("mla")
    calls = deps.retriever.calls
    primary = [c for c in calls if c["doc_ids"] == ["deepseek_v2"]]
    family = [c for c in calls if c["doc_ids"] == list(SURVEY_DOC_IDS)]
    assert primary and family
    assert all(c["doc_ids"] is not None for c in calls), (
        "전체 4편 무차별 검색은 하지 않는다"
    )


def test_rewritten_queries_take_priority():
    _, deps, _ = run("mla", rewritten_queries=["재작성 질의 A", "재작성 질의 B"])
    first_two = [c["query"] for c in deps.retriever.calls[:2]]
    assert first_two == ["재작성 질의 A", "재작성 질의 B"]


# ------------------------------------------------------------------ 근거 검증


def test_evidence_ids_follow_naming_rule_and_chunks_exist(stub_retriever):
    out, _, _ = run("mla")
    for i, e in enumerate(out.tech_profile.evidence, start=1):
        assert e.evidence_id == f"mla-PROFILE-{i:02d}"
    for r in out.trl_eval:
        for i, e in enumerate(r.evidence, start=1):
            assert e.evidence_id == f"mla-{r.criterion_id}-{i:02d}"
            if e.chunk_id:
                chunk = stub_retriever.get_chunk(e.chunk_id)  # V5
                assert chunk is not None
                assert e.quote in chunk.text  # V6
                assert e.locator.startswith(f"{chunk.doc_id} p.{chunk.page}")


def test_fabricated_chunk_id_is_dropped():
    table = drafts("mla")
    table["T2"]["citations"] = [
        {"source": "chunk", "ref": "deepseek_v2:099:99", "quote": "anything"},
        {
            "source": "chunk",
            "ref": "deepseek_v2:006:09",
            "quote": "requires a significantly smaller amount of KV cache",
        },
    ]
    out, _, _ = run("mla", table)
    t2 = next(r for r in out.trl_eval if r.criterion_id == "T2")
    assert [e.chunk_id for e in t2.evidence] == ["deepseek_v2:006:09"]
    assert t2.evidence[0].evidence_id == "mla-T2-01"  # 번호는 살아남은 것 기준


def test_paraphrased_quote_is_dropped_and_all_dropped_becomes_not_public():
    table = drafts("mla")
    table["T2"]["citations"] = [
        {
            "source": "chunk",
            "ref": "deepseek_v2:006:09",
            "quote": "MLA needs much less KV cache",
        },  # 다듬은 인용
    ]
    out, _, _ = run("mla", table)
    t2 = next(r for r in out.trl_eval if r.criterion_id == "T2")
    assert t2.level == "not_public" and t2.confidence == "low"
    ev = t2.evidence[0]
    assert ev.source_type == "not_public" and ev.search_query and ev.searched_at


def test_web_citation_must_come_from_search_results():
    table = drafts("mla")
    table["T3"]["citations"].append(
        {"source": "web", "ref": "https://example.com/made-up", "quote": "x"}
    )
    out, _, _ = run("mla", table)
    t3 = next(r for r in out.trl_eval if r.criterion_id == "T3")
    assert all(e.url != "https://example.com/made-up" for e in t3.evidence)
    assert len(t3.evidence) == 4


# ------------------------------------------------------------------ T3: 코드 계산


@pytest.mark.parametrize(
    "checklist,expected",
    [
        (dict.fromkeys(ARTIFACT_KEYS, "Y"), "L3"),
        (
            {
                "code": "Y",
                "model_or_design": "Y",
                "eval_scripts_data": "N",
                "third_party_reproduction": "Y",
            },
            "L3",
        ),
        (
            {
                "code": "Y",
                "model_or_design": "N",
                "eval_scripts_data": "unknown",
                "third_party_reproduction": "N",
            },
            "L2",
        ),
        (dict.fromkeys(ARTIFACT_KEYS, "N"), "L1"),
        (dict.fromkeys(ARTIFACT_KEYS, "unknown"), "L0"),
    ],
)
def test_compute_t3_level(checklist, expected):
    assert compute_t3_level(checklist) == expected


def test_t3_level_is_computed_not_taken_from_llm():
    out, _, _ = run("mla")
    t3 = next(r for r in out.trl_eval if r.criterion_id == "T3")
    assert t3.level == "L3" and t3.details["checklist"]["eval_scripts_data"] == "N"
    assert out.tech_profile.public_artifacts == t3.details["checklist"]
    assert out.tech_profile.public_artifact_urls["code"].startswith(
        "https://github.com/"
    )

    out2, _, _ = run("pim_cxl")
    t3b = next(r for r in out2.trl_eval if r.criterion_id == "T3")
    assert t3b.level == "L2"  # Y 1개


def test_t3_without_any_valid_evidence_is_l0_with_not_public_record():
    table = drafts("mla")
    table["T3"]["citations"] = []
    out, _, _ = run("mla", table)
    t3 = next(r for r in out.trl_eval if r.criterion_id == "T3")
    assert t3.level == "L0"
    assert t3.details["checklist"] == dict.fromkeys(ARTIFACT_KEYS, "unknown")
    assert t3.evidence[0].source_type == "not_public"


# ------------------------------------------------------------------ 기준별 필수 필드


def test_t1_requires_why_not_higher_else_result_is_omitted():
    table = drafts("mla")
    table["T1"]["details"].pop("why_not_higher")
    out, _, _ = run("mla", table)
    assert "T1" not in [r.criterion_id for r in out.trl_eval]
    assert [r.criterion_id for r in out.trl_eval] == ["T2", "T3", "T4"]


def test_t1_level_estimate_must_say_public_info_estimate():
    table = drafts("mla")
    table["T1"]["level_estimate"] = "TRL 6"
    out, _, _ = run("mla", table)
    assert "T1" not in [r.criterion_id for r in out.trl_eval]


def test_t4_separates_remaining_tasks_from_not_public_items():
    out, _, _ = run("mla")
    t4 = next(r for r in out.trl_eval if r.criterion_id == "T4")
    assert t4.level == "narrative"
    assert t4.details["remaining_tasks"] and t4.details["not_public_items"]
    assert "검색어:" in t4.details["not_public_items"][0]


# ------------------------------------------------------------------ 수치


def test_profile_measurements_link_to_evidence_and_note_missing_conditions():
    out, _, _ = run("mla")
    ev_ids = {e.evidence_id for e in out.tech_profile.evidence}
    m = out.tech_profile.measurements
    assert len(m) == 2
    assert all(x.evidence_id in ev_ids for x in m)
    kv = next(x for x in m if x.metric == "KV cache per token reduction")
    assert (
        kv.hardware is None
        and "원문 미기재" in kv.condition_note
        and "hardware" in kv.condition_note
    )


def test_measurement_whose_citation_failed_is_dropped():
    table = drafts("mla")
    table["PROFILE"]["citations"][2]["quote"] = (
        "reduces KV cache per token by 99%"  # 원문에 없음
    )
    out, _, _ = run("mla", table)
    assert [x.metric for x in out.tech_profile.measurements] == [
        "training GPU hours per trillion tokens"
    ]


# ------------------------------------------------------------------ 재실행


def test_missing_criteria_returns_only_those_and_refreshes_profile():
    first, _, _ = run("mla")
    table = drafts("mla")
    out, _, llm = run(
        "mla",
        table,
        tech_profile=first.tech_profile,
        missing_criteria=["T3"],
        retry_count=1,
    )
    assert [r.criterion_id for r in out.trl_eval] == ["T3"]
    assert out.trl_eval[0].retry_count == 1
    assert out.tech_profile.retry_count == 1
    assert out.tech_profile.principle == first.tech_profile.principle  # 기존 값 유지
    assert len(out.tech_profile.evidence) > len(
        first.tech_profile.evidence
    )  # 근거는 보강
    called = {
        k for k in ("T1", "T2", "T4") if any(f"# {k} —" in p for p in prompts(llm))
    }
    assert not called, "T3만 요청했는데 다른 기준 프롬프트가 호출됨"


def test_missing_criteria_outside_trl_is_rejected():
    with pytest.raises(ValueError):
        run("mla", missing_criteria=["D1"])


# ------------------------------------------------------------------ 편향 방지


def test_prompts_never_mention_the_other_technology():
    _, _, llm_a = run("mla")
    _, _, llm_b = run("pim_cxl")
    for p in prompts(llm_a):
        assert "pim_cxl" not in p and "PIM/CXL" not in p
    for p in prompts(llm_b):
        assert "deepseek_v2" not in p and "DeepSeek-V2" not in p and "MLA" not in p
