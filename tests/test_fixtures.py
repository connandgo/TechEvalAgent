"""모든 `tests/fixtures/*`가 CONTRACTS 스키마를 통과하는지 검증한다 (main 머지 조건).

아직 제공되지 않은 픽스처는 소유 역할을 표시하고 skip한다. 제공된 픽스처가 스키마를 깨면 실패한다.
"""

import json
from pathlib import Path

import pytest

from techeval.schemas import (
    PERSPECTIVE_CRITERIA,
    STAKEHOLDERS,
    CriterionResult,
    Evidence,
    EvidenceGap,
    JudgeResult,
    SynthesisResult,
    TechProfile,
    TechRef,
    compute_confidence,
)
from techeval.stub_llm import FIXTURE_OWNERS

FIXTURES = Path(__file__).parent / "fixtures"
TECH_IDS = ("mla", "pim_cxl")

# CONTRACTS §2 `details` 필수 키
REQUIRED_DETAILS: dict[str, tuple[str, ...]] = {
    "T1": ("trl_band", "estimate", "why_not_higher"),
    "T2": ("env_level",),
    "T3": ("checklist",),
    "T4": ("remaining_tasks", "not_public_items"),
    "M1": ("market_figures",),
    "M2": ("adopters",),
    "M3": ("checklist",),
    "S1": ("roles",),
    "S2": ("benefits",),
    "S3": ("burdens",),
    "S4": ("tradeoffs",),
    "D1": ("directness",),
    "D2": ("directness",),
    "D3": ("directness",),
    "D4": ("checklist",),
}


def _load(name: str):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} 미제공 (역할 {FIXTURE_OWNERS.get(name, '?')})")
    if name.endswith(".md"):
        return path.read_text(encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_criterion_list(name: str, perspective: str) -> list[CriterionResult]:
    results = [CriterionResult.model_validate(r) for r in _load(name)]
    expected = {(t, c) for t in TECH_IDS for c in PERSPECTIVE_CRITERIA[perspective]}
    got = {(r.tech_id, r.criterion_id) for r in results}
    assert got == expected, f"{name}: (tech, criterion) 집합 불일치. 누락={expected - got}, 초과={got - expected}"
    for r in results:
        assert r.perspective == perspective
        assert r.confidence == compute_confidence(r.evidence), (
            f"{name} {r.tech_id}/{r.criterion_id}: confidence={r.confidence} 이지만 "
            f"compute_confidence={compute_confidence(r.evidence)} (V7)"
        )
        missing = [k for k in REQUIRED_DETAILS[r.criterion_id] if k not in r.details]
        assert not missing, f"{name} {r.tech_id}/{r.criterion_id}: details 필수 키 누락 {missing}"
        if r.level != "not_public":
            for e in r.evidence:
                assert e.evidence_id.startswith(f"{r.tech_id}-{r.criterion_id}-"), (
                    f"{name}: evidence_id 규칙 위반 {e.evidence_id} (기대 접두사 {r.tech_id}-{r.criterion_id}-)"
                )
    return results


# --- A ---------------------------------------------------------------------


def test_chunks():
    retriever = pytest.importorskip("techeval.retrieval.retriever", reason="역할 A 미제공")
    chunks = [retriever.RetrievedChunk.model_validate(c) for c in _load("chunks.json")]
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
    assert set(by_doc) == {"deepseek_v2", "pim_cxl_1m", "io_survey", "kv_survey"}, by_doc
    assert all(n >= 5 for n in by_doc.values()), f"doc당 5청크 이상 필요: {by_doc}"
    assert any(c.chunk_type == "table" for c in chunks), "표 청크(chunk_type='table') 포함 필요"
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk_id 중복"


# --- C ---------------------------------------------------------------------


def test_web_results():
    ws = pytest.importorskip("techeval.tools.web_search", reason="역할 C 미제공")
    data = _load("web_results.json")
    assert isinstance(data, dict) and data
    for _query, results in data.items():
        for r in results:
            ws.WebResult.model_validate(r)


def test_market_eval():
    _check_criterion_list("market_eval.json", "market")


def test_stakeholder_eval():
    results = _check_criterion_list("stakeholder_eval.json", "stakeholder")
    for r in results:
        if r.criterion_id in ("S2", "S3"):
            key = "benefits" if r.criterion_id == "S2" else "burdens"
            for s in STAKEHOLDERS:
                assert r.details[key].get(s), f"{r.tech_id}/{r.criterion_id}: {s} 항목 ≥1 필요 (V4)"
        if r.criterion_id == "S4":
            assert len(r.details["tradeoffs"]) >= 1, f"{r.tech_id}/S4: tradeoffs ≥1 필요 (V4)"


# --- B ---------------------------------------------------------------------


def test_tech_profiles():
    profiles = [TechProfile.model_validate(p) for p in _load("tech_profiles.json")]
    assert {p.tech_id for p in profiles} == set(TECH_IDS)
    for p in profiles:
        assert p.principle and p.validation_env
        assert p.limitations and p.measurements
        assert set(p.public_artifacts) >= {"code", "model_or_design", "eval_scripts_data", "third_party_reproduction"}


def test_trl_eval():
    _check_criterion_list("trl_eval.json", "trl")


def test_domain_eval():
    results = _check_criterion_list("domain_eval.json", "domain")
    for r in results:
        if r.criterion_id in ("D1", "D2", "D3") and r.level in ("L2", "L3"):
            assert r.measurements, f"{r.tech_id}/{r.criterion_id}: level {r.level} 이면 measurements 필요 (V3)"
            if r.details["directness"] == "L2":
                assert r.details.get("extrapolation_logic"), f"{r.tech_id}/{r.criterion_id}: L2는 외삽 논리 필수"


# --- D ---------------------------------------------------------------------


def test_synthesis():
    s = SynthesisResult.model_validate(_load("synthesis.json"))
    assert s.conflicts and s.unit_notes and s.evidence_asymmetry_note


def test_report_md():
    md = _load("report_md.md")
    for chapter in ("SUMMARY", "1.", "2.", "3.", "4.", "5.", "6.", "REFERENCE"):
        assert chapter in md, f"report_md.md에 '{chapter}' 챕터 표기 필요"


# --- E ---------------------------------------------------------------------


def test_counter_evidence():
    evs = [Evidence.model_validate(e) for e in _load("counter_evidence.json")]
    assert evs
    for e in evs:
        assert "-COUNTER-" in e.evidence_id, e.evidence_id
        assert e.unit == "family"


def test_evidence_gap():
    g = EvidenceGap.model_validate(_load("evidence_gap.json"))
    assert set(g.evidence_count) == set(TECH_IDS)
    for item in g.opposing_missing:
        assert {"tech_id", "criterion_id", "reason"} <= set(item)


def test_judge_result():
    j = JudgeResult.model_validate(_load("judge_result.json"))
    assert set(j.scores) == {"evidence", "criteria_compliance", "neutrality", "specificity", "format"}
    assert j.passed == (all(v >= 3 for v in j.scores.values()) and not j.missing_required)


def test_state_initial():
    s = _load("state_initial.json")
    techs = [TechRef.model_validate(t) for t in s["technologies"]]
    assert [t.tech_id for t in techs] == list(TECH_IDS)
    assert tuple(s["stakeholders"]) == STAKEHOLDERS
    assert set(s["missing_criteria"]) == set(PERSPECTIVE_CRITERIA)
    assert all(v == 0 for v in s["retry_counts"].values())
    assert {"counter", "report"} <= set(s["retry_counts"])
    for key in ("tech_profiles", "trl_eval", "market_eval", "stakeholder_eval", "domain_eval", "counter_evidence"):
        assert s[key] == []
