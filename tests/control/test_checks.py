"""제어 노드 단위 테스트.

tech_evidence_check / perspective_check / evidence_gap / query_rewrite / counter_evidence / judge
"""

from techeval.agents._deps import Deps, ReportInput
from techeval.control.counter_evidence import search_counter_evidence
from techeval.control.evidence_gap import check_evidence_gap
from techeval.control.judge import judge_report, static_checks
from techeval.control.perspective_check import check_perspectives
from techeval.control.query_rewrite import rewrite_queries
from techeval.control.sources import SourceRegistry, evidence_problems, normalize_url
from techeval.control.tech_evidence_check import check_tech_evidence
from techeval.schemas import (
    PERSPECTIVE_CRITERIA,
    EvidenceGap,
    Gap,
    JudgeResult,
    SynthesisResult,
)
from techeval.stub_llm import FakeStructuredLLM
from tests.helpers import (
    MLA,
    NOW,
    PIM,
    MemRetriever,
    make_chunks,
    make_web_search,
    not_public_evidence,
    paper_evidence,
    profile,
    result,
    web_evidence,
)

TECHS = [MLA, PIM]


def _retriever() -> MemRetriever:
    return MemRetriever(make_chunks())


def _chunk(r: MemRetriever, doc: str, page: int = 1):
    return r.by_id[f"{doc}:{page:03d}:01"]


def _trl(r: MemRetriever, tech, cids=("T1", "T2", "T3", "T4")):
    doc = tech.primary_doc_id
    return [result(tech, c, "trl", [paper_evidence(f"{tech.tech_id}-{c}-01", _chunk(r, doc))]) for c in cids]


# --- sources ---------------------------------------------------------------


def test_registry_records_and_normalizes_urls():
    reg = SourceRegistry()
    ws = reg.wrap(make_web_search(n=1))
    ws("q")
    assert reg.has("https://site0.example.com/1") and reg.has("http://www.site0.example.com/1/")
    assert not reg.has("https://other.example.com/1")
    assert normalize_url("HTTPS://www.A.com/x/#frag") == "a.com/x"


def test_evidence_problems_paper_and_web():
    r = _retriever()
    reg = SourceRegistry(preload_urls=["https://ok.com/a"])
    ok = paper_evidence("mla-T1-01", _chunk(r, "deepseek_v2"))
    assert evidence_problems(ok, r, reg) == []
    bad_chunk = ok.model_copy(update={"chunk_id": "deepseek_v2:099:01"})
    assert any("V5" in p for p in evidence_problems(bad_chunk, r, reg))
    bad_quote = ok.model_copy(update={"quote": "text that is not in the chunk"})
    assert any("V6" in p for p in evidence_problems(bad_quote, r, reg))
    assert evidence_problems(web_evidence("mla-M1-01", "https://ok.com/a"), r, reg) == []
    assert any("V5" in p for p in evidence_problems(web_evidence("mla-M1-02", "https://nope.com/a"), r, reg))
    assert evidence_problems(web_evidence("mla-M1-02", "https://nope.com/a"), r, None) == []  # registry 없으면 생략
    assert evidence_problems(not_public_evidence("mla-T3-NP"), r, reg) == []


# --- tech_evidence_check -----------------------------------------------------


def test_tech_check_sufficient():
    r = _retriever()
    profiles = [profile(t, [paper_evidence(f"{t.tech_id}-PROFILE-01", _chunk(r, t.primary_doc_id))]) for t in TECHS]
    res = check_tech_evidence(TECHS, profiles, _trl(r, MLA) + _trl(r, PIM), r)
    assert res.sufficient and res.missing == {"mla": [], "pim_cxl": []}


def test_tech_check_reports_missing_criterion_profile_field_and_bad_evidence():
    r = _retriever()
    mla_profile = profile(MLA, [paper_evidence("mla-PROFILE-01", _chunk(r, "deepseek_v2"))], measurements=[])
    pim_profile = profile(PIM, [paper_evidence("pim_cxl-PROFILE-01", _chunk(r, "pim_cxl_1m"))])
    trl = _trl(r, MLA, cids=("T1", "T2", "T4")) + _trl(r, PIM)
    trl[0].evidence[0] = trl[0].evidence[0].model_copy(update={"chunk_id": "deepseek_v2:099:01"})  # mla T1 조작
    res = check_tech_evidence(TECHS, [mla_profile, pim_profile], trl, r)
    assert not res.sufficient
    assert res.missing["mla"] == ["PROFILE:measurements", "T1", "T3"]
    assert res.missing_criteria("mla") == ["T1", "T3"]
    assert res.missing["pim_cxl"] == []


def test_tech_check_missing_profile_entirely():
    r = _retriever()
    res = check_tech_evidence(TECHS, [], _trl(r, MLA) + _trl(r, PIM), r)
    assert res.missing["mla"] == [
        f"PROFILE:{f}" for f in ("principle", "limitations", "measurements", "validation_env")
    ]


def test_tech_check_not_public_is_not_rechecked():
    r = _retriever()
    trl = _trl(r, MLA, cids=("T1", "T2", "T4")) + _trl(r, PIM)
    trl.append(result(MLA, "T3", "trl", [not_public_evidence("mla-T3-NP")], level="not_public"))
    profiles = [profile(t, [paper_evidence(f"{t.tech_id}-PROFILE-01", _chunk(r, t.primary_doc_id))]) for t in TECHS]
    assert check_tech_evidence(TECHS, profiles, trl, r).sufficient


# --- perspective_check ---------------------------------------------------------


def _full_evals(r: MemRetriever, reg: SourceRegistry) -> dict:
    """30개 전부 유효한 결과 (기준별 유효 level·details는 helpers.result 기본값)."""
    evals: dict = {}
    for p, cids in PERSPECTIVE_CRITERIA.items():
        evals[p] = []
        for t in TECHS:
            for c in cids:
                if p in ("trl", "domain"):
                    ev = [paper_evidence(f"{t.tech_id}-{c}-01", _chunk(r, t.primary_doc_id))]
                else:
                    url = f"https://{t.tech_id}.example.com/{c}"
                    reg.add(url)
                    ev = [web_evidence(f"{t.tech_id}-{c}-01", url)]
                evals[p].append(result(t, c, p, ev))
    return evals


def test_perspective_check_all_sufficient():
    r, reg = _retriever(), SourceRegistry()
    res = check_perspectives(TECHS, _full_evals(r, reg), r, reg)
    assert res.sufficient and res.corrected == [] and res.problems == []
    assert res.flat_missing() == {"trl": [], "market": [], "stakeholder": [], "domain": []}


def test_perspective_check_finds_missing_v4_v5_v7():
    r, reg = _retriever(), SourceRegistry()
    evals = _full_evals(r, reg)
    # V8: pim M2 제거
    evals["market"] = [x for x in evals["market"] if not (x.tech_id == "pim_cxl" and x.criterion_id == "M2")]
    # V4: mla S2 investor 비움
    s2 = next(x for x in evals["stakeholder"] if x.tech_id == "mla" and x.criterion_id == "S2")
    s2.details["benefits"]["investor"] = []
    # V4: pim D2 외삽 논리 없음
    d2 = next(x for x in evals["domain"] if x.tech_id == "pim_cxl" and x.criterion_id == "D2")
    d2.details = {"directness": "L2"}
    # V5: mla M1 URL 미기록
    m1 = next(x for x in evals["market"] if x.tech_id == "mla" and x.criterion_id == "M1")
    m1.evidence[0] = m1.evidence[0].model_copy(update={"url": "https://fake.example.com/x"})
    # V7: pim T1 confidence 과대
    t1 = next(x for x in evals["trl"] if x.tech_id == "pim_cxl" and x.criterion_id == "T1")
    t1.confidence = "high"

    res = check_perspectives(TECHS, evals, r, reg)
    assert res.missing["market"] == {"mla": ["M1"], "pim_cxl": ["M2"]}
    assert res.missing["stakeholder"] == {"mla": ["S2"], "pim_cxl": []}
    assert res.missing["domain"] == {"mla": [], "pim_cxl": ["D2"]}
    assert res.missing["trl"] == {"mla": [], "pim_cxl": []}
    assert [(c.tech_id, c.criterion_id, c.confidence) for c in res.corrected] == [("pim_cxl", "T1", "medium")]
    assert res.flat_missing()["market"] == ["M1", "M2"]


# --- evidence_gap ------------------------------------------------------------------


def test_evidence_gap_balanced_with_independent_sources():
    r, reg = _retriever(), SourceRegistry()
    evals = _full_evals(r, reg)
    # trl·domain 근거에 서베이 청크를 하나씩 더해 독립 출처를 만든다
    for p in ("trl", "domain"):
        for x in evals[p]:
            x.evidence.append(paper_evidence(f"{x.evidence[0].evidence_id}-s", _chunk(r, "kv_survey"), unit="family"))
    gap = check_evidence_gap(TECHS, evals, None)
    assert gap.asymmetry is False and gap.needs_counter_search is False
    assert gap.evidence_count == {"mla": 23, "pim_cxl": 23}  # 기준당 1건(15) + 서베이 8건
    assert gap.vendor_source_ratio == {"mla": 0.348, "pim_cxl": 0.348}  # 8/23 (T·D 원문 청크 8건)


def test_evidence_gap_ratio_and_opposing_missing():
    r, reg = _retriever(), SourceRegistry()
    evals = _full_evals(r, reg)
    for x in evals["market"] + evals["stakeholder"]:
        if x.tech_id == "mla":
            x.evidence += [
                web_evidence(f"{x.evidence[0].evidence_id}-{i}", f"https://mla.example.com/{i}") for i in range(3)
            ]
    gap = check_evidence_gap(TECHS, evals, None)
    assert gap.asymmetry is True and gap.needs_counter_search is True
    # T·D는 원문만 인용 → 기술당 상한 3개 (도메인 우선)
    assert [(i["tech_id"], i["criterion_id"]) for i in gap.opposing_missing] == [
        ("mla", "D1"),
        ("mla", "D2"),
        ("mla", "D3"),
        ("pim_cxl", "D1"),
        ("pim_cxl", "D2"),
        ("pim_cxl", "D3"),
    ]
    assert "2:1" in gap.note


def test_evidence_gap_uses_synthesis_gaps_first_and_skips_covered_tech():
    r, reg = _retriever(), SourceRegistry()
    evals = _full_evals(r, reg)
    synth = SynthesisResult(
        agreements=[],
        conflicts=[
            {
                "tech_id": "mla",
                "perspective_a": "market",
                "criterion_a": "M2",
                "perspective_b": "domain",
                "criterion_b": "D2",
                "statement": "s",
                "cause": "c",
                "evidence_ids": ["a", "b"],
            }
        ],
        gaps=[Gap(tech_id="pim_cxl", criterion_id="M2", gap_type="opposing_missing", description="벤더 자료만")],
        unit_notes="u",
        evidence_asymmetry_note="a",
        generated_at=NOW,
    )
    gap = check_evidence_gap(TECHS, evals, synth)
    assert gap.opposing_missing[0] == {"tech_id": "pim_cxl", "criterion_id": "M2", "reason": "벤더 자료만"}
    covered = [web_evidence("mla-COUNTER-01", "https://c.example.com/1")]
    gap2 = check_evidence_gap(TECHS, evals, synth, covered)
    assert all(i["tech_id"] != "mla" for i in gap2.opposing_missing)
    assert gap2.needs_counter_search is True  # pim은 아직 탐색 전


# --- query_rewrite ------------------------------------------------------------------


def test_rewrite_queries_avoids_previous_and_changes_with_retry():
    q1 = rewrite_queries(MLA, ["T3"], ["DeepSeek-V2 MLA open source code release github"], retry_count=1)
    assert q1 and all("open source code release github" not in q or "DeepSeek-V2 MLA" not in q for q in q1)
    assert all(MLA.name in q or any(a in q for a in MLA.search_aliases) for q in q1)
    q2 = rewrite_queries(MLA, ["T3"], ["DeepSeek-V2 MLA open source code release github"], retry_count=2)
    assert q1 != q2 and set(q1) == set(q2)
    assert rewrite_queries(MLA, [], [], retry_count=0) == []
    prof = rewrite_queries(PIM, ["PROFILE:measurements", "T2"], [], retry_count=1)
    assert len(prof) <= 6 and any("throughput" in q or "footprint" in q for q in prof)


# --- counter_evidence -------------------------------------------------------------


def test_counter_evidence_search_uses_web_and_surveys_only():
    r = _retriever()
    deps = Deps(retriever=r, web_search=make_web_search(n=2), llm=None, judge_llm=None, now=lambda: NOW)
    gap = EvidenceGap(
        asymmetry=True,
        evidence_count={},
        vendor_source_ratio={},
        needs_counter_search=True,
        note="",
        opposing_missing=[
            {"tech_id": "mla", "criterion_id": "D2", "reason": "x"},
            {"tech_id": "pim_cxl", "criterion_id": "M2", "reason": "y"},
        ],
    )
    evs = search_counter_evidence(gap, deps)
    assert [e.evidence_id for e in evs] == [
        "mla-COUNTER-01",
        "mla-COUNTER-02",
        "mla-COUNTER-03",
        "mla-COUNTER-04",
        "pim_cxl-COUNTER-01",
        "pim_cxl-COUNTER-02",
        "pim_cxl-COUNTER-03",
        "pim_cxl-COUNTER-04",
    ]
    assert all(e.unit == "family" for e in evs)
    papers = [e for e in evs if e.source_type == "paper"]
    assert papers and all(e.doc_id in ("io_survey", "kv_survey") for e in papers)
    assert all(e.quote and r.get_chunk(e.chunk_id) and e.quote in r.get_chunk(e.chunk_id).text for e in papers)
    assert r.calls and all(q.startswith(("MLA 계열", "CXL·PIM/PNM 메모리 계열")) for q in r.calls)


# --- judge --------------------------------------------------------------------------


def _report_input(r: MemRetriever, reg: SourceRegistry, counter=()) -> ReportInput:
    evals = _full_evals(r, reg)
    profiles = [profile(t, [paper_evidence(f"{t.tech_id}-PROFILE-01", _chunk(r, t.primary_doc_id))]) for t in TECHS]
    synth = SynthesisResult(
        agreements=[],
        gaps=[],
        unit_notes="u",
        evidence_asymmetry_note="a",
        generated_at=NOW,
        conflicts=[
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
    )
    gap = EvidenceGap(
        asymmetry=False,
        evidence_count={"mla": 15, "pim_cxl": 15},
        vendor_source_ratio={"mla": 0.5, "pim_cxl": 0.5},
        opposing_missing=[],
        needs_counter_search=False,
        note="n",
    )
    return ReportInput(
        technologies=TECHS,
        domain="d",
        tech_profiles=profiles,
        trl_eval=evals["trl"],
        market_eval=evals["market"],
        stakeholder_eval=evals["stakeholder"],
        domain_eval=evals["domain"],
        counter_evidence=list(counter),
        synthesis=synth,
        evidence_gap=gap,
    )


TABLE = (
    "| 기준 | 기술 | 레벨 | 신뢰도 | 근거 단위 | 근거 |\n|---|---|---|---|---|---|\n"
    "| T1 | mla | L2 | medium | paper | [E: mla-T1-01] |"
)
GOOD_MD = f"""# 보고서
## SUMMARY
요약 [E: mla-T1-01]
## 1. 분석 배경
## 2. 기술 선정
## 3. 기술 개요
## 4. 관점별 평가
### 4.1 TRL
{TABLE}
### 4.2 시장성
{TABLE}
### 4.3 이해관계자
{TABLE}
### 4.4 도메인 적합성
{TABLE}
## 5. 시사점
[E: mla-M2-01] [E: mla-D2-01]
## 6. 한계점
## REFERENCE
1. deepseek_v2 p.1 (근거 ID: mla-T1-01, mla-D2-01)
2. https://mla.example.com/M2 (근거 ID: mla-M2-01)
"""


def test_static_checks_pass_on_good_report():
    r, reg = _retriever(), SourceRegistry()
    missing, hits, problems = static_checks(GOOD_MD, _report_input(r, reg))
    assert missing == [] and hits == [] and problems == []


def test_static_checks_catch_chapter_neutrality_and_citation_violations():
    r, reg = _retriever(), SourceRegistry()
    md = (
        GOOD_MD.replace("### 4.3 이해관계자\n", "").replace("요약", "MLA가 더 우수하다. 종합 점수 4.2")
        + "\n[E: ghost-X1-01]\n3. x (근거 ID: pim_cxl-S1-01)\n"
    )
    missing, hits, problems = static_checks(md, _report_input(r, reg))
    assert missing == ["4.3"]
    assert len(hits) == 1 and "더 우수하다" in hits[0]
    assert any("ghost-X1-01" in p for p in problems) and any("pim_cxl-S1-01" in p for p in problems)


def test_judge_recomputes_passed_and_overrides_scores():
    r, reg = _retriever(), SourceRegistry()
    inp = _report_input(r, reg)
    llm = FakeStructuredLLM()  # judge_result.json: passed=True 픽스처

    ok = judge_report(GOOD_MD, inp, llm, NOW)
    assert ok.passed is True and ok.judged_at == NOW
    prompt = llm.prompts_for(JudgeResult)[0]
    assert "mla T1: level=" in prompt and "- mla-T1-01:" in prompt and "# SUMMARY" in prompt

    bad = judge_report(GOOD_MD.replace("요약", "PIM/CXL을 추천한다"), inp, llm, NOW)
    assert bad.passed is False and bad.scores["neutrality"] == 1
    assert any("추천" in i for i in bad.revision_instructions)

    llm.register(
        JudgeResult,
        lambda text: {
            "scores": {"evidence": 3, "criteria_compliance": 3, "neutrality": 5, "specificity": 3, "format": 5},
            "reasons": {},
            "missing_required": ["pim_cxl/T1 why_not_higher"],
            "revision_instructions": ["보강"],
            "passed": True,  # LLM이 잘못 판정해도 코드가 재계산
            "judged_at": NOW,
        },
    )
    fixed = judge_report(GOOD_MD, inp, llm, NOW)
    assert fixed.passed is False and fixed.missing_required == ["pim_cxl/T1 why_not_higher"]


def test_paper_evidence_from_web_is_validated_by_url():
    """C가 arXiv 검색 결과를 source_type=paper(chunk_id 없음, url 있음)로 변환한 경우 — URL 경로로 검증 (H4)."""
    r = _retriever()
    reg = SourceRegistry(preload_urls=["https://arxiv.org/abs/2502.07864"])
    ok = web_evidence("mla-M3-01", "https://arxiv.org/abs/2502.07864", source_type="paper")
    assert evidence_problems(ok, r, reg) == []
    bad = web_evidence("mla-M3-02", "https://arxiv.org/abs/9999.00000", source_type="paper")
    assert any("V5" in p for p in evidence_problems(bad, r, reg))
    neither = ok.model_copy(update={"url": None, "locator": "somewhere"})
    assert any("neither" in p for p in evidence_problems(neither, r, reg))


def test_confidence_counts_each_web_paper_as_its_own_source():
    """논문은 URL(=논문 1편) 단위로 독립 출처. 같은 arxiv.org 라도 다른 논문이면 2개. 웹 자료는 도메인 단위."""
    from techeval.schemas import compute_confidence

    a = web_evidence("x-1", "https://arxiv.org/abs/1", source_type="paper")
    b = web_evidence("x-2", "https://arxiv.org/abs/2", source_type="paper")
    assert compute_confidence([a, b]) == "high"
    assert compute_confidence([a, a.model_copy(update={"evidence_id": "x-1b"})]) == "medium"  # 같은 논문
    n1 = web_evidence("x-3", "https://news.example.com/p1")
    n2 = web_evidence("x-4", "https://news.example.com/p2")
    assert compute_confidence([n1, n2]) == "medium"  # 같은 사이트 두 기사 = 출처 1개
