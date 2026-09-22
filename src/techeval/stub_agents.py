"""픽스처/합성 기반 스텁 에이전트 (E). `run.py --stub`과 `tests/test_graph.py`에서 B·C·D 에이전트 대신 쓴다.

각 함수는 CONTRACTS §5 시그니처를 그대로 따른다. 동작:
1. `tests/fixtures/`에 해당 픽스처가 있으면 그것을 돌려준다 (generated_at·retry_count만 갱신).
2. 없으면 `deps.retriever`/`deps.web_search`가 실제로 돌려준 청크·URL만으로 최소 결과를 합성한다.
   검색 결과가 없으면 `not_public`으로 기록한다 — 스텁도 근거를 지어내지 않는다 (AGENTS.md 규칙 1·2).
"""

import json
import logging
from pathlib import Path

from techeval.agents._deps import AgentInput, Deps, ReportInput, SynthesisInput, TechResearchOutput
from techeval.control._common import _empty_details, make_not_public_result
from techeval.control.counter_evidence import chunk_to_evidence, web_to_evidence
from techeval.schemas import (
    PERSPECTIVE_CRITERIA,
    STAKEHOLDERS,
    SURVEY_DOC_IDS,
    Conflict,
    CriterionResult,
    Evidence,
    Measurement,
    SynthesisResult,
    TechProfile,
    TechRef,
    compute_confidence,
)
from techeval.stub_llm import DEFAULT_FIXTURES_DIR

logger = logging.getLogger(__name__)

FIXTURES_DIR = DEFAULT_FIXTURES_DIR
STUB_MARK = "[stub]"
try:  # 요약표 머리행은 D(report.tables)의 형식을 따른다
    from techeval.report.tables import TABLE_HEADER
except ImportError:
    TABLE_HEADER = "| 기준 | 기술 | 레벨 | 신뢰도 | 근거 단위 | 근거 |"

_LEVELS = {
    "trl": {"T1": "TRL 4-6", "T2": "L2", "T3": "L2", "T4": "narrative"},
    "market": {"M1": "L2", "M2": "L2", "M3": "L2"},
    "stakeholder": {"S1": "assigned", "S2": "narrative", "S3": "narrative", "S4": "narrative"},
    "domain": {"D1": "L2", "D2": "L2", "D3": "L2", "D4": "checklist"},
}


def _fixture(name: str) -> list | dict | None:
    path = Path(FIXTURES_DIR) / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_results(
    name: str, tech_id: str, criteria: list[str], now: str, retry: int
) -> list[CriterionResult] | None:
    data = _fixture(name)
    if data is None:
        return None
    out = [
        CriterionResult.model_validate(r).model_copy(update={"generated_at": now, "retry_count": retry})
        for r in data
        if r["tech_id"] == tech_id and r["criterion_id"] in criteria
    ]
    found = {r.criterion_id for r in out}
    if found != set(criteria):
        logger.warning("%s: %s 픽스처에 %s 누락 — 합성으로 대체", name, tech_id, sorted(set(criteria) - found))
        return None
    return out


# --- 합성 -------------------------------------------------------------------


def _paper_evidence(tech: TechRef, cid: str, deps: Deps, n: int = 2) -> list[Evidence]:
    """선정 논문 청크 n개 + 서베이 청크 1개(기술 계열 맥락, unit=family)."""
    query = f"{tech.name} {cid}"
    chunks = deps.retriever.search(query, top_k=n, doc_ids=[tech.primary_doc_id])[:n]
    surveys = deps.retriever.search(query, top_k=1, doc_ids=list(SURVEY_DOC_IDS))[:1]
    out = [
        chunk_to_evidence(c, evidence_id=f"{tech.tech_id}-{cid}-{i + 1:02d}", accessed=deps.now())
        for i, c in enumerate([*chunks, *surveys])
    ]
    for e, c in zip(out, [*chunks, *surveys], strict=True):
        e.unit = "family" if c.doc_id in SURVEY_DOC_IDS else "paper"
    return out


def _web_evidence(tech: TechRef, cid: str, deps: Deps, n: int = 2) -> list[Evidence]:
    query = f"{tech.family} {cid}"
    results = deps.web_search(query, max_results=n)
    return [
        web_to_evidence(r, evidence_id=f"{tech.tech_id}-{cid}-{i + 1:02d}", accessed=deps.now())
        for i, r in enumerate(results[:n])
    ]


def _details(cid: str, evidence: list[Evidence]) -> dict:
    """V4·REQUIRED_DETAILS를 만족하는 최소 details. 값은 stub 표시."""
    eid = evidence[0].evidence_id
    entry = [{"text": f"{STUB_MARK} placeholder", "evidence_id": eid, "is_inference": True}]
    match cid:
        case "T1":
            return {"trl_band": "4-6", "estimate": 5, "why_not_higher": f"{STUB_MARK} 실서비스 운영 근거 미확인"}
        case "T2":
            return {"env_level": "L2"}
        case "T3":
            return {
                "checklist": {
                    "code": "unknown",
                    "model_or_design": "unknown",
                    "eval_scripts_data": "unknown",
                    "third_party_reproduction": "unknown",
                }
            }
        case "T4":
            return {"remaining_tasks": [f"{STUB_MARK} task"], "not_public_items": []}
        case "M1":
            return {
                "market_figures": [
                    {"figure": f"{STUB_MARK} n/a", "publisher": "stub", "published_date": "2026", "evidence_id": eid}
                ]
            }
        case "M2":
            return {"adopters": [{"name": f"{STUB_MARK} adopter", "stage": "PoC", "evidence_id": eid}]}
        case "M3":
            return {
                "checklist": {
                    "framework": "N",
                    "vendor_product": "N",
                    "standardization": "N",
                    "third_party_research_tools": "N",
                }
            }
        case "S1":
            return {"roles": {s: "affected" for s in STAKEHOLDERS}}
        case "S2":
            return {"benefits": {s: entry for s in STAKEHOLDERS}}
        case "S3":
            return {"burdens": {s: entry for s in STAKEHOLDERS}}
        case "S4":
            return {
                "tradeoffs": [
                    {
                        "beneficiary": STAKEHOLDERS[0],
                        "burdened": STAKEHOLDERS[1],
                        "text": f"{STUB_MARK} tradeoff",
                        "evidence_ids": [eid],
                    }
                ]
            }
        case "D1" | "D2" | "D3":
            return {"directness": "L2", "extrapolation_logic": f"{STUB_MARK} 외삽 논리"}
        case "D4":
            return {
                "checklist": {
                    k: "unknown"
                    for k in (
                        "model_retrain",
                        "model_convert",
                        "serving_engine_change",
                        "hw_replace",
                        "memory_add",
                        "other",
                    )
                }
            }
    return _empty_details(cid)


def _synth_result(
    tech: TechRef, perspective: str, cid: str, evidence: list[Evidence], deps: Deps, retry: int
) -> CriterionResult:
    if not evidence:
        return make_not_public_result(
            tech, cid, queries=[f"{tech.name} {cid}"], now=deps.now(), retry_count=retry, reason="스텁 검색 결과 없음"
        )
    unit = "paper" if perspective in ("trl", "domain") else "family"
    measurements = []
    if cid in ("D1", "D2", "D3"):
        measurements = [
            Measurement(
                metric=f"{STUB_MARK} {cid}", value="n/a", condition_note="stub", evidence_id=evidence[0].evidence_id
            )
        ]
    return CriterionResult(
        tech_id=tech.tech_id,
        perspective=perspective,
        criterion_id=cid,
        level=_LEVELS[perspective][cid],
        level_estimate="TRL 5 (stub)" if cid == "T1" else None,
        content=f"{STUB_MARK} {tech.name} {cid} 합성 결과. 근거 {len(evidence)}건은 실제 검색 결과에서 가져옴.",
        evidence=evidence,
        confidence=compute_confidence(evidence),
        evidence_unit=unit,
        measurements=measurements,
        details=_details(cid, evidence),
        generated_at=deps.now(),
        retry_count=retry,
    )


def _criteria(inp: AgentInput, perspective: str) -> list[str]:
    return list(inp.missing_criteria) if inp.missing_criteria else list(PERSPECTIVE_CRITERIA[perspective])


def _perspective_eval(
    inp: AgentInput, deps: Deps, perspective: str, fixture: str, paper: bool
) -> list[CriterionResult]:
    criteria = _criteria(inp, perspective)
    fixed = _fixture_results(fixture, inp.tech.tech_id, criteria, deps.now(), inp.retry_count)
    if fixed is not None:
        return fixed
    out = []
    for cid in criteria:
        ev = _paper_evidence(inp.tech, cid, deps) if paper else _web_evidence(inp.tech, cid, deps)
        out.append(_synth_result(inp.tech, perspective, cid, ev, deps, inp.retry_count))
    return out


# --- CONTRACTS §5 시그니처 -------------------------------------------------------


def run_tech_research(inp: AgentInput, deps: Deps) -> TechResearchOutput:
    now = deps.now()
    criteria = _criteria(inp, "trl")  # missing_criteria 가 있으면 그 기준만 (프로필은 항상 갱신)
    profiles = _fixture("tech_profiles.json")
    trl = _fixture_results("trl_eval.json", inp.tech.tech_id, criteria, now, inp.retry_count)
    raw = next((p for p in profiles or [] if p["tech_id"] == inp.tech.tech_id), None)
    if raw is not None and trl is not None:
        profile = TechProfile.model_validate(raw).model_copy(
            update={
                "generated_at": now,
                "retry_count": inp.retry_count,
                "search_queries_used": [*raw.get("search_queries_used", []), *inp.rewritten_queries],
            }
        )
        return TechResearchOutput(tech_profile=profile, trl_eval=trl)

    queries = inp.rewritten_queries or [f"{inp.tech.name} architecture"]
    chunks = deps.retriever.search(queries[0], top_k=3, doc_ids=[inp.tech.primary_doc_id])
    evidence = [
        chunk_to_evidence(c, evidence_id=f"{inp.tech.tech_id}-PROFILE-{i + 1:02d}", accessed=now)
        for i, c in enumerate(chunks)
    ]
    if not evidence:
        evidence = [
            Evidence(
                evidence_id=f"{inp.tech.tech_id}-PROFILE-NP",
                source_type="not_public",
                unit="paper",
                quote="",
                locator="not_found",
                search_query=" | ".join(queries),
                searched_at=now,
                search_scope="stub retriever",
            )
        ]
    profile = TechProfile(
        tech_id=inp.tech.tech_id,
        principle=f"{STUB_MARK} {inp.tech.name} principle",
        scope=f"{STUB_MARK} scope",
        limitations=[f"{STUB_MARK} limitation"],
        measurements=[Measurement(metric=f"{STUB_MARK} metric", value="n/a", evidence_id=evidence[0].evidence_id)],
        validation_env=f"{STUB_MARK} validation env",
        public_artifacts={
            k: "unknown" for k in ("code", "model_or_design", "eval_scripts_data", "third_party_reproduction")
        },
        evidence=evidence,
        search_queries_used=queries,
        generated_at=now,
        retry_count=inp.retry_count,
    )
    trl_eval = [
        _synth_result(inp.tech, "trl", cid, _paper_evidence(inp.tech, cid, deps), deps, inp.retry_count)
        for cid in criteria
    ]
    return TechResearchOutput(tech_profile=profile, trl_eval=trl_eval)


def run_market_eval(inp: AgentInput, deps: Deps) -> list[CriterionResult]:
    return _perspective_eval(inp, deps, "market", "market_eval.json", paper=False)


def run_stakeholder_eval(inp: AgentInput, deps: Deps) -> list[CriterionResult]:
    return _perspective_eval(inp, deps, "stakeholder", "stakeholder_eval.json", paper=False)


def run_domain_eval(inp: AgentInput, deps: Deps) -> list[CriterionResult]:
    return _perspective_eval(inp, deps, "domain", "domain_eval.json", paper=True)


def run_synthesis(inp: SynthesisInput, deps: Deps) -> SynthesisResult:
    data = _fixture("synthesis.json")
    if data is not None:
        return SynthesisResult.model_validate(data).model_copy(update={"generated_at": deps.now()})
    conflicts = []
    for t in inp.technologies:
        s4 = next((r for r in inp.stakeholder_eval if r.tech_id == t.tech_id and r.criterion_id == "S4"), None)
        d2 = next((r for r in inp.domain_eval if r.tech_id == t.tech_id and r.criterion_id == "D2"), None)
        if s4 and d2:
            conflicts.append(
                Conflict(
                    tech_id=t.tech_id,
                    perspective_a="stakeholder",
                    criterion_a="S4",
                    perspective_b="domain",
                    criterion_b="D2",
                    statement=f"{STUB_MARK} {t.name}: 이해관계자 상충 지점과 도메인 동시 처리 근거의 직접성이 다르다.",
                    cause="평가 단위 차이(family vs paper)",
                    evidence_ids=[s4.evidence[0].evidence_id, d2.evidence[0].evidence_id],
                )
            )
    return SynthesisResult(
        agreements=[],
        conflicts=conflicts
        or [
            Conflict(
                tech_id=inp.technologies[0].tech_id,
                perspective_a="trl",
                criterion_a="T1",
                perspective_b="market",
                criterion_b="M2",
                statement=f"{STUB_MARK} placeholder conflict",
                cause="근거 유형 차이",
                evidence_ids=[inp.trl_eval[0].evidence[0].evidence_id, inp.market_eval[0].evidence[0].evidence_id]
                if inp.trl_eval and inp.market_eval
                else ["stub-1", "stub-2"],
            )
        ],
        gaps=[],
        unit_notes=f"{STUB_MARK} TRL·도메인은 paper 단위, 시장·이해관계자는 family 단위로 평가함.",
        evidence_asymmetry_note=f"{STUB_MARK} 근거 비대칭은 evidence_gap 노드 결과를 참조.",
        generated_at=deps.now(),
    )


def _all_results(inp: ReportInput) -> list[CriterionResult]:
    return [*inp.trl_eval, *inp.market_eval, *inp.stakeholder_eval, *inp.domain_eval]


def run_report(inp: ReportInput, deps: Deps) -> str:
    """D의 report_md.md 픽스처가 입력의 evidence_id만 인용하면 그것을, 아니면 필수 챕터를 갖춘 합성 보고서."""
    from techeval.control.judge import cited_ids, evidence_index

    idx = evidence_index(inp)
    path = Path(FIXTURES_DIR) / "report_md.md"
    if path.exists():
        md = path.read_text(encoding="utf-8")
        if cited_ids(md) <= set(idx):
            return md
        logger.warning("report_md.md 픽스처의 각주가 입력 근거와 맞지 않아 합성 보고서로 대체")

    results = _all_results(inp)
    by_p: dict[str, list[CriterionResult]] = {p: [] for p in PERSPECTIVE_CRITERIA}
    for r in results:
        by_p[r.perspective].append(r)

    def table(rs: list[CriterionResult]) -> str:
        rows = [TABLE_HEADER, "|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: (r.criterion_id, r.tech_id)):
            cites = " ".join(f"[E: {e.evidence_id}]" for e in r.evidence)
            rows.append(
                f"| {r.criterion_id} | {r.tech_id} | {r.level} | {r.confidence} | {r.evidence_unit} | {cites} |"
            )
        return "\n".join(rows)

    revision = ""
    if inp.judge_result is not None:
        revision = "\n\n> 재생성: " + " / ".join(inp.judge_result.revision_instructions)

    sections = [
        "# KV cache 최적화 기술 다관점 평가 보고서 (stub)",
        f"## SUMMARY\n\n{STUB_MARK} 두 기술은 관점별로 근거 단위와 근거 유형이 달라 판정 단계가 다르다.{revision}",
        "## 1. 분석 배경\n\n" + f"{STUB_MARK} KV cache 병목과 SW/HW 접근의 분기. 도메인: {inp.domain}.",
        "## 2. 기술 선정\n\n"
        + "\n".join(f"- {t.name} ({t.tech_id}, {t.approach}): {t.paper_title}" for t in inp.technologies),
        "## 3. 기술 개요\n\n"
        + "\n".join(
            f"- {p.tech_id}: {p.principle} " + " ".join(f"[E: {e.evidence_id}]" for e in p.evidence)
            for p in inp.tech_profiles
        ),
        "## 4. 관점별 평가\n\n### 4.1 기술 성숙도(TRL)\n\n"
        + table(by_p["trl"])
        + "\n\n### 4.2 시장성\n\n"
        + table(by_p["market"])
        + "\n\n### 4.3 이해관계자\n\n"
        + table(by_p["stakeholder"])
        + "\n\n### 4.4 도메인 적합성\n\n"
        + table(by_p["domain"]),
        "## 5. 시사점\n\n"
        + "\n".join(
            f"- {c.tech_id} {c.criterion_a}↔{c.criterion_b}: {c.statement} ({c.cause}) "
            + " ".join(f"[E: {i}]" for i in c.evidence_ids if i in idx)
            for c in inp.synthesis.conflicts
        ),
        "## 6. 한계점\n\n"
        + f"{STUB_MARK} 근거 비대칭: {inp.evidence_gap.note}. 벤더 자료 비율 {inp.evidence_gap.vendor_source_ratio}. "
        + " ".join(f"[E: {e.evidence_id}]" for e in inp.counter_evidence),
    ]
    md = "\n\n".join(sections)
    try:  # REFERENCE 절은 D(report.citation)의 형식을 따른다
        from techeval.report.citation import build_reference_section

        return md + "\n\n" + build_reference_section(md, idx)
    except ImportError:
        cited = sorted(cited_ids(md))
        refs = "\n".join(
            f"{i + 1}. {idx[e].title or idx[e].locator} (근거 ID: {e})" for i, e in enumerate(cited) if e in idx
        )
        return md + "\n\n## REFERENCE\n\n" + refs


def render_pdf(report_md: str, out_path: str) -> str:
    """PDF 렌더링 스텁: D의 `report.pdf.render_pdf`가 없을 때 마크다운을 그대로 저장한다."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    target = p.with_suffix(".stub.md") if p.suffix == ".pdf" else p
    target.write_text(report_md, encoding="utf-8")
    logger.warning("render_pdf 스텁: PDF 대신 %s 에 마크다운 저장", target)
    return str(target)
