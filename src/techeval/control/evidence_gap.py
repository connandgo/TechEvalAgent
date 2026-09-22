"""근거 비대칭 검사 노드 (E). CONTRACTS §6.

기술별 evidence 수(중복 id 제거), 벤더·제안사 자료 비율, 한쪽 기술만 not_public인 기준, 반대 방향 근거가 없는 기준을
계산해 `EvidenceGap`을 만든다. 비율 > 2:1 이거나 반대 근거가 0인 (기술, 기준)이 있으면 `needs_counter_search=True`.
판정·의견은 만들지 않는다.
"""

import logging

from techeval.schemas import (
    PERSPECTIVE_CRITERIA,
    CriterionResult,
    Evidence,
    EvidenceGap,
    SynthesisResult,
    TechRef,
)

logger = logging.getLogger(__name__)

ASYMMETRY_RATIO = 2.0  # evidence 수 비율 상한 (2:1)
MAX_OPPOSING_ITEMS_PER_TECH = 3  # 반대 근거 탐색 대상 상한 (호출 수 제어)
# 반대 근거 탐색 우선순위: 종합 노드가 지목한 gap → 도메인 → 시장 → 이해관계자 → TRL
_PRIORITY = {
    c: i
    for i, c in enumerate(
        [
            *PERSPECTIVE_CRITERIA["domain"],
            *PERSPECTIVE_CRITERIA["market"],
            *PERSPECTIVE_CRITERIA["stakeholder"],
            *PERSPECTIVE_CRITERIA["trl"],
        ]
    )
}


def is_proponent_source(e: Evidence, tech: TechRef) -> bool:
    """제안사·벤더 자료인가: 선정 논문 원문(paper, primary_doc_id) 또는 official."""
    if e.source_type == "official":
        return True
    return e.source_type == "paper" and e.doc_id == tech.primary_doc_id


def has_independent_evidence(r: CriterionResult, tech: TechRef) -> bool:
    """제안사 자료가 아닌 근거(서베이·웹·특허)가 하나라도 있는가."""
    return any(e.source_type in ("paper", "web", "patent") and not is_proponent_source(e, tech) for e in r.evidence)


def check_evidence_gap(
    technologies: list[TechRef],
    evals: dict[str, list[CriterionResult]],
    synthesis: SynthesisResult | None,
    counter_evidence: list[Evidence] | None = None,
) -> EvidenceGap:
    """`evals` = {"trl": [...], "market": [...], "stakeholder": [...], "domain": [...]} (latest_by_criterion 적용본).

    `counter_evidence`가 이미 있는 기술은 "반대 근거 0건" 대상에서 제외한다 (탐색을 이미 한 것이므로).
    """
    covered = {e.evidence_id.split("-COUNTER-")[0] for e in (counter_evidence or []) if "-COUNTER-" in e.evidence_id}
    all_results = [r for rs in evals.values() for r in rs]
    by_tech: dict[str, list[CriterionResult]] = {t.tech_id: [] for t in technologies}
    for r in all_results:
        by_tech.setdefault(r.tech_id, []).append(r)

    evidence_count: dict[str, int] = {}
    vendor_ratio: dict[str, float] = {}
    for tech in technologies:
        uniq: dict[str, Evidence] = {}
        for r in by_tech[tech.tech_id]:
            for e in r.evidence:
                if e.source_type in ("inference", "not_public"):
                    continue
                uniq.setdefault(e.evidence_id, e)
        evidence_count[tech.tech_id] = len(uniq)
        vendor = sum(1 for e in uniq.values() if is_proponent_source(e, tech))
        vendor_ratio[tech.tech_id] = round(vendor / len(uniq), 3) if uniq else 0.0

    counts = sorted(evidence_count.values())
    ratio = (counts[-1] / counts[0]) if counts and counts[0] > 0 else (float("inf") if counts and counts[-1] else 1.0)
    asymmetry = ratio > ASYMMETRY_RATIO

    # 기준별로 한쪽 기술만 not_public 인 항목
    level_index = {(r.tech_id, r.criterion_id): r.level for r in all_results}
    one_sided_np: list[tuple[str, str]] = []
    tids = [t.tech_id for t in technologies]
    for cids in PERSPECTIVE_CRITERIA.values():
        for cid in cids:
            np = [tid for tid in tids if level_index.get((tid, cid)) == "not_public"]
            if len(np) == 1:
                one_sided_np.append((np[0], cid))

    # 반대 방향 근거가 0인 (기술, 기준): 종합 노드 gap(opposing_missing) 우선 + 제안사 자료만 있는 기준
    opposing: list[dict] = []
    seen: set[tuple[str, str]] = set()
    if synthesis is not None:
        for g in synthesis.gaps:
            if g.gap_type == "opposing_missing" and (g.tech_id, g.criterion_id) not in seen:
                seen.add((g.tech_id, g.criterion_id))
                opposing.append({"tech_id": g.tech_id, "criterion_id": g.criterion_id, "reason": g.description})
    for tech in technologies:
        if tech.tech_id in covered:
            continue
        candidates = [
            r
            for r in by_tech[tech.tech_id]
            if r.level != "not_public"
            and (tech.tech_id, r.criterion_id) not in seen
            and not has_independent_evidence(r, tech)
        ]
        candidates.sort(key=lambda r: _PRIORITY[r.criterion_id])
        for r in candidates:
            seen.add((tech.tech_id, r.criterion_id))
            opposing.append(
                {
                    "tech_id": tech.tech_id,
                    "criterion_id": r.criterion_id,
                    "reason": (
                        f"{tech.tech_id}/{r.criterion_id}의 근거가 모두 제안사·벤더 자료"
                        f"({', '.join(sorted({e.source_type for e in r.evidence}))})이며 "
                        "독립 출처의 반대 방향 근거가 0건"
                    ),
                }
            )

    # 기술당 상한 적용 (종합 노드 지목분 우선)
    per_tech: dict[str, int] = {}
    limited: list[dict] = []
    for item in opposing:
        n = per_tech.get(item["tech_id"], 0)
        if n < MAX_OPPOSING_ITEMS_PER_TECH:
            limited.append(item)
            per_tech[item["tech_id"]] = n + 1

    needs = asymmetry or bool(limited)
    parts = [f"고유 evidence 수 {evidence_count} (비율 {ratio:.2f}:1, 상한 {ASYMMETRY_RATIO:.0f}:1)"]
    if one_sided_np:
        parts.append("한쪽 기술만 not_public: " + ", ".join(f"{t}/{c}" for t, c in one_sided_np))
    if limited:
        parts.append("반대 근거 0건: " + ", ".join(f"{i['tech_id']}/{i['criterion_id']}" for i in limited))
    if len(opposing) > len(limited):
        parts.append(f"(탐색 대상은 기술당 {MAX_OPPOSING_ITEMS_PER_TECH}개로 제한, 전체 {len(opposing)}건)")
    parts.append("벤더·제안사 자료 비율 " + ", ".join(f"{t}={v:.0%}" for t, v in vendor_ratio.items()))
    note = "; ".join(parts)
    logger.info("evidence_gap_check: asymmetry=%s needs_counter_search=%s — %s", asymmetry, needs, note)

    return EvidenceGap(
        asymmetry=asymmetry,
        evidence_count=evidence_count,
        vendor_source_ratio=vendor_ratio,
        opposing_missing=limited,
        needs_counter_search=needs,
        note=note,
    )
