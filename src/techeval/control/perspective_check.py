"""관점별 근거 검사 노드 (E). CONTRACTS §6·§10 V4~V8.

입력: 4개 `*_eval`(중복 제거 후) / 출력: 관점·기술별 부족 기준, confidence 재계산본.
재계산 허용 범위는 `confidence`뿐이다 (V7). 레벨·내용은 건드리지 않는다.
"""

import logging
from typing import Any

from pydantic import BaseModel

from techeval.control.sources import SourceRegistry, evidence_problems
from techeval.schemas import (
    PERSPECTIVE_CRITERIA,
    STAKEHOLDERS,
    CriterionResult,
    TechRef,
    compute_confidence,
)

logger = logging.getLogger(__name__)

# CRITERIA §2 기준별 허용 level 문자열 (not_public은 전 기준 공통)
CRITERION_LEVELS: dict[str, frozenset[str]] = {
    "T1": frozenset({"TRL 1-3", "TRL 4-6", "TRL 7-9"}),
    "T2": frozenset({"L1", "L2", "L3", "L4"}),
    "T3": frozenset({"L0", "L1", "L2", "L3"}),
    "T4": frozenset({"narrative"}),
    "M1": frozenset({"L0", "L1", "L2", "L3"}),
    "M2": frozenset({"L0", "L1", "L2", "L3", "L4"}),
    "M3": frozenset({"L0", "L1", "L2", "L3"}),
    "S1": frozenset({"assigned"}),
    "S2": frozenset({"narrative"}),
    "S3": frozenset({"narrative"}),
    "S4": frozenset({"narrative"}),
    "D1": frozenset({"L1", "L2", "L3"}),
    "D2": frozenset({"L1", "L2", "L3"}),
    "D3": frozenset({"L1", "L2", "L3"}),
    "D4": frozenset({"checklist"}),
}

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


class PerspectiveCheckResult(BaseModel):
    missing: dict[str, dict[str, list[str]]]  # perspective -> tech_id -> [criterion_id]
    problems: list[str]
    corrected: list[CriterionResult]  # confidence를 재계산해 덮어쓴 결과 (State에 append)

    @property
    def sufficient(self) -> bool:
        return not any(cids for techs in self.missing.values() for cids in techs.values())

    def flat_missing(self) -> dict[str, list[str]]:
        """CONTRACTS State.missing_criteria 형식: perspective -> 기준 목록(기술 무관 합집합)."""
        return {p: sorted({c for cids in techs.values() for c in cids}) for p, techs in self.missing.items()}


def _v4_problems(r: CriterionResult) -> list[str]:
    """level 체계·details 필수 키(CRITERIA §2), S2/S3 4주체 각 ≥1, S4 ≥1쌍, D1~D3 measurements, L2 외삽 논리."""
    if r.level == "not_public":
        return []
    d = r.details or {}
    tag = f"{r.tech_id}/{r.criterion_id}"
    probs: list[str] = []
    if r.level not in CRITERION_LEVELS[r.criterion_id]:
        probs.append(f"{tag}: level {r.level!r}는 허용 값 {sorted(CRITERION_LEVELS[r.criterion_id])} 밖 (CRITERIA §2)")
    missing_keys = [k for k in REQUIRED_DETAILS[r.criterion_id] if k not in d]
    if missing_keys:
        probs.append(f"{tag}: details 필수 키 누락 {missing_keys} (CONTRACTS §2)")
    if r.criterion_id == "T1" and not (d.get("why_not_higher") or "").strip():
        probs.append(f"{tag}: T1은 '한 단계 위로 올리지 못한 이유'(why_not_higher) 필수")
    if r.criterion_id in ("S2", "S3"):
        key = "benefits" if r.criterion_id == "S2" else "burdens"
        entries = d.get(key) or {}
        for s in STAKEHOLDERS:
            if not entries.get(s):
                probs.append(f"{tag}: details.{key}[{s}] 비어 있음 (V4)")
    elif r.criterion_id == "S4":
        if not d.get("tradeoffs"):
            probs.append(f"{tag}: details.tradeoffs ≥1 필요 (V4)")
    elif r.criterion_id in ("D1", "D2", "D3"):
        if r.level in ("L2", "L3") and not r.measurements:
            probs.append(f"{tag}: level {r.level} 이면 measurements 필요 (V3)")
        if d.get("directness") == "L2" and not d.get("extrapolation_logic"):
            probs.append(f"{tag}: directness=L2 이면 extrapolation_logic 필수")
    return probs


def check_perspectives(
    technologies: list[TechRef],
    evals: dict[str, list[CriterionResult]],
    retriever: Any,
    registry: SourceRegistry | None = None,
) -> PerspectiveCheckResult:
    """`evals`는 {"trl": [...], "market": [...], "stakeholder": [...], "domain": [...]} (latest_by_criterion 적용본)."""
    missing: dict[str, dict[str, list[str]]] = {p: {t.tech_id: [] for t in technologies} for p in PERSPECTIVE_CRITERIA}
    problems: list[str] = []
    corrected: list[CriterionResult] = []

    for perspective, cids in PERSPECTIVE_CRITERIA.items():
        index = {(r.tech_id, r.criterion_id): r for r in evals.get(perspective, [])}
        for tech in technologies:
            for cid in cids:
                r = index.get((tech.tech_id, cid))
                tag = f"{tech.tech_id}/{cid}"
                if r is None:
                    missing[perspective][tech.tech_id].append(cid)
                    problems.append(f"{tag}: result missing (V8)")
                    continue

                probs = _v4_problems(r)
                if r.level != "not_public":
                    probs += [p for e in r.evidence for p in evidence_problems(e, retriever, registry)]
                if probs:
                    missing[perspective][tech.tech_id].append(cid)
                    problems += probs
                    continue

                expected = compute_confidence(r.evidence)
                if r.confidence != expected:  # V7: 경고 후 재계산 값으로 덮어씀
                    logger.warning("%s: confidence %s -> %s 로 재계산 (V7)", tag, r.confidence, expected)
                    corrected.append(r.model_copy(update={"confidence": expected}))

    for perspective, techs in missing.items():
        for tid, cids in techs.items():
            if cids:
                logger.info("perspective_check %s/%s: 부족 %s", perspective, tid, cids)
    for p in problems:
        logger.debug("  - %s", p)
    if not problems:
        logger.info("perspective_check: 30개 기준 모두 충분")

    return PerspectiveCheckResult(missing=missing, problems=problems, corrected=corrected)
