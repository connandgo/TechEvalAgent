"""기술 근거 검사 노드 (E). CONTRACTS §6 첫 행.

입력: `tech_profiles`, `trl_eval` (중복 제거 후) / 출력: 기술별 부족 항목.
부족 항목 표기: 기준 ID(`"T3"`) 또는 프로필 필드(`"PROFILE:measurements"`).
"""

import logging
from typing import Any

from pydantic import BaseModel

from techeval.control.sources import SourceRegistry, evidence_problems
from techeval.schemas import PERSPECTIVE_CRITERIA, CriterionResult, TechProfile, TechRef

logger = logging.getLogger(__name__)

PROFILE_REQUIRED_FIELDS: tuple[str, ...] = ("principle", "limitations", "measurements", "validation_env")


class TechCheckResult(BaseModel):
    missing: dict[str, list[str]]  # tech_id -> ["T3", "PROFILE:measurements", ...]
    problems: dict[str, list[str]]  # tech_id -> 사람이 읽을 위반 사유

    @property
    def sufficient(self) -> bool:
        return not any(self.missing.values())

    def missing_criteria(self, tech_id: str) -> list[str]:
        return [m for m in self.missing.get(tech_id, []) if not m.startswith("PROFILE:")]


def _profile_missing(profile: TechProfile | None) -> list[str]:
    if profile is None:
        return [f"PROFILE:{f}" for f in PROFILE_REQUIRED_FIELDS]
    missing = []
    for f in PROFILE_REQUIRED_FIELDS:
        v = getattr(profile, f)
        if not v or (isinstance(v, str) and not v.strip()):
            missing.append(f"PROFILE:{f}")
    return missing


def check_tech_evidence(
    technologies: list[TechRef],
    profiles: list[TechProfile],
    trl_eval: list[CriterionResult],
    retriever: Any,
    registry: SourceRegistry | None = None,
) -> TechCheckResult:
    """TechProfile 필수 필드 + T1~T4 존재 + 각 evidence 실존(V5)·인용 일치(V6)를 검사한다."""
    by_tech_profile = {p.tech_id: p for p in profiles}
    by_tech_results: dict[str, dict[str, CriterionResult]] = {}
    for r in trl_eval:
        by_tech_results.setdefault(r.tech_id, {})[r.criterion_id] = r

    missing: dict[str, list[str]] = {}
    problems: dict[str, list[str]] = {}
    for tech in technologies:
        tid = tech.tech_id
        tech_missing: list[str] = []
        tech_problems: list[str] = []

        profile = by_tech_profile.get(tid)
        tech_missing += _profile_missing(profile)
        if profile is not None:
            for e in profile.evidence:
                probs = evidence_problems(e, retriever, registry)
                if probs:
                    tech_problems += probs
                    if "PROFILE:evidence" not in tech_missing:
                        tech_missing.append("PROFILE:evidence")

        results = by_tech_results.get(tid, {})
        for cid in PERSPECTIVE_CRITERIA["trl"]:
            r = results.get(cid)
            if r is None:
                tech_missing.append(cid)
                tech_problems.append(f"{cid}: result missing")
                continue
            if r.level == "not_public":
                continue  # 이미 확정된 비공개 항목은 재검색 대상이 아니다
            probs = [p for e in r.evidence for p in evidence_problems(e, retriever, registry)]
            if probs:
                tech_missing.append(cid)
                tech_problems += probs

        missing[tid] = tech_missing
        problems[tid] = tech_problems
        if tech_missing:
            logger.info("tech_evidence_check %s: 부족 %s", tid, tech_missing)
            for p in tech_problems:
                logger.debug("  - %s", p)
        else:
            logger.info("tech_evidence_check %s: 충분", tid)

    return TechCheckResult(missing=missing, problems=problems)
