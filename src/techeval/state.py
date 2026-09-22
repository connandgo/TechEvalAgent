"""LangGraph State 스키마와 State 헬퍼. `docs/CONTRACTS.md` §7과 1:1로 대응한다."""

import operator
from typing import Annotated, TypedDict

from techeval.schemas import (
    DOMAIN,
    PERSPECTIVE_CRITERIA,
    STAKEHOLDERS,
    TECHNOLOGIES,
    CriterionResult,
    Evidence,
    EvidenceGap,
    JudgeResult,
    SynthesisResult,
    TechProfile,
    TechRef,
)


class GraphState(TypedDict, total=False):
    # 초기 입력 (overwrite)
    domain: str
    technologies: list[TechRef]  # Human 고정 (mla, pim_cxl)
    stakeholders: tuple[str, ...]  # STAKEHOLDERS 상수

    # 기술 조사 (append, 기술별 Send)
    tech_profiles: Annotated[list[TechProfile], operator.add]
    trl_eval: Annotated[list[CriterionResult], operator.add]

    # 관점 평가 (append, 기술별 Send) — 병렬 실행, 서로 다른 키
    market_eval: Annotated[list[CriterionResult], operator.add]
    stakeholder_eval: Annotated[list[CriterionResult], operator.add]
    domain_eval: Annotated[list[CriterionResult], operator.add]

    # 제어 (overwrite)
    missing_criteria: dict[str, list[str]]  # {"trl": ["T3"], "market": [], ...} 관점별
    retry_counts: dict[str, int]  # {"trl:mla": 1, "market:pim_cxl": 0, "counter": 0, "report": 0}
    counter_evidence: list[Evidence]
    synthesis: SynthesisResult
    evidence_gap: EvidenceGap
    report_md: str
    judge_result: JudgeResult
    report_pdf_path: str


# operator.add 로 누적되는 키 목록 (중복 제거가 필요한 키)
APPEND_KEYS: tuple[str, ...] = ("tech_profiles", "trl_eval", "market_eval", "stakeholder_eval", "domain_eval")


def latest_by_criterion(results: list[CriterionResult]) -> list[CriterionResult]:
    """`(tech_id, criterion_id)`별로 `generated_at`이 가장 최신인 결과 1개만 남긴다.

    같은 시각이면 뒤에 온 것(리스트 후순위 = 나중에 append된 것)을 취한다.
    반환 순서는 입력에서 각 키가 처음 등장한 순서를 따른다.
    """
    chosen: dict[tuple[str, str], CriterionResult] = {}
    for r in results:
        key = (r.tech_id, r.criterion_id)
        prev = chosen.get(key)
        if prev is None or r.generated_at >= prev.generated_at:
            chosen[key] = r
    return list(chosen.values())


def latest_by_tech(profiles: list[TechProfile]) -> list[TechProfile]:
    """`tech_id`별로 `generated_at`이 가장 최신인 TechProfile 1개만 남긴다."""
    chosen: dict[str, TechProfile] = {}
    for p in profiles:
        prev = chosen.get(p.tech_id)
        if prev is None or p.generated_at >= prev.generated_at:
            chosen[p.tech_id] = p
    return list(chosen.values())


def initial_retry_counts(technologies: list[TechRef] | None = None) -> dict[str, int]:
    """`"<perspective>:<tech_id>"`, `"counter"`, `"report"` 키를 전부 0으로 초기화한다."""
    techs = technologies or TECHNOLOGIES
    counts = {f"{p}:{t.tech_id}": 0 for p in PERSPECTIVE_CRITERIA for t in techs}
    counts["counter"] = 0
    counts["report"] = 0
    return counts


def build_initial_state() -> GraphState:
    """그래프 시작 State. `domain`, `technologies`, `stakeholders`와 빈 컬렉션, `retry_counts` 0."""
    return GraphState(
        domain=DOMAIN,
        technologies=list(TECHNOLOGIES),
        stakeholders=STAKEHOLDERS,
        tech_profiles=[],
        trl_eval=[],
        market_eval=[],
        stakeholder_eval=[],
        domain_eval=[],
        missing_criteria={p: [] for p in PERSPECTIVE_CRITERIA},
        retry_counts=initial_retry_counts(),
        counter_evidence=[],
    )
