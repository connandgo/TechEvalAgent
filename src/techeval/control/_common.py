"""제어 노드 공용 헬퍼. 의견을 생성하지 않는다 — 상한 도달 시 `not_public` 확정 기록만 만든다."""

import logging

from techeval.schemas import (
    CRITERION_PERSPECTIVE,
    STAKEHOLDERS,
    CriterionResult,
    Evidence,
    TechRef,
)

logger = logging.getLogger(__name__)

# 기준별 검색 범위 서술 (not_public 기록용)
CRITERION_SCOPE: dict[str, str] = {
    "T1": "선정 논문 원문, 공개 구현·모델 카드",
    "T2": "선정 논문 원문 실험 절",
    "T3": "선정 논문, 공개 코드 저장소, 제3자 재현 보고",
    "T4": "선정 논문 한계·향후 과제 절",
    "M1": "시장 조사 보고서, 애널리스트 전망, 벤더 IR 자료",
    "M2": "벤더·클라우드 발표, 기술 블로그, 뉴스",
    "M3": "서빙 프레임워크 문서, 표준화 기구, 제3자 연구",
    "S1": "벤더·운영사 발표, 산업 분석",
    "S2": "벤더·운영사 발표, 산업 분석",
    "S3": "벤더·운영사 발표, 산업 분석, 도입 사례 보고",
    "S4": "산업 분석, 도입 사례 보고",
    "D1": "선정 논문 원문, 서베이 2편",
    "D2": "선정 논문 원문, 서베이 2편",
    "D3": "선정 논문 원문, 서베이 2편",
    "D4": "선정 논문 원문, 공개 구현 문서",
}


def _empty_details(criterion_id: str) -> dict:
    """스키마·검사 노드가 요구하는 details 필수 키를 '미확인' 값으로 채운다 (판정 아님)."""
    unknown_list = [{"text": "not_public", "evidence_id": None, "is_inference": False}]
    match criterion_id:
        case "T1":
            return {"trl_band": "not_public", "estimate": None, "why_not_higher": "not_public"}
        case "T2":
            return {"env_level": "not_public"}
        case "T3" | "D4" | "M3":
            return {"checklist": {}}
        case "T4":
            return {"remaining_tasks": [], "not_public_items": ["all"]}
        case "M1":
            return {"market_figures": []}
        case "M2":
            return {"adopters": []}
        case "S1":
            return {"roles": {s: "unrelated" for s in STAKEHOLDERS}}
        case "S2":
            return {"benefits": {s: unknown_list for s in STAKEHOLDERS}}
        case "S3":
            return {"burdens": {s: unknown_list for s in STAKEHOLDERS}}
        case "S4":
            return {"tradeoffs": [{"beneficiary": None, "burdened": None, "text": "not_public", "evidence_ids": []}]}
        case "D1" | "D2" | "D3":
            return {"directness": "L1", "extrapolation_logic": None}
    return {}


def make_not_public_result(
    tech: TechRef,
    criterion_id: str,
    *,
    queries: list[str],
    now: str,
    retry_count: int,
    reason: str = "재검색 상한 도달",
) -> CriterionResult:
    """재시도 상한에 도달한 기준을 `not_public`으로 확정하는 기록. 검색어·검색일·범위를 남긴다 (AGENTS.md 규칙 2)."""
    perspective = CRITERION_PERSPECTIVE[criterion_id]
    unit = "paper" if perspective in ("trl", "domain") else "family"
    evidence = Evidence(
        evidence_id=f"{tech.tech_id}-{criterion_id}-NP",
        source_type="not_public",
        unit=unit,
        quote="",
        locator="not_found",
        search_query=" | ".join(queries) if queries else f"{tech.name} {criterion_id}",
        searched_at=now,
        search_scope=CRITERION_SCOPE.get(criterion_id, ""),
    )
    logger.warning("%s/%s -> not_public 확정 (%s, retry=%d)", tech.tech_id, criterion_id, reason, retry_count)
    return CriterionResult(
        tech_id=tech.tech_id,
        perspective=perspective,
        criterion_id=criterion_id,
        level="not_public",
        content=(
            f"{reason}: {tech.name}의 {criterion_id} 항목은 확인된 범위({evidence.search_scope})에서 "
            "공개 정보를 찾지 못했다."
        ),
        evidence=[evidence],
        confidence="low",
        evidence_unit=unit,
        details=_empty_details(criterion_id),
        generated_at=now,
        retry_count=retry_count,
    )
