"""제어 노드 공용 헬퍼. 의견을 생성하지 않는다 — 상한 도달 시 `not_public` 확정 기록만 만든다."""

import logging

from techeval.schemas import (
    CRITERION_PERSPECTIVE,
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
    """not_public 확정 기록의 details. 필수 키는 두되 값은 전부 '판정 없음'을 뜻하는 빈 값/None으로 채운다.

    S1 역할, D 직접성 등급, S4 상충 쌍처럼 **판정에 해당하는 값은 절대 넣지 않는다** (제어 노드는 의견을 만들지 않음).
    """
    match criterion_id:
        case "T1":
            return {"trl_band": None, "estimate": None, "why_not_higher": None}
        case "T2":
            return {"env_level": None}
        case "T3" | "D4" | "M3":
            return {"checklist": {}}
        case "T4":
            return {"remaining_tasks": [], "not_public_items": []}
        case "M1":
            return {"market_figures": []}
        case "M2":
            return {"adopters": []}
        case "S1":
            return {"roles": {}}
        case "S2":
            return {"benefits": {}}
        case "S3":
            return {"burdens": {}}
        case "S4":
            return {"tradeoffs": []}
        case "D1" | "D2" | "D3":
            return {"directness": None, "extrapolation_logic": None}
    return {}


def make_not_public_result(
    tech: TechRef,
    criterion_id: str,
    *,
    queries: list[str],
    now: str,
    retry_count: int,
    reason: str = "재검색 상한 도달",
    problems: list[str] | None = None,
) -> CriterionResult:
    """재시도 상한에 도달한 기준을 `not_public`으로 확정하는 **기록**(판정 아님).

    - `queries`: 실제로 사용된 검색어. 모르면 빈 리스트로 넘기고, 기록에는 '검색어 미기록'으로 남긴다 (지어내지 않음).
    - `problems`: 마지막 검사에서 걸린 위반 목록. 있으면 "근거 검증 실패", 없으면 "결과 없음"으로 사유를 구분한다.
    """
    perspective = CRITERION_PERSPECTIVE[criterion_id]
    unit = "paper" if perspective in ("trl", "domain") else "family"
    cause = "근거 검증 실패" if problems else "결과 없음"
    evidence = Evidence(
        evidence_id=f"{tech.tech_id}-{criterion_id}-NP",
        source_type="not_public",
        unit=unit,
        quote="",
        locator="not_found",
        search_query=" | ".join(queries) if queries else "(검색어 미기록 — 에이전트 재실행 상한 도달)",
        searched_at=now,
        search_scope=CRITERION_SCOPE.get(criterion_id, ""),
    )
    detail = f" 마지막 위반: {'; '.join(problems[:3])}" if problems else ""
    logger.warning(
        "%s/%s -> not_public 확정 (%s/%s, retry=%d)%s", tech.tech_id, criterion_id, reason, cause, retry_count, detail
    )
    return CriterionResult(
        tech_id=tech.tech_id,
        perspective=perspective,
        criterion_id=criterion_id,
        level="not_public",
        content=(
            f"[제어 노드 기록] {reason} ({cause}, 재시도 {retry_count}회): {tech.name}의 {criterion_id} 항목에 대해 "
            f"검증을 통과한 근거가 확보되지 않아 not_public으로 확정함. "
            f"확인 범위: {evidence.search_scope or '미기록'}.{detail}"
        ),
        evidence=[evidence],
        confidence="low",
        evidence_unit=unit,
        details=_empty_details(criterion_id),
        generated_at=now,
        retry_count=retry_count,
    )
