"""LangGraph State 스키마와 State 헬퍼. `docs/CONTRACTS.md` §7과 1:1로 대응한다.

Agent 실습(Supervisor 패턴)에서 하나로 두껍던 State를 레이어로 나눴다.

- `ControlState` : 조정·종료·재개에 필요한 최소 제어 메타. 라우팅 키는 supervisor가 쓰고,
  하위 에이전트는 reducer 키(`node_status`, `last_error`)로 자기 상태만 보고한다.
- `PayloadState` : 에이전트 산출물. 각 에이전트는 자기 키에만 쓴다.
- 하위 에이전트는 이 State 전체를 보지 않는다. supervisor가 `AgentInput`(담당 기술·부족 기준·이전 결과)만 만들어 넘긴다.

결정 로그 본문(어떤 근거로 어디로 보냈는지)은 State에 쌓지 않고 `trace_id`를 붙여 logger·LangSmith로 보낸다.
State에는 마지막 판단 사유 한 줄(`decision_reason`)만 둔다.
"""

import operator
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal, TypedDict

from techeval.schemas import (
    DOMAIN,
    PERSPECTIVE_CRITERIA,
    STAKEHOLDERS,
    TECHNOLOGIES,
    CriterionResult,
    Evidence,
    EvidenceGap,
    JudgeResult,
    QualityResult,
    SupervisorAction,
    SynthesisResult,
    TechProfile,
    TechRef,
)


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """같은 슈퍼스텝에 여러 노드(기술별 Send)가 쓰는 dict 키를 덮어쓰지 않고 병합한다."""
    return {**(left or {}), **(right or {})}


def union_list(left: list[str] | None, right: list[str] | None) -> list[str]:
    """병렬 노드가 각자 쓴 목록을 합집합으로 병합한다 (순서는 정렬)."""
    return sorted({*(left or []), *(right or [])})


def merge_lists(left: dict | None, right: dict | None) -> dict[str, list[str]]:
    """{키: 목록} dict를 키별로 이어 붙인다 (중복 제거, 순서 유지). 병렬 노드의 검색 이력 병합용."""
    out = {k: list(v) for k, v in (left or {}).items()}
    for k, v in (right or {}).items():
        out[k] = list(dict.fromkeys([*out.get(k, []), *v]))
    return out


def last_value(left: str | None, right: str | None) -> str | None:
    """병렬 노드가 동시에 써도 InvalidUpdateError 없이 마지막 값을 남긴다."""
    return right if right is not None else left


class ControlState(TypedDict, total=False):
    """제어 메타 — 조정(라우팅)·종료·재개에 필요한 최소치."""

    trace_id: str  # 외부 로그·LangSmith 메타데이터와 State를 잇는 상관 키
    next: SupervisorAction  # supervisor가 고른 다음 담당 (SupervisorDecision으로 검증 후 기록)
    dispatch: dict[str, dict[str, list[str]]]  # 이번 배정 {노드: {tech_id: 부족 항목}}, 빈 리스트 = 첫 실행
    decision_reason: str  # 마지막 판단 사유 한 줄 (상세 로그는 외부)
    step_count: int  # supervisor 판단 횟수 — MAX_SUPERVISOR_STEPS로 종료 보장
    retry_counts: dict[str, int]  # {"trl:mla": 1, "market:pim_cxl": 0, "counter": 0, "report": 0}
    missing_criteria: dict[str, list[str]]  # 재작업 대상 {"market:mla": ["M2"], "trl:pim_cxl": [...]}
    synthesis_stale: bool  # 반대 근거·관점 재작업이 종합 뒤에 들어와 종합을 다시 해야 함
    report_stale: bool  # 품질 미달 재작업 후 보고서를 다시 써야 함
    quality_pending: bool  # 새 보고서가 아직 품질 평가를 받지 않음
    source_urls: Annotated[
        list[str], union_list
    ]  # 웹 검색이 돌려준 URL (V5 근거 실존 검사 기준). 재개 시 출처 레지스트리를 여기서 복원
    search_log: Annotated[dict[str, list[str]], merge_lists]  # {"market:mla": [실제 실행한 검색어]} — not_public 기록용
    run_status: Literal[
        "succeeded", "incomplete"
    ]  # 출력 시점 판정: 품질 검수 통과 = succeeded, 그 밖(검수 오류·미달·상한) = incomplete
    end_reason: str  # 출력으로 간 사유 (검수 통과 / 검수 실행 오류 / 재작성 상한 / supervisor 상한)
    node_status: Annotated[dict[str, str], merge_dict]  # {"market:mla": "ok"|"error"} 재개·재시도 판단용
    last_error: Annotated[str | None, last_value]  # 마지막 노드 오류 (재개 시 원인 확인)


class PayloadState(TypedDict, total=False):
    """작업 페이로드 — 에이전트 산출물."""

    # 초기 입력 (overwrite)
    domain: str
    technologies: list[TechRef]  # Human 고정 (mla, pim_cxl)
    stakeholders: tuple[str, ...]  # STAKEHOLDERS 상수

    # 기술 조사 (append, 기술별 Send)
    tech_profiles: Annotated[list[TechProfile], operator.add]
    trl_eval: Annotated[list[CriterionResult], operator.add]

    # 관점 평가 (append, 기술별 Send) — 재작업 결과도 누적, 읽을 때 latest_by_criterion으로 최신본만 사용
    market_eval: Annotated[list[CriterionResult], operator.add]
    stakeholder_eval: Annotated[list[CriterionResult], operator.add]
    domain_eval: Annotated[list[CriterionResult], operator.add]

    counter_evidence: list[Evidence]
    synthesis: SynthesisResult
    evidence_gap: EvidenceGap
    report_md: str
    judge_result: JudgeResult  # 보고서 재생성 지시 전달용 (report 에이전트가 읽음)
    quality_result: QualityResult  # 품질 평가 4개 항목 결과
    report_pdf_path: str


class GraphState(ControlState, PayloadState, total=False):
    """그래프 State = 제어 레이어 + 페이로드 레이어."""


def _ts(value: str) -> tuple[int, datetime | str]:
    """generated_at 비교 키. ISO datetime이면 aware UTC로 정규화해 비교하고, 파싱 실패 시 문자열 비교로 폴백한다."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return (0, value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (1, dt)


# operator.add 로 누적되는 키 목록 (중복 제거가 필요한 키)
APPEND_KEYS: tuple[str, ...] = ("tech_profiles", "trl_eval", "market_eval", "stakeholder_eval", "domain_eval")


def latest_by_criterion(results: list[CriterionResult]) -> list[CriterionResult]:
    """`(tech_id, criterion_id)`별로 `generated_at`이 가장 최신인 결과 1개만 남긴다.

    `generated_at`은 ISO datetime으로 파싱해 비교한다(offset 표기가 달라도 안전). 파싱 불가면 문자열 비교.
    같은 시각이면 뒤에 온 것(나중에 append된 것)을 취한다.
    반환 순서는 입력에서 각 키가 처음 등장한 순서를 따른다.
    """
    chosen: dict[tuple[str, str], CriterionResult] = {}
    for r in results:
        key = (r.tech_id, r.criterion_id)
        prev = chosen.get(key)
        if prev is None or _ts(r.generated_at) >= _ts(prev.generated_at):
            chosen[key] = r
    return list(chosen.values())


def latest_by_tech(profiles: list[TechProfile]) -> list[TechProfile]:
    """`tech_id`별로 `generated_at`이 가장 최신인 TechProfile 1개만 남긴다."""
    chosen: dict[str, TechProfile] = {}
    for p in profiles:
        prev = chosen.get(p.tech_id)
        if prev is None or _ts(p.generated_at) >= _ts(prev.generated_at):
            chosen[p.tech_id] = p
    return list(chosen.values())


def initial_retry_counts(technologies: list[TechRef] | None = None) -> dict[str, int]:
    """`"<perspective>:<tech_id>"`, `"counter"`, `"report"` 키를 전부 0으로 초기화한다."""
    techs = technologies or TECHNOLOGIES
    counts = {f"{p}:{t.tech_id}": 0 for p in PERSPECTIVE_CRITERIA for t in techs}
    counts["counter"] = 0
    counts["report"] = 0
    return counts


def build_initial_state(trace_id: str | None = None) -> GraphState:
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
        trace_id=trace_id or uuid.uuid4().hex[:12],
        step_count=0,
        synthesis_stale=False,
        report_stale=False,
        quality_pending=False,
        source_urls=[],
        search_log={},
        node_status={},
        last_error=None,
    )
