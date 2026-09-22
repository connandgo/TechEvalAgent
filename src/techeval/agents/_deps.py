"""에이전트 함수에 주입되는 의존성과 입출력 모델. `docs/CONTRACTS.md` §5와 1:1로 대응한다.

에이전트(B, C, D)는 그래프를 모르는 순수 함수 `run_xxx(inp, deps)`이며, 여기 정의된 타입만 본다.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from techeval.schemas import (
    DOMAIN,
    CriterionId,
    CriterionResult,
    Evidence,
    EvidenceGap,
    JudgeResult,
    SynthesisResult,
    TechProfile,
    TechRef,
)

# `retrieval.retriever.BaseRetriever` / `tools.web_search.WebSearchFn`은 A·C 소유 모듈이라 런타임 의존을 피하고
# Any로 받는다. 실제 타입은 CONTRACTS §3·§4 참조.
WebSearchFn = Callable[..., list[Any]]


class Deps(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    retriever: Any  # BaseRetriever (A)
    web_search: WebSearchFn  # web_search / stub_web_search (C)
    llm: Any  # langchain BaseChatModel (LLM_MODEL)
    judge_llm: Any | None = None  # JUDGE_MODEL, judge 노드에만 주입
    now: Callable[[], str]  # ISO datetime 생성기 (테스트에서 고정)


class AgentInput(BaseModel):
    tech: TechRef
    domain: str = DOMAIN
    tech_profile: TechProfile | None = None  # 기술 조사 완료 후 관점 에이전트에 전달
    trl_eval: list[CriterionResult] = []  # 도메인·시장 에이전트가 참고 가능
    missing_criteria: list[CriterionId] | None = None  # 재실행 시 이 기준만 생성
    previous_results: list[CriterionResult] = []  # 재실행 시 기존 결과 (부분 갱신용)
    retry_count: int = 0
    rewritten_queries: list[str] = []  # 질의 재작성 노드가 제안한 검색어


class TechResearchOutput(BaseModel):
    tech_profile: TechProfile
    trl_eval: list[CriterionResult]  # T1~T4


class SynthesisInput(BaseModel):
    technologies: list[TechRef]
    tech_profiles: list[TechProfile]
    trl_eval: list[CriterionResult]
    market_eval: list[CriterionResult]
    stakeholder_eval: list[CriterionResult]
    domain_eval: list[CriterionResult]
    counter_evidence: list[Evidence] = []


class ReportInput(BaseModel):
    technologies: list[TechRef]
    domain: str
    tech_profiles: list[TechProfile]
    trl_eval: list[CriterionResult]
    market_eval: list[CriterionResult]
    stakeholder_eval: list[CriterionResult]
    domain_eval: list[CriterionResult]
    counter_evidence: list[Evidence]
    synthesis: SynthesisResult
    evidence_gap: EvidenceGap
    judge_result: JudgeResult | None = None  # 재생성 시 수정 지시 반영
    previous_report_md: str | None = None
