"""도메인 평가 Agent (B) — run_domain_eval.

배치 크기·컨텍스트 길이·하드웨어 구성 등 논문 실험 조건을 근거로, 대규모 데이터센터 장문맥 추론
도메인에 대한 근거의 직접성(D1~D3)과 도입 변경 범위(D4)를 판정한다.

- inp.tech_profile 의 근거 청크를 retriever.get_chunk 로 되살려 우선 활용하고, 배치·컨텍스트·HW
  키워드로 추가 검색한다 (docs/roles/B-tech-domain.md §5.2).
- 두 기술에 같은 프롬프트 템플릿을 쓴다. 기술별 분기 프롬프트 없음.
- D4 체크리스트 6항목은 Pydantic으로 강제한다. 빠진 키를 기본값으로 채우지 않는다.
"""

import logging
from typing import Literal

from pydantic import BaseModel, ValidationError

from ..schemas import SURVEY_DOC_IDS, CriterionResult
from ._deps import AgentInput, Deps
from .tech_research import (
    CriterionDraft,
    SearchContext,
    _finalize_criterion,
    invoke_structured,
    load_prompt,
    pick_alias,
    profile_summary,
    retry_note,
    select_targets,
)

logger = logging.getLogger(__name__)

DOMAIN_CRITERIA: tuple[str, ...] = ("D1", "D2", "D3", "D4")
D4_KEYS: tuple[str, ...] = (
    "model_retrain",
    "model_convert",
    "serving_engine_change",
    "hw_replace",
    "memory_add",
    "other",
)

YNU = Literal["Y", "N", "unknown"]


class D4Checklist(BaseModel):
    model_retrain: YNU
    model_convert: YNU
    serving_engine_change: YNU
    hw_replace: YNU
    memory_add: YNU
    other: YNU


_PAPER_QUERIES: dict[str, list[str]] = {
    # 별칭+영문(정확도) 와 "{name}"+한국어(다양성) 를 섞는다 (tech_research._PAPER_QUERIES 주석 참조).
    "D1": [
        "{alias} KV cache per token elements context length",
        "{name} 장문맥 KV cache 메모리 부담 long context",
    ],
    "D2": [
        "{alias} batch size throughput tokens per second",
        "{name} 동시 처리 처리량 concurrent requests generation throughput",
    ],
    "D3": [
        "{alias} latency energy per token cost",
        "{name} 지연 시간 에너지 비용 operating cost electricity hardware cost",
    ],
    "D4": [
        "{alias} deployment requirements training serving hardware",
        "{name} 도입 시 변경 범위 integration serving framework memory expansion",
    ],
}
_FAMILY_QUERIES = [
    "{family} 계열 데이터센터 적용 조건",
    "{alias} memory hierarchy interconnect limitations",
]
# D4 서빙 엔진 지원 확인. 기술이 무엇인지에 대한 힌트를 넣지 않고 지원 여부만 묻는다.
_WEB_QUERIES: dict[str, list[str]] = {
    "D4": [
        "{name} serving engine support vLLM SGLang",
        "{alias} inference framework integration",
    ]
}


def _fmt(templates: list[str], tech, retry_count: int = 0) -> list[str]:
    alias = pick_alias(tech, retry_count)
    return [
        t.format(name=tech.name, alias=alias, family=tech.family) for t in templates
    ]


def gather_domain_context(
    inp: AgentInput, deps: Deps, targets: list[str]
) -> SearchContext:
    tech = inp.tech
    ctx = SearchContext()
    # 1) 기술 개요의 근거 청크를 되살린다 — 수치·측정 조건이 이미 여기 있다.
    if inp.tech_profile is not None:
        ctx.add_prior(inp.tech_profile.evidence)
        for e in inp.tech_profile.evidence:
            if e.chunk_id:
                chunk = deps.retriever.get_chunk(e.chunk_id)
                if chunk is not None:
                    ctx.add_chunks([chunk], unit=e.unit)
                else:
                    logger.warning(
                        "[%s] tech_profile 근거 chunk_id 가 retriever 에 없음: %s",
                        tech.tech_id,
                        e.chunk_id,
                    )
    # 2) 추가 검색 (E의 재작성 질의가 있으면 우선)
    queries = list(inp.rewritten_queries)
    for cid in targets:
        queries += _fmt(_PAPER_QUERIES[cid], tech, inp.retry_count)
    for q in dict.fromkeys(queries):
        ctx.add_chunks(
            deps.retriever.search(q, top_k=8, doc_ids=[tech.primary_doc_id]),
            unit="paper",
        )
        ctx.queries.append(q)
    for q in _fmt(_FAMILY_QUERIES, tech, inp.retry_count):
        ctx.add_chunks(
            deps.retriever.search(q, top_k=3, doc_ids=list(SURVEY_DOC_IDS)),
            unit="family",
        )
        ctx.queries.append(q)
    for cid in targets:
        for q in _fmt(_WEB_QUERIES.get(cid, []), tech, inp.retry_count):
            ctx.add_web(deps.web_search(q, max_results=5, fetch_content=True))
            ctx.queries.append(q)
    return ctx


def _check_d_details(cid: str, r: CriterionResult) -> None:
    """CONTRACTS §2 details 규약 + CRITERIA §2.4 판정 방식 검사. 위반 시 ValueError."""
    if r.level == "not_public":
        return
    d = r.details
    if cid in ("D1", "D2", "D3"):
        if r.level not in ("L1", "L2", "L3"):
            raise ValueError(f"{cid} level 값 이상: {r.level}")
        if d.get("directness") != r.level:
            raise ValueError(f"{cid} details.directness 가 level 과 다름")
        if r.level == "L2" and not str(d.get("extrapolation_logic", "")).strip():
            raise ValueError(f"{cid} L2(간접 근거)는 details.extrapolation_logic 필수")
    elif cid == "D4":
        if r.level != "checklist":
            raise ValueError("D4 level 은 'checklist' 여야 함")
        D4Checklist.model_validate(d.get("checklist") or {})


def run_domain_eval(inp: AgentInput, deps: Deps) -> list[CriterionResult]:
    tech = inp.tech
    targets = select_targets(inp.missing_criteria, DOMAIN_CRITERIA)
    now = deps.now()
    ctx = gather_domain_context(inp, deps, targets)
    system = load_prompt("domain", "system").format(domain=inp.domain)
    summary = profile_summary(inp.tech_profile)
    scope = "선정 논문 원문 + 서베이 2편 + 서빙 엔진 공식 문서"
    logger.info(
        "[%s] domain_eval targets=%s retry=%d chunks=%d web=%d",
        tech.tech_id,
        targets,
        inp.retry_count,
        len(ctx.chunks),
        len(ctx.web),
    )

    results: list[CriterionResult] = []
    for cid in targets:
        try:
            user = load_prompt("domain", cid).format(
                tech_id=tech.tech_id, context=ctx.render(), profile_summary=summary
            )
            draft = invoke_structured(
                deps.llm, CriterionDraft, system, user + retry_note(inp)
            )
            if cid == "D4":
                draft.level = "checklist"
                draft.measurements = []
            # V3: D1~D3 는 measurements 필수. 수치 인용이 전부 검증 실패하면 not_public 으로 기록한다.
            r = _finalize_criterion(
                draft=draft,
                tech_id=tech.tech_id,
                perspective="domain",
                criterion_id=cid,
                ctx=ctx,
                unit="paper",
                now=now,
                retry_count=inp.retry_count,
                scope=scope,
                # V3(CONTRACT CHANGE): L2/L3만 수치 필수. L1(근거 없음)은 수치 없이 판정 가능.
                require_measurements=cid in ("D1", "D2", "D3")
                and draft.level in ("L2", "L3"),
            )
            _check_d_details(cid, r)
            results.append(r)
        except (ValidationError, ValueError) as exc:
            logger.error("[%s] %s 결과 폐기: %s", tech.tech_id, cid, exc)
    return results
