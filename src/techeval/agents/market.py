"""시장 평가 Agent (C) — run_market_eval.

기술 계열 단위(`unit="family"`)로 M1 성장성 / M2 채택 / M3 생태계를 판정한다.

- 시장·채택 자료는 웹에서 찾고, M3의 프레임워크 지원은 `deps.retriever`로 선정 논문 원문과
  교차 확인한다 (docs/roles/C-market-stakeholder.md §5.2).
- **M2·M3의 레벨은 LLM이 정하지 않고 코드가 계산한다.** M2는 확인된 최고 채택 단계로,
  M3은 체크리스트의 Y 개수로 정한다.
- 채택 주체가 없는 사례는 인정하지 않는다. 계열 상용화 근거로 논문 시제품의 채택을 주장하지 않는다.
"""

import logging
from typing import Literal

from pydantic import BaseModel, ValidationError

from ..schemas import CriterionResult
from ._deps import AgentInput, Deps
from .tech_research import (
    CriterionDraft,
    SearchContext,
    _finalize_criterion,
    invoke_structured,
    load_prompt,
    select_targets,
)

logger = logging.getLogger(__name__)

MARKET_CRITERIA: tuple[str, ...] = ("M1", "M2", "M3")

#: M2 채택 단계 — 값이 클수록 상용에 가깝다. 레벨 계산에 쓴다.
ADOPTION_STAGES: dict[str, int] = {
    "research": 1,
    "poc": 2,
    "product": 3,
    "production": 4,
}
#: 확인된 최고 단계 -> level (CRITERIA.md §2.2)
M2_LEVELS: dict[int, str] = {0: "L0", 1: "L1", 2: "L2", 3: "L3", 4: "L4"}

M3_KEYS: tuple[str, ...] = (
    "framework",
    "vendor_product",
    "standardization",
    "third_party_research_tools",
)

YN = Literal["Y", "N"]


class M3Checklist(BaseModel):
    """빠진 키를 기본값으로 채우지 않는다 — 4개 전부 있어야 한다."""

    framework: YN
    vendor_product: YN
    standardization: YN
    third_party_research_tools: YN


class Adopter(BaseModel):
    name: str
    stage: Literal["research", "poc", "product", "production"]
    evidence_id: str | None = None


def m2_level(adopters: list[dict]) -> str:
    """확인된 채택 사례 중 가장 높은 단계로 레벨을 정한다. 주체 이름이 없는 사례는 세지 않는다."""
    best = 0
    for a in adopters:
        if not (a.get("name") or "").strip():
            logger.warning("채택 주체 이름이 없는 사례는 제외한다: %s", a)
            continue
        best = max(best, ADOPTION_STAGES.get(a.get("stage", ""), 0))
    return M2_LEVELS[best]


def m3_level(checklist: dict) -> str:
    """Y 개수로 레벨을 정한다. 3~4 -> L3, 1~2 -> L2, 0 -> L1 (CRITERIA.md §2.2)."""
    yes = sum(1 for k in M3_KEYS if checklist.get(k) == "Y")
    if yes >= 3:
        return "L3"
    if yes >= 1:
        return "L2"
    return "L1"


#: 기준별 검색어. 계열 단위 조사이므로 논문 제목이 아니라 계열 별칭으로 던진다.
_WEB_QUERIES: dict[str, list[str]] = {
    "M1": ["{alias} market forecast demand drivers"],
    "M2": ["{alias} adoption production deployment"],
    "M3": ["{alias} framework vendor standardization support"],
}
#: M3 프레임워크 지원을 논문 원문과 교차 확인할 때 쓰는 질의
_PAPER_QUERIES: list[str] = [
    "{alias} key-value cache representation",
    "{name} architecture design",
]


def _fmt(templates: list[str], tech) -> list[str]:
    alias = tech.search_aliases[0] if tech.search_aliases else tech.name
    return [
        t.format(name=tech.name, alias=alias, family=tech.family) for t in templates
    ]


def gather_market_context(
    inp: AgentInput, deps: Deps, targets: list[str]
) -> SearchContext:
    """웹 검색으로 계열 근거를 모으고, M3이 대상이면 논문 청크도 함께 모은다."""
    tech = inp.tech
    ctx = SearchContext()
    queries = list(inp.rewritten_queries)
    for cid in targets:
        queries += _fmt(_WEB_QUERIES.get(cid, []), tech)
    for q in queries:
        try:
            ctx.add_web(deps.web_search(q, max_results=5, fetch_content=True))
        except Exception as exc:  # noqa: BLE001 — provider가 어떤 예외를 낼지 모른다
            # 질의 하나가 실패해도 나머지는 계속한다. 근거를 하나도 못 모으면
            # _finalize_criterion 이 not_public 으로 기록한다 (AGENTS.md 규칙 2).
            logger.warning("[%s] 웹 검색 실패 q=%r: %s", tech.tech_id, q, exc)
        ctx.queries.append(q)

    if "M3" in targets:
        for q in _fmt(_PAPER_QUERIES, tech):
            chunks = deps.retriever.search(q, top_k=5, doc_ids=[tech.primary_doc_id])
            ctx.add_chunks(chunks, unit="paper")
    return ctx


def _check_m_details(cid: str, r: CriterionResult) -> None:
    """레벨을 코드로 계산하고 details 형식을 강제한다. 어긋나면 결과를 폐기한다."""
    if r.level == "not_public":
        return  # 근거를 못 모은 결과에는 details 요건을 적용하지 않는다
    d = r.details or {}
    if cid == "M1":
        figures = d.get("market_figures")
        if figures is None:
            raise ValueError("M1: details.market_figures 가 없다")
        for f in figures:
            if not f.get("publisher"):
                raise ValueError(f"M1: 시장 수치에 발행 주체가 없다: {f}")
    elif cid == "M2":
        adopters = d.get("adopters")
        if adopters is None:
            raise ValueError("M2: details.adopters 가 없다")
        for a in adopters:
            Adopter.model_validate(a)
        r.details["adopters"] = [a for a in adopters if (a.get("name") or "").strip()]
        r.level = m2_level(r.details["adopters"])
    elif cid == "M3":
        checklist = M3Checklist.model_validate(d.get("checklist") or {}).model_dump()
        r.details["checklist"] = checklist
        r.level = m3_level(checklist)


def stub_overrides() -> dict:
    """FakeStructuredLLM 자동 등록용. 이 모듈은 공용 CriterionDraft만 쓰므로 추가 역변환이 없다."""
    return {}


def run_market_eval(inp: AgentInput, deps: Deps) -> list[CriterionResult]:
    tech = inp.tech
    targets = select_targets(inp.missing_criteria, MARKET_CRITERIA)
    now = deps.now()
    ctx = gather_market_context(inp, deps, targets)
    system = load_prompt("market", "system")
    scope = "웹 검색(벤더 공식·언론·조사기관) + 선정 논문 원문 교차 확인"
    logger.info(
        "[%s] market_eval targets=%s retry=%d web=%d chunks=%d",
        tech.tech_id,
        targets,
        inp.retry_count,
        len(ctx.web),
        len(ctx.chunks),
    )

    results: list[CriterionResult] = []
    for cid in targets:
        try:
            user = load_prompt("market", cid).format(
                tech_id=tech.tech_id, context=ctx.render()
            )
            draft = invoke_structured(deps.llm, CriterionDraft, system, user)
            draft.measurements = []  # 시장 기준은 Measurement 대상이 아니다
            r = _finalize_criterion(
                draft=draft,
                tech_id=tech.tech_id,
                perspective="market",
                criterion_id=cid,
                ctx=ctx,
                unit="family",
                now=now,
                retry_count=inp.retry_count,
                scope=scope,
            )
            _check_m_details(cid, r)
            results.append(r)
        except (ValidationError, ValueError) as exc:
            logger.error("[%s] %s 결과 폐기: %s", tech.tech_id, cid, exc)
    return results
