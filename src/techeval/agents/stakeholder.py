"""이해관계자 평가 Agent (C) — run_stakeholder_eval.

고정 4주체(`schemas.STAKEHOLDERS`)에 대해 S1 역할 / S2 편익 / S3 부담 / S4 상충을 판정한다.

- **RAG를 쓰지 않는다.** 근거는 웹 검색으로만 모은다 (docs/roles/C-market-stakeholder.md §5.3).
- S2·S3은 4주체 각각 최소 1개를 강제한다. 직접 자료가 없으면 `is_inference=True`로 표시하고
  추론의 출발점이 된 evidence_id를 달게 한다. inference가 섞이면 `compute_confidence`가 `low`로 낮춘다.
- S4는 최소 1쌍. `beneficiary`와 `burdened`가 서로 달라야 한다.
- 두 기술에 같은 프롬프트 템플릿, 같은 4주체를 쓴다.
"""

import logging
from typing import Literal

from pydantic import BaseModel, ValidationError

from ..schemas import STAKEHOLDERS, CriterionResult, Evidence
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

STAKEHOLDER_CRITERIA: tuple[str, ...] = ("S1", "S2", "S3", "S4")

Role = Literal["decision_maker", "affected", "supplier", "unrelated"]


class StakeholderEntry(BaseModel):
    """S2 편익 / S3 부담의 항목 1개."""

    text: str
    evidence_id: str | None = None
    is_inference: bool = False


class Tradeoff(BaseModel):
    """S4 상충 1쌍."""

    beneficiary: str
    burdened: str
    text: str
    evidence_ids: list[str] = []


#: 기준별 검색어. 4주체 모두를 한 번에 다루도록 넓게 던진다.
_WEB_QUERIES: dict[str, list[str]] = {
    "S1": ["{alias} stakeholder decision maker supplier"],
    "S2": ["{alias} benefits for operators and vendors"],
    "S3": ["{alias} costs burdens migration risk"],
    "S4": ["{alias} conflicting interests tradeoffs"],
}


def _fmt(templates: list[str], tech) -> list[str]:
    alias = tech.search_aliases[0] if tech.search_aliases else tech.name
    return [
        t.format(name=tech.name, alias=alias, family=tech.family) for t in templates
    ]


def gather_stakeholder_context(
    inp: AgentInput, deps: Deps, targets: list[str]
) -> SearchContext:
    """웹 검색만 쓴다. retriever는 호출하지 않는다 (역할 C 문서 §5.3)."""
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
    return ctx


def _inference_evidence(evidence_id: str, basis: str | None, text: str) -> Evidence:
    """추론 항목을 Evidence로 남긴다. 어디서 출발한 추론인지 locator에 적는다."""
    return Evidence(
        evidence_id=evidence_id,
        source_type="inference",
        unit="family",
        quote=text,
        locator=f"inference from {basis}" if basis else "inference",
    )


def _check_s_details(cid: str, r: CriterionResult) -> None:
    """4주체 보장·형식을 강제한다. 어긋나면 결과를 폐기한다 (V4)."""
    if r.level == "not_public":
        return  # 근거를 못 모은 결과에는 details 요건을 적용하지 않는다
    d = r.details or {}
    if cid == "S1":
        roles = d.get("roles") or {}
        missing = [s for s in STAKEHOLDERS if s not in roles]
        if missing:
            raise ValueError(f"S1: details.roles 에 빠진 주체: {missing}")
        bad = {s: v for s, v in roles.items() if v not in Role.__args__}
        if bad:
            raise ValueError(f"S1: 알 수 없는 역할 값: {bad}")
        r.details["roles"] = {s: roles[s] for s in STAKEHOLDERS}  # 순서 고정
    elif cid in ("S2", "S3"):
        key = "benefits" if cid == "S2" else "burdens"
        entries = d.get(key) or {}
        cleaned: dict[str, list[dict]] = {}
        for s in STAKEHOLDERS:
            items = [StakeholderEntry.model_validate(e) for e in (entries.get(s) or [])]
            if not items:
                raise ValueError(
                    f"{cid}: details.{key}[{s}] 가 비어 있다 (4주체 각 ≥1)"
                )
            cleaned[s] = [e.model_dump() for e in items]
        r.details[key] = cleaned
        _attach_inference_evidence(r, cleaned)
    elif cid == "S4":
        pairs = [Tradeoff.model_validate(t) for t in (d.get("tradeoffs") or [])]
        if not pairs:
            raise ValueError("S4: details.tradeoffs 가 최소 1쌍 필요하다")
        for t in pairs:
            if t.beneficiary == t.burdened:
                raise ValueError(f"S4: 편익 주체와 부담 주체가 같다: {t.beneficiary}")
            for who in (t.beneficiary, t.burdened):
                if who not in STAKEHOLDERS:
                    raise ValueError(f"S4: 알 수 없는 주체: {who}")
        r.details["tradeoffs"] = [t.model_dump() for t in pairs]


def _attach_inference_evidence(
    r: CriterionResult, entries: dict[str, list[dict]]
) -> None:
    """추론 항목마다 inference Evidence를 붙인다. compute_confidence가 이를 보고 low로 낮춘다."""
    from ..schemas import compute_confidence

    seq = len(r.evidence)
    added = False
    for items in entries.values():
        for e in items:
            if not e.get("is_inference"):
                continue
            seq += 1
            eid = f"{r.tech_id}-{r.criterion_id}-{seq:02d}"
            r.evidence.append(_inference_evidence(eid, e.get("evidence_id"), e["text"]))
            e["evidence_id"] = eid
            added = True
    if added:
        r.confidence = compute_confidence(r.evidence)


def stub_overrides() -> dict:
    """FakeStructuredLLM 자동 등록용. 이 모듈은 공용 CriterionDraft만 쓴다."""
    return {}


def run_stakeholder_eval(inp: AgentInput, deps: Deps) -> list[CriterionResult]:
    tech = inp.tech
    targets = select_targets(inp.missing_criteria, STAKEHOLDER_CRITERIA)
    now = deps.now()
    ctx = gather_stakeholder_context(inp, deps, targets)
    system = load_prompt("stakeholder", "system")
    scope = "웹 검색(벤더 공식·언론·조사기관). 논문 원문은 사용하지 않음"
    logger.info(
        "[%s] stakeholder_eval targets=%s retry=%d web=%d",
        tech.tech_id,
        targets,
        inp.retry_count,
        len(ctx.web),
    )

    results: list[CriterionResult] = []
    for cid in targets:
        try:
            user = load_prompt("stakeholder", cid).format(
                tech_id=tech.tech_id, context=ctx.render()
            )
            draft = invoke_structured(deps.llm, CriterionDraft, system, user)
            draft.level = "assigned" if cid == "S1" else "narrative"
            draft.measurements = []
            r = _finalize_criterion(
                draft=draft,
                tech_id=tech.tech_id,
                perspective="stakeholder",
                criterion_id=cid,
                ctx=ctx,
                unit="family",
                now=now,
                retry_count=inp.retry_count,
                scope=scope,
            )
            _check_s_details(cid, r)
            results.append(r)
        except (ValidationError, ValueError) as exc:
            logger.error("[%s] %s 결과 폐기: %s", tech.tech_id, cid, exc)
    return results
