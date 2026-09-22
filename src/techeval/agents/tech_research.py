"""기술 조사 Agent (B) — run_tech_research.

논문 원문에서 기술 원리·범위·한계·정량 수치·측정 조건을 추출해 TechProfile을 만들고,
검증 환경과 공개 구현 자료를 근거로 T1~T4를 평가한다.

설계 원칙 (AGENTS.md §2, §6.2):
- LLM은 "어느 청크의 어느 문장"을 인용했는지(Citation)만 낸다. Evidence는 코드가 만든다.
  → 인용의 ref가 실제 검색 결과에 있는지, quote가 원문 부분 문자열인지 코드에서 검증한다.
- T3 레벨은 체크리스트 Y 개수로 코드가 계산한다. LLM에게 맡기지 않는다.
- confidence는 compute_confidence()로 계산한다.
- 검증 실패한 기준은 조용히 기본값으로 채우지 않고 로그 후 결과에서 뺀다(E의 검사 노드가 재실행).
"""

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..retrieval.retriever import RetrievedChunk
from ..schemas import (
    SURVEY_DOC_IDS,
    CriterionResult,
    Evidence,
    EvidenceUnit,
    Measurement,
    TechProfile,
    TechRef,
    compute_confidence,
)
from ..tools.web_search import WebResult, not_public_evidence
from ._deps import AgentInput, Deps, TechResearchOutput

logger = logging.getLogger(__name__)

TRL_CRITERIA: tuple[str, ...] = ("T1", "T2", "T3", "T4")
ARTIFACT_KEYS: tuple[str, ...] = (
    "code",
    "model_or_design",
    "eval_scripts_data",
    "third_party_reproduction",
)
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

YNU = Literal["Y", "N", "unknown"]


def normalize_ws(text: str) -> str:
    """V6(인용문 부분 문자열) 비교용 공백 정규화."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# LLM 구조화 출력용 초안 모델. 최종 스키마(schemas.py)와 분리한다.
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """LLM이 낸 인용 1건. ref는 chunk_id 또는 url, quote는 원문 부분 문자열."""

    source: Literal["chunk", "web"]
    ref: str
    quote: str


class MeasurementDraft(BaseModel):
    metric: str
    value: str
    unit: str | None = None
    model_size: str | None = None
    context_length: str | None = None
    hardware: str | None = None
    baseline: str | None = None
    batch_size: str | None = None
    condition_note: str | None = None
    citation_index: int = Field(
        ge=0, description="citations 리스트에서 이 수치의 출처 인덱스"
    )


class ProfileDraft(BaseModel):
    principle: str
    scope: str
    limitations: list[str] = []
    validation_env: str
    measurements: list[MeasurementDraft] = []
    citations: list[Citation] = []


class CriterionDraft(BaseModel):
    """T1, T2, T4 및 D1~D4 공용 초안."""

    level: str
    level_estimate: str | None = None
    content: str
    details: dict = {}
    measurements: list[MeasurementDraft] = []
    citations: list[Citation] = []


class ArtifactChecklist(BaseModel):
    code: YNU
    model_or_design: YNU
    eval_scripts_data: YNU
    third_party_reproduction: YNU


class ArtifactDraft(BaseModel):
    """T3 전용 초안. level은 없다 — 코드가 계산한다."""

    checklist: ArtifactChecklist
    urls: dict[str, str] = {}
    content: str
    citations: list[Citation] = []


# ---------------------------------------------------------------------------
# 검색 컨텍스트: 검색 결과를 모아 프롬프트에 넣고, 인용을 Evidence로 되돌린다.
# ---------------------------------------------------------------------------


class SearchContext:
    def __init__(self) -> None:
        self.chunks: dict[str, RetrievedChunk] = {}
        self.chunk_unit: dict[str, EvidenceUnit] = {}
        self.web: dict[str, WebResult] = {}
        self.web_alt: dict[
            str, list[WebResult]
        ] = {}  # 같은 URL이 다른 snippet으로 또 나온 경우
        self.queries: list[str] = []
        self.prior: dict[
            str, list[Evidence]
        ] = {}  # 상류(기술 개요)에서 검증된 근거 — 재인용 허용

    def add_chunks(self, chunks: list[RetrievedChunk], *, unit: EvidenceUnit) -> None:
        for c in chunks:
            self.chunks.setdefault(c.chunk_id, c)
            self.chunk_unit.setdefault(c.chunk_id, unit)

    def add_web(self, results: list[WebResult]) -> None:
        for r in results:
            if r.url in self.web:
                if r is not self.web[r.url]:
                    self.web_alt.setdefault(r.url, []).append(r)
            else:
                self.web[r.url] = r

    def add_prior(self, evidence: list[Evidence]) -> None:
        """기술 개요의 근거를 재인용 후보로 등록한다. chunk_id/url 이 실제 검색 결과였음은 상류에서 검증됐다."""
        for e in evidence:
            key = e.chunk_id or e.url
            if key and e.source_type != "not_public":
                self.prior.setdefault(key, []).append(e)

    def _from_prior(self, cit: Citation, *, evidence_id: str) -> Evidence | None:
        for e in self.prior.get(cit.ref, []):
            if normalize_ws(cit.quote) and normalize_ws(cit.quote) in normalize_ws(
                e.quote
            ):
                return e.model_copy(
                    update={"evidence_id": evidence_id, "quote": cit.quote}
                )
        return None

    def is_empty(self) -> bool:
        return not self.chunks and not self.web

    def render(self) -> str:
        parts: list[str] = []
        if self.chunks:
            parts.append("### 논문 청크\n")
            for c in self.chunks.values():
                tag = (
                    "[계열 맥락·서베이]"
                    if self.chunk_unit.get(c.chunk_id) == "family"
                    else "[선정 논문]"
                )
                sec = f" §{c.section}" if c.section else ""
                parts.append(
                    f"--- chunk_id: {c.chunk_id} {tag} ({c.doc_id} p.{c.page}{sec}, {c.chunk_type})\n{c.text}\n"
                )
        if self.web:
            parts.append("### 웹 검색 결과\n")
            for r in self.web.values():
                body = r.content or r.snippet
                parts.append(
                    f"--- url: {r.url} [{r.source_kind}] {r.title} ({r.publisher or '-'}, {r.published_date or '날짜 미상'})\n{body}\n"
                )
        return "\n".join(parts) if parts else "(검색 결과 없음)"

    def resolve(
        self, cit: Citation, *, evidence_id: str, default_unit: EvidenceUnit
    ) -> Evidence | None:
        """인용 1건을 Evidence로. ref가 실존하지 않거나 quote가 원문에 없으면 None (AGENTS.md 규칙 1)."""
        if cit.source == "chunk":
            chunk = self.chunks.get(cit.ref)
            if chunk is None:
                logger.warning(
                    "%s: 인용된 chunk_id가 검색 결과에 없음: %s", evidence_id, cit.ref
                )
                return None
            try:
                return chunk.to_evidence(
                    evidence_id=evidence_id,
                    quote=cit.quote,
                    unit=self.chunk_unit.get(cit.ref, default_unit),
                )
            except ValueError as exc:
                logger.warning("%s: %s", evidence_id, exc)
                return None
        result = self.web.get(cit.ref)
        if result is None:
            prior = self._from_prior(cit, evidence_id=evidence_id)
            if prior is None:
                logger.warning(
                    "%s: 인용된 url이 검색 결과에 없음: %s", evidence_id, cit.ref
                )
            return prior
        variants = [result, *self.web_alt.get(cit.ref, [])]
        haystack = normalize_ws(
            " ".join(f"{v.snippet} {v.content or ''}" for v in variants)
        )
        if not normalize_ws(cit.quote) or normalize_ws(cit.quote) not in haystack:
            prior = self._from_prior(cit, evidence_id=evidence_id)
            if prior is None:
                logger.warning(
                    "%s: quote가 웹 결과 본문에 없음: %s", evidence_id, cit.ref
                )
            return prior
        return result.to_evidence(
            evidence_id=evidence_id, quote=cit.quote, unit=default_unit
        )


def resolve_citations(
    ctx: SearchContext,
    citations: list[Citation],
    *,
    prefix: str,
    default_unit: EvidenceUnit,
) -> tuple[list[Evidence], list[Evidence | None]]:
    """인용 목록 → Evidence 목록. 두 번째 반환값은 초안 인덱스와 정렬된 리스트(measurement 연결용)."""
    aligned: list[Evidence | None] = []
    seen: set[tuple[str, str]] = set()
    n = 0
    for cit in citations:
        key = (cit.ref, normalize_ws(cit.quote))
        if key in seen:
            aligned.append(
                next(
                    e
                    for e in aligned
                    if e
                    and (e.chunk_id or e.url) == cit.ref
                    and normalize_ws(e.quote) == key[1]
                )
            )
            continue
        n += 1
        ev = ctx.resolve(
            cit, evidence_id=f"{prefix}-{n:02d}", default_unit=default_unit
        )
        if ev is None:
            n -= 1
        else:
            seen.add(key)
        aligned.append(ev)
    return [e for e in aligned if e is not None], aligned


def build_measurements(
    drafts: list[MeasurementDraft], aligned: list[Evidence | None]
) -> list[Measurement]:
    out: list[Measurement] = []
    for m in drafts:
        ev = aligned[m.citation_index] if 0 <= m.citation_index < len(aligned) else None
        if ev is None:
            logger.warning(
                "수치 '%s=%s' 의 출처 인용이 검증 실패로 폐기됨 → 수치도 폐기",
                m.metric,
                m.value,
            )
            continue
        data = m.model_dump(exclude={"citation_index"})
        missing = [
            k
            for k in (
                "model_size",
                "context_length",
                "hardware",
                "baseline",
                "batch_size",
            )
            if not data.get(k)
        ]
        if missing and not data.get("condition_note"):
            data["condition_note"] = "원문 미기재: " + ", ".join(missing)
        out.append(Measurement(**data, evidence_id=ev.evidence_id))
    return out


# ---------------------------------------------------------------------------
# 프롬프트 / LLM 호출
# ---------------------------------------------------------------------------


def load_prompt(agent: str, name: str) -> str:
    return (PROMPT_DIR / agent / f"{name}.md").read_text(encoding="utf-8")


def invoke_structured(llm, model: type[BaseModel], system: str, user: str):
    """llm.with_structured_output(Model) 호출 후 Pydantic으로 재검증한다 (AGENTS.md §6.2)."""
    raw = llm.with_structured_output(model).invoke(
        [("system", system), ("human", user)]
    )
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    return model.model_validate(raw)


def select_targets(
    missing: list[str] | None, all_criteria: tuple[str, ...]
) -> list[str]:
    if not missing:
        return list(all_criteria)
    bad = [c for c in missing if c not in all_criteria]
    if bad:
        raise ValueError(
            f"missing_criteria에 이 에이전트 소관이 아닌 기준이 있음: {bad}"
        )
    return [c for c in all_criteria if c in missing]


def profile_summary(p: TechProfile | None) -> str:
    if p is None:
        return "(기술 개요 없음)"
    lines = [
        f"- 원리: {p.principle}",
        f"- 범위: {p.scope}",
        f"- 검증 환경: {p.validation_env}",
    ]
    if p.limitations:
        lines.append("- 명시된 한계: " + " / ".join(p.limitations))
    for m in p.measurements:
        cond = ", ".join(
            f"{k}={v}"
            for k, v in (
                ("model", m.model_size),
                ("ctx", m.context_length),
                ("hw", m.hardware),
                ("baseline", m.baseline),
                ("batch", m.batch_size),
            )
            if v
        )
        lines.append(
            f"- 수치: {m.metric} = {m.value}{' ' + m.unit if m.unit else ''} [{cond or '조건 미기재'}] (근거 {m.evidence_id})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 검색 전략 (docs/roles/B-tech-domain.md §5.1)
# ---------------------------------------------------------------------------

_PAPER_QUERIES: dict[str, list[str]] = {
    "profile": [
        "{name} 핵심 원리 KV cache",
        "{alias} key-value cache mechanism design",
        "{alias} results throughput memory reduction",
        "{name} 한계 limitations",
    ],
    "T2": [
        "실험 환경 GPU 구성",
        "evaluation setup hardware prototype simulation",
        "{alias} deployment serving measurement",
    ],
    "T4": ["{alias} limitations future work", "{name} 남은 과제 미공개 항목"],
}
_FAMILY_QUERIES = ["{alias} survey taxonomy", "{family} 계열 한계 반대 근거"]
_WEB_QUERIES: dict[str, list[str]] = {
    "T3": [
        "{name} code availability github",
        "{name} model weights design availability",
        "{alias} third-party implementation reproduction",
    ],
    "T4": ["{name} production deployment"],
}


_RETRY_EXTRA_QUERIES = [
    "{alias} experimental setup evaluation details",
    "{alias} reported numbers table results",
    "{name} 측정 조건 하드웨어 배치 컨텍스트",
]


def pick_alias(tech: TechRef, retry_count: int = 0) -> str:
    """재시도마다 다른 별칭으로 검색어를 만든다 (docs/roles/B §5.1: retry_count 반영)."""
    aliases = tech.search_aliases or [tech.name]
    return aliases[retry_count % len(aliases)]


def _fmt(templates: list[str], tech: TechRef, retry_count: int = 0) -> list[str]:
    alias = pick_alias(tech, retry_count)
    return [
        t.format(name=tech.name, alias=alias, family=tech.family) for t in templates
    ]


def retry_note(inp: AgentInput) -> str:
    """재실행일 때 프롬프트 끝에 붙이는 안내. 첫 실행이면 빈 문자열."""
    if inp.retry_count <= 0:
        return ""
    missing = ", ".join(inp.missing_criteria or []) or "(전체)"
    return (
        f"\n\n## 재시도 안내\n\n이번은 {inp.retry_count}번째 재시도다. 이전 시도에서 근거가 부족했던 기준: {missing}. "
        "이전과 같은 문장을 되풀이하지 말고, 다른 절·표·웹 결과에서 근거를 찾아라. "
        "그래도 없으면 추측하지 말고 인용을 비워 두어라."
    )


def gather_context(inp: AgentInput, deps: Deps, targets: list[str]) -> SearchContext:
    tech = inp.tech
    ctx = SearchContext()
    # E의 질의 재작성 결과가 있으면 그것을 우선 사용한다.
    paper_queries = list(inp.rewritten_queries) or _fmt(
        _PAPER_QUERIES["profile"], tech, inp.retry_count
    )
    for cid in targets:
        paper_queries += _fmt(_PAPER_QUERIES.get(cid, []), tech, inp.retry_count)
    if inp.retry_count > 0:
        paper_queries += _fmt(_RETRY_EXTRA_QUERIES, tech, inp.retry_count)
    for q in dict.fromkeys(paper_queries):
        ctx.add_chunks(
            deps.retriever.search(q, top_k=6, doc_ids=[tech.primary_doc_id]),
            unit="paper",
        )
        ctx.queries.append(q)
    # 서베이는 기술 계열 맥락·한계 보강용 2차 검색 (unit=family)
    for q in _fmt(_FAMILY_QUERIES, tech, inp.retry_count):
        ctx.add_chunks(
            deps.retriever.search(q, top_k=3, doc_ids=list(SURVEY_DOC_IDS)),
            unit="family",
        )
        ctx.queries.append(q)
    for cid in targets:
        for q in _fmt(_WEB_QUERIES.get(cid, []), tech, inp.retry_count):
            ctx.add_web(deps.web_search(q, max_results=5))
            ctx.queries.append(q)
    return ctx


# ---------------------------------------------------------------------------
# T3 레벨: 코드로 계산
# ---------------------------------------------------------------------------


def compute_t3_level(checklist: dict[str, str]) -> str:
    values = [checklist.get(k, "unknown") for k in ARTIFACT_KEYS]
    if all(v == "unknown" for v in values):
        return "L0"
    yes = sum(v == "Y" for v in values)
    if yes >= 3:
        return "L3"
    if yes >= 1:
        return "L2"
    return "L1"


# ---------------------------------------------------------------------------
# 결과 조립
# ---------------------------------------------------------------------------


def not_public_result(
    *,
    tech_id: str,
    perspective: str,
    criterion_id: str,
    queries: list[str],
    scope: str,
    unit: EvidenceUnit,
    now: str,
    retry_count: int,
    level: str = "not_public",
    details: dict | None = None,
) -> CriterionResult:
    ev = not_public_evidence(
        evidence_id=f"{tech_id}-{criterion_id}-01",
        queries=queries,
        scope=scope,
        unit=unit,
    )
    return CriterionResult(
        tech_id=tech_id,
        perspective=perspective,
        criterion_id=criterion_id,
        level=level,
        content=f"검색 범위({scope}) 안에서 {criterion_id} 판정에 쓸 근거를 찾지 못했다. 검색어: {' | '.join(queries)}.",
        evidence=[ev],
        confidence="low",
        evidence_unit=unit,
        details=details or {"search_queries": queries},
        generated_at=now,
        retry_count=retry_count,
    )


def _finalize_criterion(
    *,
    draft: CriterionDraft,
    tech_id: str,
    perspective: str,
    criterion_id: str,
    ctx: SearchContext,
    unit: EvidenceUnit,
    now: str,
    retry_count: int,
    scope: str,
    require_measurements: bool = False,
) -> CriterionResult:
    evidence, aligned = resolve_citations(
        ctx, draft.citations, prefix=f"{tech_id}-{criterion_id}", default_unit=unit
    )
    measurements = build_measurements(draft.measurements, aligned) if evidence else []
    if not evidence or (require_measurements and not measurements):
        why = (
            "검증을 통과한 근거가 없음"
            if not evidence
            else "검증을 통과한 수치가 없음 (V3)"
        )
        logger.warning("%s/%s: %s → not_public", tech_id, criterion_id, why)
        return not_public_result(
            tech_id=tech_id,
            perspective=perspective,
            criterion_id=criterion_id,
            queries=ctx.queries,
            scope=scope,
            unit=unit,
            now=now,
            retry_count=retry_count,
        )
    return CriterionResult(
        tech_id=tech_id,
        perspective=perspective,
        criterion_id=criterion_id,
        level=draft.level,
        level_estimate=draft.level_estimate,
        content=draft.content,
        evidence=evidence,
        confidence=compute_confidence(evidence),
        evidence_unit=unit,
        measurements=measurements,
        details=draft.details,
        generated_at=now,
        retry_count=retry_count,
    )


def _check_t_details(cid: str, r: CriterionResult) -> None:
    """CONTRACTS §2 details 필수 키 검사. 빠지면 ValueError → 호출측이 결과에서 제외."""
    d = r.details
    if r.level == "not_public":
        return
    if cid == "T1":
        for k in ("trl_band", "estimate", "why_not_higher"):
            if not d.get(k):
                raise ValueError(f"T1 details.{k} 누락")
        if r.level not in ("TRL 1-3", "TRL 4-6", "TRL 7-9"):
            raise ValueError(f"T1 level 값 이상: {r.level}")
        if not r.level_estimate or "공개 정보 기반 추정" not in r.level_estimate:
            raise ValueError("T1 level_estimate에 '공개 정보 기반 추정' 문구 필요")
    elif cid == "T2":
        if r.level not in ("L1", "L2", "L3", "L4"):
            raise ValueError(f"T2 level 값 이상: {r.level}")
        if d.get("env_level") != r.level:
            raise ValueError("T2 details.env_level 이 level 과 다름")
    elif cid == "T4":
        if r.level != "narrative":
            raise ValueError("T4 level 은 'narrative' 여야 함")
        if not isinstance(d.get("remaining_tasks"), list) or not isinstance(
            d.get("not_public_items"), list
        ):
            raise ValueError(
                "T4 details.remaining_tasks / not_public_items 는 list 여야 함"
            )


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------


def _build_profile(
    inp: AgentInput, deps: Deps, ctx: SearchContext, system: str, now: str
) -> TechProfile:
    tech = inp.tech
    user = load_prompt("tech_research", "profile").format(
        tech_name=tech.name,
        tech_id=tech.tech_id,
        paper_title=tech.paper_title,
        paper_date=tech.paper_date,
        context=ctx.render(),
    )
    draft = invoke_structured(deps.llm, ProfileDraft, system, user + retry_note(inp))
    evidence, aligned = resolve_citations(
        ctx, draft.citations, prefix=f"{tech.tech_id}-PROFILE", default_unit="paper"
    )
    if not evidence:
        raise ValueError(
            f"{tech.tech_id}: TechProfile 인용이 전부 검증 실패 — 검색 결과에 없는 chunk_id/quote"
        )
    return TechProfile(
        tech_id=tech.tech_id,
        principle=draft.principle,
        scope=draft.scope,
        limitations=draft.limitations,
        measurements=build_measurements(draft.measurements, aligned),
        validation_env=draft.validation_env,
        public_artifacts={k: "unknown" for k in ARTIFACT_KEYS},
        evidence=evidence,
        search_queries_used=list(ctx.queries),
        generated_at=now,
        retry_count=inp.retry_count,
    )


def _refresh_profile(
    existing: TechProfile,
    inp: AgentInput,
    deps: Deps,
    ctx: SearchContext,
    system: str,
    now: str,
) -> TechProfile:
    """재실행: 누락된 필드만 보강한다. 이미 있는 값은 덮어쓰지 않는다."""
    fresh = _build_profile(inp, deps, ctx, system, now)
    data = existing.model_dump()
    for key in ("principle", "scope", "validation_env"):
        if not data[key].strip():
            data[key] = getattr(fresh, key)
    if not data["limitations"]:
        data["limitations"] = fresh.limitations
    if not data["measurements"]:
        data["measurements"] = [m.model_dump() for m in fresh.measurements]
    known = {e.evidence_id for e in existing.evidence}
    offset = len(known)
    for i, e in enumerate(fresh.evidence, start=1):
        e2 = e.model_copy(
            update={"evidence_id": f"{existing.tech_id}-PROFILE-{offset + i:02d}"}
        )
        data["evidence"].append(e2.model_dump())
    data["search_queries_used"] = list(
        dict.fromkeys(data["search_queries_used"] + ctx.queries)
    )
    data["generated_at"] = now
    data["retry_count"] = inp.retry_count
    return TechProfile.model_validate(data)


def _eval_t3(
    inp: AgentInput, deps: Deps, ctx: SearchContext, system: str, now: str
) -> tuple[CriterionResult, ArtifactDraft]:
    tech = inp.tech
    user = load_prompt("tech_research", "T3").format(
        tech_id=tech.tech_id, context=ctx.render()
    )
    draft = invoke_structured(deps.llm, ArtifactDraft, system, user + retry_note(inp))
    checklist = draft.checklist.model_dump()
    level = compute_t3_level(checklist)
    evidence, _ = resolve_citations(
        ctx, draft.citations, prefix=f"{tech.tech_id}-T3", default_unit="paper"
    )
    scope = "선정 논문 원문, GitHub, Hugging Face, 벤더 공식 자료"
    if not evidence:
        # 근거 없이 Y/N 을 말할 수는 없다. 미확인(L0) + not_public 근거 기록.
        result = not_public_result(
            tech_id=tech.tech_id,
            perspective="trl",
            criterion_id="T3",
            queries=ctx.queries,
            scope=scope,
            unit="paper",
            now=now,
            retry_count=inp.retry_count,
            level="L0",
            details={
                "checklist": {k: "unknown" for k in ARTIFACT_KEYS},
                "search_queries": ctx.queries,
            },
        )
        return result, draft
    result = CriterionResult(
        tech_id=tech.tech_id,
        perspective="trl",
        criterion_id="T3",
        level=level,
        content=draft.content,
        evidence=evidence,
        confidence=compute_confidence(evidence),
        evidence_unit="paper",
        details={"checklist": checklist, "urls": draft.urls},
        generated_at=now,
        retry_count=inp.retry_count,
    )
    return result, draft


def run_tech_research(inp: AgentInput, deps: Deps) -> TechResearchOutput:
    tech = inp.tech
    targets = select_targets(inp.missing_criteria, TRL_CRITERIA)
    now = deps.now()
    ctx = gather_context(inp, deps, targets)
    system = load_prompt("tech_research", "system")
    logger.info(
        "[%s] tech_research targets=%s retry=%d chunks=%d web=%d",
        tech.tech_id,
        targets,
        inp.retry_count,
        len(ctx.chunks),
        len(ctx.web),
    )

    if inp.tech_profile is None or inp.missing_criteria is None:
        profile = _build_profile(inp, deps, ctx, system, now)
    else:
        profile = _refresh_profile(inp.tech_profile, inp, deps, ctx, system, now)

    results: list[CriterionResult] = []
    scope = "선정 논문 원문 + 서베이 2편 + 웹(GitHub, HF, 벤더 공식 자료)"
    for cid in targets:
        try:
            if cid == "T3":
                r, artifact = _eval_t3(inp, deps, ctx, system, now)
                profile = profile.model_copy(
                    update={
                        "public_artifacts": artifact.checklist.model_dump(),
                        "public_artifact_urls": dict(artifact.urls),
                    }
                )
            else:
                fmt_args = {
                    "tech_id": tech.tech_id,
                    "context": ctx.render(),
                    "profile_summary": profile_summary(profile),
                    "validation_env": profile.validation_env,
                    "limitations": "\n".join(f"- {x}" for x in profile.limitations)
                    or "(원문에 명시된 한계 없음)",
                }
                user = load_prompt("tech_research", cid).format(**fmt_args)
                draft = invoke_structured(
                    deps.llm, CriterionDraft, system, user + retry_note(inp)
                )
                if cid == "T4":
                    draft.level = "narrative"
                r = _finalize_criterion(
                    draft=draft,
                    tech_id=tech.tech_id,
                    perspective="trl",
                    criterion_id=cid,
                    ctx=ctx,
                    unit="paper",
                    now=now,
                    retry_count=inp.retry_count,
                    scope=scope,
                )
                _check_t_details(cid, r)
            results.append(r)
        except (ValidationError, ValueError) as exc:
            # 규칙 8: 조용히 기본값으로 채우지 않는다. 빠진 기준은 E의 검사 노드가 재실행을 건다.
            logger.error("[%s] %s 결과 폐기: %s", tech.tech_id, cid, exc)

    return TechResearchOutput(tech_profile=profile, trl_eval=results)


# ---------------------------------------------------------------------------
# --stub 실행 지원: B 픽스처(tech_profiles/trl_eval/domain_eval.json)를 드래프트로 역변환해
# E의 FakeStructuredLLM(overrides=...)에 꽂는다. E의 build_deps(stub=True)가 호출한다.
# ---------------------------------------------------------------------------

_STUB_TECH_ID = re.compile(r"tech_id\W{0,4}(mla|pim_cxl)", re.IGNORECASE)
_STUB_HEADING = re.compile(r"^# (T\d|D\d|기술 개요)", re.MULTILINE)
DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures"


def _citations_from_evidence(
    evidence: list[Evidence],
) -> tuple[list[Citation], dict[str, int]]:
    cits: list[Citation] = []
    index: dict[str, int] = {}
    for e in evidence:
        if e.source_type == "not_public" or not e.quote:
            continue
        ref = e.chunk_id or e.url
        if not ref:
            continue
        index[e.evidence_id] = len(cits)
        cits.append(
            Citation(source="chunk" if e.chunk_id else "web", ref=ref, quote=e.quote)
        )
    return cits, index


def _measurement_drafts(
    measurements: list[Measurement], index: dict[str, int]
) -> list[MeasurementDraft]:
    out: list[MeasurementDraft] = []
    for m in measurements:
        if m.evidence_id in index:
            out.append(
                MeasurementDraft(
                    **m.model_dump(exclude={"evidence_id"}),
                    citation_index=index[m.evidence_id],
                )
            )
    return out


def profile_to_draft(p: TechProfile) -> ProfileDraft:
    cits, index = _citations_from_evidence(p.evidence)
    return ProfileDraft(
        principle=p.principle,
        scope=p.scope,
        limitations=p.limitations,
        validation_env=p.validation_env,
        measurements=_measurement_drafts(p.measurements, index),
        citations=cits,
    )


def criterion_to_draft(r: CriterionResult) -> CriterionDraft:
    cits, index = _citations_from_evidence(r.evidence)
    return CriterionDraft(
        level=r.level,
        level_estimate=r.level_estimate,
        content=r.content,
        details=r.details,
        measurements=_measurement_drafts(r.measurements, index),
        citations=cits,
    )


def criterion_to_artifact_draft(r: CriterionResult) -> ArtifactDraft:
    cits, _ = _citations_from_evidence(r.evidence)
    checklist = r.details.get("checklist") or {k: "unknown" for k in ARTIFACT_KEYS}
    return ArtifactDraft(
        checklist=ArtifactChecklist(**checklist),
        urls=r.details.get("urls", {}),
        content=r.content,
        citations=cits,
    )


def stub_overrides(
    fixtures_dir: str | Path | None = None,
) -> dict[type, Callable[[str], BaseModel]]:
    """FakeStructuredLLM(overrides=stub_overrides()) 로 쓴다. 프롬프트의 `tech_id:` 표기와 첫 제목으로 항목을 고른다."""
    import json

    d = Path(fixtures_dir) if fixtures_dir else DEFAULT_FIXTURES_DIR
    profiles = {
        p["tech_id"]: TechProfile.model_validate(p)
        for p in json.loads((d / "tech_profiles.json").read_text(encoding="utf-8"))
    }
    results: dict[tuple[str, str], CriterionResult] = {}
    for name in ("trl_eval.json", "domain_eval.json"):
        for raw in json.loads((d / name).read_text(encoding="utf-8")):
            r = CriterionResult.model_validate(raw)
            results[(r.tech_id, r.criterion_id)] = r

    def _key(text: str) -> tuple[str, str]:
        m = _STUB_TECH_ID.search(text)
        h = _STUB_HEADING.search(text)
        if not m or not h:
            raise LookupError(
                "B stub: 프롬프트에서 tech_id 또는 기준 제목을 찾을 수 없음"
            )
        return m.group(1).lower(), h.group(1)

    def profile(text: str) -> ProfileDraft:
        tech_id, _ = _key(text)
        return profile_to_draft(profiles[tech_id])

    def criterion(text: str) -> CriterionDraft:
        tech_id, cid = _key(text)
        return criterion_to_draft(results[(tech_id, cid)])

    def artifact(text: str) -> ArtifactDraft:
        tech_id, _ = _key(text)
        return criterion_to_artifact_draft(results[(tech_id, "T3")])

    return {ProfileDraft: profile, CriterionDraft: criterion, ArtifactDraft: artifact}
