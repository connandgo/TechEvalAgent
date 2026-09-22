"""보고서 생성 Agent (역할 D). SUMMARY ~ REFERENCE를 챕터(절) 단위로 생성해 조립한다. 추가 검색 없음.

- 챕터마다 별도 프롬프트를 쓰고, 각 프롬프트에는 그 챕터에 필요한 데이터만 넣는다.
- 요약표·측정 조건 표·REFERENCE·1/2장 고정 문단은 코드가 결정적으로 만든다.
- 재생성(judge_result + previous_report_md)이면 문제 챕터만 다시 만들고 나머지는 이전 텍스트를 그대로 둔다.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from techeval.agents._deps import Deps, ReportInput
from techeval.report.citation import (
    CITATION_RE,
    build_evidence_index,
    build_reference_section,
)
from techeval.report.lint import LintResult, lint_report
from techeval.report.sections import Section, join_sections, split_sections
from techeval.report.tables import (
    CRITERION_NAMES,
    PERSPECTIVE_TITLES,
    format_measurement,
    measurement_table,
    perspective_table,
)
from techeval.report.templates import (
    BACKGROUND_TEMPLATE,
    CH4_INTRO,
    SELECTION_REASONS,
    SELECTION_TEMPLATE,
    tech_rows,
)
from techeval.schemas import CriterionResult, Evidence, TechProfile, TechRef

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "report"
REPORT_TITLE = "KV cache 최적화 기술 다관점 평가 보고서"
SURVEY_DOCS = ("io_survey", "kv_survey")
CH4_PERSPECTIVES = {
    "4.1": "trl",
    "4.2": "market",
    "4.3": "stakeholder",
    "4.4": "domain",
}
SUMMARY_CONFLICTS = 4
QUOTE_PREVIEW = 300

# 절 key → 제목. "3"과 "4"는 하위 절을 묶는 제목 전용 절이다.
HEADINGS: dict[str, str] = {
    "SUMMARY": "## SUMMARY",
    "1": "## 1. 분석 배경",
    "2": "## 2. 기술 선정",
    "3": "## 3. 기술 개요",
    "4": "## 4. 관점별 평가",
    "4.1": "### 4.1 기술 성숙도(TRL)",
    "4.2": "### 4.2 시장성",
    "4.3": "### 4.3 이해관계자",
    "4.4": "### 4.4 도메인 적합성",
    "5": "## 5. 시사점",
    "6": "## 6. 한계점",
}
LLM_UNITS: tuple[str, ...] = (
    "SUMMARY",
    "1",
    "2",
    "3.1",
    "3.2",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "5",
    "6",
)
SECTION_ORDER: tuple[str, ...] = (
    "title",
    "SUMMARY",
    "1",
    "2",
    "3",
    "3.1",
    "3.2",
    "4",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "5",
    "6",
)

# 재생성 지시 → 절 매핑. 명시적 챕터 표기가 있으면 그것만, 없으면 키워드로 추정한다.
EXPLICIT_REFS: list[tuple[str, tuple[str, ...]]] = [
    (r"SUMMARY|요약문", ("SUMMARY",)),
    (r"(?<![\d.])3\.1", ("3.1",)),
    (r"(?<![\d.])3\.2", ("3.2",)),
    (r"(?<![\d.])4\.1", ("4.1",)),
    (r"(?<![\d.])4\.2", ("4.2",)),
    (r"(?<![\d.])4\.3", ("4.3",)),
    (r"(?<![\d.])4\.4", ("4.4",)),
    (r"(?<![\d.])1\s*장|(?<![\d.])1\.\s*분석", ("1",)),
    (r"(?<![\d.])2\s*장|(?<![\d.])2\.\s*기술 선정", ("2",)),
    (r"(?<![\d.])3\s*장|(?<![\d.])3\.\s*기술 개요", ("3.1", "3.2")),
    (r"(?<![\d.])4\s*장|(?<![\d.])4\.\s*관점별", ("4.1", "4.2", "4.3", "4.4")),
    (r"(?<![\d.])5\s*장|(?<![\d.])5\.\s*시사점", ("5",)),
    (r"(?<![\d.])6\s*장|(?<![\d.])6\.\s*한계", ("6",)),
]
KEYWORD_REFS: list[tuple[str, tuple[str, ...]]] = [
    (r"분석 배경|배경", ("1",)),
    (r"기술 선정|후보", ("2",)),
    (r"기술 개요|원리", ("3.1", "3.2")),
    (r"\bT[1-4]\b|TRL|기술 성숙도", ("4.1",)),
    (r"\bM[1-3]\b|시장", ("4.2",)),
    (r"\bS[1-4]\b|이해관계자", ("4.3",)),
    (r"\bD[1-4]\b|도메인", ("4.4",)),
    (r"요약표", ("4.1", "4.2", "4.3", "4.4")),
    (r"시사점|상충", ("5",)),
    (r"한계", ("6",)),
]


@dataclass
class Revision:
    previous_text: str
    instructions: list[str]


@dataclass
class _Ctx:
    inp: ReportInput
    deps: Deps
    index: dict[str, Evidence]
    results: list[CriterionResult]
    system: str

    @property
    def techs(self) -> list[TechRef]:
        return self.inp.technologies

    def name(self, tech_id: str) -> str:
        return next((t.name for t in self.techs if t.tech_id == tech_id), tech_id)

    def profile(self, tech_id: str) -> TechProfile | None:
        return next((p for p in self.inp.tech_profiles if p.tech_id == tech_id), None)


# ---------------------------------------------------------------- 공통 유틸


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _response_text(resp: object) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def sanitize_citations(text: str, index: dict[str, Evidence]) -> str:
    """존재하지 않는 evidence_id 인용을 지운다(허위 인용보다 인용 없음이 낫다). 제거는 경고 로그."""

    def repl(m: re.Match) -> str:
        ids = [i.strip() for i in m.group(1).split(",") if i.strip()]
        kept = [i for i in ids if i in index]
        if len(kept) != len(ids):
            logger.warning(
                "존재하지 않는 인용 제거: %s", [i for i in ids if i not in index]
            )
        return f"[E: {', '.join(kept)}]" if kept else ""

    return CITATION_RE.sub(repl, text)


def _clean_llm_text(text: str, index: dict[str, Evidence]) -> str:
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # 절 제목은 코드가 붙인다. LLM이 쓴 #~### 제목은 #### 소제목으로 낮춰 절 구조를 깨지 않게 한다.
    text = re.sub(r"^#{1,3}\s+", "#### ", text, flags=re.MULTILINE)
    return sanitize_citations(text, index)


def _evidence_line(e: Evidence) -> str:
    quote = e.quote if len(e.quote) <= QUOTE_PREVIEW else e.quote[:QUOTE_PREVIEW] + "…"
    extra = ""
    if e.source_type == "not_public":
        extra = f" [not_public: 검색어={e.search_query}, 검색일={e.searched_at}, 범위={e.search_scope}]"
    elif e.source_type == "inference":
        extra = " [inference: 본문에서 '추론'으로 표기]"
    meta = ", ".join(x for x in (e.publisher, e.published_date) if x)
    return f"- {e.evidence_id} ({e.source_type}, unit={e.unit}, {e.locator}{', ' + meta if meta else ''}){extra}: {quote}"


def _result_block(r: CriterionResult) -> str:
    lines = [
        f"#### {r.tech_id} / {r.criterion_id} {CRITERION_NAMES.get(r.criterion_id, '')}",
        f"- level: {r.level}"
        + (f" (추정: {r.level_estimate})" if r.level_estimate else ""),
        f"- confidence: {r.confidence} / evidence_unit: {r.evidence_unit}",
        f"- content: {r.content}",
    ]
    if r.details:
        lines.append(f"- details: {json.dumps(r.details, ensure_ascii=False)}")
    lines += [
        f"- measurement: {format_measurement(m)} [E: {m.evidence_id}]"
        for m in r.measurements
    ]
    lines.append("- evidence:")
    lines += ["  " + _evidence_line(e) for e in r.evidence]
    return "\n".join(lines)


def _llm_text(ctx: _Ctx, prompt_name: str, data: str, revision: Revision | None) -> str:
    human = f"{_load_prompt(prompt_name)}\n\n# 입력 데이터\n\n{data}"
    if revision:
        human += "\n\n" + _load_prompt("revision.md").format(
            previous_text=revision.previous_text.strip() or "(이전 텍스트 없음)",
            instructions="\n".join(f"- {i}" for i in revision.instructions),
        )
    resp = ctx.deps.llm.invoke([("system", ctx.system), ("human", human)])
    return _clean_llm_text(_response_text(resp), ctx.index)


def _cite(ids: list[str]) -> str:
    return f"[E: {', '.join(ids)}]" if ids else ""


# ---------------------------------------------------------------- 절 생성기


def _gen_summary(ctx: _Ctx, revision: Revision | None) -> str:
    syn = ctx.inp.synthesis
    # S4 기반 상충을 기술별로 먼저, 이어서 나머지 순서대로 상위 3~4개
    is_s4 = [("S4" in (c.criterion_a, c.criterion_b)) for c in syn.conflicts]
    ordered = [c for c, s4 in zip(syn.conflicts, is_s4) if s4] + [
        c for c, s4 in zip(syn.conflicts, is_s4) if not s4
    ]
    ordered = ordered[:SUMMARY_CONFLICTS]
    data = "## 상충 지점 (synthesis.conflicts 상위)\n" + "\n".join(
        f"- [{c.tech_id}] {c.perspective_a}/{c.criterion_a} ↔ {c.perspective_b}/{c.criterion_b}: {c.statement} "
        f"(원인: {c.cause}) {_cite(c.evidence_ids)}"
        for c in ordered
    )
    return _llm_text(ctx, "summary.md", data, revision)


def _survey_evidence(ctx: _Ctx) -> list[Evidence]:
    pool = [e for p in ctx.inp.tech_profiles for e in p.evidence] + list(
        ctx.inp.counter_evidence
    )
    return list({e.evidence_id: e for e in pool if e.doc_id in SURVEY_DOCS}.values())


def _gen_background(ctx: _Ctx, revision: Revision | None) -> str:
    text = BACKGROUND_TEMPLATE.format(domain=ctx.inp.domain).strip()
    survey = _survey_evidence(ctx)
    if not survey:
        if revision:
            logger.warning(
                "1장 재생성 지시가 있으나 서베이 근거가 없어 고정 텍스트만 유지"
            )
        return text
    data = "## 서베이 근거 (io_survey / kv_survey)\n" + "\n".join(
        _evidence_line(e) for e in survey
    )
    return f"{text}\n\n{_llm_text(ctx, 'ch1.md', data, revision)}"


def _kv_claim(ctx: _Ctx, tech_id: str) -> str:
    profile = ctx.profile(tech_id)
    m = next(
        (
            m
            for m in (profile.measurements if profile else [])
            if "kv" in m.metric.lower()
        ),
        None,
    )
    return (
        f". 선정 논문은 {format_measurement(m)}를 보고한다[E: {m.evidence_id}]"
        if m
        else ""
    )


def _gen_selection(ctx: _Ctx, revision: Revision | None) -> str:
    reasons = "\n".join(
        SELECTION_REASONS[t.tech_id].format(
            name=t.name, kv_claim=_kv_claim(ctx, t.tech_id)
        )
        for t in ctx.techs
        if t.tech_id in SELECTION_REASONS
    )
    text = SELECTION_TEMPLATE.format(
        tech_rows=tech_rows(ctx.techs), reasons=reasons
    ).strip()
    kv = [e for e in _survey_evidence(ctx) if e.doc_id == "kv_survey"]
    if not kv:
        return text
    data = "## kv_survey 근거\n" + "\n".join(_evidence_line(e) for e in kv)
    return f"{text}\n\n{_llm_text(ctx, 'ch2.md', data, revision)}"


def _gen_overview(tech: TechRef) -> Callable[[_Ctx, Revision | None], str]:
    def gen(ctx: _Ctx, revision: Revision | None) -> str:
        p = ctx.profile(tech.tech_id)
        if p is None:
            logger.warning("3장: %s tech_profile 없음", tech.tech_id)
            return "이 기술의 기술 개요(tech_profile)가 State에 없어 서술하지 않는다."
        data = "\n".join(
            [
                f"## 기술: {tech.name} (tech_id={tech.tech_id}, 선정 논문: {tech.paper_title}, {tech.paper_date})",
                f"- principle: {p.principle}",
                f"- scope: {p.scope}",
                "- limitations:\n" + "\n".join(f"  - {x}" for x in p.limitations),
                f"- validation_env: {p.validation_env}",
                f"- public_artifacts: {json.dumps(p.public_artifacts, ensure_ascii=False)}",
                "- measurements:\n"
                + "\n".join(
                    f"  - {format_measurement(m)} [E: {m.evidence_id}]"
                    for m in p.measurements
                ),
                "## 인용 가능한 evidence\n"
                + "\n".join(_evidence_line(e) for e in p.evidence),
            ]
        )
        text = _llm_text(ctx, "ch3.md", data, revision)
        if p.measurements:
            text += "\n\n**정량 성과와 측정 조건** (서로 다른 실험의 값이므로 행 사이를 직접 비교하지 않는다)\n\n"
            text += measurement_table(p.measurements)
        return text

    return gen


def _gen_perspective(number: str) -> Callable[[_Ctx, Revision | None], str]:
    perspective = CH4_PERSPECTIVES[number]

    def gen(ctx: _Ctx, revision: Revision | None) -> str:
        results = [r for r in ctx.results if r.perspective == perspective]
        blocks = []
        for t in ctx.techs:
            blocks.append(f"### {t.name} (tech_id={t.tech_id}, 계열: {t.family})")
            blocks += [_result_block(r) for r in results if r.tech_id == t.tech_id]
        data = (
            f"## 관점: {PERSPECTIVE_TITLES[perspective]} ({perspective})\n\n"
            + "\n\n".join(blocks)
        )
        text = _llm_text(
            ctx,
            "ch4.md",
            f"{_load_prompt(f'ch4_{perspective}.md')}\n\n{data}",
            revision,
        )
        return f"{text}\n\n{perspective_table(number, perspective, ctx.techs, ctx.results)}"

    return gen


def _gen_implications(ctx: _Ctx, revision: Revision | None) -> str:
    syn = ctx.inp.synthesis
    data = "\n".join(
        [
            "## 상충 (synthesis.conflicts)",
            *(
                f"- [{c.tech_id}] {c.perspective_a}/{c.criterion_a} ↔ {c.perspective_b}/{c.criterion_b}: {c.statement} "
                f"(원인: {c.cause}) {_cite(c.evidence_ids)}"
                for c in syn.conflicts
            ),
            "## 일치 (synthesis.agreements)",
            *(
                f"- [{a.tech_id or '공통'}] {'/'.join(a.perspectives)} {','.join(a.criterion_ids)}: {a.statement} "
                f"{_cite(a.evidence_ids)}"
                for a in syn.agreements
            ),
            f"## 평가 단위 차이 (unit_notes)\n{syn.unit_notes}",
            f"## 근거 비대칭\n{syn.evidence_asymmetry_note}",
        ]
    )
    return _llm_text(ctx, "ch5.md", data, revision)


def _gap_table(ctx: _Ctx) -> str:
    gaps = ctx.inp.synthesis.gaps
    if not gaps:
        return ""
    rows = ["| 기술 | 기준 | 공백 유형 | 내용 |", "|---|---|---|---|"]
    rows += [
        f"| {ctx.name(g.tech_id)} | {g.criterion_id} | {g.gap_type} | {g.description.replace('|', '/')} |"
        for g in gaps
    ]
    return "**남은 근거 공백 (synthesis.gaps)**\n\n" + "\n".join(rows)


def _gen_limitations(ctx: _Ctx, revision: Revision | None) -> str:
    inp = ctx.inp
    gap = inp.evidence_gap
    t1 = [r for r in inp.trl_eval if r.criterion_id == "T1"]
    ratio = ", ".join(
        f"{ctx.name(t)} {v:.1%}" for t, v in gap.vendor_source_ratio.items()
    )
    counts = ", ".join(f"{ctx.name(t)} {n}건" for t, n in gap.evidence_count.items())
    checks = [
        "기술 조사 근거 검사(원리·한계·수치·측정 조건 누락 시 재검색, 상한 2회)",
        "관점별 근거 검사(15기준 × 2기술, evidence 실존·인용 일치, 신뢰도 재계산)",
        f"근거 비대칭 검사(비대칭={gap.asymmetry}, 반대 근거 탐색 필요={gap.needs_counter_search})",
        f"반대 근거 탐색 결과 {len(inp.counter_evidence)}건 반영",
        "보고서 검수(별도 검수 모델, 5차원)"
        + (" — 1회 재생성 반영" if inp.judge_result else ""),
    ]
    data = "\n".join(
        [
            "## (1) TRL 추정 근거 — T1",
            *(
                f"- {r.tech_id}: level={r.level}, 추정={r.level_estimate}, "
                f"올리지 못한 이유={r.details.get('why_not_higher')} {_cite([e.evidence_id for e in r.evidence])}"
                for r in t1
            ),
            f"## (2) 평가 단위 확장 (unit_notes)\n{inp.synthesis.unit_notes}",
            f"## (3) 제안사·벤더(official) 자료 의존 비율 — 그대로 인용\n- vendor_source_ratio: {ratio}\n- evidence 수: {counts}",
            f"- evidence_gap.note: {gap.note}",
            "## (4) 적용한 검사\n" + "\n".join(f"- {c}" for c in checks),
            "## (4) 남은 공백 (synthesis.gaps)\n"
            + "\n".join(
                f"- {g.tech_id} {g.criterion_id} {g.gap_type}: {g.description}"
                for g in inp.synthesis.gaps
            ),
        ]
    )
    text = _llm_text(ctx, "ch6.md", data, revision)
    table = _gap_table(ctx)
    return f"{text}\n\n{table}" if table else text


def _generators(ctx: _Ctx) -> dict[str, Callable[[_Ctx, Revision | None], str]]:
    gens: dict[str, Callable[[_Ctx, Revision | None], str]] = {
        "SUMMARY": _gen_summary,
        "1": _gen_background,
        "2": _gen_selection,
        "5": _gen_implications,
        "6": _gen_limitations,
    }
    for i, tech in enumerate(ctx.techs, start=1):
        gens[f"3.{i}"] = _gen_overview(tech)
    for number in CH4_PERSPECTIVES:
        gens[number] = _gen_perspective(number)
    return gens


def _heading(ctx: _Ctx, key: str) -> str:
    if key.startswith("3."):
        return f"### {key} {ctx.techs[int(key.split('.')[1]) - 1].name}"
    return HEADINGS[key]


def _static_section(ctx: _Ctx, key: str) -> Section:
    if key == "title":
        names = " / ".join(t.name for t in ctx.techs)
        body = f"\n- 평가 도메인: {ctx.inp.domain}\n- 대상 기술: {names}\n- 생성 시각: {ctx.deps.now()}\n\n"
        return Section(key="title", heading=f"# {REPORT_TITLE}\n", body=body)
    body = f"\n{CH4_INTRO}\n\n" if key == "4" else "\n"
    return Section(key=key, heading=_heading(ctx, key) + "\n", body=body)


def _make_section(ctx: _Ctx, key: str, revision: Revision | None) -> Section:
    logger.info("보고서 절 생성: %s%s", key, " (재생성)" if revision else "")
    body = _generators(ctx)[key](ctx, revision)
    return Section(
        key=key, heading=_heading(ctx, key) + "\n", body=f"\n{body.strip()}\n\n"
    )


def _assemble(ctx: _Ctx, sections: dict[str, Section]) -> str:
    body = join_sections([sections[k] for k in SECTION_ORDER if k in sections])
    return body.rstrip() + "\n\n" + build_reference_section(body, ctx.index)


# ---------------------------------------------------------------- 재생성


def map_instructions_to_units(
    instructions: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """지시문 → {절 key: [지시]} 와 매핑하지 못한 지시 목록."""
    mapping: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for ins in instructions:
        units = [u for pat, us in EXPLICIT_REFS if re.search(pat, ins) for u in us]
        if not units:
            units = [u for pat, us in KEYWORD_REFS if re.search(pat, ins) for u in us]
        if not units:
            if re.search(r"REFERENCE|참고문헌", ins):
                continue  # REFERENCE는 매번 본문 인용으로 다시 만든다
            unmapped.append(ins)
        for u in dict.fromkeys(units):
            mapping.setdefault(u, []).append(ins)
    return mapping, unmapped


def _revision_targets(ctx: _Ctx, previous: dict[str, Section]) -> dict[str, list[str]]:
    judge = ctx.inp.judge_result
    instructions = (
        [*judge.revision_instructions, *judge.missing_required] if judge else []
    )
    targets, unmapped = map_instructions_to_units(instructions)

    prev_lint = lint_report(ctx.inp.previous_report_md or "", ctx.index)
    for issue in prev_lint.errors:
        if issue.section in LLM_UNITS:
            targets.setdefault(issue.section, []).append(issue.message)
    for key in LLM_UNITS:
        if key not in previous:
            targets.setdefault(key, []).append("이전 보고서에 이 절이 없어 새로 작성")
    if unmapped:
        fallback = [k for k in targets] or ["SUMMARY", "5"]
        logger.warning(
            "챕터를 특정하지 못한 수정 지시 %d건 → %s에 적용: %s",
            len(unmapped),
            fallback,
            unmapped,
        )
        for key in fallback:
            targets.setdefault(key, []).extend(unmapped)
    return targets


def _regenerate(ctx: _Ctx) -> str:
    previous = {s.key: s for s in split_sections(ctx.inp.previous_report_md or "")}
    targets = _revision_targets(ctx, previous)
    logger.info("보고서 재생성 대상 절: %s", sorted(targets))
    sections: dict[str, Section] = {}
    for key in SECTION_ORDER:
        if key in targets:
            prev_text = previous[key].body if key in previous else ""
            sections[key] = _make_section(ctx, key, Revision(prev_text, targets[key]))
        elif key in previous:
            sections[key] = previous[key]
        else:
            sections[key] = _static_section(ctx, key)
    return _assemble(ctx, sections)


# ---------------------------------------------------------------- 진입점


def _self_fix(
    ctx: _Ctx, sections: dict[str, Section], lint: LintResult
) -> dict[str, Section]:
    """lint 오류(금칙어 등)가 난 LLM 절만 1회 고쳐 쓴다."""
    fixed = dict(sections)
    for key in sorted(lint.sections_with_errors() & set(LLM_UNITS)):
        msgs = [i.message for i in lint.errors if i.section == key]
        fixed[key] = _make_section(ctx, key, Revision(sections[key].body, msgs))
    return fixed


def run_report(inp: ReportInput, deps: Deps) -> str:
    """State의 평가 결과와 근거만으로 보고서 마크다운을 만든다. 파일 저장은 E가 한다."""
    results = [*inp.trl_eval, *inp.market_eval, *inp.stakeholder_eval, *inp.domain_eval]
    index = build_evidence_index(
        results=results,
        profiles=inp.tech_profiles,
        counter_evidence=inp.counter_evidence,
    )
    ctx = _Ctx(
        inp=inp,
        deps=deps,
        index=index,
        results=results,
        system=_load_prompt("system.md"),
    )

    if inp.judge_result is not None and inp.previous_report_md:
        report_md = _regenerate(ctx)
    else:
        if inp.judge_result is not None:
            logger.warning(
                "judge_result는 있으나 previous_report_md가 없어 전체를 새로 생성"
            )
        sections = {
            key: _make_section(ctx, key, None)
            if key in LLM_UNITS
            else _static_section(ctx, key)
            for key in SECTION_ORDER
        }
        report_md = _assemble(ctx, sections)
        lint = lint_report(report_md, index)
        if lint.sections_with_errors() & set(LLM_UNITS):
            logger.warning(
                "lint 오류 → 해당 절 1회 수정: %s", [i.message for i in lint.errors]
            )
            report_md = _assemble(ctx, _self_fix(ctx, sections, lint))

    final = lint_report(report_md, index)
    for issue in final.errors:
        logger.warning("lint 오류 남음: %s", issue.message)
    for issue in final.warnings:
        logger.info("lint 경고: %s", issue.message)
    for item in final.missing_required:
        logger.warning("lint 필수 항목 누락: %s", item)
    return report_md
