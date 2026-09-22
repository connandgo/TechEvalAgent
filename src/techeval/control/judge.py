"""보고서 검수 노드 (E). CRITERIA §6 5차원 1/3/5.

`deps.judge_llm`(JUDGE_MODEL)로 채점하고, 코드로 강제 가능한 규칙(V9~V11, 필수 챕터)은 코드가 덮어쓴다.
`passed`는 LLM 값을 믿지 않고 `all(score >= 3) and not missing_required`로 재계산한다.
"""

import logging
import re
from pathlib import Path
from typing import Any

from techeval.agents._deps import ReportInput
from techeval.schemas import JUDGE_DIMENSIONS, CriterionResult, Evidence, JudgeResult

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "control" / "judge.md"

# V10 금칙어 (우열·추천·순위). D의 lint와 독립적으로 judge에서도 검사한다.
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"더\s*우수",
    r"더\s*뛰어나",
    r"우월",
    r"추천",
    r"권장",
    r"\b1\s*위\b",
    r"[0-9]\s*위(?![가-힣])",
    r"순위",
    r"종합\s*점수",
    r"총점",
)
# V11 합산·평균
AGGREGATE_PATTERNS: tuple[str, ...] = (r"합산", r"평균\s*(점수|레벨)", r"레벨\s*(합|평균)")

REQUIRED_CHAPTERS: tuple[tuple[str, str], ...] = (
    ("SUMMARY", r"^#+\s*SUMMARY"),
    ("1. 분석 배경", r"^#+\s*1\.\s*분석\s*배경"),
    ("2. 기술 선정", r"^#+\s*2\.\s*기술\s*선정"),
    ("3. 기술 개요", r"^#+\s*3\.\s*기술\s*개요"),
    ("4. 관점별 평가", r"^#+\s*4\.\s*관점별\s*평가"),
    ("4.1", r"^#+\s*4\.1"),
    ("4.2", r"^#+\s*4\.2"),
    ("4.3", r"^#+\s*4\.3"),
    ("4.4", r"^#+\s*4\.4"),
    ("5. 시사점", r"^#+\s*5\.\s*시사점"),
    ("6. 한계점", r"^#+\s*6\.\s*한계점"),
    ("REFERENCE", r"^#+\s*REFERENCE"),
)

_CITATION = re.compile(r"\[E:\s*([A-Za-z0-9_\-]+)\]")


def _all_results(inp: ReportInput) -> list[CriterionResult]:
    return [*inp.trl_eval, *inp.market_eval, *inp.stakeholder_eval, *inp.domain_eval]


def evidence_index(inp: ReportInput) -> dict[str, Evidence]:
    idx: dict[str, Evidence] = {}
    for r in _all_results(inp):
        for e in r.evidence:
            idx.setdefault(e.evidence_id, e)
    for p in inp.tech_profiles:
        for e in p.evidence:
            idx.setdefault(e.evidence_id, e)
    for e in inp.counter_evidence:
        idx.setdefault(e.evidence_id, e)
    return idx


def cited_ids(report_md: str) -> set[str]:
    return set(_CITATION.findall(report_md))


def static_checks(report_md: str, inp: ReportInput) -> tuple[list[str], list[str], list[str]]:
    """코드로 판정 가능한 위반. (missing_required, neutrality_hits, evidence_problems)."""
    missing: list[str] = []
    for name, pat in REQUIRED_CHAPTERS:
        if not re.search(pat, report_md, flags=re.MULTILINE | re.IGNORECASE):
            missing.append(name)

    # V8 대응: 본문에 30개 (tech, criterion) 결과가 모두 반영됐는지는 LLM이 보고, 여기서는 입력 자체의 누락만 본다
    present = {(r.tech_id, r.criterion_id) for r in _all_results(inp)}
    for t in inp.technologies:
        for cid in ("T1", "T2", "T3", "T4", "M1", "M2", "M3", "S1", "S2", "S3", "S4", "D1", "D2", "D3", "D4"):
            if (t.tech_id, cid) not in present:
                missing.append(f"{t.tech_id}/{cid}")

    neutrality_hits: list[str] = []
    for line in report_md.splitlines():
        for pat in (*FORBIDDEN_PATTERNS, *AGGREGATE_PATTERNS):
            if re.search(pat, line):
                neutrality_hits.append(f"{pat!r}: {line.strip()[:120]}")
                break

    # V9: 본문 각주가 근거 인덱스에 있어야 하고, REFERENCE에는 본문 인용분만
    idx = evidence_index(inp)
    cited = cited_ids(report_md)
    problems: list[str] = [f"본문 각주 {eid}가 근거 인덱스에 없음" for eid in sorted(cited - set(idx))]
    ref_match = re.search(r"^#+\s*REFERENCE.*$", report_md, flags=re.MULTILINE | re.IGNORECASE)
    if ref_match:
        ref_section = report_md[ref_match.end() :]
        ref_ids = set(re.findall(r"\b([a-z_]+-(?:[TMSD][1-4]|PROFILE|COUNTER)-[0-9A-Z]+)\b", ref_section))
        uncited = sorted(ref_ids - cited)
        if uncited:
            problems.append("REFERENCE에 본문 미인용 항목: " + ", ".join(uncited))
    return missing, neutrality_hits, problems


def _eval_summary(inp: ReportInput) -> str:
    lines = []
    for r in sorted(_all_results(inp), key=lambda r: (r.tech_id, r.perspective, r.criterion_id)):
        ids = ", ".join(e.evidence_id for e in r.evidence)
        lines.append(f"- {r.tech_id} {r.criterion_id}: level={r.level} confidence={r.confidence} evidence=[{ids}]")
    return "\n".join(lines)


def _evidence_index_text(inp: ReportInput) -> str:
    lines = []
    for eid, e in sorted(evidence_index(inp).items()):
        src = e.url or e.locator
        lines.append(f"- {eid}: [{e.source_type}/{e.unit}] {e.title or ''} — {src}")
    return "\n".join(lines)


def _run_lint(report_md: str, idx: dict[str, Evidence]) -> tuple[list[str], list[str]]:
    """D의 `report.lint.lint_report(report_md, evidence_index) -> LintResult`. 없으면 빈 결과.

    반환: (missing_required, errors 메시지). errors(금칙어·미인용 각주 등)는 judge 점수를 1점으로 내린다.
    """
    try:
        from techeval.report.lint import lint_report
    except ImportError:
        return [], []
    out = lint_report(report_md, idx)
    missing = [str(m) for m in out.missing_required]
    errors = [f"[{i.section}] {i.kind}: {i.message}" for i in out.errors]
    return missing, errors


def judge_report(report_md: str, inp: ReportInput, judge_llm: Any, now: str) -> JudgeResult:
    """JUDGE_MODEL로 채점 → 코드 검사 결과로 덮어쓰기 → passed 재계산."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        now=now,
        eval_summary=_eval_summary(inp),
        evidence_index=_evidence_index_text(inp),
        evidence_gap=inp.evidence_gap.model_dump_json(indent=1),
        report_md=report_md,
    )
    raw = judge_llm.with_structured_output(JudgeResult).invoke(prompt)
    result = JudgeResult.model_validate(raw if isinstance(raw, dict) else raw.model_dump())

    scores = dict(result.scores)
    reasons = dict(result.reasons)
    missing = list(result.missing_required)
    instructions = list(result.revision_instructions)

    static_missing, neutrality_hits, ev_problems = static_checks(report_md, inp)
    lint_missing, lint_errors = _run_lint(report_md, evidence_index(inp))
    for m in [*static_missing, *lint_missing]:
        if m not in missing:
            missing.append(m)
    if lint_errors:
        # D lint의 errors: banned_term → 중립성, unknown_citation/reference_mismatch → 근거성
        banned = [e for e in lint_errors if "banned_term" in e]
        cites = [e for e in lint_errors if "banned_term" not in e]
        if banned:
            neutrality_hits = [*neutrality_hits, *banned]
        if cites:
            ev_problems = [*ev_problems, *cites]
    # 누락 항목을 차원별로 분리: 챕터·요약표 → format, "<tech>/<criterion>" 기준 누락 → criteria_compliance
    all_missing = [*static_missing, *lint_missing]
    criteria_missing = [m for m in all_missing if re.fullmatch(r"[a-z_]+/[TMSD][1-4]", m)]
    format_missing = [m for m in all_missing if m not in criteria_missing]
    if format_missing:
        scores["format"] = 1
        reasons["format"] = (
            reasons.get("format", "") + f" [코드 검사] 필수 챕터·요약표 누락: {format_missing}"
        ).strip()
        instructions.append(f"누락된 필수 챕터·요약표를 추가하라: {', '.join(format_missing)}")
    if criteria_missing:
        scores["criteria_compliance"] = 1
        reasons["criteria_compliance"] = (
            reasons.get("criteria_compliance", "") + f" [코드 검사] 기준 결과 누락: {criteria_missing}"
        ).strip()
        instructions.append(f"누락된 기준 결과를 반영하라: {', '.join(criteria_missing)}")
    if neutrality_hits:
        scores["neutrality"] = 1
        reasons["neutrality"] = (
            reasons.get("neutrality", "")
            + f" [코드 검사] 금칙어 {len(neutrality_hits)}건: "
            + " / ".join(neutrality_hits[:5])
        ).strip()
        instructions.append(
            "우열·추천·순위·합산 표현을 제거하고 기준·단위·근거 유형 차이로만 서술하라: "
            + " / ".join(neutrality_hits[:5])
        )
    if ev_problems:
        scores["evidence"] = 1
        reasons["evidence"] = (reasons.get("evidence", "") + " [코드 검사] " + " / ".join(ev_problems)).strip()
        instructions.append(
            "근거 인덱스에 없는 각주를 제거하고 REFERENCE를 본문 인용분으로 한정하라: " + " / ".join(ev_problems)
        )

    for dim in JUDGE_DIMENSIONS:
        scores.setdefault(dim, 1)
        reasons.setdefault(dim, "")
    passed = all(v >= 3 for v in scores.values()) and not missing
    logger.info("judge: scores=%s missing=%s passed=%s", scores, missing, passed)
    return JudgeResult(
        scores=scores,
        reasons=reasons,
        missing_required=missing,
        revision_instructions=instructions,
        passed=passed,
        judged_at=now,
    )
