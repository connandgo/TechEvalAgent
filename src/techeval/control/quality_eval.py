"""보고서 품질 평가 노드 (E). 보고서 생성 직후 4개 항목을 평가하고, 미달 원인(`QualityIssue`)을 supervisor에 넘긴다.

| 항목 | 방식 | 판정 |
|---|---|---|
| groundedness (근거성) | hybrid | 코드: 본문 각주 [E:id]가 근거 인덱스에 있고 REFERENCE는 본문 인용분만 (static_checks + D lint)
|  |  | LLM: judge `evidence` 점수 >= 3 |
| neutrality (중립성) | hybrid | 코드: 우열·추천·순위·합산 금칙어 0건 / LLM: judge `neutrality` 점수 >= 3 |
| bias_control (편향 통제) | code | 근거 확보와 한계 공개를 따로 본다. 문제(근거 수 비대칭·반대 근거 미확보·벤더 자료 편중)마다 |
|  |  | (1) 수치상 문제가 없으면 통과, (2) 문제가 있으면 반대 근거 탐색을 실제로 했는지(확보 시도)와 |
|  |  | (3) 6장 한계점의 같은 문단에 문제 유형 + 해당 기술명(+ 벤더는 비율 수치)을 밝혔는지(공개)를 모두 요구 |
| perspective_coverage (관점 충족) | code | 4.1~4.4 절이 모두 있고, 기술 × 15개 기준 결과 누락 0건 |

LLM 호출은 `judge_report` 1회뿐이다. `passed`는 LLM 값을 믿지 않고 코드가 계산한다:
4개 항목 통과 + 필수 구조(SUMMARY·각 장·REFERENCE, D lint) + 기존 Judge `passed`(5차원 모두 3점 이상)가 모두 참일 때만 True.
기존 보고서 재생성 경로(report agent가 `judge_result.revision_instructions`를 읽음)를 그대로 쓰기 위해
`passed`·`revision_instructions`를 품질 평가 결과로 바꾼 JudgeResult도 함께 돌려준다.
"""

import logging
import re
from typing import Any

from techeval.agents._deps import ReportInput
from techeval.control._common import secured_counter_techs
from techeval.control.judge import _run_lint, evidence_index, judge_report, static_checks
from techeval.schemas import JudgeResult, QualityCheck, QualityIssue, QualityResult, TechRef

logger = logging.getLogger(__name__)

MAX_EVIDENCE_RATIO = 2.0
MAX_VENDOR_RATIO = 0.5
COVERAGE_CHAPTERS: tuple[str, ...] = ("4.1", "4.2", "4.3", "4.4")

_LIMITS_HEADING = re.compile(r"^(#+)\s*6\.\s*한계점.*$", flags=re.MULTILINE)
_CRITERION_MISSING = re.compile(r"([a-z_]+)/([TMSD][1-4])")
_ASYMMETRY_WORDS = re.compile(r"비대칭|편중|불균형")
_COUNTER_WORDS = re.compile(r"반대\s*근거")
_VENDOR_WORDS = re.compile(r"벤더|제안사|자사")
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")


def _limits_section(report_md: str) -> str:
    """6장 한계점 본문 (다음 같은/상위 레벨 제목 전까지). 없으면 빈 문자열."""
    m = _LIMITS_HEADING.search(report_md)
    if not m:
        return ""
    level = len(m.group(1))
    nxt = re.compile(rf"^#{{1,{level}}}\s", flags=re.MULTILINE).search(report_md, m.end())
    return report_md[m.end() : nxt.start() if nxt else len(report_md)]


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _names(tech: TechRef) -> list[str]:
    return [tech.name, tech.tech_id, *tech.search_aliases]


def _disclosed(paras: list[str], words: re.Pattern[str], tech: TechRef, *, need_number: bool = False) -> bool:
    """한계점의 한 문단 안에 문제 유형 키워드와 해당 기술명(필요하면 비율 수치)이 함께 있는지."""
    return any(
        words.search(p) and any(n in p for n in _names(tech)) and (not need_number or _PERCENT.search(p)) for p in paras
    )


def _check_bias(
    report_md: str, inp: ReportInput, counter_attempts: int
) -> tuple[QualityCheck, list[str], list[QualityIssue]]:
    gap = inp.evidence_gap
    paras = _paragraphs(_limits_section(report_md))
    techs = {t.tech_id: t for t in inp.technologies}
    attempted = counter_attempts > 0
    instructions: list[str] = []
    issues: list[QualityIssue] = []
    notes: list[str] = []

    def problem(cause: str, tech: TechRef, ok_disclosed: bool, what: str, instr: str) -> None:
        """확보 시도 전이면 원인 그대로(supervisor가 반대 근거 탐색), 시도 후 미공개면 undisclosed(보고서 수정)."""
        if attempted and ok_disclosed:  # 확보 성공이 아니라 '탐색했으나 미확보 + 공개'로 통과
            notes.append(f"{what}: 탐색했으나 미확보, 한계점 공개됨(확보 아님)")
            return
        if not attempted:  # 탐색 후 재작성 기회가 1회뿐이므로 공개 지시도 미리 넘긴다
            notes.append(f"{what}: 반대 근거 탐색 전")
            issues.append(QualityIssue(criterion="bias_control", cause=cause, tech_id=tech.tech_id, detail=what))
            if not ok_disclosed:
                instructions.append(instr)
            return
        notes.append(f"{what}: 한계점 공개 안 됨")
        issues.append(QualityIssue(criterion="bias_control", cause="undisclosed", tech_id=tech.tech_id, detail=what))
        instructions.append(instr)

    counts = {tid: gap.evidence_count.get(tid, 0) for tid in techs}
    lo_tid = min(counts, key=counts.get) if counts else None
    lo, hi = (min(counts.values()), max(counts.values())) if counts else (0, 0)
    ratio = hi / lo if lo else (float("inf") if hi else 1.0)
    if lo_tid is not None and ratio > MAX_EVIDENCE_RATIO:
        t = techs[lo_tid]
        problem(
            "asymmetry",
            t,
            _disclosed(paras, _ASYMMETRY_WORDS, t),
            f"근거 수 {counts} 비율 {ratio:.2f}:1",
            f"6장 한계점에 {t.name}의 근거 수가 적다는 비대칭({counts}, {ratio:.2f}:1)을 기술명과 함께 한 문단으로 명시하라.",
        )

    secured = secured_counter_techs(inp.counter_evidence)  # not_public·inference COUNTER는 확보로 보지 않음
    unresolved: dict[str, list[str]] = {}
    for o in gap.opposing_missing:
        tid = str(o.get("tech_id"))
        if tid in techs and tid not in secured:
            unresolved.setdefault(tid, []).append(str(o.get("criterion_id")))
    for tid, cids in unresolved.items():
        t = techs[tid]
        problem(
            "counter_missing",
            t,
            _disclosed(paras, _COUNTER_WORDS, t),
            f"{tid} 반대 근거 미확보 {cids}",
            f"6장 한계점에 {t.name}의 반대 근거 미확보 기준({', '.join(cids)})을 기술명과 함께 한 문단으로 명시하라.",
        )

    for tid, v in gap.vendor_source_ratio.items():
        if v > MAX_VENDOR_RATIO and tid in techs:
            t = techs[tid]
            problem(
                "vendor_heavy",
                t,
                _disclosed(paras, _VENDOR_WORDS, t, need_number=True),
                f"{tid} 벤더·제안사 자료 비율 {v:.0%}",
                f"6장 한계점에 {t.name}의 벤더·제안사 자료 비율({v:.1%})을 기술명·수치와 함께 한 문단으로 명시하라.",
            )

    detail = (
        f"반대 근거 탐색 {counter_attempts}회, 실제 확보 기술 {sorted(secured) or '없음'}; 근거 수 {counts} (상한 {MAX_EVIDENCE_RATIO}:1); "
        f"벤더 비율 {gap.vendor_source_ratio} (상한 {MAX_VENDOR_RATIO:.0%}); " + ("; ".join(notes) or "문제 없음")
    )
    check = QualityCheck(criterion="bias_control", passed=not issues, method="code", detail=detail)
    return check, instructions, issues


def evaluate_report_quality(
    report_md: str, inp: ReportInput, judge_llm: Any, now: str, *, counter_attempts: int = 0
) -> tuple[QualityResult, JudgeResult]:
    """4개 항목 품질 평가. 반환: (QualityResult, 재생성 경로용으로 passed·지시를 바꾼 JudgeResult).

    `counter_attempts`: 지금까지 반대 근거 탐색을 실행한 횟수 (State `retry_counts["counter"]`).
    """
    judge = judge_report(report_md, inp, judge_llm, now)
    missing, neutrality_hits, ev_problems = static_checks(report_md, inp)
    lint_missing, lint_errors = _run_lint(report_md, evidence_index(inp))
    neutrality_hits = [*neutrality_hits, *(e for e in lint_errors if "banned_term" in e)]
    ev_problems = [*ev_problems, *(e for e in lint_errors if "banned_term" not in e)]

    checks: dict[str, QualityCheck] = {}
    instructions: list[str] = []
    issues: list[QualityIssue] = []

    ev_score = judge.scores.get("evidence", 1)
    ok = not ev_problems and ev_score >= 3
    checks["groundedness"] = QualityCheck(
        criterion="groundedness",
        passed=ok,
        method="hybrid",
        detail=f"코드: 각주·REFERENCE 위반 {len(ev_problems)}건 {ev_problems[:3]}; LLM evidence 점수 {ev_score}",
    )
    if ev_problems:
        issues.append(QualityIssue(criterion="groundedness", cause="citation", detail="; ".join(ev_problems[:3])))
        instructions.append(
            "근거 인덱스에 없는 각주를 제거하고 REFERENCE를 본문 인용분으로 한정하라: " + " / ".join(ev_problems[:5])
        )
    elif not ok:
        issues.append(
            QualityIssue(criterion="groundedness", cause="weak_evidence", detail=judge.reasons.get("evidence", ""))
        )
        instructions.append(
            "판정 문장마다 [E: evidence_id] 각주를 달아 근거와 연결하라: " + judge.reasons.get("evidence", "")
        )

    neu_score = judge.scores.get("neutrality", 1)
    ok = not neutrality_hits and neu_score >= 3
    checks["neutrality"] = QualityCheck(
        criterion="neutrality",
        passed=ok,
        method="hybrid",
        detail=f"코드: 금칙어 {len(neutrality_hits)}건 {neutrality_hits[:3]}; LLM neutrality 점수 {neu_score}",
    )
    if neutrality_hits:
        issues.append(QualityIssue(criterion="neutrality", cause="banned_term", detail="; ".join(neutrality_hits[:3])))
        instructions.append(
            "우열·추천·순위·합산 표현을 제거하고 기준·단위·근거 유형 차이로만 서술하라: "
            + " / ".join(neutrality_hits[:5])
        )
    elif not ok:
        issues.append(QualityIssue(criterion="neutrality", cause="tone", detail=judge.reasons.get("neutrality", "")))
        instructions.append("관점별 레벨을 비교·평가하는 어조를 제거하라: " + judge.reasons.get("neutrality", ""))

    bias_check, bias_instr, bias_issues = _check_bias(report_md, inp, counter_attempts)
    checks["bias_control"] = bias_check
    instructions.extend(bias_instr)
    issues.extend(bias_issues)

    missing_chapters = [c for c in COVERAGE_CHAPTERS if c in missing]
    missing_criteria = [m for m in missing if _CRITERION_MISSING.fullmatch(m)]
    checks["perspective_coverage"] = QualityCheck(
        criterion="perspective_coverage",
        passed=not missing_chapters and not missing_criteria,
        method="code",
        detail=f"누락 절 {missing_chapters or '없음'}; 누락 기준 결과 {len(missing_criteria)}건 {missing_criteria[:5]}",
    )
    for c in missing_chapters:
        issues.append(QualityIssue(criterion="perspective_coverage", cause="missing_section", detail=c))
    if missing_chapters:
        instructions.append(
            f"4장 관점별 평가에 누락된 절을 추가하라(기술×기준 요약표 포함): {', '.join(missing_chapters)}"
        )
    for m in missing_criteria:  # static_checks는 입력(State)에 결과가 없는 기술×기준을 올린다
        tid, cid = _CRITERION_MISSING.fullmatch(m).groups()  # type: ignore[union-attr]
        issues.append(
            QualityIssue(criterion="perspective_coverage", cause="missing_result", tech_id=tid, criterion_id=cid)
        )

    # 4개 항목과 별개로, 기존 judge가 강제하던 구조 요건(SUMMARY·각 장·REFERENCE, D lint 필수 항목)도 게이트에 남긴다
    structural_missing = [
        m for m in [*missing, *lint_missing] if m not in COVERAGE_CHAPTERS and not _CRITERION_MISSING.fullmatch(m)
    ]
    structural_missing = list(dict.fromkeys(structural_missing))
    if structural_missing:
        issues.append(QualityIssue(criterion="structure", cause="structure", detail=", ".join(structural_missing)))
        instructions.append(f"누락된 필수 챕터·요약표를 추가하라: {', '.join(structural_missing)}")

    # 기존 Judge(5차원: evidence·criteria_compliance·neutrality·specificity·format)의 탈락도 그대로 탈락으로 둔다
    low_dims = [d for d, s in judge.scores.items() if s < 3]
    for d in low_dims:
        issues.append(
            QualityIssue(
                criterion="judge", cause="legacy_judge", detail=f"{d}={judge.scores[d]}: {judge.reasons.get(d, '')}"
            )
        )
    if not judge.passed and not low_dims and not structural_missing:  # LLM이 올린 필수 항목 누락만 남은 경우
        issues.append(
            QualityIssue(criterion="judge", cause="legacy_judge", detail=f"missing_required={judge.missing_required}")
        )

    passed = all(c.passed for c in checks.values()) and not structural_missing and judge.passed
    if not passed:
        # LLM judge의 세부 수정 지시도 전달 (중복 제거)
        instructions.extend(i for i in judge.revision_instructions if i not in instructions)

    quality = QualityResult(
        checks=checks,
        passed=passed,
        revision_instructions=instructions if not passed else [],
        evaluated_at=now,
        legacy_judge_passed=judge.passed,
        issues=issues,
    )
    logger.info(
        "quality_eval: %s 구조 누락=%s 기존 judge=%s passed=%s",
        {k: c.passed for k, c in checks.items()},
        structural_missing,
        judge.passed,
        passed,
    )
    adapted = judge.model_copy(update={"passed": passed, "revision_instructions": quality.revision_instructions})
    return quality, adapted
