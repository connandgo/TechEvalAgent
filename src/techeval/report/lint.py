"""judge 전 자체 검사 (결정적). 위반은 예외가 아니라 재생성 지시 목록으로 돌려준다."""

import re

from pydantic import BaseModel

from techeval.report.citation import (
    CITATION_RE,
    NON_REFERENCE_TYPES,
    collect_cited_ids,
    reference_ids,
)
from techeval.report.sections import (
    CH4_SUBSECTIONS,
    REQUIRED_CHAPTERS,
    Section,
    split_sections,
)
from techeval.report.tables import TABLE_HEADER
from techeval.schemas import Evidence

# 금칙어 (역할 문서 §5.3 + E의 control/judge.py FORBIDDEN/AGGREGATE 패턴). 부정문("합산하지 않는다")도 걸린다.
BANNED_PATTERNS: tuple[str, ...] = (
    r"더\s*우수",
    r"더\s*낫",
    r"더\s*뛰어나",
    r"우월",
    r"우세",
    r"열세",
    r"추천",
    r"권장",
    r"[0-9]\s*위(?![가-힣])",
    r"순위",
    r"종합\s*점수",
    r"총점",
    r"합산",
    r"평균\s*(점수|레벨)",
    r"레벨\s*(합|평균)",
)
_BANNED_RE = re.compile("|".join(f"(?:{p})" for p in BANNED_PATTERNS))
CHAPTER_TITLES: dict[str, str] = {
    "SUMMARY": "SUMMARY",
    "1": "1. 분석 배경",
    "2": "2. 기술 선정",
    "3": "3. 기술 개요",
    "4": "4. 관점별 평가",
    "5": "5. 시사점",
    "6": "6. 한계점",
    "REFERENCE": "REFERENCE",
}
# 수치 패턴: 93.3%, 5.76x, 2.1배. 앞이 영숫자인 경우(예: 'H800')는 제외.
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s?(?:%|x\b|×|배)")
NEAR_CHARS = 20
# SUMMARY: 전체 보고서의 핵심 요약, A4 반 페이지 이내, 도입(인트로) 문장으로 시작 금지.
# 반 페이지 ≈ 10pt 본문 폭 기준 700자 안팎(인용 표기·마크다운 제외). 분량은 judge 채점 대상이 아니므로(CRITERIA §6)
# errors가 아니라 warnings로 두고, run_report가 이 경고를 보고 SUMMARY만 1회 다시 쓴다.
SUMMARY_MAX_CHARS = 700
SUMMARY_INTRO_RE = re.compile(r"^(?:본|이|이번|해당)\s*(?:보고서|요약|평가|문서|절|장)|^(?:다음은|아래는|요약하면)")
FIXABLE_WARNING_KINDS: tuple[str, ...] = ("summary_too_long", "summary_intro", "summary_uncited")
# 인용 없는 수치를 D가 스스로 고칠 절. 6장 벤더 비율은 evidence가 아니라 E 검사 산출값이라 제외한다.
NUMBER_FIX_SECTIONS: tuple[str, ...] = ("SUMMARY", "3.1", "3.2", "4.1", "4.2", "4.3", "4.4", "5")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class LintIssue(BaseModel):
    section: str  # Section.key ("4.2", "REFERENCE" ...)
    kind: str  # banned_term / unknown_citation / reference_mismatch / number_without_citation
    message: str


class LintResult(BaseModel):
    errors: list[LintIssue]
    warnings: list[LintIssue]
    missing_required: list[str]
    revision_instructions: list[str]

    @property
    def passed(self) -> bool:
        return not self.errors and not self.missing_required

    def sections_with_errors(self) -> set[str]:
        return {i.section for i in self.errors}

    def fix_issues(self) -> list[LintIssue]:
        """D가 스스로 고칠 문제: 오류 전부 + SUMMARY 분량·인트로·무인용 경고 + 본문 절의 인용 없는 수치."""
        fixable = [
            w
            for w in self.warnings
            if w.kind in FIXABLE_WARNING_KINDS
            or (w.kind == "number_without_citation" and w.section in NUMBER_FIX_SECTIONS)
        ]
        return [*self.errors, *fixable]

    def sections_to_fix(self) -> set[str]:
        return {i.section for i in self.fix_issues()}


def find_banned_terms(text: str) -> list[str]:
    """검출된 금칙어 표현(원문 그대로)을 중복 없이 반환한다."""
    return list(dict.fromkeys(m.group(0) for m in _BANNED_RE.finditer(text)))


def _number_warnings(section: Section) -> list[LintIssue]:
    """수치 근처(앞 20자 ~ 같은 문장 끝)에 [E: 인용이 없으면 경고.
    역할 문서는 '20자 내'로 정했지만 조건 병기 괄호가 길어지면 인용이 문장 끝에 오므로 문장 끝까지 본다."""
    issues = []
    for line in section.body.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for m in NUMBER_RE.finditer(line):
            end = re.search(r"\.(?=\s|$)", line[m.end() :])
            stop = m.end() + end.end() if end else len(line)
            window = line[max(0, m.start() - NEAR_CHARS) : stop]
            if "[E:" not in window:
                issues.append(
                    LintIssue(
                        section=section.key,
                        kind="number_without_citation",
                        message=f"[{section.key}] 수치 '{m.group(0)}' 근처에 [E: id] 인용이 없음 — "
                        "수치가 나온 그 문장 끝에 해당 measurement의 [E: id]를 붙여라(다음 문장으로 미루지 말 것)",
                    )
                )
    return issues


def summary_plain_text(body: str) -> str:
    """인용 표기·마크다운 기호를 뺀 SUMMARY 본문."""
    text = CITATION_RE.sub("", body)
    text = re.sub(r"[*_`>#]|^\s*[-\d.]+\s", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def _summary_warnings(section: Section) -> list[LintIssue]:
    text = summary_plain_text(section.body)
    issues = []
    if len(text) > SUMMARY_MAX_CHARS:
        issues.append(
            LintIssue(
                section="SUMMARY",
                kind="summary_too_long",
                message=f"[SUMMARY] {len(text)}자로 A4 반 페이지 기준({SUMMARY_MAX_CHARS}자)을 넘음 — 핵심 결론만 남겨 줄여라",
            )
        )
    # 문장마다 인용이 있어야 한다. 첫 문장에 인용이 없으면 총론·도입 문장으로 본다("두 기술은 … 평가를 받는다").
    body = re.sub(r"\s+", " ", section.body).strip()
    sentences = [s for s in SENTENCE_SPLIT_RE.split(body) if s.strip()]
    uncited = [s for s in sentences if "[E:" not in s]
    first_uncited = bool(sentences) and "[E:" not in sentences[0]
    if SUMMARY_INTRO_RE.match(text) or first_uncited:
        issues.append(
            LintIssue(
                section="SUMMARY",
                kind="summary_intro",
                message=f"[SUMMARY] 도입·총론 문장으로 시작함('{text[:30]}…') — 첫 문장부터 근거 있는 핵심 결론을 써라",
            )
        )
    rest = [s for s in uncited if not (first_uncited and s == sentences[0])]
    if rest:
        issues.append(
            LintIssue(
                section="SUMMARY",
                kind="summary_uncited",
                message=f"[SUMMARY] 인용 없는 문장 {len(rest)}개('{summary_plain_text(rest[0])[:30]}…') — "
                "문장마다 [E: id]를 붙이거나 근거 없는 일반론은 지워라",
            )
        )
    return issues


def lint_report(report_md: str, evidence_index: dict[str, Evidence]) -> LintResult:
    sections = split_sections(report_md)
    keys = {s.key for s in sections}
    errors: list[LintIssue] = []
    warnings: list[LintIssue] = []
    missing: list[str] = []

    # 1) 필수 챕터 7개 + REFERENCE
    for ch in REQUIRED_CHAPTERS:
        if ch not in keys:
            missing.append(f"챕터 누락: {CHAPTER_TITLES[ch]}")

    # 2) 4장 절별 요약표
    for sub in CH4_SUBSECTIONS:
        sec = next((s for s in sections if s.key == sub), None)
        if sec is None or TABLE_HEADER not in sec.body:
            missing.append(f"요약표 누락: {sub}")

    body_sections = [s for s in sections if s.key != "REFERENCE"]
    for s in body_sections:
        # 3) 금칙어
        for term in find_banned_terms(s.text):
            errors.append(
                LintIssue(
                    section=s.key,
                    kind="banned_term",
                    message=f"[{s.key}] 금칙어 '{term}' 사용",
                )
            )
        # 4) 존재하지 않는 evidence_id
        for eid in collect_cited_ids(s.text):
            if eid not in evidence_index:
                errors.append(
                    LintIssue(
                        section=s.key,
                        kind="unknown_citation",
                        message=f"[{s.key}] 존재하지 않는 evidence_id 인용: {eid}",
                    )
                )
        warnings += _number_warnings(s)
        if s.key == "SUMMARY":
            warnings += _summary_warnings(s)

    # 5) REFERENCE = 본문 인용 집합 (inference/not_public 제외)
    ref = next((s for s in sections if s.key == "REFERENCE"), None)
    if ref is not None:
        body = "".join(s.text for s in body_sections)
        cited = {
            eid
            for eid in collect_cited_ids(body)
            if eid in evidence_index and evidence_index[eid].source_type not in NON_REFERENCE_TYPES
        }
        listed = set(reference_ids(ref.body))
        if CITATION_RE.search(ref.body):
            errors.append(
                LintIssue(
                    section="REFERENCE",
                    kind="reference_mismatch",
                    message="REFERENCE 안에 [E: ] 인용 표기가 있음",
                )
            )
        if cited - listed:
            errors.append(
                LintIssue(
                    section="REFERENCE",
                    kind="reference_mismatch",
                    message=f"본문 인용이 REFERENCE에 없음: {sorted(cited - listed)}",
                )
            )
        if listed - cited:
            errors.append(
                LintIssue(
                    section="REFERENCE",
                    kind="reference_mismatch",
                    message=f"REFERENCE에 본문 미인용 항목: {sorted(listed - cited)}",
                )
            )

    instructions = [f"{m} — 해당 챕터를 추가하라" for m in missing]
    for i in errors:
        if i.kind == "banned_term":
            instructions.append(f"{i.message}: 우열·추천·순위 표현을 지우고 기준·단위·근거 유형 차이로 다시 서술하라")
        elif i.kind == "unknown_citation":
            instructions.append(f"{i.message}: 입력 evidence 목록에 있는 id만 인용하라")
        else:
            instructions.append(i.message)
    instructions += [w.message for w in warnings if w.kind in FIXABLE_WARNING_KINDS]
    return LintResult(
        errors=errors,
        warnings=warnings,
        missing_required=missing,
        revision_instructions=instructions,
    )
