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

BANNED_TERMS: tuple[str, ...] = (
    "더 우수",
    "더 낫",
    "추천",
    "권장",
    "1위",
    "순위",
    "종합 점수",
    "총점",
    "평균 레벨",
    "우세",
    "열세",
)
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


def find_banned_terms(text: str) -> list[str]:
    return [t for t in BANNED_TERMS if t in text]


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
                        message=f"[{section.key}] 수치 '{m.group(0)}' 근처에 [E: id] 인용이 없음",
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

    # 5) REFERENCE = 본문 인용 집합 (inference/not_public 제외)
    ref = next((s for s in sections if s.key == "REFERENCE"), None)
    if ref is not None:
        body = "".join(s.text for s in body_sections)
        cited = {
            eid
            for eid in collect_cited_ids(body)
            if eid in evidence_index
            and evidence_index[eid].source_type not in NON_REFERENCE_TYPES
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
            instructions.append(
                f"{i.message}: 우열·추천·순위 표현을 지우고 기준·단위·근거 유형 차이로 다시 서술하라"
            )
        elif i.kind == "unknown_citation":
            instructions.append(f"{i.message}: 입력 evidence 목록에 있는 id만 인용하라")
        else:
            instructions.append(i.message)
    return LintResult(
        errors=errors,
        warnings=warnings,
        missing_required=missing,
        revision_instructions=instructions,
    )
