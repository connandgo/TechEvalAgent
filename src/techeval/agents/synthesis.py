"""평가 종합 Agent (역할 D). 4관점 결과의 일치·상충·공백을 연결한다. 추가 검색을 하지 않는다."""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from techeval.agents._deps import Deps, SynthesisInput
from techeval.report.citation import build_evidence_index
from techeval.report.lint import find_banned_terms
from techeval.schemas import (
    PERSPECTIVE_CRITERIA,
    Agreement,
    Conflict,
    CriterionResult,
    Evidence,
    Gap,
    SynthesisResult,
    TechProfile,
)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "synthesis"
MAX_LLM_ATTEMPTS = 2
# 관점별 기대 평가 단위 (CRITERIA §1). domain은 원문 근거가 기본이지만 계열 근거도 허용되므로 혼합 여부만 본다.
EXPECTED_UNIT = {"trl": "paper", "market": "family", "stakeholder": "family"}
QUOTE_PREVIEW = 240


class SynthesisDraft(BaseModel):
    """LLM 구조화 출력. 코드 검증·보강을 거쳐 SynthesisResult가 된다."""

    agreements: list[Agreement] = []
    conflicts: list[Conflict] = []
    gaps: list[Gap] = []
    unit_notes: str
    evidence_asymmetry_note: str


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _all_results(inp: SynthesisInput) -> list[CriterionResult]:
    return [*inp.trl_eval, *inp.market_eval, *inp.stakeholder_eval, *inp.domain_eval]


def build_matrix(
    results: list[CriterionResult],
) -> dict[tuple[str, str, str], CriterionResult]:
    """(tech_id, perspective, criterion_id) → CriterionResult."""
    return {(r.tech_id, r.perspective, r.criterion_id): r for r in results}


def _source_count(r: CriterionResult) -> int:
    quoted = [e for e in r.evidence if e.source_type not in ("inference", "not_public")]
    return len({e.doc_id or e.url or e.locator for e in quoted})


def detect_gaps(results: list[CriterionResult]) -> list[Gap]:
    """규칙으로 판정 가능한 공백. opposing_missing은 근거 방향 판단이 필요해 LLM이 채운다."""
    gaps: list[Gap] = []
    for r in results:
        types = {e.source_type for e in r.evidence}
        units = {e.unit for e in r.evidence}
        if r.level == "not_public":
            gaps.append(
                Gap(
                    tech_id=r.tech_id,
                    criterion_id=r.criterion_id,
                    gap_type="not_public",
                    description=f"{r.criterion_id} 판정 자체가 not_public(공개 근거 미확인)",
                )
            )
        elif "not_public" in types or r.details.get("not_public_items"):
            # 일부 항목만 미공개: not_public evidence로 남기거나, T4처럼 details.not_public_items에 적는다(CRITERIA §2.1)
            items = r.details.get("not_public_items") or []
            gaps.append(
                Gap(
                    tech_id=r.tech_id,
                    criterion_id=r.criterion_id,
                    gap_type="not_public",
                    description=f"{r.criterion_id} 판정 일부 항목이 not_public으로 기록됨"
                    + (f": {'; '.join(items)}" if items else ""),
                )
            )
        if types <= {"inference", "not_public"} and "inference" in types:
            gaps.append(
                Gap(
                    tech_id=r.tech_id,
                    criterion_id=r.criterion_id,
                    gap_type="inference_only",
                    description=f"{r.criterion_id} 근거가 추론(inference)뿐이고 직접 인용 자료가 없음",
                )
            )
        elif (
            "not_public" not in types
            and not r.details.get("not_public_items")
            and r.confidence in ("low", "medium")
            and _source_count(r) == 1
        ):
            gaps.append(
                Gap(
                    tech_id=r.tech_id,
                    criterion_id=r.criterion_id,
                    gap_type="single_source",
                    description=f"{r.criterion_id} 인용 출처가 1개뿐(confidence={r.confidence})",
                )
            )
        expected = EXPECTED_UNIT.get(r.perspective)
        if len(units) > 1 or (expected and r.evidence_unit != expected):
            gaps.append(
                Gap(
                    tech_id=r.tech_id,
                    criterion_id=r.criterion_id,
                    gap_type="unit_mismatch",
                    description=f"{r.criterion_id} 근거 단위 {sorted(units)}, 판정 단위 {r.evidence_unit}"
                    + (f" (관점 기대 단위 {expected})" if expected else ""),
                )
            )
    return gaps


def _evidence_owner(eid: str) -> str:
    return eid.split("-", 1)[0]


def evidence_stats(index: dict[str, Evidence]) -> dict[str, dict[str, float]]:
    """기술별 evidence 수와 official 비율. evidence_id 접두어(tech_id)로 기술을 가른다."""
    stats: dict[str, dict[str, float]] = {}
    for eid, e in index.items():
        s = stats.setdefault(
            _evidence_owner(eid),
            {"count": 0, "official": 0, "inference": 0, "not_public": 0},
        )
        s["count"] += 1
        if e.source_type in ("official", "inference", "not_public"):
            s[e.source_type] += 1
    for s in stats.values():
        s["official_ratio"] = s["official"] / s["count"] if s["count"] else 0.0
    return stats


def _stats_sentence(stats: dict[str, dict[str, float]], names: dict[str, str]) -> str:
    parts = [
        f"{names.get(t, t)} {int(s['count'])}건(official {s['official_ratio']:.0%}, "
        f"inference {int(s['inference'])}건, not_public {int(s['not_public'])}건)"
        for t, s in stats.items()
    ]
    return "근거 수: " + ", ".join(parts) + "."


def _evidence_brief(e: Evidence) -> str:
    quote = e.quote if len(e.quote) <= QUOTE_PREVIEW else e.quote[:QUOTE_PREVIEW] + "…"
    return f"{e.evidence_id} ({e.source_type}, unit={e.unit}, {e.locator}): {quote}"


def _matrix_table(results: list[CriterionResult]) -> str:
    rows = [
        "| tech_id | perspective | criterion | level | confidence | evidence_unit | evidence_ids | content |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        level = f"{r.level} / {r.level_estimate}" if r.level_estimate else r.level
        ids = ", ".join(e.evidence_id for e in r.evidence)
        content = r.content.replace("|", "/").replace("\n", " ")
        rows.append(
            f"| {r.tech_id} | {r.perspective} | {r.criterion_id} | {level} | {r.confidence} "
            f"| {r.evidence_unit} | {ids} | {content} |"
        )
    return "\n".join(rows)


def _details_block(results: list[CriterionResult]) -> str:
    """상충 판단에 필요한 details: S4 tradeoffs, D 직접성, T1 올리지 못한 이유."""
    lines = []
    for r in results:
        if r.criterion_id == "S4":
            lines.append(
                f"- {r.tech_id} S4 tradeoffs: {json.dumps(r.details.get('tradeoffs', []), ensure_ascii=False)}"
            )
        elif r.criterion_id in ("D1", "D2", "D3"):
            lines.append(
                f"- {r.tech_id} {r.criterion_id} directness={r.details.get('directness')} "
                f"extrapolation_logic={r.details.get('extrapolation_logic')}"
            )
        elif r.criterion_id == "T1":
            lines.append(f"- {r.tech_id} T1 why_not_higher={r.details.get('why_not_higher')}")
    return "\n".join(lines)


def _profiles_block(profiles: list[TechProfile]) -> str:
    return "\n".join(
        f"- {p.tech_id}: validation_env={p.validation_env} / limitations={'; '.join(p.limitations)}" for p in profiles
    )


def _build_messages(
    inp: SynthesisInput,
    results: list[CriterionResult],
    index: dict[str, Evidence],
    detected: list[Gap],
    stats_sentence: str,
    feedback: list[str],
) -> list[tuple[str, str]]:
    counter = "\n".join(f"- {_evidence_brief(e)}" for e in inp.counter_evidence) or "- (없음)"
    evidence = "\n".join(f"- {_evidence_brief(e)}" for e in index.values())
    gaps = "\n".join(f"- {g.tech_id} {g.criterion_id} {g.gap_type}: {g.description}" for g in detected) or "- (없음)"
    techs = "\n".join(f"- {t.tech_id}: {t.name} ({t.approach}, 계열: {t.family})" for t in inp.technologies)
    human = "\n\n".join(
        [
            _load_prompt("agreements.md"),
            _load_prompt("conflicts.md"),
            _load_prompt("gaps.md"),
            f"## 평가 대상 기술\n{techs}",
            f"## 기술 × 관점 × 기준 매트릭스\n{_matrix_table(results)}",
            f"## 상충 판단용 세부 필드\n{_details_block(results)}",
            f"## 기술 개요 요약\n{_profiles_block(inp.tech_profiles)}",
            f"## 코드가 이미 찾은 공백 (다시 쓰지 말 것)\n{gaps}",
            f"## 반대 근거 탐색 결과 (counter_evidence)\n{counter}",
            f"## 근거 통계 (코드 계산)\n{stats_sentence}",
            f"## 인용 가능한 evidence 목록 (이 id만 사용)\n{evidence}",
        ]
    )
    if feedback:
        human += "\n\n## 이전 출력의 문제 (반드시 고칠 것)\n" + "\n".join(f"- {f}" for f in feedback)
    return [("system", _load_prompt("system.md")), ("human", human)]


def _clean_ids(ids: list[str], index: dict[str, Evidence], where: str) -> list[str]:
    kept = [i for i in dict.fromkeys(ids) if i in index]
    dropped = [i for i in ids if i not in index]
    if dropped:
        logger.warning("%s: 존재하지 않는 evidence_id 제거 %s", where, dropped)
    return kept


def _validate_draft(
    draft: SynthesisDraft,
    index: dict[str, Evidence],
    matrix: dict[tuple[str, str, str], CriterionResult],
    tech_ids: list[str],
) -> tuple[list[Agreement], list[Conflict], list[Gap], list[str]]:
    """evidence_id 실존·기준 소속·S4 출발 conflict·금칙어를 검사한다. (정리된 항목들, 문제 목록)"""
    problems: list[str] = []
    agreements: list[Agreement] = []
    for i, a in enumerate(draft.agreements):
        ids = _clean_ids(a.evidence_ids, index, f"agreement[{i}]")
        if not ids:
            problems.append(f"agreement[{i}] '{a.statement[:40]}'의 evidence_ids가 모두 존재하지 않음")
            continue
        agreements.append(a.model_copy(update={"evidence_ids": ids}))

    conflicts: list[Conflict] = []
    for i, c in enumerate(draft.conflicts):
        where = f"conflict[{i}]"
        bad_ref = [
            f"{p}/{cid}"
            for p, cid in (
                (c.perspective_a, c.criterion_a),
                (c.perspective_b, c.criterion_b),
            )
            if cid not in PERSPECTIVE_CRITERIA[p] or (c.tech_id, p, cid) not in matrix
        ]
        if bad_ref:
            problems.append(f"{where}: 입력에 없는 관점/기준 {bad_ref}")
            continue
        ids = _clean_ids(c.evidence_ids, index, where)
        if len(ids) < 2:
            problems.append(f"{where}: 실존 evidence_id가 2개 미만")
            continue
        banned = find_banned_terms(c.statement + c.cause)
        if banned:
            problems.append(f"{where}: 우열 표현 {banned} 사용 — 원인은 기준·단위·근거 유형·직접성 차이로 서술")
            continue
        conflicts.append(c.model_copy(update={"evidence_ids": ids}))

    for tid in tech_ids:
        if (tid, "stakeholder", "S4") in matrix and not any(
            c.tech_id == tid and "S4" in (c.criterion_a, c.criterion_b) for c in conflicts
        ):
            problems.append(f"{tid}: S4(details.tradeoffs)를 출발점으로 한 conflict가 없음(기술별 최소 1개)")
    if not conflicts:
        problems.append("유효한 conflict가 0개")

    gaps = [g for g in draft.gaps if any(k[0] == g.tech_id and k[2] == g.criterion_id for k in matrix)]
    return agreements, conflicts, gaps, problems


# _matrix_table 행: | tech | perspective | criterion | level | confidence | evidence_unit | evidence_ids | content |
MATRIX_ROW_RE = re.compile(
    r"^\| (mla|pim_cxl) \| (\w+) \| ([TMSD][1-4]) \|(?:[^|]*\|){3} ([^|]*) \|",
    re.MULTILINE,
)


def _stub_draft(prompt: str) -> SynthesisDraft:
    """스텁 LLM용 결정적 드래프트: 프롬프트의 매트릭스 표에서 기술별 S4 ↔ 다른 관점 conflict를 1개씩 만든다."""
    rows: dict[tuple[str, str], tuple[str, list[str]]] = {}
    for tid, persp, cid, ids in MATRIX_ROW_RE.findall(prompt):
        rows[(tid, cid)] = (persp, [i.strip() for i in ids.split(",") if i.strip()])
    conflicts = []
    for tid in dict.fromkeys(t for t, _ in rows):
        other = next((c for c in ("D4", "M2", "T1") if (tid, c) in rows), None)
        if (tid, "S4") not in rows or other is None:
            continue
        persp_b, ids_b = rows[(tid, other)]
        conflicts.append(
            Conflict(
                tech_id=tid,
                perspective_a="stakeholder",
                criterion_a="S4",
                perspective_b=persp_b,
                criterion_b=other,
                statement=f"(스텁) {tid}의 S4 상충과 {other} 판정이 서로 다른 지점을 가리킨다.",
                cause="(스텁) 평가 단위 차이: S4는 기술 계열(family), 상대 기준은 논문·구현(paper) 단위로 판정했다.",
                evidence_ids=list(dict.fromkeys([*rows[(tid, "S4")][1], *ids_b])),
            )
        )
    if not conflicts:
        raise ValueError("stub synthesis: 프롬프트에서 기술 × 관점 × 기준 매트릭스를 찾지 못함")
    return SynthesisDraft(
        conflicts=conflicts,
        unit_notes="(스텁) TRL·도메인은 논문 단위, 시장·이해관계자는 계열 단위로 판정했다.",
        evidence_asymmetry_note="(스텁) 근거 방향성은 판단하지 않았다.",
    )


def stub_overrides() -> dict[type, Callable[[str], Any]]:
    """E의 FakeStructuredLLM이 자동 등록한다(CONTRACTS §9). 스텁 실행에서도 입력 evidence로만 conflict를 만든다."""
    return {SynthesisDraft: _stub_draft}


def _merge_gaps(detected: list[Gap], llm_gaps: list[Gap]) -> list[Gap]:
    merged: dict[tuple[str, str, str], Gap] = {}
    for g in [*detected, *llm_gaps]:
        merged.setdefault((g.tech_id, g.criterion_id, g.gap_type), g)
    return list(merged.values())


def run_synthesis(inp: SynthesisInput, deps: Deps) -> SynthesisResult:
    """4관점 결과 → SynthesisResult. deps.retriever / deps.web_search는 호출하지 않는다."""
    results = _all_results(inp)
    matrix = build_matrix(results)
    index = build_evidence_index(
        results=results,
        profiles=inp.tech_profiles,
        counter_evidence=inp.counter_evidence,
    )
    tech_ids = [t.tech_id for t in inp.technologies]
    names = {t.tech_id: t.name for t in inp.technologies}
    detected = detect_gaps(results)
    stats_sentence = _stats_sentence(evidence_stats(index), names)
    logger.info(
        "synthesis 입력: 결과 %d개, evidence %d개, 규칙 공백 %d개, counter %d개",
        len(results),
        len(index),
        len(detected),
        len(inp.counter_evidence),
    )

    structured = deps.llm.with_structured_output(SynthesisDraft)
    feedback: list[str] = []
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        raw = structured.invoke(_build_messages(inp, results, index, detected, stats_sentence, feedback))
        draft = SynthesisDraft.model_validate(raw.model_dump() if isinstance(raw, BaseModel) else raw)
        agreements, conflicts, llm_gaps, problems = _validate_draft(draft, index, matrix, tech_ids)
        if not problems:
            break
        logger.warning("synthesis 검증 실패(시도 %d/%d): %s", attempt, MAX_LLM_ATTEMPTS, problems)
        feedback = problems
    else:
        raise ValueError(f"synthesis 결과가 검증을 통과하지 못함: {feedback}")

    return SynthesisResult(
        agreements=agreements,
        conflicts=conflicts,
        gaps=_merge_gaps(detected, llm_gaps),
        unit_notes=draft.unit_notes,
        evidence_asymmetry_note=f"{stats_sentence} {draft.evidence_asymmetry_note}".strip(),
        generated_at=deps.now(),
    )
