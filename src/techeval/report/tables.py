"""4장 각 절 말미의 기술 × 기준 요약표 (마크다운). LLM 없이 결정적으로 생성한다."""

import logging

from techeval.schemas import PERSPECTIVE_CRITERIA, CriterionResult, Measurement, TechRef

logger = logging.getLogger(__name__)

PERSPECTIVE_TITLES: dict[str, str] = {
    "trl": "기술 성숙도(TRL)",
    "market": "시장성",
    "stakeholder": "이해관계자",
    "domain": "도메인 적합성",
}
CRITERION_NAMES: dict[str, str] = {
    "T1": "현재 TRL",
    "T2": "검증 환경",
    "T3": "재현성",
    "T4": "다음 과제",
    "M1": "성장성",
    "M2": "채택",
    "M3": "생태계",
    "S1": "의사결정",
    "S2": "편익",
    "S3": "부담",
    "S4": "상충",
    "D1": "장문맥 병목",
    "D2": "동시 처리",
    "D3": "지연·에너지·비용",
    "D4": "변경 범위",
}
UNIT_KO = {"paper": "논문·구현(paper)", "family": "기술 계열(family)"}

# lint가 4장 요약표 개수를 셀 때 이 헤더 행을 찾는다. 바꾸면 lint도 함께 바뀐다.
TABLE_HEADER = "| 기준 | 기술 | 레벨 | 신뢰도 | 근거 단위 | 근거 |"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _level(r: CriterionResult) -> str:
    return f"{r.level} (추정: {r.level_estimate})" if r.level_estimate else r.level


def criterion_table(
    tech_ids: list[str],
    criteria: list[str],
    results: list[CriterionResult],
    *,
    tech_names: dict[str, str] | None = None,
) -> str:
    """기준 × 기술 행으로 레벨·신뢰도·근거 단위·evidence_id를 나열한 표. 결과가 없으면 '결과 없음' 행."""
    by_key = {(r.tech_id, r.criterion_id): r for r in results}
    names = tech_names or {}
    rows = [TABLE_HEADER, "|---|---|---|---|---|---|"]
    for cid in criteria:
        for tid in tech_ids:
            label = f"{cid} {CRITERION_NAMES.get(cid, '')}".strip()
            tech = names.get(tid, tid)
            r = by_key.get((tid, cid))
            if r is None:
                logger.warning("요약표: %s × %s 결과 없음", tid, cid)
                rows.append(f"| {label} | {_cell(tech)} | 결과 없음 | — | — | — |")
                continue
            cites = ", ".join(e.evidence_id for e in r.evidence)
            rows.append(
                f"| {label} | {_cell(tech)} | {_cell(_level(r))} | {r.confidence} "
                f"| {UNIT_KO[r.evidence_unit]} | [E: {cites}] |"
            )
    return "\n".join(rows)


def perspective_table(
    number: str,
    perspective: str,
    technologies: list[TechRef],
    results: list[CriterionResult],
) -> str:
    """캡션(**표 4.x ...**) + 요약표. number는 '4.1' 형식."""
    caption = f"**표 {number} 기술 × 기준 요약 — {PERSPECTIVE_TITLES[perspective]}**"
    table = criterion_table(
        [t.tech_id for t in technologies],
        PERSPECTIVE_CRITERIA[perspective],
        [r for r in results if r.perspective == perspective],
        tech_names={t.tech_id: t.name for t in technologies},
    )
    note = "레벨은 기준별 판정 단계이며 관점·기준 간 합산하지 않는다."
    return f"{caption}\n\n{table}\n\n{note}"


def _value(m: Measurement) -> str:
    return f"{m.value} {m.unit}" if m.unit and m.unit not in m.value else m.value


def format_measurement(m: Measurement) -> str:
    """수치 + 측정 조건 병기 문자열. 예: 'KV cache reduction 93.3% (모델: 236B, ...)'."""
    conds = [
        ("모델", m.model_size),
        ("컨텍스트", m.context_length),
        ("HW", m.hardware),
        ("베이스라인", m.baseline),
        ("배치", m.batch_size),
        ("조건", m.condition_note),
    ]
    cond = ", ".join(f"{k}: {v}" for k, v in conds if v)
    return f"{m.metric} {_value(m)}" + (f" ({cond})" if cond else " (조건 미기재)")


def measurement_table(measurements: list[Measurement]) -> str:
    """3장 기술 개요의 정량 성과 표. 각 행에 측정 조건과 근거를 붙인다."""
    rows = [
        "| 지표 | 값 | 모델 규모 | 컨텍스트 | 하드웨어 | 베이스라인 | 근거 |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in measurements:
        cells = [
            m.metric,
            _value(m),
            m.model_size or "—",
            m.context_length or "—",
            m.hardware or "—",
            m.baseline or "—",
        ]
        rows.append(
            "| " + " | ".join(_cell(c) for c in cells) + f" | [E: {m.evidence_id}] |"
        )
    return "\n".join(rows)
