"""질의 재작성 노드 (E). 부족 항목별 대체 검색어 2~3개를 만든다. 의견 생성 없음.

기본은 결정적(별칭 × 항목 키워드 조합, 이전 검색어 제외)이고, `llm`이 주어지면 `prompts/control/query_rewrite.md`로
추가 후보를 받아 합친다. LLM 호출 결과도 Pydantic으로 검증한다.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from techeval.schemas import TechRef

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "control" / "query_rewrite.md"
QUERIES_PER_ITEM = 3

# 부족 항목 → 영문 검색 키워드 (원문 코퍼스가 영문이므로 영문 위주)
ITEM_KEYWORDS: dict[str, list[str]] = {
    "T1": ["deployment status production", "readiness maturity level", "released model serving"],
    "T2": ["experimental setup evaluation environment", "prototype hardware testbed", "simulation methodology"],
    "T3": ["open source code release github", "reproduction third-party benchmark", "model weights checkpoint release"],
    "T4": ["limitations future work", "open challenges remaining", "not yet addressed"],
    "PROFILE:principle": ["architecture overview mechanism", "how it works core idea"],
    "PROFILE:limitations": ["limitations drawbacks", "overhead cost trade-off"],
    "PROFILE:measurements": ["throughput latency results table", "KV cache reduction memory footprint numbers"],
    "PROFILE:validation_env": ["experimental setup hardware GPU", "evaluation platform configuration"],
    "PROFILE:evidence": ["method section", "results section"],
    "M1": ["market size forecast CAGR", "demand drivers growth outlook"],
    "M2": ["adoption production deployment case", "customer proof of concept pilot"],
    "M3": ["framework support vLLM SGLang TensorRT-LLM", "standardization consortium spec", "third-party tools"],
    "S1": ["stakeholder decision maker vendor operator", "who deploys who supplies"],
    "S2": ["benefits cost savings operator", "advantage for model developers vendors"],
    "S3": ["migration cost burden", "integration effort risk"],
    "S4": ["trade-off conflict between vendors operators", "winners losers ecosystem"],
    "D1": ["long context KV cache memory footprint 128K", "KV cache reduction per token"],
    "D2": ["batch size concurrency throughput", "concurrent sessions tokens per second"],
    "D3": ["TTFT TPOT latency energy per token", "total cost of ownership inference"],
    "D4": ["retraining required conversion", "serving engine modification hardware change"],
}


class QueryRewriteOutput(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=12)


def _dedupe(queries: list[str], exclude: set[str]) -> list[str]:
    out: list[str] = []
    seen = {q.strip().lower() for q in exclude}
    for q in queries:
        k = q.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(q.strip())
    return out


def deterministic_queries(tech: TechRef, missing: list[str], previous: list[str]) -> list[str]:
    """별칭 × 항목 키워드 조합. 이전 검색어와 중복되는 것은 제외. 항목당 최대 QUERIES_PER_ITEM개."""
    aliases = [tech.name, *tech.search_aliases] or [tech.name]
    out: list[str] = []
    for item in missing:
        kws = ITEM_KEYWORDS.get(item, [item])
        cands = [f"{alias} {kw}" for kw in kws for alias in aliases]
        out += _dedupe(cands, set(previous) | set(out))[:QUERIES_PER_ITEM]
    return out


def rewrite_queries(
    tech: TechRef,
    missing: list[str],
    previous: list[str],
    *,
    retry_count: int,
    llm: Any | None = None,
) -> list[str]:
    """부족 항목별 대체 검색어. `retry_count`가 커질수록 뒤쪽 키워드를 우선해 검색어를 바꾼다."""
    if not missing:
        return []
    # 재시도 회차에 따라 키워드 순서를 회전시켜 같은 검색어 반복을 피한다
    rotated_missing = list(missing)
    base = deterministic_queries(tech, rotated_missing, previous)
    if retry_count > 0 and len(base) > 1:
        shift = retry_count % len(base)
        base = base[shift:] + base[:shift]

    if llm is None:
        logger.info("query_rewrite %s retry=%d: %s", tech.tech_id, retry_count, base)
        return base

    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        tech_id=tech.tech_id,
        tech_name=tech.name,
        aliases=", ".join(tech.search_aliases),
        missing="\n".join(f"- {m}" for m in missing),
        previous="\n".join(f"- {q}" for q in previous) or "- (없음)",
        retry_count=retry_count,
    )
    out = llm.with_structured_output(QueryRewriteOutput).invoke(prompt)
    out = QueryRewriteOutput.model_validate(out if isinstance(out, dict) else out.model_dump())
    merged = _dedupe([*out.queries, *base], set(previous))
    logger.info("query_rewrite %s retry=%d (llm): %s", tech.tech_id, retry_count, merged)
    return merged
