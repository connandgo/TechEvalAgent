"""픽스처 기반 가짜 LLM. `--stub` 실행과 단위 테스트에서 실제 모델 대신 주입한다.

`with_structured_output(Model)`이 반환하는 러너블은 프롬프트 문자열에서 `tech_id`·`criterion_id`를 추출해
`tests/fixtures/`의 해당 픽스처를 `Model` 인스턴스로 돌려준다. 매칭에 실패하면 조용히 기본값을 만들지 않고
명확한 예외를 낸다(AGENTS.md 규칙 8).

각 역할은 `calls`를 읽어 "프롬프트에 무엇이 들어갔는지"를 검증할 수 있고, `register(Model, fn)`으로
특정 모델의 응답을 덮어써 실패 경로(예: judge 미달)를 테스트할 수 있다.
"""

import json
import logging
import re
import types
import typing
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from techeval.agents._deps import TechResearchOutput
from techeval.schemas import (
    CRITERION_PERSPECTIVE,
    PERSPECTIVE_CRITERIA,
    CriterionResult,
    EvidenceGap,
    JudgeResult,
    SynthesisResult,
    TechProfile,
)

logger = logging.getLogger(__name__)

DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

# 픽스처 파일 → 소유 역할 (누락 시 에러 메시지에 표시)
FIXTURE_OWNERS: dict[str, str] = {
    "chunks.json": "A",
    "web_results.json": "C",
    "tech_profiles.json": "B",
    "trl_eval.json": "B",
    "domain_eval.json": "B",
    "market_eval.json": "C",
    "stakeholder_eval.json": "C",
    "synthesis.json": "D",
    "report_md.md": "D",
    "counter_evidence.json": "E",
    "evidence_gap.json": "E",
    "judge_result.json": "E",
    "state_initial.json": "E",
}

PERSPECTIVE_FIXTURE: dict[str, str] = {
    "trl": "trl_eval.json",
    "market": "market_eval.json",
    "stakeholder": "stakeholder_eval.json",
    "domain": "domain_eval.json",
}

# 프롬프트에서 관점을 유추할 때 쓰는 키워드 (criterion_id가 전혀 없을 때만 사용)
PERSPECTIVE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "trl": ("TRL", "기술 성숙도", "재현성", "검증 환경"),
    "market": ("시장성", "시장", "채택", "생태계"),
    "stakeholder": ("이해관계자", "4주체", "편익", "부담", "상충"),
    "domain": ("도메인 적합성", "도메인", "장문맥 병목", "변경 범위"),
}

# 한글이 바로 붙는 경우("M3만")에도 잡히도록 \b 대신 ASCII 단어문자 기준 경계를 쓴다
_ASCII_B = r"(?<![A-Za-z0-9_])"
_ASCII_E = r"(?![A-Za-z0-9_])"
_TECH_ID_EXPLICIT = re.compile(r"tech_id\W{0,4}(mla|pim_cxl)" + _ASCII_E, re.IGNORECASE)
_TECH_ID_LOOSE = re.compile(_ASCII_B + r"(mla|pim_cxl|pim|cxl|deepseek)" + _ASCII_E, re.IGNORECASE)
_CRITERION_ID = re.compile(_ASCII_B + r"([TMSD][1-4])" + _ASCII_E)


class FixtureLookupError(LookupError):
    """프롬프트·모델로 픽스처를 결정하지 못했을 때."""


def prompt_to_text(prompt: Any) -> str:
    """str / PromptValue / 메시지 리스트 / dict 등 어떤 입력이든 검색 가능한 문자열로 편다."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, PromptValue):
        return prompt.to_string()
    if isinstance(prompt, BaseMessage):
        return prompt.content if isinstance(prompt.content, str) else json.dumps(prompt.content, ensure_ascii=False)
    if isinstance(prompt, dict):
        return json.dumps(prompt, ensure_ascii=False, default=str)
    if isinstance(prompt, list | tuple):
        return "\n".join(prompt_to_text(p) for p in prompt)
    return str(prompt)


def extract_tech_id(text: str) -> str | None:
    """`tech_id: mla` 같은 명시 표기를 우선하고, 없으면 별칭 등장 빈도로 결정한다. 동률이면 None."""
    explicit = _TECH_ID_EXPLICIT.findall(text)
    if explicit:
        return Counter(t.lower() for t in explicit).most_common(1)[0][0]
    counts: Counter[str] = Counter()
    for m in _TECH_ID_LOOSE.findall(text):
        m = m.lower()
        counts["pim_cxl" if m in ("pim_cxl", "pim", "cxl") else "mla"] += 1
    if not counts:
        return None
    top = counts.most_common(2)
    if len(top) == 2 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def extract_criterion_ids(text: str) -> Counter:
    return Counter(_CRITERION_ID.findall(text))


def extract_perspective(text: str) -> str | None:
    """criterion_id가 없을 때 키워드로 관점을 유추한다."""
    scores = {p: sum(text.count(k) for k in ks) for p, ks in PERSPECTIVE_KEYWORDS.items()}
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return None
    if sum(1 for v in scores.values() if v == best[1]) > 1:
        return None
    return best[0]


class FakeStructuredLLM:
    """`with_structured_output` / `invoke`만 흉내 내는 최소 LLM. langchain BaseChatModel을 상속하지 않는다."""

    def __init__(
        self,
        fixtures_dir: str | Path | None = None,
        overrides: dict[type, Callable[[str], Any]] | None = None,
    ):
        self.fixtures_dir = Path(fixtures_dir) if fixtures_dir else DEFAULT_FIXTURES_DIR
        self.calls: list[dict[str, Any]] = []  # {"schema": str, "prompt": str}
        self._registry: dict[type, Callable[[str], Any]] = dict(overrides or {})
        self._cache: dict[str, Any] = {}

    # -- 등록/기록 -----------------------------------------------------------

    def register(self, schema: type, fn: Callable[[str], Any]) -> None:
        """특정 스키마의 응답을 덮어쓴다. `fn(prompt_text)`는 인스턴스 또는 dict를 반환한다."""
        self._registry[schema] = fn

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def prompts_for(self, schema: type) -> list[str]:
        return [c["prompt"] for c in self.calls if c["schema"] == schema.__name__]

    # -- langchain 호환 표면 ---------------------------------------------------

    def with_structured_output(self, schema: type, **_: Any) -> RunnableLambda:
        def _run(prompt: Any) -> Any:
            text = prompt_to_text(prompt)
            self.calls.append({"schema": getattr(schema, "__name__", str(schema)), "prompt": text})
            return self.resolve(schema, text)

        return RunnableLambda(_run)

    def invoke(self, prompt: Any, **_: Any) -> AIMessage:
        """비구조화 호출. 등록된 `str` 응답이 있으면 그것을, 없으면 빈 메시지를 돌려준다."""
        text = prompt_to_text(prompt)
        self.calls.append({"schema": "str", "prompt": text})
        fn = self._registry.get(str)
        return AIMessage(content=fn(text) if fn else "")

    def bind(self, **_: Any) -> "FakeStructuredLLM":
        return self

    def bind_tools(self, *_: Any, **__: Any) -> "FakeStructuredLLM":
        return self

    # -- 픽스처 해석 -----------------------------------------------------------

    def load_fixture(self, name: str) -> Any:
        if name not in self._cache:
            path = self.fixtures_dir / name
            if not path.exists():
                owner = FIXTURE_OWNERS.get(name, "?")
                raise FixtureLookupError(f"fixture not found: {path} (owner: {owner})")
            self._cache[name] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[name]

    def resolve(self, schema: Any, text: str) -> Any:
        """스키마와 프롬프트로 응답 객체를 만든다."""
        if schema in self._registry:
            out = self._registry[schema](text)
            if isinstance(out, dict) and isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(out)
            return out

        origin = typing.get_origin(schema)
        if origin in (list, typing.List):  # noqa: UP006 — get_origin 비교용
            (item,) = typing.get_args(schema)
            return self._resolve_list(item, text)

        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise FixtureLookupError(f"unsupported schema for FakeStructuredLLM: {schema!r}")

        if schema is CriterionResult:
            return self._resolve_criterion(text)
        if schema is TechProfile:
            return self._resolve_profile(text)
        if schema is TechResearchOutput:
            return TechResearchOutput(
                tech_profile=self._resolve_profile(text),
                trl_eval=self._resolve_list(CriterionResult, text, perspective="trl"),
            )
        if schema is SynthesisResult:
            return SynthesisResult.model_validate(self.load_fixture("synthesis.json"))
        if schema is EvidenceGap:
            return EvidenceGap.model_validate(self.load_fixture("evidence_gap.json"))
        if schema is JudgeResult:
            return JudgeResult.model_validate(self.load_fixture("judge_result.json"))

        # 그 외 BaseModel: 필드별로 재귀 해석 (각 역할이 정의한 래퍼 모델 지원)
        return self._resolve_generic(schema, text)

    def _resolve_generic(self, schema: type[BaseModel], text: str) -> BaseModel:
        values: dict[str, Any] = {}
        for name, field in schema.model_fields.items():
            ann = field.annotation
            try:
                values[name] = self._resolve_annotation(ann, text)
            except FixtureLookupError:
                if field.is_required():
                    raise FixtureLookupError(
                        f"cannot fill required field {schema.__name__}.{name}: {ann!r}. "
                        f"use FakeStructuredLLM.register({schema.__name__}, fn) in your test."
                    ) from None
                # 기본값 있는 필드는 비워 둔다
        return schema.model_validate(values)

    def _resolve_annotation(self, ann: Any, text: str) -> Any:
        origin = typing.get_origin(ann)
        if origin in (types.UnionType, typing.Union):
            args = [a for a in typing.get_args(ann) if a is not type(None)]
            if len(args) == 1:
                return self._resolve_annotation(args[0], text)
            raise FixtureLookupError(f"ambiguous union {ann!r}")
        if origin in (list, typing.List):  # noqa: UP006
            (item,) = typing.get_args(ann)
            if isinstance(item, type) and issubclass(item, BaseModel):
                return self._resolve_list(item, text)
            raise FixtureLookupError(f"unsupported list item {item!r}")
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            return self.resolve(ann, text)
        raise FixtureLookupError(f"unsupported annotation {ann!r}")

    def _resolve_list(self, item: type, text: str, perspective: str | None = None) -> list[Any]:
        if item is CriterionResult:
            tech_id = self._require_tech(text)
            ids = extract_criterion_ids(text)
            if perspective:
                wanted = [c for c in PERSPECTIVE_CRITERIA[perspective] if not ids or c in ids]
            elif ids:
                wanted = sorted(ids, key=lambda c: (CRITERION_PERSPECTIVE[c], c))
            else:
                p = extract_perspective(text)
                if p is None:
                    raise FixtureLookupError("no criterion_id or perspective keyword found in prompt")
                wanted = PERSPECTIVE_CRITERIA[p]
            return [self._criterion_from_fixture(tech_id, c) for c in wanted]
        if item is TechProfile:
            return [self._resolve_profile(text)]
        if isinstance(item, type) and issubclass(item, BaseModel):
            return [self.resolve(item, text)]
        raise FixtureLookupError(f"unsupported list item {item!r}")

    def _require_tech(self, text: str) -> str:
        tech_id = extract_tech_id(text)
        if tech_id is None:
            raise FixtureLookupError("cannot determine tech_id from prompt (mention 'tech_id: mla' or 'pim_cxl')")
        return tech_id

    def _resolve_criterion(self, text: str) -> CriterionResult:
        tech_id = self._require_tech(text)
        ids = extract_criterion_ids(text)
        if not ids:
            raise FixtureLookupError("cannot determine criterion_id from prompt")
        top = ids.most_common(2)
        if len(top) == 2 and top[0][1] == top[1][1]:
            raise FixtureLookupError(f"ambiguous criterion_id in prompt: {top[0][0]} vs {top[1][0]}")
        return self._criterion_from_fixture(tech_id, top[0][0])

    def _criterion_from_fixture(self, tech_id: str, criterion_id: str) -> CriterionResult:
        fixture = PERSPECTIVE_FIXTURE[CRITERION_PERSPECTIVE[criterion_id]]
        for raw in self.load_fixture(fixture):
            if raw["tech_id"] == tech_id and raw["criterion_id"] == criterion_id:
                return CriterionResult.model_validate(raw)
        raise FixtureLookupError(f"no {tech_id}/{criterion_id} in {fixture}")

    def _resolve_profile(self, text: str) -> TechProfile:
        tech_id = self._require_tech(text)
        for raw in self.load_fixture("tech_profiles.json"):
            if raw["tech_id"] == tech_id:
                return TechProfile.model_validate(raw)
        raise FixtureLookupError(f"no profile for {tech_id} in tech_profiles.json")
