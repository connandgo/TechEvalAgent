"""환경 설정과 의존성 팩토리.

`.env`(또는 환경변수)에서 Settings를 읽고, `get_llm()/get_judge_llm()/build_deps()`로 모델·검색기를 만든다.
모듈 전역에서 실제 모델을 초기화하지 않는다.
"""

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

from techeval.agents._deps import Deps

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


AGENT_NAMES: tuple[str, ...] = ("tech_research", "domain", "market", "stakeholder", "synthesis", "report")


class Settings(BaseModel):
    llm_provider: str = ""
    llm_model: str = ""  # 기본 생성 모델. 아래 에이전트별 키가 비어 있으면 이 값으로 폴백
    llm_model_tech_research: str = ""
    llm_model_domain: str = ""
    llm_model_market: str = ""
    llm_model_stakeholder: str = ""
    llm_model_synthesis: str = ""
    llm_model_report: str = ""
    judge_model: str = ""
    embedding_model: str = "BAAI/bge-m3"
    chroma_dir: str = "data/chroma"
    papers_dir: str = "data/papers"
    web_search_provider: str = ""
    web_search_api_key: str = ""
    output_dir: str = "outputs"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _warn_same_judge(self):
        report_model = self.model_for("report")
        if report_model and self.judge_model and report_model == self.judge_model:
            logger.warning(
                "JUDGE_MODEL(%s)이 실효 보고서 생성 모델과 같습니다. "
                "AGENTS.md 규칙 9: 보고서 생성 모델과 검수 모델은 분리해야 합니다.",
                self.judge_model,
            )
        return self

    def model_for(self, agent: str | None) -> str:
        """에이전트별 모델(`LLM_MODEL_<AGENT>`), 없으면 `LLM_MODEL`."""
        if agent:
            if agent not in AGENT_NAMES:
                raise KeyError(f"unknown agent: {agent} (choose from {AGENT_NAMES})")
            override = getattr(self, f"llm_model_{agent}")
            if override:
                return override
        return self.llm_model

    def agent_overrides(self) -> dict[str, str]:
        """LLM_MODEL과 다른 모델을 지정한 에이전트만 {agent: model}로."""
        return {a: self.model_for(a) for a in AGENT_NAMES if self.model_for(a) != self.llm_model}

    def require_llm(self) -> None:
        missing = [k for k in ("LLM_PROVIDER", "LLM_MODEL", "JUDGE_MODEL") if not getattr(self, k.lower())]
        if missing:
            raise RuntimeError(f".env에 {', '.join(missing)} 설정이 필요합니다 (.env.example 참조)")


def load_settings(env_file: str | Path | None = None) -> Settings:
    """`.env`를 로드한 뒤 환경변수에서 Settings를 만든다. 이미 설정된 환경변수가 우선한다."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    fields = Settings.model_fields
    values = {name: os.environ[name.upper()] for name in fields if name.upper() in os.environ}
    settings = Settings(**values)
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return settings


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_OPENAI_GPT5_MODEL = re.compile(r"^gpt-5(?:$|[-.])", re.IGNORECASE)


def _model_init_kwargs(model: str, provider: str) -> dict[str, Any]:
    """모델별 `init_chat_model` 인자.

    OpenAI의 GPT-5 계열(`gpt-5-mini` 포함)은 `temperature`를 전달하면 API 오류가 나므로 생략한다.
    그 외 모델은 평가 재현성을 위해 temperature=0을 유지한다.
    """
    kwargs: dict[str, Any] = {"model_provider": provider}
    if not (provider.lower() == "openai" and _OPENAI_GPT5_MODEL.match(model)):
        kwargs["temperature"] = 0
    return kwargs


class _ToolCallingLLM:
    """`with_structured_output`의 기본 method를 네이티브 tool calling(`function_calling`)으로 고정하는 래퍼.

    langchain-openai 1.x는 기본 method가 `json_schema`(strict)라서 에이전트 출력 스키마의 자유형 dict 필드
    (예: `details: dict`)를 "additionalProperties is required to be false"로 거부한다. 그 밖의 속성·메서드
    (`invoke` 등)는 원본 모델에 그대로 위임한다.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("method", "function_calling")
        return self._llm.with_structured_output(schema, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def _init_chat_model(model: str, provider: str) -> Any:
    from langchain.chat_models import init_chat_model

    return _ToolCallingLLM(init_chat_model(model, **_model_init_kwargs(model, provider)))


def get_llm(settings: Settings | None = None, agent: str | None = None) -> Any:
    """평가·생성용 모델. `agent`를 주면 `LLM_MODEL_<AGENT>`(없으면 `LLM_MODEL`)를 쓴다.

    어떤 모델이든 `with_structured_output`이 네이티브 tool calling으로 동작하는 provider여야 한다
    (JSON 모드 흉내만 내는 provider는 중첩 리스트 출력에서 깨진다).
    """
    s = settings or load_settings()
    s.require_llm()
    return _init_chat_model(s.model_for(agent), s.llm_provider)


def get_judge_llm(settings: Settings | None = None) -> Any:
    """보고서 검수용 모델 (`JUDGE_MODEL`). 실효 보고서 생성 모델과 달라야 한다."""
    s = settings or load_settings()
    s.require_llm()
    return _init_chat_model(s.judge_model, s.llm_provider)


def build_agent_deps(deps: Deps, settings: Settings | None = None) -> dict[str, Deps]:
    """에이전트별 모델이 지정된 경우에만 그 에이전트용 Deps(llm만 교체)를 만든다. 나머지는 공용 deps를 쓴다."""
    s = settings or load_settings()
    out: dict[str, Deps] = {}
    for agent, model in s.agent_overrides().items():
        logger.info("에이전트별 모델: %s -> %s", agent, model)
        out[agent] = deps.model_copy(update={"llm": _init_chat_model(model, s.llm_provider)})
    return out


def build_deps(stub: bool = False, settings: Settings | None = None, *, use_cache: bool = True) -> Deps:
    """에이전트 의존성 조립.

    stub=True: `StubRetriever` + `stub_web_search` + `FakeStructuredLLM` (네트워크·모델 불필요).
    stub=False: `VectorRetriever` + `web_search` + 실제 모델.
    """
    if stub:
        from techeval.stub_llm import FakeStructuredLLM, collect_stub_overrides

        overrides = collect_stub_overrides()  # B/C/D 에이전트 모듈의 stub_overrides() 자동 수집
        return Deps(
            retriever=make_stub_retriever(),
            web_search=make_stub_web_search(),
            llm=FakeStructuredLLM(overrides=overrides),
            judge_llm=FakeStructuredLLM(),
            now=lambda: "2026-01-01T00:00:00",
        )

    s = settings or load_settings()
    from techeval.retrieval.retriever import VectorRetriever
    from techeval.tools.web_search import web_search

    if not use_cache:
        # C의 web_search가 캐시(outputs/web_cache/)를 건너뛰도록 환경변수로 알린다 (`run.py --no-cache`).
        os.environ["WEB_SEARCH_NO_CACHE"] = "1"

    retriever = VectorRetriever(chroma_dir=s.chroma_dir, embedding_model=s.embedding_model)
    # 임베딩 모델을 메인 스레드에서 1회 미리 적재한다. Send fan-out(mla / pim_cxl)이 병렬 스레드에서 동시에
    # 최초 적재를 시도하면 torch MPS 커널 캐시가 경쟁해 프로세스가 SIGABRT로 죽는다 (macOS에서 재현).
    logger.info("임베딩 모델 warm-up: %s", s.embedding_model)
    retriever.search("KV cache", top_k=1, mode="dense")

    return Deps(
        retriever=retriever,
        web_search=web_search,
        llm=get_llm(s),
        judge_llm=get_judge_llm(s),
        now=now_iso,
    )


# --- 스텁 폴백 (A·C의 스텁이 아직 머지되지 않았을 때) --------------------------------


class _FallbackRetriever:
    """A의 `StubRetriever`가 없을 때 쓰는 빈 검색기. 모든 검색이 빈 결과 → 결과는 not_public으로 기록된다."""

    def search(self, query: str, *, top_k: int = 5, doc_ids: list[str] | None = None, mode: str = "hybrid") -> list:
        return []

    def get_chunk(self, chunk_id: str) -> None:
        return None


def _fallback_web_search(query: str, **kwargs: Any) -> list:
    """C의 `stub_web_search`가 없을 때 쓰는 빈 검색 함수."""
    return []


def make_stub_retriever() -> Any:
    try:
        from techeval.retrieval.stub import StubRetriever

        return StubRetriever()
    except ImportError:
        logger.warning("techeval.retrieval.stub.StubRetriever 없음 (역할 A) — 빈 검색기로 대체")
        return _FallbackRetriever()


def make_stub_web_search() -> Any:
    try:
        from techeval.tools.stub import stub_web_search

        return stub_web_search
    except ImportError:
        logger.warning("techeval.tools.stub.stub_web_search 없음 (역할 C) — 빈 검색 함수로 대체")
        return _fallback_web_search
