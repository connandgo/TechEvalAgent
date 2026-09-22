"""환경 설정과 의존성 팩토리.

`.env`(또는 환경변수)에서 Settings를 읽고, `get_llm()/get_judge_llm()/build_deps()`로 모델·검색기를 만든다.
모듈 전역에서 실제 모델을 초기화하지 않는다.
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

from techeval.agents._deps import Deps

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    llm_provider: str = ""
    llm_model: str = ""
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
        if self.llm_model and self.judge_model and self.llm_model == self.judge_model:
            logger.warning(
                "JUDGE_MODEL(%s)이 LLM_MODEL과 같습니다. AGENTS.md 규칙 9: 생성 모델과 검수 모델은 분리해야 합니다.",
                self.judge_model,
            )
        return self

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


def _init_chat_model(model: str, provider: str) -> Any:
    from langchain.chat_models import init_chat_model

    return init_chat_model(model, model_provider=provider, temperature=0)


def get_llm(settings: Settings | None = None) -> Any:
    """평가·생성용 모델 (`LLM_MODEL`)."""
    s = settings or load_settings()
    s.require_llm()
    return _init_chat_model(s.llm_model, s.llm_provider)


def get_judge_llm(settings: Settings | None = None) -> Any:
    """보고서 검수용 모델 (`JUDGE_MODEL`). `LLM_MODEL`과 달라야 한다."""
    s = settings or load_settings()
    s.require_llm()
    return _init_chat_model(s.judge_model, s.llm_provider)


def build_deps(stub: bool = False, settings: Settings | None = None, *, use_cache: bool = True) -> Deps:
    """에이전트 의존성 조립.

    stub=True: `StubRetriever` + `stub_web_search` + `FakeStructuredLLM` (네트워크·모델 불필요).
    stub=False: `VectorRetriever` + `web_search` + 실제 모델.
    """
    if stub:
        from techeval.stub_llm import FakeStructuredLLM

        try:
            from techeval.retrieval.stub import StubRetriever
        except ImportError as e:  # A 미제공
            raise ImportError("techeval.retrieval.stub.StubRetriever (역할 A)가 아직 없습니다") from e
        try:
            from techeval.tools.stub import stub_web_search
        except ImportError as e:  # C 미제공
            raise ImportError("techeval.tools.stub.stub_web_search (역할 C)가 아직 없습니다") from e

        return Deps(
            retriever=StubRetriever(),
            web_search=stub_web_search,
            llm=FakeStructuredLLM(),
            judge_llm=FakeStructuredLLM(),
            now=lambda: "2026-01-01T00:00:00",
        )

    s = settings or load_settings()
    from techeval.retrieval.retriever import VectorRetriever
    from techeval.tools.web_search import web_search

    if not use_cache:
        # C의 web_search가 캐시(outputs/web_cache/)를 건너뛰도록 환경변수로 알린다 (`run.py --no-cache`).
        os.environ["WEB_SEARCH_NO_CACHE"] = "1"

    return Deps(
        retriever=VectorRetriever(chroma_dir=s.chroma_dir, embedding_model=s.embedding_model),
        web_search=web_search,
        llm=get_llm(s),
        judge_llm=get_judge_llm(s),
        now=now_iso,
    )
