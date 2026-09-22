"""B 통합 테스트 — 실제 retriever + 실제 LLM으로 `mla` 1회 (docs/roles/B-tech-domain.md §6).

실행 조건: `scripts/ingest.py` 완료 + `.env`에 LLM_PROVIDER/LLM_MODEL. 없으면 skip.
    uv run pytest -m integration tests/agents/test_integration_b.py -s
"""

import logging
from pathlib import Path

import pytest

from techeval.agents._deps import AgentInput
from techeval.agents.domain import DOMAIN_CRITERIA, run_domain_eval
from techeval.agents.tech_research import TRL_CRITERIA, run_tech_research
from techeval.schemas import CriterionResult, TechProfile, compute_confidence, get_tech

pytestmark = pytest.mark.integration


class _DropCounter(logging.Handler):
    """인용 폐기 WARNING 개수 = 프롬프트 튜닝·모델 선정의 판단 근거."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.dropped = 0

    def emit(self, record):
        if "없음" in record.getMessage() or "폐기" in record.getMessage():
            self.dropped += 1


@pytest.fixture(scope="module")
def real_deps():
    from techeval.config import Settings, build_deps

    s = Settings()
    if not (s.llm_provider and s.llm_model):
        pytest.skip(".env 에 LLM_PROVIDER / LLM_MODEL 필요")
    if not Path(s.chroma_dir).exists():
        pytest.skip("scripts/ingest.py 를 먼저 실행")
    return build_deps(stub=False, settings=s)


def test_real_mla_tech_research_then_domain(real_deps):
    counter = _DropCounter()
    logging.getLogger("techeval.agents").addHandler(counter)
    tech = get_tech("mla")

    out = run_tech_research(AgentInput(tech=tech), real_deps)
    TechProfile.model_validate(out.tech_profile.model_dump())
    assert [r.criterion_id for r in out.trl_eval] == list(TRL_CRITERIA)
    for r in out.trl_eval:
        CriterionResult.model_validate(r.model_dump())
        assert r.confidence == compute_confidence(r.evidence)

    d = run_domain_eval(
        AgentInput(tech=tech, tech_profile=out.tech_profile, trl_eval=out.trl_eval),
        real_deps,
    )
    assert [r.criterion_id for r in d] == list(DOMAIN_CRITERIA)
    for r in d:
        CriterionResult.model_validate(r.model_dump())

    not_public = [r.criterion_id for r in out.trl_eval + d if r.level == "not_public"]
    print(f"\n[B integration] 인용 폐기 {counter.dropped}건, not_public={not_public}")
    print(f"[B integration] T: {[(r.criterion_id, r.level) for r in out.trl_eval]}")
    print(f"[B integration] D: {[(r.criterion_id, r.level) for r in d]}")
