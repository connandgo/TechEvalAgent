"""integration: 실제 모델·검색기로 파이프라인 1회 실행 (`uv run pytest -m integration`).

.env에 LLM 설정이 없거나 A/C/D 실제 모듈이 아직 없으면 skip한다. 산출물과 judge 점수를 로그로 남긴다.
"""

import logging
import os

import pytest

from techeval.config import build_deps, load_settings
from techeval.graph import GraphConfig, build_graph, invoke_config, load_agents

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration


def test_e2e_real_run(tmp_path):
    settings = load_settings()
    if not (settings.llm_model and settings.judge_model and settings.llm_provider):
        pytest.skip("LLM_PROVIDER/LLM_MODEL/JUDGE_MODEL 미설정")
    try:
        agents, stubbed = load_agents(stub=False, allow_stub_fallback=bool(os.environ.get("E2E_ALLOW_STUB")))
    except ImportError as e:
        pytest.skip(str(e))
    deps = build_deps(stub=False, settings=settings)
    cfg = GraphConfig(output_dir=str(tmp_path), skip_pdf=bool(os.environ.get("E2E_SKIP_PDF")))
    final = build_graph(deps, agents, cfg).invoke({}, config=invoke_config(cfg))

    assert (tmp_path / "report.md").exists()
    jr = final["judge_result"]
    logger.info("E2E judge: passed=%s scores=%s missing=%s", jr.passed, jr.scores, jr.missing_required)
    logger.info("E2E retry_counts=%s stubbed=%s pdf=%s", final["retry_counts"], stubbed, final.get("report_pdf_path"))
    assert final["report_md"]
    if not cfg.skip_pdf:
        assert final["report_pdf_path"]
