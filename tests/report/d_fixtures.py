"""D 테스트용 픽스처 로더.

C 픽스처(market_eval / stakeholder_eval)가 main에 올라오기 전까지는 D가 CONTRACTS 스키마로 만든 샘플
(`tests/report/upstream_samples/`)을 쓴다. C 픽스처가 올라오면 샘플을 지우고 D 픽스처(synthesis.json, report_md.md)를
다시 생성한다(`uv run python -m tests.report.make_d_fixtures`).
B 픽스처(tech_profiles / trl_eval / domain_eval)와 E 픽스처(counter_evidence / evidence_gap / judge_result)는
`tests/fixtures/`의 것을 그대로 쓴다.
"""

import json
from pathlib import Path

from techeval.report.citation import build_evidence_index
from techeval.schemas import CriterionResult, Evidence, TechProfile

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLES = Path(__file__).resolve().parent / "upstream_samples"
UPSTREAM_SAMPLES = {"market_eval.json", "stakeholder_eval.json"}
EVAL_FILES = ("trl_eval", "market_eval", "stakeholder_eval", "domain_eval")


def fixture_path(name: str) -> Path:
    return (SAMPLES if name in UPSTREAM_SAMPLES else FIXTURES) / name


def load(name: str, model=None):
    path = fixture_path(name)
    if name.endswith(".md"):
        return path.read_text(encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    if model is None:
        return data
    if isinstance(data, list):
        return [model.model_validate(x) for x in data]
    return model.model_validate(data)


def load_evals() -> dict[str, list[CriterionResult]]:
    return {k: load(f"{k}.json", CriterionResult) for k in EVAL_FILES}


def load_index() -> dict[str, Evidence]:
    results = [r for rs in load_evals().values() for r in rs]
    return build_evidence_index(
        results=results,
        profiles=load("tech_profiles.json", TechProfile),
        counter_evidence=load("counter_evidence.json", Evidence),
    )
