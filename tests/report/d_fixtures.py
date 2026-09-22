"""D 테스트용 픽스처 로더. A·B·C·E 픽스처는 모두 `tests/fixtures/`의 것을 쓴다.

D 픽스처(synthesis.json, report_md.md)는 이 픽스처들로 생성했으므로, 상류 픽스처가 바뀌면 다시 생성한다:
`uv run python -m tests.report.make_d_fixtures`
"""

import json
from pathlib import Path

from techeval.report.citation import build_evidence_index
from techeval.schemas import CriterionResult, Evidence, TechProfile

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EVAL_FILES = ("trl_eval", "market_eval", "stakeholder_eval", "domain_eval")


def fixture_path(name: str) -> Path:
    return FIXTURES / name


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
