"""B 픽스처 3종 생성기 — tests/fixtures/{tech_profiles,trl_eval,domain_eval}.json.

`_b_drafts.py`의 초안을 실제 에이전트 함수(run_tech_research / run_domain_eval)에 통과시켜 만든다.
따라서 픽스처의 모든 Evidence 는 A 의 chunks.json / C 의 web_results.json 에 실존하는 인용이다.

    uv run python -m tests.agents.make_b_fixtures
"""

import json
from pathlib import Path

from techeval.agents._deps import AgentInput
from techeval.agents.domain import run_domain_eval
from techeval.agents.tech_research import run_tech_research
from techeval.schemas import TECHNOLOGIES
from tests.agents._b_drafts import drafts
from tests.agents._b_helpers import FIXTURES, make_deps, make_llm


def build() -> dict[str, list[dict]]:
    profiles, trl, domain = [], [], []
    for tech in TECHNOLOGIES:
        deps = make_deps(make_llm(drafts(tech.tech_id)))
        out = run_tech_research(AgentInput(tech=tech), deps)
        assert len(out.trl_eval) == 4, (
            tech.tech_id,
            [r.criterion_id for r in out.trl_eval],
        )
        d = run_domain_eval(
            AgentInput(tech=tech, tech_profile=out.tech_profile, trl_eval=out.trl_eval),
            deps,
        )
        assert len(d) == 4, (tech.tech_id, [r.criterion_id for r in d])
        profiles.append(out.tech_profile.model_dump())
        trl += [r.model_dump() for r in out.trl_eval]
        domain += [r.model_dump() for r in d]
    return {
        "tech_profiles.json": profiles,
        "trl_eval.json": trl,
        "domain_eval.json": domain,
    }


def main(out_dir: Path = FIXTURES) -> None:
    for name, data in build().items():
        (out_dir / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {out_dir / name} ({len(data)} items)")


if __name__ == "__main__":
    main()
