import re
from pathlib import Path

import pytest

from techeval.agents._deps import Deps, ReportInput
from techeval.agents.report import map_instructions_to_units, run_report
from techeval.report.citation import build_evidence_index
from techeval.report.lint import lint_report
from techeval.report.sections import split_sections
from techeval.report.tables import TABLE_HEADER
from techeval.schemas import (
    DOMAIN,
    TECHNOLOGIES,
    CriterionResult,
    Evidence,
    EvidenceGap,
    JudgeResult,
    SynthesisResult,
    TechProfile,
)
from tests.report.d_fixtures import load

ID_RE = re.compile(r"\b(?:mla|pim_cxl)-[A-Z]+[0-9]*-\d{2}\b")


class TextStub:
    """절 프롬프트마다 입력 데이터에 있는 evidence_id 하나를 인용한 문장을 돌려준다."""

    def __init__(self, tag: str = "v1", extra: str = ""):
        self.tag, self.extra = tag, extra
        self.calls: list[str] = []

    def with_structured_output(self, model):
        raise AssertionError("run_report는 구조화 출력을 쓰지 않는다")

    def invoke(self, messages):
        human = messages[-1][1]
        self.calls.append(human)
        head = human.splitlines()[0]
        data = human.split("# 입력 데이터", 1)[1]
        eid = ID_RE.search(data).group(0)

        class R:
            content = f"{self.tag} {head} 서술이다[E: {eid}].{self.extra}"

        return R()


class SearchSpy:
    def __init__(self):
        self.called = False

    def __call__(self, *a, **k):
        self.called = True
        raise AssertionError("report는 검색을 호출하면 안 된다")

    search = __call__


@pytest.fixture
def inp() -> ReportInput:
    return ReportInput(
        technologies=TECHNOLOGIES,
        domain=DOMAIN,
        tech_profiles=load("tech_profiles.json", TechProfile),
        trl_eval=load("trl_eval.json", CriterionResult),
        market_eval=load("market_eval.json", CriterionResult),
        stakeholder_eval=load("stakeholder_eval.json", CriterionResult),
        domain_eval=load("domain_eval.json", CriterionResult),
        counter_evidence=load("counter_evidence.json", Evidence),
        synthesis=load("synthesis.json", SynthesisResult),
        evidence_gap=load("evidence_gap.json", EvidenceGap),
    )


def _index(inp: ReportInput) -> dict[str, Evidence]:
    results = [*inp.trl_eval, *inp.market_eval, *inp.stakeholder_eval, *inp.domain_eval]
    return build_evidence_index(
        results=results,
        profiles=inp.tech_profiles,
        counter_evidence=inp.counter_evidence,
    )


def _deps(llm) -> tuple[Deps, SearchSpy]:
    spy = SearchSpy()
    return Deps(
        retriever=spy, web_search=spy, llm=llm, now=lambda: "2026-01-01T00:00:00"
    ), spy


def test_report_structure_and_lint(inp):
    llm = TextStub()
    deps, spy = _deps(llm)
    md = run_report(inp, deps)
    assert not spy.called
    res = lint_report(md, _index(inp))
    assert res.passed, (res.errors, res.missing_required)
    keys = [s.key for s in split_sections(md)]
    for k in (
        "SUMMARY",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "REFERENCE",
        "4.1",
        "4.2",
        "4.3",
        "4.4",
    ):
        assert k in keys
    assert md.count(TABLE_HEADER) == 4
    # 챕터 단위 호출: SUMMARY, 1, 2(kv_survey 근거 있음), 3.1, 3.2, 4.1~4.4, 5, 6 = 11회
    assert len(llm.calls) == 11
    # 각 절 프롬프트에는 그 절에 필요한 데이터만: 4.2(시장) 프롬프트에 S4 결과가 들어가지 않는다
    market_prompt = next(c for c in llm.calls if "(market)\n" in c)
    assert "/ S4" not in market_prompt and "/ M2" in market_prompt


def test_unknown_citation_from_llm_is_removed(inp):
    llm = TextStub(extra=" 추가 문장이다[E: mla-FAKE-01].")
    md = run_report(inp, _deps(llm)[0])
    assert "mla-FAKE-01" not in md
    assert lint_report(md, _index(inp)).passed


def test_banned_term_triggers_one_fix_pass(inp):
    class BannedOnce(TextStub):
        def invoke(self, messages):
            resp = super().invoke(messages)
            if "5. 시사점" in messages[-1][1].splitlines()[0]:
                resp.content += " 따라서 MLA를 추천한다."
            return resp

    llm = BannedOnce()
    md = run_report(inp, _deps(llm)[0])
    assert any("# 수정 모드" in c and "금칙어 '추천'" in c for c in llm.calls)
    # 스텁은 수정 요청에도 같은 문장을 붙이므로 남아 있을 수 있다 — 단, 수정 호출은 5장에만 1회
    assert sum("# 수정 모드" in c for c in llm.calls) == 1
    assert md


def test_regeneration_only_changes_target_chapters(inp):
    first = run_report(inp, _deps(TextStub("v1"))[0])
    judge = load("judge_result.json", JudgeResult)  # 지시: 5장, 4.4
    inp2 = inp.model_copy(update={"judge_result": judge, "previous_report_md": first})
    llm2 = TextStub("v2")
    second = run_report(inp2, _deps(llm2)[0])

    before = {s.key: s.text for s in split_sections(first)}
    after = {s.key: s.text for s in split_sections(second)}
    changed = {k for k in after if after[k] != before.get(k) and k != "REFERENCE"}
    assert changed == {"5", "4.4"}
    assert len(llm2.calls) == 2 and all("# 수정 모드" in c for c in llm2.calls)
    assert "v1 " in llm2.calls[0]  # 이전 텍스트를 기반으로 수정
    assert lint_report(second, _index(inp)).passed


def test_map_instructions_to_units():
    mapping, unmapped = map_instructions_to_units(
        [
            "5장 시사점: 원인을 평가 단위 차이로",
            "4.4 도메인 적합성: D2 조건 병기",
            "SUMMARY 문장을 4문장 이내로",
            "T1 올리지 못한 이유 누락",
            "REFERENCE 형식 맞추기",
            "전반적으로 문장을 다듬어라",
        ]
    )
    assert set(mapping) == {
        "5",
        "4.4",
        "SUMMARY",
        "4.1",
    }  # 명시 표기(4.4)가 있으면 키워드(D2 등)로 넓히지 않음
    assert mapping["4.1"] == ["T1 올리지 못한 이유 누락"]
    assert unmapped == ["전반적으로 문장을 다듬어라"]


@pytest.mark.integration
def test_real_llm_report_passes_lint(inp, tmp_path):
    config = pytest.importorskip("techeval.config")
    from techeval.agents._deps import SynthesisInput
    from techeval.agents.synthesis import run_synthesis
    from techeval.report.pdf import render_pdf

    deps = Deps(
        retriever=None,
        web_search=None,
        llm=config.get_llm(),
        now=lambda: "2026-01-01T00:00:00",
    )
    syn = run_synthesis(
        SynthesisInput(**inp.model_dump(include=set(SynthesisInput.model_fields))), deps
    )
    md = run_report(inp.model_copy(update={"synthesis": syn}), deps)
    res = lint_report(md, _index(inp))
    assert res.passed, (res.errors, res.missing_required)
    assert Path(render_pdf(md, str(tmp_path / "report.pdf"))).stat().st_size > 0
