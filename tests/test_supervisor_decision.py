"""supervisor 판단 스키마(`SupervisorDecision`) 검증: 잘못된 action·기술 ID·담당 밖 기준 차단."""

import pytest
from pydantic import ValidationError

from techeval.schemas import SupervisorDecision


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"next": "perspectives", "dispatch": {"market_eval": {"gpu": ["M1"]}}}, "알 수 없는 기술"),
        ({"next": "perspectives", "dispatch": {"market_eval": {"mla": ["S1"]}}}, "담당 관점 밖"),
        ({"next": "perspectives", "dispatch": {"tech_research": {"mla": []}}}, "배정할 수 없는 노드"),
        ({"next": "tech_research", "dispatch": {}}, "배정이 비어"),
        ({"next": "report", "dispatch": {"market_eval": {"mla": []}}}, "배정할 수 없는 노드"),
    ],
)
def test_supervisor_decision_rejects_bad_dispatch(kwargs, msg):
    with pytest.raises(ValidationError, match=msg):
        SupervisorDecision(reason="r", **kwargs)


def test_supervisor_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        SupervisorDecision(next="delete_everything", reason="r")
    ok = SupervisorDecision(
        next="tech_research", dispatch={"tech_research": {"mla": ["T2", "PROFILE:measurements"]}}, reason="r"
    )
    assert ok.next == "tech_research"
