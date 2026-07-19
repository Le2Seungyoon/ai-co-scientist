import pytest
from a2a.types import Message, Role

from ai_co_scientist.a2a.base import build_agent_card, extract_data_payload


def test_agent_card_fields():
    card = build_agent_card("research", "가설 담당", 9002)
    assert card.name == "research"
    # a2a-sdk 1.1.1: AgentCard에 최상위 url 필드가 없다 —
    # 실제 엔드포인트는 supported_interfaces[].url에 실린다 (deviation, see task-6-report.md)
    assert "9002" in card.supported_interfaces[0].url
    assert len(card.skills) == 1


def test_extract_data_payload():
    # a2a-sdk 1.1.1: Part는 discriminated union(.root)이 아니라
    # protobuf 메시지 — data 필드에 google.protobuf.Value를 직접 담는다.
    # DataPart 헬퍼 대신 a2a.helpers.new_data_message로 구성한다.
    from a2a.helpers import new_data_message

    msg = new_data_message(
        {"type": "cycle_context", "data": {"cycle_id": 1}},
        role=Role.ROLE_USER,
    )
    payload = extract_data_payload(msg)
    assert payload["type"] == "cycle_context"


def test_extract_without_datapart_raises():
    msg = Message(role=Role.ROLE_USER, message_id="m-2", parts=[])
    with pytest.raises(ValueError):
        extract_data_payload(msg)
