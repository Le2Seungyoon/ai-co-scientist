from a2a.helpers import get_data_parts, new_data_part, new_message
from a2a.types import Role, TaskState

from ai_co_scientist.a2a.base import PayloadExecutor, TaskPayloadExecutor, extract_data_payload
from ai_co_scientist.core.failure import AgentInputRequired, AgentTaskFailure


class FakeQueue:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


class FakeContext:
    def __init__(self, payload: dict):
        self.message = new_message(parts=[new_data_part(payload)], role=Role.ROLE_USER)


PAYLOAD = {"type": "code_artifact", "data": {"cycle_id": 1}}


async def test_task_executor_success_sequence():
    async def handler(payload):
        return {"type": "run_result", "data": {"cycle_id": 1, "metrics": {"m": 1.0}}}

    q = FakeQueue()
    await TaskPayloadExecutor(handler).execute(FakeContext(PAYLOAD), q)
    kinds = [type(e).__name__ for e in q.events]
    assert kinds == ["Task", "TaskStatusUpdateEvent", "TaskArtifactUpdateEvent", "TaskStatusUpdateEvent"]
    assert q.events[0].status.state == TaskState.TASK_STATE_SUBMITTED
    assert q.events[1].status.state == TaskState.TASK_STATE_WORKING
    assert q.events[-1].status.state == TaskState.TASK_STATE_COMPLETED
    artifact_payload = get_data_parts(q.events[2].artifact.parts)[0]
    assert artifact_payload["type"] == "run_result"


async def test_task_executor_failure_carries_reason():
    async def handler(payload):
        raise AgentTaskFailure({"type": "failure_event", "data": {"cycle_id": 1, "category": "infra_oom", "detail": "OOM"}})

    q = FakeQueue()
    await TaskPayloadExecutor(handler).execute(FakeContext(PAYLOAD), q)
    final = q.events[-1]
    assert final.status.state == TaskState.TASK_STATE_FAILED
    reason = get_data_parts(final.status.message.parts)[0]
    assert reason["data"]["category"] == "infra_oom"


async def test_task_executor_input_required():
    async def handler(payload):
        raise AgentInputRequired({"type": "error", "data": {"detail": "사람 확인 필요"}})

    q = FakeQueue()
    await TaskPayloadExecutor(handler).execute(FakeContext(PAYLOAD), q)
    assert q.events[-1].status.state == TaskState.TASK_STATE_INPUT_REQUIRED


async def test_task_executor_unexpected_exception_still_has_message():
    async def handler(payload):
        raise RuntimeError("예상 못한 버그")

    q = FakeQueue()
    await TaskPayloadExecutor(handler).execute(FakeContext(PAYLOAD), q)
    final = q.events[-1]
    assert final.status.state == TaskState.TASK_STATE_FAILED
    assert "예상 못한 버그" in get_data_parts(final.status.message.parts)[0]["data"]["detail"]


async def test_message_executor_converts_agent_failure_to_payload():
    async def handler(payload):
        raise AgentTaskFailure({"type": "failure_event", "data": {"cycle_id": 1, "category": "impl_bug", "detail": "x"}})

    q = FakeQueue()
    await PayloadExecutor(handler).execute(FakeContext(PAYLOAD), q)
    reply = extract_data_payload(q.events[-1])
    assert reply["type"] == "failure_event"


async def test_task_executor_unserializable_result_fails_with_reason():
    async def handler(payload):
        return {"type": "run_result", "data": {"bad": object()}}  # Struct 직렬화 불가

    q = FakeQueue()
    await TaskPayloadExecutor(handler).execute(FakeContext(PAYLOAD), q)
    final = q.events[-1]
    assert final.status.state == TaskState.TASK_STATE_FAILED
    assert "직렬화 실패" in get_data_parts(final.status.message.parts)[0]["data"]["detail"]
