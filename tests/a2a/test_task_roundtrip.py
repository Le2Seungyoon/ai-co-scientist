import threading
import time

import httpx
import pytest
import uvicorn

from ai_co_scientist.a2a.base import build_app
from ai_co_scientist.a2a.client import PMClient
from ai_co_scientist.core.failure import AgentTaskFailure

PORT = 9098


async def slow_handler(payload: dict) -> dict:
    import asyncio
    await asyncio.sleep(0.3)
    if payload.get("data", {}).get("boom"):
        raise AgentTaskFailure({"type": "failure_event",
                                "data": {"cycle_id": 1, "category": "infra_oom", "detail": "주입"}})
    return {"type": "run_result", "data": {"cycle_id": 1, "metrics": {"val_mse": 0.1}}}


@pytest.fixture(scope="module")
def task_server():
    config = uvicorn.Config(
        build_app("executor-echo", "task 모드 테스트", PORT, slow_handler, mode="task"),
        host="127.0.0.1", port=PORT, log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/.well-known/agent-card.json",
                         timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        pytest.fail("task 서버가 10초 내에 뜨지 않음")
    yield
    server.should_exit = True
    thread.join(timeout=5)


async def test_submit_then_poll_completed(task_server):
    client = PMClient(f"http://127.0.0.1:{PORT}")
    try:
        task_id = await client.submit({"type": "code_artifact", "data": {"cycle_id": 1}})
        assert isinstance(task_id, str) and task_id
        outcome = await client.poll(task_id, interval=0.1, timeout=15)
        assert outcome.state == "completed"
        assert outcome.payload["type"] == "run_result"
    finally:
        await client.aclose()


async def test_poll_failed_carries_failure_payload(task_server):
    client = PMClient(f"http://127.0.0.1:{PORT}")
    try:
        task_id = await client.submit({"type": "code_artifact", "data": {"cycle_id": 1, "boom": True}})
        outcome = await client.poll(task_id, interval=0.1, timeout=15)
        assert outcome.state == "failed"
        assert outcome.payload["data"]["category"] == "infra_oom"
    finally:
        await client.aclose()


async def test_client_is_cached_across_calls(task_server):
    client = PMClient(f"http://127.0.0.1:{PORT}")
    try:
        await client.submit({"type": "code_artifact", "data": {"cycle_id": 1}})
        first = client._client
        await client.submit({"type": "code_artifact", "data": {"cycle_id": 2}})
        assert client._client is first  # 카드 재리졸브 없음
    finally:
        await client.aclose()


async def test_poll_timeout_zero_on_finished_task_returns(task_server):
    """이미 terminal인 task는 timeout=0이어도 즉시 결과를 반환해야 한다."""
    client = PMClient(f"http://127.0.0.1:{PORT}")
    try:
        task_id = await client.submit({"type": "code_artifact", "data": {"cycle_id": 9}})
        first = await client.poll(task_id, interval=0.1, timeout=15)
        assert first.state == "completed"
        # timeout=0이어도 이미 terminal인 task는 GetTask가 최소 1회 실행되어야 함
        again = await client.poll(task_id, interval=0.1, timeout=0)
        assert again.state == "completed"
    finally:
        await client.aclose()
