import threading
import time

import httpx
import pytest
import uvicorn

from ai_co_scientist.a2a.base import build_app
from ai_co_scientist.a2a.client import PMClient

PORT = 9099


async def echo_handler(payload: dict) -> dict:
    return {"echo": payload}


@pytest.fixture(scope="module")
def echo_server():
    config = uvicorn.Config(
        build_app("echo", "테스트용 에코", PORT, echo_handler),
        host="127.0.0.1", port=PORT, log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    # a2a-sdk 1.1.1: 카드 경로가 /.well-known/agent-card.json으로 바뀜
    # (브리프 원안의 /.well-known/agent.json에서 deviation, task-6-report.md 참고)
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{PORT}/.well-known/agent-card.json", timeout=0.5)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    else:
        pytest.fail("에코 서버가 10초 내에 뜨지 않음")
    yield
    server.should_exit = True
    thread.join(timeout=5)


async def test_send_payload_roundtrip(echo_server):
    client = PMClient(f"http://127.0.0.1:{PORT}")
    result = await client.send({"type": "cycle_context", "data": {"cycle_id": 7}})
    # 주의(a2a-sdk 1.1.1): int는 Part.data(google.protobuf.Value)를 왕복하며 float가 된다
    # (7 -> 7.0). 아래 assert는 7.0 == 7이라 그대로 통과한다.
    assert result["echo"]["data"]["cycle_id"] == 7
