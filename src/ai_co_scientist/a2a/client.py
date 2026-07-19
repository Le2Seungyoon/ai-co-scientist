"""PM 전용 A2A 클라이언트 — star topology의 구조적 보장을 위해 PM 패키지만 import한다.

주의(a2a-sdk 1.1.1): 이 SDK 버전은 브리프가 가정한 pydantic 기반 API(0.2.x대,
DataPart/TextPart/FilePart 클래스, Part.root, `A2AClient`/`Message(role=Role.user, ...)`
등)가 아니라 protobuf 메시지 기반 API다. 세부 편차는 task-7-report.md의
"SDK 편차 로그" 참고 (base.py/task-6-report.md와 동일 계열의 편차).

- `A2ACardResolver` + `ClientFactory`는 브리프 그대로 존재하지만 생성자 인자
  순서/키워드가 다르다(`A2ACardResolver(httpx_client, base_url)`는 위치 인자).
- `ClientFactory(config).create(card)`가 반환하는 `Client.send_message()`는
  `Message`가 아니라 `SendMessageRequest`(message 필드를 감싼 protobuf 요청)를
  받고, `AsyncIterator[StreamResponse]`(task/message/status_update/artifact_update
  중 하나가 set된 protobuf)를 yield한다.
- `DataPart`/`Part(root=...)`는 존재하지 않는다. 대신 `a2a.helpers.new_data_part`로
  data Part를 만들고, `a2a.helpers.get_data_parts`로 dict를 꺼낸다(base.py와 동일 패턴).
- `Role.user`/`Role.agent`가 아니라 `Role.ROLE_USER`/`Role.ROLE_AGENT`.
- 서버(base.py)가 스트리밍 미지원(capabilities.streaming=False)이므로
  `ClientConfig(streaming=False)`로 맞춰 단일 응답만 받는다(M1 스코프 = 동기 왕복).
  이 경우 서버가 Message를 바로 반환하면 응답은 `StreamResponse.message`에 실린다
  (Task로 승격되지 않음 — DefaultRequestHandler.on_message_send 참고).

M2: 카드/클라이언트 캐싱 + task 모드(submit → GetTask 폴링) 추가 (scout §3·§4).
task-3-brief.md의 draft 코드를 .venv의 실제 SDK와 대조 검증했고 편차 없음
(GetTaskRequest(id=...), TaskState.TASK_STATE_*, SendMessageRequest.configuration.
return_immediately, StreamResponse.HasField("task") 모두 브리프와 동일).
"""
import asyncio
import time
from dataclasses import dataclass

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import get_data_parts, new_data_part, new_message
from a2a.types import GetTaskRequest, Role, SendMessageRequest, TaskState

_STATE_NAMES = {
    TaskState.TASK_STATE_COMPLETED: "completed",
    TaskState.TASK_STATE_FAILED: "failed",
    TaskState.TASK_STATE_CANCELED: "canceled",
    TaskState.TASK_STATE_REJECTED: "rejected",
    TaskState.TASK_STATE_INPUT_REQUIRED: "input_required",
    TaskState.TASK_STATE_AUTH_REQUIRED: "auth_required",
}


@dataclass
class TaskOutcome:
    state: str
    payload: dict | None


def _extract_data(stream_response) -> dict | None:
    """StreamResponse(message 또는 task)에서 첫 DataPart(dict)를 꺼낸다. 없으면 None.

    M1 send()의 응답 추출 로직 — message 모드(task로 승격되지 않는 동기 왕복) 그대로 유지.
    """
    if stream_response.HasField("message"):
        parts = get_data_parts(stream_response.message.parts)
        if parts:
            return parts[0]
    elif stream_response.HasField("task"):
        for artifact in stream_response.task.artifacts:
            parts = get_data_parts(artifact.parts)
            if parts:
                return parts[0]
        for msg in reversed(stream_response.task.history):
            if msg.role == Role.ROLE_AGENT:
                parts = get_data_parts(msg.parts)
                if parts:
                    return parts[0]
    return None


def _task_payload(task) -> dict | None:
    """완료면 artifacts에서, 실패/입력요구면 status.message → history 순으로 DataPart 추출.

    # 주의: artifacts 우선 추출은 '실패 task에는 artifact가 없다'는 TaskPayloadExecutor 불변식에 의존 (base.py)
    """
    for artifact in task.artifacts:
        parts = get_data_parts(artifact.parts)
        if parts:
            return parts[0]
    if task.status.HasField("message"):
        parts = get_data_parts(task.status.message.parts)
        if parts:
            return parts[0]
    for msg in reversed(task.history):
        if msg.role == Role.ROLE_AGENT:
            parts = get_data_parts(msg.parts)
            if parts:
                return parts[0]
    return None


class PMClient:
    """PM(프로젝트 매니저) 에이전트가 다른 에이전트의 A2A 서버와 통신하기 위한 클라이언트.

    star topology 보장을 위해 이 모듈은 `a2a.*`와 `httpx` 외의 다른 에이전트 패키지를
    import하지 않는다.

    카드/클라이언트는 인스턴스 내 1회만 리졸브하고 캐싱한다(§9 반영사항 4) — 매 호출마다
    `A2ACardResolver.get_agent_card()`를 다시 부르지 않는다. `aclose()`로 내부 httpx
    클라이언트를 정리하면 다음 호출에서 재리졸브한다.
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
            card = await A2ACardResolver(
                self._http, self._base_url).get_agent_card()
            self._client = ClientFactory(
                ClientConfig(httpx_client=self._http, streaming=False)).create(card)
        return self._client

    async def aclose(self) -> None:
        """내부 httpx 클라이언트를 정리하고 캐시를 비운다."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
            self._client = None

    def _build_request(self, payload: dict) -> SendMessageRequest:
        return SendMessageRequest(message=new_message(
            parts=[new_data_part(payload)], role=Role.ROLE_USER))

    async def send(self, payload: dict) -> dict:
        """payload(dict)를 DataPart로 보내고 응답의 DataPart(dict)를 반환. 동기 왕복(M1 계약 유지)."""
        client = await self._ensure_client()
        async for stream_response in client.send_message(self._build_request(payload)):
            data = _extract_data(stream_response)
            if data is not None:
                return data
        raise RuntimeError("A2A 응답에서 DataPart를 찾지 못함")

    async def submit(self, payload: dict) -> str:
        """비차단 제출 — 첫 Task 이벤트(SUBMITTED)에서 task_id 반환. executor는 백그라운드 계속."""
        client = await self._ensure_client()
        request = self._build_request(payload)
        request.configuration.return_immediately = True
        async for stream_response in client.send_message(request):
            if stream_response.HasField("task"):
                return stream_response.task.id
        raise RuntimeError("submit 응답에서 Task 이벤트를 받지 못함")

    async def poll(self, task_id: str, interval: float = 0.5,
                    timeout: float = 120.0) -> TaskOutcome:
        """GetTask 반복 — terminal/INPUT_REQUIRED 도달 시 TaskOutcome 반환. 초과 시 TimeoutError.

        최소 1회 GetTask 호출 보장(timeout=0일 때도 이미 terminal인 task는 즉시 결과 반환).
        """
        client = await self._ensure_client()
        deadline = time.monotonic() + timeout
        while True:
            task = await client.get_task(GetTaskRequest(id=task_id))
            name = _STATE_NAMES.get(task.status.state)
            if name is not None:
                return TaskOutcome(state=name, payload=_task_payload(task))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"task {task_id} 폴링 타임아웃 ({timeout}s)")
            await asyncio.sleep(interval)
