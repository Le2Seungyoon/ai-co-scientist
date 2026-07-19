"""A2A 서버 공통 골격 — 모든 에이전트 server.py가 이걸로 기동한다.

에이전트는 handler(payload dict → 응답 dict) 하나만 구현하면 되고,
A2A 프로토콜 처리(카드, task, 메시지 파싱)는 여기서 흡수한다.

주의(a2a-sdk 1.1.1): 이 SDK 버전은 브리프가 가정한 pydantic 기반 API(0.2.x대,
DataPart/TextPart/FilePart, Part.root, A2AStarletteApplication, new_agent_parts_message 등)
가 아니라 protobuf 메시지 기반 API다. 세부 편차는 task-6-report.md의
"SDK 편차 로그" 참고.
"""
from collections.abc import Awaitable, Callable

import uvicorn
from a2a.helpers import get_data_parts, new_data_message, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Message
from a2a.utils import TransportProtocol
from a2a.utils.constants import DEFAULT_RPC_URL, PROTOCOL_VERSION_1_0
from starlette.applications import Starlette

from ai_co_scientist.core.config import ensure_utf8_console
from ai_co_scientist.core.failure import AgentInputRequired, AgentTaskFailure

PayloadHandler = Callable[[dict], Awaitable[dict]]


def extract_data_payload(message: Message) -> dict:
    """Message에서 첫 DataPart의 data(dict)를 꺼낸다.

    1.1.1에서 Part는 discriminated union이 아니라 flat proto 메시지이므로
    `a2a.helpers.get_data_parts`(내부적으로 `part.HasField("data")` 체크 +
    `google.protobuf.json_format.MessageToDict`)를 사용한다.
    """
    parts = get_data_parts(message.parts)
    if not parts:
        raise ValueError("메시지에 DataPart가 없음")
    return parts[0]


class PayloadExecutor(AgentExecutor):
    """DataPart(JSON) 요청 → handler 호출 → DataPart 응답."""

    def __init__(self, handler: PayloadHandler):
        self._handler = handler

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        payload = extract_data_payload(context.message)
        try:
            result = await self._handler(payload)
        except AgentTaskFailure as e:
            # 메시지 모드의 구조적 실패 신고 채널: failure_event payload를 정상 응답으로
            result = e.payload
        await event_queue.enqueue_event(new_data_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("M1에서는 cancel 미지원")


class TaskPayloadExecutor(AgentExecutor):
    """장기 작업용 task 모드 executor — submit → 폴링(GetTask) 흐름의 서버 측.

    이벤트 규약(scout §2·§5): 첫 이벤트는 반드시 Task 객체여야 하고,
    실패는 반드시 status.message를 채워서 신고한다(사유 없는 FAILED 방지).
    """

    def __init__(self, handler: PayloadHandler):
        self._handler = handler

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        try:
            payload = extract_data_payload(context.message)
            result = await self._handler(payload)
        except AgentInputRequired as e:
            await updater.requires_input(
                message=updater.new_agent_message([new_data_part(e.payload)]))
            return
        except AgentTaskFailure as e:
            await updater.failed(
                message=updater.new_agent_message([new_data_part(e.payload)]))
            return
        except Exception as e:  # noqa: BLE001 — 사유 없는 FAILED 방지가 목적
            await updater.failed(message=updater.new_agent_message(
                [new_data_part({"type": "error", "data": {"detail": str(e)}})]))
            return
        try:
            await updater.add_artifact([new_data_part(result)])
        except Exception as e:  # noqa: BLE001 — 직렬화 실패도 WORKING 방치 대신 사유 있는 FAILED로
            await updater.failed(message=updater.new_agent_message(
                [new_data_part({"type": "error",
                                "data": {"detail": f"결과 직렬화 실패: {e}"}})]))
            return
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("M2에서는 cancel 미지원")


def build_agent_card(name: str, description: str, port: int) -> AgentCard:
    """1.1.1 AgentCard에는 최상위 `url` 필드가 없다 — 엔드포인트는
    `supported_interfaces`(AgentInterface 목록)에 실린다."""
    url = f"http://127.0.0.1:{port}/"
    return AgentCard(
        name=name,
        description=description,
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
        version="0.1.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id=f"{name}-main",
                name=f"{name} 기본 스킬",
                description=description,
                tags=[name],
            )
        ],
    )


def build_app(name: str, description: str, port: int, handler: PayloadHandler,
              mode: str = "message"):
    """테스트에서 in-process로 띄울 수 있게 ASGI app을 분리.

    `A2AStarletteApplication`은 1.1.1에 존재하지 않는다 — 대신
    `a2a.server.routes`의 라우트 팩토리들로 Starlette app을 직접 조립한다.
    REST 트랜스포트(create_rest_routes)는 M1 스코프 밖이라 제외(JSON-RPC만 노출).

    mode="task"면 장기 작업용 TaskPayloadExecutor(submit → GetTask 폴링)를,
    기본값 "message"면 blocking 요청-응답용 PayloadExecutor를 사용한다.
    """
    card = build_agent_card(name, description, port)
    executor = TaskPayloadExecutor(handler) if mode == "task" else PayloadExecutor(handler)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(request_handler, rpc_url=DEFAULT_RPC_URL),
    ]
    return Starlette(routes=routes)


def serve(name: str, description: str, port: int, handler: PayloadHandler,
          mode: str = "message") -> None:
    """에이전트 server.py의 main — blocking."""
    ensure_utf8_console()
    uvicorn.run(
        build_app(name, description, port, handler, mode=mode),
        host="127.0.0.1", port=port, log_level="warning",
    )
