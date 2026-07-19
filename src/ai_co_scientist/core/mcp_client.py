"""MCP stdio 클라이언트 헬퍼 — 에이전트가 MCP 서버를 서브프로세스로 spawn해 tool 호출."""
import json
import sys
from contextlib import asynccontextmanager
from os import environ
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def mcp_session(server_module: str, extra_env: dict | None = None):
    """`python -m <server_module>`을 stdio로 spawn하고 초기화된 세션을 yield."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", server_module],
        env={**environ, **(extra_env or {})},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def tool_result_data(result) -> Any:
    """call_tool 결과에서 데이터 추출 — structuredContent 우선, 없으면 text JSON 파싱."""
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
