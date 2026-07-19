from ai_co_scientist.core.mcp_client import mcp_session, tool_result_data

SERVER = "ai_co_scientist.mcp_servers.websearch.server"


async def test_search_ranks_by_overlap():
    async with mcp_session(SERVER) as session:
        res = await session.call_tool("search", {"query": "overfitting 방지", "top_k": 2})
        rows = tool_result_data(res)
        if isinstance(rows, dict):
            rows = rows.get("result", rows)
        assert rows and "overfit" in rows[0]["title"].lower()
