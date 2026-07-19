from ai_co_scientist.core.mcp_client import mcp_session, tool_result_data

SERVER = "ai_co_scientist.mcp_servers.shared_log.server"


async def test_log_append_and_query_via_mcp(tmp_path):
    env = {"COSCIENTIST_LOG_DB": str(tmp_path / "log.sqlite3")}
    async with mcp_session(SERVER, extra_env=env) as session:
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        assert {"log_append", "query_ledger", "get_consensus",
                "update_consensus", "update_task_status"} <= names

        res = await session.call_tool("log_append", {
            "cycle_id": 1, "record_type": "hypothesis", "owner": "research",
            "content": {"statement": "더미 가설"},
        })
        assert not res.isError

        res = await session.call_tool("query_ledger", {"record_type": "hypothesis"})
        rows = tool_result_data(res)
        # FastMCP는 list 반환을 {"result": [...]}로 감쌀 수 있음 — 둘 다 허용
        if isinstance(rows, dict):
            rows = rows.get("result", rows)
        assert len(rows) == 1
        assert rows[0]["content"]["statement"] == "더미 가설"


async def test_db_file_is_created_at_env_path(tmp_path):
    db_file = tmp_path / "log.sqlite3"
    env = {"COSCIENTIST_LOG_DB": str(db_file)}
    async with mcp_session(SERVER, extra_env=env) as session:
        await session.call_tool("update_consensus", {
            "key": "best_pipeline", "value": {"desc": "baseline"},
        })
    assert db_file.exists()
