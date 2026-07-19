from ai_co_scientist.core.mcp_client import mcp_session, tool_result_data

SERVER = "ai_co_scientist.mcp_servers.wandb_tools.server"


async def test_log_and_query_metrics(tmp_path):
    env = {"COSCIENTIST_WANDB_DB": str(tmp_path / "wandb.sqlite3")}
    async with mcp_session(SERVER, extra_env=env) as session:
        res = await session.call_tool("log_metrics", {
            "run_id": "run-1", "cycle_id": 1, "metrics": {"val_mse": 0.2}})
        assert not res.isError
        res = await session.call_tool("query_runs", {"cycle_id": 1})
        rows = tool_result_data(res)
        if isinstance(rows, dict):
            rows = rows.get("result", rows)
        assert len(rows) == 1 and rows[0]["metrics"]["val_mse"] == 0.2
