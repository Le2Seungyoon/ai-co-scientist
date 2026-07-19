from ai_co_scientist.core.mcp_client import mcp_session, tool_result_data

SERVER = "ai_co_scientist.mcp_servers.lightning.server"

GOOD = 'import json\nprint(json.dumps({"train_mse": 0.1, "val_mse": 0.2, "predictions_path": "p.csv"}))\n'
CRASH = "raise RuntimeError('boom')\n"
SLOW = "import time\ntime.sleep(3)\n"


def _write(tmp_path, content):
    p = tmp_path / "exp.py"
    p.write_text(content, encoding="utf-8")
    return str(p)


def _unwrap(res):
    data = tool_result_data(res)
    return data.get("result", data) if isinstance(data, dict) and "result" in data else data


async def test_submit_and_poll_completed(tmp_path):
    env = {"COSCIENTIST_LIGHTNING_DB": str(tmp_path / "l.sqlite3")}
    async with mcp_session(SERVER, extra_env=env) as session:
        sub = _unwrap(await session.call_tool("submit_job", {
            "entrypoint_path": _write(tmp_path, GOOD), "timeout_s": 5}))
        job = _unwrap(await session.call_tool("poll_job", {"job_id": sub["job_id"]}))
        assert job["status"] == "completed"
        assert job["metrics"]["val_mse"] == 0.2
        assert job["artifacts"]["predictions_path"] == "p.csv"   # 문자열 값은 artifacts로 분리


async def test_crash_reported_failed(tmp_path):
    env = {"COSCIENTIST_LIGHTNING_DB": str(tmp_path / "l.sqlite3")}
    async with mcp_session(SERVER, extra_env=env) as session:
        sub = _unwrap(await session.call_tool("submit_job", {
            "entrypoint_path": _write(tmp_path, CRASH), "timeout_s": 5}))
        job = _unwrap(await session.call_tool("poll_job", {"job_id": sub["job_id"]}))
        assert job["status"] == "failed" and "boom" in job["detail"]


async def test_timeout_status(tmp_path):
    env = {"COSCIENTIST_LIGHTNING_DB": str(tmp_path / "l.sqlite3")}
    async with mcp_session(SERVER, extra_env=env) as session:
        sub = _unwrap(await session.call_tool("submit_job", {
            "entrypoint_path": _write(tmp_path, SLOW), "timeout_s": 0.3}))
        job = _unwrap(await session.call_tool("poll_job", {"job_id": sub["job_id"]}))
        assert job["status"] == "timeout"


async def test_credits_deducted_and_rejected_at_zero(tmp_path):
    env = {"COSCIENTIST_LIGHTNING_DB": str(tmp_path / "l.sqlite3"),
           "COSCIENTIST_CREDITS": "1"}
    async with mcp_session(SERVER, extra_env=env) as session:
        path = _write(tmp_path, GOOD)
        first = _unwrap(await session.call_tool("submit_job", {"entrypoint_path": path, "timeout_s": 5}))
        assert first["job_id"]
        credits = _unwrap(await session.call_tool("get_credits", {}))
        assert credits["remaining"] == 0.0
        second = _unwrap(await session.call_tool("submit_job", {"entrypoint_path": path, "timeout_s": 5}))
        assert second["job_id"] == "" and second["rejected"] == "credit"


def test_deduct_conditional_update_prevents_overdraft(tmp_path):
    from ai_co_scientist.mcp_servers.lightning import db as ldb
    path = str(tmp_path / "l.sqlite3")
    ldb.init_credits(path, 1.5)
    assert ldb.deduct(path, 1.0) is True
    assert ldb.deduct(path, 1.0) is False   # 0.5 < 1.0 — 차감 없이 거절
    assert ldb.get_credits(path) == 0.5
