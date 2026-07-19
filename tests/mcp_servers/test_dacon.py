from ai_co_scientist.core.mcp_client import mcp_session, tool_result_data

SERVER = "ai_co_scientist.mcp_servers.dacon.server"


def _unwrap(res):
    data = tool_result_data(res)
    return data.get("result", data) if isinstance(data, dict) and "result" in data else data


def _setup(tmp_path, preds):
    toy = tmp_path / "toy"
    toy.mkdir()
    (toy / "holdout.csv").write_text("x,y\n0,1.0\n1,2.0\n2,3.0\n3,4.0\n", encoding="utf-8")
    p = tmp_path / "preds.csv"
    p.write_text("id,pred\n" + "\n".join(f"{i},{v}" for i, v in enumerate(preds)) + "\n",
                 encoding="utf-8")
    return {"COSCIENTIST_TOY_DATA": str(toy),
            "COSCIENTIST_DACON_DB": str(tmp_path / "d.sqlite3")}, str(p)


async def test_submit_scores_public_and_private(tmp_path):
    # 전반부(1.0,2.0)는 정답, 후반부(3.0,4.0)는 +1 오차 → public 0, private 1
    env, preds = _setup(tmp_path, [1.0, 2.0, 4.0, 5.0])
    async with mcp_session(SERVER, extra_env=env) as session:
        sub = _unwrap(await session.call_tool("submit", {
            "predictions_path": preds, "cycle_id": 1}))
        assert sub["public_score"] == 0.0
        score = _unwrap(await session.call_tool("get_score", {
            "submission_id": sub["submission_id"]}))
        assert score["private_score"] == 1.0


async def test_missing_holdout_is_error(tmp_path):
    env = {"COSCIENTIST_TOY_DATA": str(tmp_path / "none"),
           "COSCIENTIST_DACON_DB": str(tmp_path / "d.sqlite3")}
    async with mcp_session(SERVER, extra_env=env) as session:
        res = await session.call_tool("submit", {
            "predictions_path": str(tmp_path / "p.csv"), "cycle_id": 1})
        assert res.isError


async def test_single_row_holdout_is_clean_error(tmp_path):
    toy = tmp_path / "toy"
    toy.mkdir()
    (toy / "holdout.csv").write_text("x,y\n0,1.0\n", encoding="utf-8")
    p = tmp_path / "preds.csv"
    p.write_text("id,pred\n0,1.0\n", encoding="utf-8")
    env = {"COSCIENTIST_TOY_DATA": str(toy), "COSCIENTIST_DACON_DB": str(tmp_path / "d.sqlite3")}
    async with mcp_session(SERVER, extra_env=env) as session:
        res = await session.call_tool("submit", {"predictions_path": str(p), "cycle_id": 1})
        assert res.isError  # ZeroDivisionError가 아닌 도메인 에러로 깔끔히 거절


def _real_env(tmp_path, **overrides):
    env = {
        "COSCIENTIST_DACON_DB": str(tmp_path / "d.sqlite3"),
        "COSCIENTIST_DACON_MOCK": "0",
        "DACON_API_TOKEN": "dummy-token",
        "DACON_CPT_ID": "235954",
        "DACON_TEAM_NAME": "ContinualLight",
    }
    env.update(overrides)
    return env


async def test_real_submit_success_is_pending_not_scored(tmp_path):
    p = tmp_path / "submission.zip"
    p.write_bytes(b"pk\x03\x04")  # 내용은 fake HTTP 경로라 무의미, 파일 존재만 필요
    env = _real_env(tmp_path, COSCIENTIST_DACON_FAKE_HTTP="ok")
    async with mcp_session(SERVER, extra_env=env) as session:
        sub = _unwrap(await session.call_tool("submit", {
            "predictions_path": str(p), "cycle_id": 1}))
        assert sub["isSubmitted"] is True
        assert sub["public_score"] is None
        score = _unwrap(await session.call_tool("get_score", {
            "submission_id": sub["submission_id"]}))
        assert score["public_score"] is None and score["private_score"] is None
        assert score["status"] == "submitted"


async def test_real_submit_wrong_message_is_rejected(tmp_path):
    p = tmp_path / "submission.zip"
    p.write_bytes(b"pk\x03\x04")
    env = _real_env(tmp_path, COSCIENTIST_DACON_FAKE_HTTP="wrong:3")
    async with mcp_session(SERVER, extra_env=env) as session:
        sub = _unwrap(await session.call_tool("submit", {
            "predictions_path": str(p), "cycle_id": 1}))
        assert sub["isSubmitted"] is False
        assert "row" in sub["detail"]
        score = _unwrap(await session.call_tool("get_score", {
            "submission_id": sub["submission_id"]}))
        assert score["status"] == "rejected"


async def test_real_submit_missing_credentials_is_clean_error(tmp_path):
    p = tmp_path / "submission.zip"
    p.write_bytes(b"pk\x03\x04")
    # 빈 문자열로 명시 설정 — 개발자 로컬 .env에 실제 토큰이 있어도 이 테스트는 격리됨
    env = _real_env(tmp_path, DACON_API_TOKEN="", COSCIENTIST_DACON_FAKE_HTTP="ok")
    async with mcp_session(SERVER, extra_env=env) as session:
        res = await session.call_tool("submit", {"predictions_path": str(p), "cycle_id": 1})
        assert res.isError
