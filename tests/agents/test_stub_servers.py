import os
from unittest.mock import patch

import pytest

from ai_co_scientist.agents.analysis.server import handle as analysis_handle
from ai_co_scientist.agents.coder import server as coder_server
from ai_co_scientist.agents.coder.server import handle as coder_handle
from ai_co_scientist.agents.critic import server as critic_server
from ai_co_scientist.agents.critic.server import handle as critic_handle
from ai_co_scientist.agents.executor.server import handle as executor_handle
from ai_co_scientist.agents.harness_engineer import server as harness_server
from ai_co_scientist.core.failure import AgentTaskFailure
from ai_co_scientist.core.schema import (
    CodeArtifact, ExperimentDesign, RunResult, to_payload,
)
from ai_co_scientist.mcp_servers.shared_log import db


@pytest.fixture(autouse=True)
def _reset_coder_graph(monkeypatch):
    """coder server의 _graph는 프로세스 수명 캐시 — 테스트 간 시나리오 오염 방지."""
    monkeypatch.setattr(coder_server, "_graph", None)
    monkeypatch.setattr(critic_server, "_router", None)
    monkeypatch.setattr(critic_server, "_rules", None)
    monkeypatch.setattr(harness_server, "_router", None)
    monkeypatch.setattr(harness_server, "_rules", None)


async def test_coder_handle_returns_artifact(monkeypatch):
    monkeypatch.delenv("COSCIENTIST_INJECT_FAILURE", raising=False)
    monkeypatch.setenv("COSCIENTIST_MOCKLLM_SCENARIO", "default")
    design = to_payload(ExperimentDesign(cycle_id=1, change="x", keep_fixed=[], expected_effect="y"))
    out = await coder_handle(design)
    assert out["type"] == "code_artifact"
    assert out["data"]["lint_passed"] is True


async def test_executor_handle_success(monkeypatch):
    """실행은 lightning MCP(submit→poll) 경유, 완료 시 wandb log_metrics도 기록된다."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from ai_co_scientist.agents.executor import server as executor_server

    monkeypatch.delenv("COSCIENTIST_INJECT_FAILURE", raising=False)
    calls: list[tuple[str, str]] = []  # (server_module, tool_name)

    class FakeLightningSession:
        async def call_tool(self, name, args):
            calls.append((executor_server.LIGHTNING_SERVER, name))
            if name == "submit_job":
                return SimpleNamespace(isError=False, structuredContent={"job_id": "j1"})
            if name == "poll_job":
                return SimpleNamespace(isError=False, structuredContent={
                    "status": "completed", "metrics": {"val_mse": 0.2}, "artifacts": {}})
            raise AssertionError(f"예상치 못한 tool: {name}")

    class FakeWandbSession:
        async def call_tool(self, name, args):
            calls.append((executor_server.WANDB_SERVER, name))
            assert name == "log_metrics"
            return SimpleNamespace(isError=False, structuredContent=1)

    @asynccontextmanager
    async def fake_mcp_session(module, extra_env=None):
        if module == executor_server.LIGHTNING_SERVER:
            yield FakeLightningSession()
        elif module == executor_server.WANDB_SERVER:
            yield FakeWandbSession()
        else:
            raise AssertionError(f"예상치 못한 MCP 서버: {module}")

    monkeypatch.setattr(executor_server, "mcp_session", fake_mcp_session)
    artifact = to_payload(
        CodeArtifact(cycle_id=1, entrypoint_path="dummy.py", lint_passed=True))
    out = await executor_handle(artifact)
    assert out["type"] == "run_result"
    assert out["data"]["metrics"]["val_mse"] == 0.2
    assert ("submit_job" in [c[1] for c in calls])
    assert ("log_metrics" in [c[1] for c in calls])  # wandb 기록도 수행됨


async def test_executor_handle_failure_logs_infra_event(tmp_path, monkeypatch):
    """INJECT 선처리 경로 — lightning MCP는 호출되지 않아야 한다(기존 M2 계약 유지)."""
    from contextlib import asynccontextmanager

    from ai_co_scientist.agents.executor import server as executor_server
    from ai_co_scientist.core.mcp_client import mcp_session as real_mcp_session

    lightning_called = False

    @asynccontextmanager
    async def routed_session(module, extra_env=None):
        nonlocal lightning_called
        if module == executor_server.LIGHTNING_SERVER:
            lightning_called = True
            raise AssertionError("INJECT 선처리 경로에서 lightning MCP가 호출되면 안 됨")
        async with real_mcp_session(module, extra_env=extra_env) as session:
            yield session

    monkeypatch.setattr(executor_server, "mcp_session", routed_session)
    db_file = str(tmp_path / "log.sqlite3")
    artifact = to_payload(CodeArtifact(cycle_id=2, entrypoint_path="w.py", lint_passed=True))
    with patch.dict(os.environ, {"COSCIENTIST_LOG_DB": db_file,
                                 "COSCIENTIST_INJECT_FAILURE": "infra_oom"}):
        with pytest.raises(AgentTaskFailure):
            await executor_handle(artifact)
    rows = db.query_ledger(db_file, record_type="infra_event")
    assert len(rows) == 1 and rows[0]["failure_category"] == "infra_oom"
    assert lightning_called is False


async def test_executor_logging_failure_does_not_mask_original(monkeypatch):
    """공유로그 기록이 죽어도 원본 AgentTaskFailure가 그대로 전파돼야 한다 (M2 이월 ① 격리 검증)."""
    from ai_co_scientist.agents.executor import server as executor_server

    def broken_session(*args, **kwargs):
        raise RuntimeError("MCP down")

    monkeypatch.setattr(executor_server, "mcp_session", broken_session)
    monkeypatch.setenv("COSCIENTIST_INJECT_FAILURE", "infra_oom")
    artifact = to_payload(CodeArtifact(cycle_id=5, entrypoint_path="w.py", lint_passed=True))
    with pytest.raises(AgentTaskFailure) as exc:
        await executor_handle(artifact)
    assert exc.value.payload["data"]["category"] == "infra_oom"  # 로깅 예외(RuntimeError)가 아님


async def test_executor_timeout_exhausted_after_recovery(monkeypatch):
    """복구(×2) 후에도 타임아웃이면 infra_timeout 실패 — 이벤트 2건 기록."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from ai_co_scientist.agents.executor import server as executor_server

    monkeypatch.delenv("COSCIENTIST_INJECT_FAILURE", raising=False)
    calls: list[tuple[str, str]] = []  # (server_module, tool_name)

    class FakeLightningSession:
        async def call_tool(self, name, args):
            calls.append((executor_server.LIGHTNING_SERVER, name))
            if name == "submit_job":
                return SimpleNamespace(isError=False, structuredContent={"job_id": "j1"})
            if name == "poll_job":
                return SimpleNamespace(isError=False, structuredContent={
                    "status": "timeout", "metrics": None, "artifacts": {},
                    "detail": "타임아웃(1s)"})
            raise AssertionError(f"예상치 못한 tool: {name}")

    class FakeSharedLogSession:
        async def call_tool(self, name, args):
            calls.append((executor_server.SHARED_LOG_SERVER, name))
            assert name == "log_append"
            return SimpleNamespace(isError=False, structuredContent=1)

    @asynccontextmanager
    async def fake_mcp_session(module, extra_env=None):
        if module == executor_server.LIGHTNING_SERVER:
            yield FakeLightningSession()
        elif module == executor_server.SHARED_LOG_SERVER:
            yield FakeSharedLogSession()
        else:
            raise AssertionError(f"예상치 못한 MCP 서버: {module}")

    monkeypatch.setattr(executor_server, "mcp_session", fake_mcp_session)
    artifact = to_payload(
        CodeArtifact(cycle_id=1, entrypoint_path="dummy.py", lint_passed=True))
    with pytest.raises(AgentTaskFailure) as exc:
        await executor_handle(artifact)
    assert exc.value.payload["data"]["category"] == "infra_timeout"
    submit_calls = [c for c in calls if c[1] == "submit_job"]
    infra_events = [c for c in calls if c[1] == "log_append"]
    assert len(submit_calls) == 2   # 최초 + 복구 1회 (retries=1 소진)
    assert len(infra_events) == 2   # 실패마다 infra_event 기록


async def test_analysis_handle_logs_diagnosis(tmp_path):
    db_file = str(tmp_path / "log.sqlite3")
    rr = to_payload(RunResult(cycle_id=3, metrics={"val_mse": 0.4}))
    with patch.dict(os.environ, {"COSCIENTIST_LOG_DB": db_file}):
        out = await analysis_handle(rr)
    assert out["type"] == "verdict"
    rows = db.query_ledger(db_file, record_type="diagnosis")
    assert len(rows) == 1 and rows[0]["owner"] == "analysis"
    assert db.get_consensus(db_file)["best_pipeline"]["val_mse"] == 0.4


async def test_critic_handle_pass(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_MOCKLLM_SCENARIO", "default")
    out = await critic_handle({"type": "verdict", "data": {}})
    assert out["type"] == "critique_report" and out["data"]["verdict"] == "pass"


async def test_executor_credit_rejection_fails_without_recovery(monkeypatch, tmp_path):
    """크레딧 거절 → INFRA_CREDIT 즉시 실패(복구 없음) — 스펙 §5 'PM 소관' 계약."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from ai_co_scientist.agents.executor import server as executor_server

    monkeypatch.delenv("COSCIENTIST_INJECT_FAILURE", raising=False)
    calls: list[str] = []

    class FakeLightningSession:
        async def call_tool(self, name, args):
            calls.append(name)
            if name == "submit_job":
                return SimpleNamespace(isError=False, structuredContent={"result": {"job_id": "", "rejected": "credit"}})
            if name == "poll_job":
                return SimpleNamespace(isError=False, structuredContent={"status": "completed"})
            raise AssertionError(f"예상치 못한 tool: {name}")

    class FakeSharedLogSession:
        async def call_tool(self, name, args):
            calls.append(name)
            return SimpleNamespace(isError=False, structuredContent=1)

    @asynccontextmanager
    async def fake_mcp_session(module, extra_env=None):
        if module == executor_server.LIGHTNING_SERVER:
            yield FakeLightningSession()
        elif module == executor_server.SHARED_LOG_SERVER:
            yield FakeSharedLogSession()
        else:
            raise AssertionError(f"예상치 못한 MCP 서버: {module}")

    monkeypatch.setattr(executor_server, "mcp_session", fake_mcp_session)
    artifact = to_payload(
        CodeArtifact(cycle_id=1, entrypoint_path="dummy.py", lint_passed=True))
    with pytest.raises(AgentTaskFailure) as exc:
        await executor_handle(artifact)
    assert exc.value.payload["data"]["category"] == "infra_credit"
    assert calls.count("submit_job") == 1
    assert "poll_job" not in calls


async def test_analysis_consensus_read_failure_holds_judgment(monkeypatch, tmp_path):
    """조회 실패(isError) 시 개선 판단 보류 + update_consensus 미호출 (상태 오염 방지)."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from ai_co_scientist.agents.analysis import server as analysis_server

    calls = []

    class FakeSession:
        async def call_tool(self, name, args):
            calls.append(name)
            return SimpleNamespace(
                isError=(name == "get_consensus"), structuredContent=None,
                content=[SimpleNamespace(text="Error executing tool get_consensus: boom")])

    @asynccontextmanager
    async def fake_mcp_session(module, extra_env=None):
        yield FakeSession()

    monkeypatch.setattr(analysis_server, "mcp_session", fake_mcp_session)
    rr = to_payload(RunResult(cycle_id=7, metrics={"val_mse": 0.9}))
    out = await analysis_server.handle(rr)
    assert out["data"]["improved"] is False
    assert "보류" in out["data"]["diagnosis"]
    assert "update_consensus" not in calls   # 오염 방지
    assert "log_append" in calls             # 기록은 시도됨


async def test_analysis_submits_on_overfit_suspicion(monkeypatch):
    """오버피팅 의심 + predictions + guard 통과 → dacon 제출, diagnosis에 public 반영."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from ai_co_scientist.agents.analysis import server as analysis_server

    calls = []

    class FakeSession:
        async def call_tool(self, name, args):
            calls.append(name)
            payload = {
                "get_consensus": {"result": {}},
                "submit": {"result": {"submission_id": "s1", "public_score": 0.31}},
                "get_score": {"result": {"public_score": 0.31, "private_score": 0.55}},
            }.get(name, {"result": {}})
            return SimpleNamespace(isError=False, structuredContent=payload, content=[])

    @asynccontextmanager
    async def fake_mcp_session(module, extra_env=None):
        yield FakeSession()

    monkeypatch.setattr(analysis_server, "mcp_session", fake_mcp_session)
    rr = to_payload(RunResult(cycle_id=1, metrics={"train_mse": 0.05, "val_mse": 0.3},
                              artifacts={"predictions_path": "p.csv"}))
    out = await analysis_server.handle(rr)
    assert out["data"]["overfitting_suspected"] is True
    assert "public" in out["data"]["diagnosis"]
    assert "submit" in calls


async def test_analysis_blocks_submission_without_improvement(monkeypatch):
    """베스트보다 나쁜 결과는 guard가 제출을 차단한다."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from ai_co_scientist.agents.analysis import server as analysis_server

    calls = []

    class FakeSession:
        async def call_tool(self, name, args):
            calls.append(name)
            payload = {
                "get_consensus": {"result": {"best_pipeline": {"val_mse": 0.1, "cycle_id": 0}}},
            }.get(name, {"result": {}})
            return SimpleNamespace(isError=False, structuredContent=payload, content=[])

    @asynccontextmanager
    async def fake_mcp_session(module, extra_env=None):
        yield FakeSession()

    monkeypatch.setattr(analysis_server, "mcp_session", fake_mcp_session)
    rr = to_payload(RunResult(cycle_id=1, metrics={"train_mse": 0.05, "val_mse": 0.3},
                              artifacts={"predictions_path": "p.csv"}))
    out = await analysis_server.handle(rr)
    assert out["data"]["overfitting_suspected"] is True
    assert "submit" not in calls
    assert "차단" in out["data"]["diagnosis"]
