import os
from unittest.mock import patch

import pytest

from ai_co_scientist.agents.research.graph import ResearchGraph
from ai_co_scientist.core.failure import AgentTaskFailure
from ai_co_scientist.core.schema import CycleContext, parse_payload
from ai_co_scientist.llm.router import LLMRouter
from ai_co_scientist.mcp_servers.shared_log import db


async def test_generates_output_and_logs_hypothesis(tmp_path):
    db_file = str(tmp_path / "log.sqlite3")
    with patch.dict(os.environ, {"COSCIENTIST_LOG_DB": db_file}):
        graph = ResearchGraph(router=LLMRouter(scenario="default"))
        result = await graph.run(CycleContext(cycle_id=1))
    out = parse_payload(result)
    assert out.hypothesis.single_variable
    rows = db.query_ledger(db_file, record_type="hypothesis")
    assert len(rows) == 1 and rows[0]["owner"] == "research"


async def test_gather_feeds_consensus_into_generation(tmp_path):
    db_file = str(tmp_path / "log.sqlite3")
    db.update_consensus(db_file, key="best_pipeline", value={"val_mse": 0.3, "cycle_id": 1})
    with patch.dict(os.environ, {"COSCIENTIST_LOG_DB": db_file}):
        graph = ResearchGraph(router=LLMRouter(scenario="default"))
        result = await graph.run(CycleContext(cycle_id=2))
    # default 시나리오는 rationale에 컨센서스 요약을 삽입한다
    assert "best_pipeline" in parse_payload(result).hypothesis.rationale


async def test_gather_isolates_tool_level_errors(monkeypatch, tmp_path):
    """MCP tool이 isError 응답을 돌려줘도 에러 문자열이 컨텍스트에 새면 안 된다."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from ai_co_scientist.agents.research import graph as graph_module

    class FakeResult:
        isError = True
        structuredContent = None
        content = [SimpleNamespace(text="Error executing tool get_consensus: boom")]

    class FakeSession:
        async def call_tool(self, name, args):
            return FakeResult()

    @asynccontextmanager
    async def fake_mcp_session(module, extra_env=None):
        yield FakeSession()

    monkeypatch.setattr(graph_module, "mcp_session", fake_mcp_session)
    graph = ResearchGraph(router=LLMRouter(scenario="default"))
    result = await graph.run(CycleContext(cycle_id=1))
    rationale = parse_payload(result).hypothesis.rationale
    assert "Error executing" not in rationale   # 에러 텍스트 오염 없음
    assert "없음" in rationale                   # 빈 컨텍스트로 진행됨


async def test_invalid_generation_retries_then_fails(tmp_path):
    class BadRouter:
        def invoke(self, role, task_input):
            return {  # single_variable이 keep_fixed에 포함 — 단일 변인 규칙 위반
                "hypothesis": {"cycle_id": 1, "statement": "s", "single_variable": "lr", "rationale": "r"},
                "design": {"cycle_id": 1, "change": "c", "keep_fixed": ["lr"], "expected_effect": "e"},
            }

    with patch.dict(os.environ, {"COSCIENTIST_LOG_DB": str(tmp_path / "log.sqlite3")}):
        graph = ResearchGraph(router=BadRouter())
        with pytest.raises(AgentTaskFailure) as exc:
            await graph.run(CycleContext(cycle_id=1))
    assert exc.value.payload["data"]["category"] == "logic_inconsistent"


async def test_rules_text_is_passed_to_llm(tmp_path):
    captured = {}

    class SpyRouter:
        def invoke(self, role, task_input):
            captured.update(task_input)
            return LLMRouter(scenario="default").invoke(role, task_input)

    with patch.dict(os.environ, {"COSCIENTIST_LOG_DB": str(tmp_path / "log.sqlite3")}):
        graph = ResearchGraph(router=SpyRouter())
        await graph.run(CycleContext(cycle_id=1))
    assert "## 역할" in captured.get("rules", "")  # rules/research.md 전문이 주입됨
