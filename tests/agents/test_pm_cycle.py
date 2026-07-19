from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ai_co_scientist.agents.pm import cycle as cycle_module
from ai_co_scientist.agents.pm.cycle import PMCycle
from ai_co_scientist.a2a.client import TaskOutcome
from ai_co_scientist.core import stubs
from ai_co_scientist.core.schema import CycleContext, parse_payload


class DefaultLedgerSession:
    """check_cycle_log 대역 — 기본적으로 원장에 hypothesis+diagnosis가 모두 있어
    가드가 조용히 통과하도록 한다 (기존 PM 테스트는 원장 상태에 관심 없음)."""

    async def call_tool(self, name, args):
        if name == "query_ledger":
            return SimpleNamespace(
                isError=False,
                structuredContent={"result": [
                    {"cycle_id": args.get("cycle_id"), "record_type": "hypothesis"},
                    {"cycle_id": args.get("cycle_id"), "record_type": "diagnosis"},
                ]},
                content=[])
        return SimpleNamespace(isError=False, structuredContent=None, content=[])


@pytest.fixture(autouse=True)
def _fake_mcp_session(monkeypatch):
    """기존 PM 테스트들은 mcp_session을 패치하지 않으므로, check_cycle_log 노드가
    실제 MCP 서브프로세스/기본 DB 경로에 접근하지 않도록 기본 FakeSession을 배선한다."""
    @asynccontextmanager
    async def fake_mcp_session(module, extra_env=None):
        yield DefaultLedgerSession()

    monkeypatch.setattr(cycle_module, "mcp_session", fake_mcp_session)


class FakeClient:
    """PMClient 대역 — stub 생성기로 응답하고 호출을 기록한다."""

    def __init__(self, responder, task_responder=None):
        self._responder = responder
        self._task_responder = task_responder
        self.calls = []

    async def send(self, payload):
        self.calls.append(payload)
        return self._responder(payload)

    async def submit(self, payload):
        self.calls.append(payload)
        return "task-1"

    async def poll(self, task_id, interval=0.5, timeout=120.0):
        return self._task_responder()

    async def aclose(self):
        pass


def _happy_clients():
    return {
        "research": FakeClient(lambda p: {
            "type": "research_output",
            "data": {
                "hypothesis": {"cycle_id": 1, "statement": "s", "single_variable": "v", "rationale": "r"},
                "design": {"cycle_id": 1, "change": "c", "keep_fixed": [], "expected_effect": "e"},
            },
        }),
        "coder": FakeClient(stubs.coder_stub),
        "executor": FakeClient(None, task_responder=lambda: TaskOutcome(
            state="completed",
            payload={"type": "run_result", "data": {"cycle_id": 1, "metrics": {"val_mse": 0.1}}})),
        "analysis": FakeClient(stubs.analysis_stub),
        "critic": FakeClient(stubs.critic_stub),
    }


async def _auto_gate(summary: str) -> str:
    return "abort"


async def test_happy_path_reaches_verdict():
    pm = PMCycle(_happy_clients(), human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "ok"
    assert final["verdict"]["type"] == "verdict"


async def test_coder_failure_retries_then_escalates():
    clients = _happy_clients()
    fail = {"type": "failure_event",
            "data": {"cycle_id": 1, "category": "impl_bug", "detail": "버그"}}
    clients["coder"] = FakeClient(lambda p: fail)
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "escalated"
    # coder_retry_limit=2 → 최초 1 + 재시도 2 = 3회 호출
    assert len(clients["coder"].calls) == 3


async def test_executor_failure_feeds_back_to_research():
    clients = _happy_clients()
    outcomes = [
        TaskOutcome(state="failed", payload={
            "type": "failure_event",
            "data": {"cycle_id": 1, "category": "infra_oom", "detail": "OOM"}}),
        TaskOutcome(state="completed", payload={
            "type": "run_result", "data": {"cycle_id": 1, "metrics": {"val_mse": 0.2}}}),
    ]
    clients["executor"] = FakeClient(None, task_responder=lambda: outcomes.pop(0))
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "ok"
    # 인프라 실패 후 research가 자원 제약 피드백을 받고 재호출됐는지
    assert len(clients["research"].calls) == 2
    second_ctx = parse_payload(clients["research"].calls[1])
    assert isinstance(second_ctx, CycleContext)
    assert "인프라 실패" in second_ctx.resource_constraints


async def test_executor_impl_bug_redispatches_coder():
    """실행 중 드러난 구현 실패(exit≠0)는 재설계가 아니라 Coder 재지시 (스펙 §5)."""
    clients = _happy_clients()
    outcomes = [
        TaskOutcome(state="failed", payload={
            "type": "failure_event",
            "data": {"cycle_id": 1, "category": "impl_bug", "detail": "exit=1"}}),
        TaskOutcome(state="completed", payload={
            "type": "run_result", "data": {"cycle_id": 1, "metrics": {"val_mse": 0.2}}}),
    ]
    clients["executor"] = FakeClient(None, task_responder=lambda: outcomes.pop(0))
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "ok"
    assert len(clients["coder"].calls) == 2      # 재지시 1회
    assert len(clients["research"].calls) == 1   # 재설계 아님


async def test_research_failure_event_escalates():
    clients = _happy_clients()
    clients["research"] = FakeClient(lambda p: {
        "type": "failure_event",
        "data": {"cycle_id": 1, "category": "logic_inconsistent", "detail": "검증 실패"}})
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "escalated"


async def test_analysis_failure_event_escalates():
    clients = _happy_clients()
    clients["analysis"] = FakeClient(lambda p: {
        "type": "failure_event",
        "data": {"cycle_id": 1, "category": "unknown", "detail": "판정 불가"}})
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "escalated"


async def test_executor_impl_bug_budget_matches_direct_path():
    """실행발 impl_bug도 직접 coder 실패와 동일하게 총 3회 dispatch 후 에스컬레이션."""
    clients = _happy_clients()
    clients["executor"] = FakeClient(None, task_responder=lambda: TaskOutcome(
        state="failed", payload={
            "type": "failure_event",
            "data": {"cycle_id": 1, "category": "impl_bug", "detail": "exit=1"}}))
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "escalated"
    assert len(clients["coder"].calls) == 3  # 직접 실패 경로와 동일 예산


def _revising_critic(n_revises: int):
    """처음 n번 revise, 이후 pass를 반환하는 critic FakeClient responder."""
    count = {"n": 0}

    def responder(payload):
        count["n"] += 1
        if count["n"] <= n_revises:
            return {"type": "critique_report",
                    "data": {"target": payload.get("type", ""), "attacks": ["mock 공격"],
                             "verdict": "revise"}}
        return {"type": "critique_report",
                "data": {"target": payload.get("type", ""), "attacks": [], "verdict": "pass"}}
    return responder


async def test_critic_revise_redispatches_research_with_critique():
    clients = _happy_clients()
    clients["critic"] = FakeClient(_revising_critic(1))
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "ok"
    assert len(clients["research"].calls) == 2  # revise로 1회 재디스패치
    second_ctx = parse_payload(clients["research"].calls[1])
    assert "mock 공격" in second_ctx.critique   # 공격 목록이 재작업 입력으로 전달됨


async def test_critic_rounds_exhaust_then_finalize():
    clients = _happy_clients()
    clients["critic"] = FakeClient(_revising_critic(99))  # 항상 revise
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    # 라운드 한도(2) 소진 후에도 확정하고 끝까지 진행 — verdict 무관 확정 (스펙 §5)
    assert final["outcome"] == "ok"
    assert len(clients["research"].calls) == 1 + 2  # 최초 + revise 2라운드


async def test_critic_revise_does_not_consume_failure_budgets():
    clients = _happy_clients()
    clients["critic"] = FakeClient(_revising_critic(2))
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "ok"
    assert final.get("coder_retries", 0) == 0 and final.get("replans", 0) == 0


async def test_coder_retry_budget_resets_after_replan():
    """인프라 실패로 재설계되면 새 design의 coder 재시도 예산이 리셋되어야 한다."""
    clients = _happy_clients()
    fail = {"type": "failure_event",
            "data": {"cycle_id": 1, "category": "impl_bug", "detail": "버그"}}
    ok = {"type": "code_artifact",
          "data": {"cycle_id": 1, "entrypoint_path": "w.py", "lint_passed": True, "notes": ""}}
    # design1: 실패×2 → 성공(예산 2 소진) / design2(재설계 후): 실패×1 → 성공
    coder_seq = [fail, fail, ok, fail, ok]
    clients["coder"] = FakeClient(lambda p: coder_seq.pop(0))
    exec_outcomes = [
        TaskOutcome(state="failed", payload={
            "type": "failure_event",
            "data": {"cycle_id": 1, "category": "infra_oom", "detail": "OOM"}}),
        TaskOutcome(state="completed", payload={
            "type": "run_result", "data": {"cycle_id": 1, "metrics": {"val_mse": 0.2}}}),
    ]
    clients["executor"] = FakeClient(None, task_responder=lambda: exec_outcomes.pop(0))
    pm = PMCycle(clients, human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "ok"          # 리셋 없으면 design2 첫 실패에서 escalated
    assert len(clients["coder"].calls) == 5
    assert len(clients["research"].calls) == 2


async def test_human_gate_retry_resumes_coder():
    """게이트가 retry를 주면 코더 재시도 예산이 리셋되고 사이클이 재개된다."""
    clients = _happy_clients()
    fail = {"type": "failure_event",
            "data": {"cycle_id": 1, "category": "impl_bug", "detail": "버그"}}
    ok = {"type": "code_artifact",
          "data": {"cycle_id": 1, "entrypoint_path": "w.py", "lint_passed": True, "notes": ""}}
    coder_seq = [fail, fail, fail, ok]   # 예산 소진 → escalate → retry → 성공
    clients["coder"] = FakeClient(lambda p: coder_seq.pop(0))
    answers = ["retry"]

    async def gate(summary):
        return answers.pop(0) if answers else "abort"

    pm = PMCycle(clients, human_gate=gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "ok"
    assert len(clients["coder"].calls) == 4  # 3(소진) + 재개 후 1


async def test_human_gate_abort_and_resume_limit():
    clients = _happy_clients()
    fail = {"type": "failure_event",
            "data": {"cycle_id": 1, "category": "impl_bug", "detail": "버그"}}
    clients["coder"] = FakeClient(lambda p: fail)   # 항상 실패

    async def always_retry(summary):
        return "retry"

    pm = PMCycle(clients, human_gate=always_retry)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    # 재개 한도(1) 소진 후엔 retry 응답이어도 종료 — 무한 재개 방지
    assert final["outcome"] == "escalated"
    assert len(clients["coder"].calls) == 6  # 3 + 재개 1회분 3


async def test_check_cycle_log_records_missing(monkeypatch):
    """decide 진입 전 check_cycle_log — 빈 원장이면 누락을 infra_event로 기록 (차단 아님)."""
    calls = []

    class FakeSession:
        async def call_tool(self, name, args):
            calls.append((name, args))
            if name == "query_ledger":
                return SimpleNamespace(isError=False, structuredContent={"result": []}, content=[])
            return SimpleNamespace(isError=False, structuredContent=None, content=[])

    @asynccontextmanager
    async def fake_mcp_session(module, extra_env=None):
        yield FakeSession()

    monkeypatch.setattr(cycle_module, "mcp_session", fake_mcp_session)
    pm = PMCycle(_happy_clients(), human_gate=_auto_gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "ok"
    appended = [args for name, args in calls if name == "log_append"]
    assert appended and appended[0]["content"]["missing_records"]  # 빈 원장 → 누락 기록


async def test_human_gate_retry_resumes_research_clean():
    """비-impl_bug 실패의 재개는 research로 가며 잔존 컨텍스트 없이 재설계한다."""
    clients = _happy_clients()
    outcomes = [
        TaskOutcome(state="failed", payload={
            "type": "failure_event",
            "data": {"cycle_id": 1, "category": "infra_oom", "detail": "OOM"}}),
        TaskOutcome(state="failed", payload={
            "type": "failure_event",
            "data": {"cycle_id": 1, "category": "infra_oom", "detail": "OOM"}}),
        TaskOutcome(state="failed", payload={
            "type": "failure_event",
            "data": {"cycle_id": 1, "category": "infra_oom", "detail": "OOM"}}),
        TaskOutcome(state="completed", payload={
            "type": "run_result", "data": {"cycle_id": 1, "metrics": {"val_mse": 0.2}}}),
    ]
    clients["executor"] = FakeClient(None, task_responder=lambda: outcomes.pop(0))
    answers = ["retry"]

    async def gate(summary):
        return answers.pop(0) if answers else "abort"

    pm = PMCycle(clients, human_gate=gate)
    final = await pm.graph.ainvoke({"cycle_id": 1, "coder_retries": 0, "replans": 0})
    assert final["outcome"] == "ok"
    # replan_limit=2: executor 실패 1(replans=1→infra_feedback→research 2), 실패 2(replans=2→research 3),
    # 실패 3(replans=3>2→escalate) → retry(카운터 리셋+컨텍스트 클리어)→dispatch_research(4번째)→coder→executor 성공
    assert len(clients["research"].calls) == 4
    resumed_ctx = parse_payload(clients["research"].calls[3])
    assert resumed_ctx.resource_constraints == ""   # 잔존 infra_feedback 재생 없음
    assert resumed_ctx.critique == ""               # 잔존 critique 재생 없음
