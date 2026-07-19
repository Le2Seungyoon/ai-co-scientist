"""Coder self-correct graph — 트레이스백/lint 사유 기반 재생성 (스펙 §5, emis-api 패턴).

시도 흐름: generate → write+lint(hook) → smoke_run. 오류 사유는 다음 generate의
입력이 되고, self_correct_limit 소진 시 impl_bug FailureEvent로 실패한다.
"""
import subprocess
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ai_co_scientist.core.config import load_config, project_root
from ai_co_scientist.core.failure import AgentTaskFailure, FailureCategory
from ai_co_scientist.core.rules import load_rules
from ai_co_scientist.core.schema import CodeArtifact, FailureEvent, to_payload
from hooks.lint_check import lint_check


class CoderState(TypedDict, total=False):
    design: dict
    cycle_id: int
    code: str
    path: str
    error: str
    attempts: int
    artifact: dict | None
    failure: dict | None


class CoderGraph:
    def __init__(self, router=None):
        from ai_co_scientist.llm.router import LLMRouter

        self._router = router or LLMRouter()
        self._rules = load_rules("coder")
        cfg = load_config()["coder"]
        self._max_attempts = cfg["self_correct_limit"] + 1
        self._smoke_timeout = cfg["smoke_timeout_s"]
        self._workspace = cfg["workspace_dir"]
        self._graph = self._build()

    async def run(self, design: dict, cycle_id: int) -> dict:
        """성공 시 code_artifact payload 반환, 소진 시 AgentTaskFailure raise."""
        final = await self._graph.ainvoke(
            {"design": design, "cycle_id": cycle_id, "attempts": 0, "error": ""})
        if final.get("failure") is not None:
            raise AgentTaskFailure(final["failure"])
        return final["artifact"]

    # ── 노드 ──────────────────────────────────────────────

    async def _generate(self, state: CoderState) -> dict:
        out = self._router.invoke(
            "coder", {"design": state["design"], "error": state.get("error", ""), "rules": self._rules})
        return {"code": out["code"], "attempts": state["attempts"] + 1}

    async def _write_and_lint(self, state: CoderState) -> dict:
        workdir = project_root() / self._workspace / f"cycle_{state['cycle_id']}"
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / f"experiment_attempt{state['attempts']}.py"
        path.write_text(state["code"], encoding="utf-8")
        ok, reason = lint_check(str(path))
        return {"path": str(path), "error": "" if ok else f"lint: {reason}"}

    async def _smoke_run(self, state: CoderState) -> dict:
        """빠른 실행으로 트레이스백 검출 — 정식 실행(인프라 관점)은 Executor 소관."""
        try:
            proc = subprocess.run(
                [sys.executable, state["path"]],
                capture_output=True, text=True, encoding="utf-8", timeout=self._smoke_timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"smoke 실행 타임아웃({self._smoke_timeout}s)"}
        if proc.returncode != 0:
            return {"error": f"실행 실패:\n{proc.stderr[-800:]}"}
        return {"error": ""}

    async def _finalize(self, state: CoderState) -> dict:
        artifact = CodeArtifact(
            cycle_id=state["cycle_id"],
            entrypoint_path=state["path"],
            lint_passed=True,
            notes=f"attempts={state['attempts']}",
        )
        return {"artifact": to_payload(artifact)}

    async def _fail(self, state: CoderState) -> dict:
        failure = FailureEvent(
            cycle_id=state["cycle_id"],
            category=FailureCategory.IMPL_BUG,
            detail=f"self-correct 소진(attempts={state['attempts']}): {state['error']}",
        )
        return {"failure": to_payload(failure)}

    # ── 조건부 엣지 ────────────────────────────────────────

    def _after_lint(self, state: CoderState) -> str:
        if not state.get("error"):
            return "smoke_run"
        return "generate" if state["attempts"] < self._max_attempts else "fail"

    def _after_smoke(self, state: CoderState) -> str:
        if not state.get("error"):
            return "finalize"
        return "generate" if state["attempts"] < self._max_attempts else "fail"

    def _build(self):
        builder = StateGraph(CoderState)
        builder.add_node("generate", self._generate)
        builder.add_node("write_and_lint", self._write_and_lint)
        builder.add_node("smoke_run", self._smoke_run)
        builder.add_node("finalize", self._finalize)
        builder.add_node("fail", self._fail)
        builder.add_edge(START, "generate")
        builder.add_edge("generate", "write_and_lint")
        builder.add_conditional_edges("write_and_lint", self._after_lint)
        builder.add_conditional_edges("smoke_run", self._after_smoke)
        builder.add_edge("finalize", END)
        builder.add_edge("fail", END)
        return builder.compile()
