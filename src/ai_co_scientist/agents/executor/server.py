"""Executor 에이전트 (M5: lightning MCP 경유 실행 + wandb 기록 + 복구 룰) — task 모드."""
import asyncio
import os
import sys

from ai_co_scientist.a2a.base import serve
from ai_co_scientist.agents.executor import logic
from ai_co_scientist.core.config import load_config
from ai_co_scientist.core.failure import AgentTaskFailure, FailureCategory
from ai_co_scientist.core.mcp_client import mcp_session, tool_result_data
from ai_co_scientist.core.rules import load_rules
from ai_co_scientist.core.schema import CodeArtifact, FailureEvent, RunResult, parse_payload, to_payload
from ai_co_scientist.core.stubs import executor_stub

SHARED_LOG_SERVER = "ai_co_scientist.mcp_servers.shared_log.server"
LIGHTNING_SERVER = "ai_co_scientist.mcp_servers.lightning.server"
WANDB_SERVER = "ai_co_scientist.mcp_servers.wandb_tools.server"


async def _log_infra_event(content: dict) -> None:
    """공유로그 기록 — 실패해도 원본 실패/결과를 가리지 않게 격리 (M2 이월 ①).

    infra_event는 executor 실패 기록의 catch-all 버킷 (스키마 CHECK 제약상 impl_bug 실패도 여기 기록, failure_category로 구분).
    """
    try:
        async with mcp_session(SHARED_LOG_SERVER) as session:
            res = await session.call_tool("log_append", {
                "cycle_id": content["cycle_id"], "record_type": "infra_event",
                "owner": "executor", "content": content,
                "failure_category": content["category"],
            })
            # tool 수준 실패도 isError 응답 — 격리 유지, 관측만 (스펙 §9 M3 교훈)
            if res.isError:
                print(f"[executor] log_append tool 실패(무시): "
                      f"{res.content[0].text if res.content else ''}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — 로깅 실패는 관측 손실일 뿐, 실행 결과가 우선
        print(f"[executor] 공유로그 기록 실패(무시): {e}", file=sys.stderr)


async def _run_via_lightning(path: str, timeout: float) -> dict:
    """submit→poll 1회분. 반환: poll 결과 dict 또는 {"status":"credit"} (거절)."""
    async with mcp_session(LIGHTNING_SERVER) as session:
        res = await session.call_tool("submit_job", {
            "entrypoint_path": path, "timeout_s": timeout})
        if res.isError:
            return {"status": "unknown", "detail": "submit tool 실패", "metrics": None, "artifacts": {}}
        sub = tool_result_data(res)
        sub = sub.get("result", sub) if isinstance(sub, dict) and "result" in sub else sub
        if sub.get("rejected") == "credit":
            return {"status": "credit", "detail": "크레딧 부족", "metrics": None, "artifacts": {}}
        interval = load_config()["pm"]["poll_interval_s"]
        while True:   # mock은 즉시 terminal이지만 폴링 배선은 유지 (real 대비)
            res = await session.call_tool("poll_job", {"job_id": sub["job_id"]})
            if res.isError:
                return {"status": "unknown", "detail": "poll tool 실패", "metrics": None, "artifacts": {}}
            job = tool_result_data(res)
            job = job.get("result", job) if isinstance(job, dict) and "result" in job else job
            if job.get("status") != "running":
                return job
            await asyncio.sleep(interval)


async def _log_wandb(cycle_id: int, metrics: dict) -> None:
    """wandb 기록 — 격리 (관측 부채널)."""
    try:
        async with mcp_session(WANDB_SERVER) as session:
            res = await session.call_tool("log_metrics", {
                "run_id": f"cycle-{cycle_id}", "cycle_id": cycle_id, "metrics": metrics})
            if res.isError:
                print("[executor] wandb 기록 실패(무시)", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[executor] wandb 기록 실패(무시): {e}", file=sys.stderr)


async def handle(payload: dict) -> dict:
    artifact = parse_payload(payload)
    if not isinstance(artifact, CodeArtifact):
        raise ValueError(f"CodeArtifact 아님: {payload.get('type')}")
    # M2 E2E 계약: 주입된 인프라 실패는 실행 전 즉시 신고 (executor_stub의 분기 재사용)
    if os.environ.get("COSCIENTIST_INJECT_FAILURE", "").startswith("infra"):
        try:
            executor_stub(payload)
        except AgentTaskFailure as e:
            await _log_infra_event(e.payload["data"])
            raise
    timeout = float(os.environ.get(
        "COSCIENTIST_RUN_TIMEOUT_S", load_config()["executor"]["run_timeout_s"]))
    recovery_used = False
    while True:
        job = await _run_via_lightning(artifact.entrypoint_path, timeout)
        status = job.get("status", "unknown")
        category = (FailureCategory.INFRA_CREDIT if status == "credit"
                    else logic.classify_status(status))
        if category is None:
            result = RunResult(cycle_id=artifact.cycle_id,
                               metrics=job.get("metrics") or {},
                               artifacts=job.get("artifacts") or {})
            await _log_wandb(artifact.cycle_id, result.metrics)
            return to_payload(result)
        failure = FailureEvent(cycle_id=artifact.cycle_id, category=category,
                               detail=job.get("detail", status))
        await _log_infra_event(failure.model_dump(mode="json"))
        rule = logic.RECOVERY_RULES.get(category)
        if rule and not recovery_used:
            timeout *= rule["timeout_multiplier"]
            recovery_used = True
            continue
        raise AgentTaskFailure(to_payload(failure))


def main() -> None:
    load_rules("executor")
    serve("executor", "실험 실행 담당 (M5: lightning MCP + 복구 룰 + wandb 기록)",
          load_config()["agents"]["executor"]["port"], handle, mode="task")


if __name__ == "__main__":
    main()
