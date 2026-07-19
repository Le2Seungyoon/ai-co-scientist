"""Lightning mock MCP 서버 — 잡 제출/폴링/크레딧 (스펙 §4). real은 골격만.

mock은 submit_job이 동기 실행 후 결과를 저장하고, poll_job은 조회만 한다 —
폴링 배선(제출→폴링)은 소비자(Executor) 쪽에서 실제로 학습된다.
"""
import os
import uuid

from mcp.server.fastmcp import FastMCP

from ai_co_scientist.core.config import load_config, project_root
from ai_co_scientist.mcp_servers.lightning import db, runner

mcp = FastMCP("lightning")


def _db_path() -> str:
    return os.environ.get("COSCIENTIST_LIGHTNING_DB") or str(
        project_root() / load_config()["paths"]["lightning_db"])


def _real_submit_job(entrypoint_path: str, timeout_s: float) -> dict:
    # TODO(M5+): lightning_sdk로 원격 잡 제출 + 상태 폴링 + 크레딧 조회.
    # 스펙 §7-②: 크레딧 자동 리필 정책 확인 후 "시간으로 해결" 대기 로직 설계.
    raise NotImplementedError("real lightning 백엔드는 골격만 (검증은 사람 몫)")


@mcp.tool()
def submit_job(entrypoint_path: str, timeout_s: float) -> dict:
    """잡 제출 — 크레딧 차감 후 (mock: 동기) 실행. 부족 시 rejected."""
    path = _db_path()
    db.init_credits(path, float(os.environ.get("COSCIENTIST_CREDITS", "100")))
    if not db.deduct(path, 1.0):
        return {"job_id": "", "rejected": "credit"}
    job_id = uuid.uuid4().hex[:12]
    # 주의: 배선된 시스템에선 executor가 INJECT를 선처리하므로 이 분기는 도달하지 않음
    # — 서버 직접 테스트용 방어층
    if os.environ.get("COSCIENTIST_INJECT_FAILURE") == "infra_oom":
        db.save_job(path, job_id, "oom", None, {}, "주입된 OOM")
        return {"job_id": job_id}
    attempt = runner.execute_entrypoint(entrypoint_path, timeout_s)
    if attempt.timed_out:
        db.save_job(path, job_id, "timeout", None, {}, f"타임아웃({timeout_s}s)")
    elif attempt.returncode != 0:
        db.save_job(path, job_id, "failed", None, {}, attempt.stderr[-500:])
    else:
        metrics, artifacts = runner.parse_output(attempt.stdout)
        if metrics is None:
            db.save_job(path, job_id, "failed", None, {}, "출력 계약 위반(마지막 줄 metrics JSON 없음)")
        else:
            db.save_job(path, job_id, "completed", metrics, artifacts, "")
    return {"job_id": job_id}


@mcp.tool()
def poll_job(job_id: str) -> dict:
    """잡 상태 조회 — {"status", "metrics", "artifacts", "detail"}."""
    job = db.get_job(_db_path(), job_id)
    if job is None:
        return {"status": "unknown", "metrics": None, "artifacts": {}, "detail": "job 없음"}
    return job


@mcp.tool()
def get_credits() -> dict:
    """잔여 크레딧 조회."""
    path = _db_path()
    db.init_credits(path, float(os.environ.get("COSCIENTIST_CREDITS", "100")))
    return {"remaining": db.get_credits(path)}


if __name__ == "__main__":
    mcp.run()
