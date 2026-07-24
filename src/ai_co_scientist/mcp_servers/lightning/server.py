"""Lightning MCP 서버 — mock: 동기 로컬 실행, real: lightning_sdk로 T4 등 원격 GPU 잡 제출.

config.yaml의 mock.lightning(또는 테스트용 COSCIENTIST_LIGHTNING_MOCK 오버라이드)로 분기한다.
real 경로는 mock과 달리 **비동기**다 — submit_job은 원격 잡을 던지고 즉시 반환("running"),
poll_job이 매 호출마다 실제 상태를 조회한다(§ 스펙 §7-②, 크레딧 리필 정책은 여전히 미확인이라
잔액은 그대로 조회만 하고 "시간으로 해결" 대기 로직은 Executor/PM 소관으로 남긴다).
"""
import base64
import os
import time
import uuid
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from ai_co_scientist.core.config import load_config, load_dotenv, project_root
from ai_co_scientist.mcp_servers.lightning import db, runner

mcp = FastMCP("lightning")

# 원격 잡이 실제 학습 전에 이미지 pull + pip install로 쓰는 여유 시간 — timeout_s는
# 순수 스크립트 실행 시간 기준(mock과 동일 계약)이라 원격 기동 오버헤드만큼 더 기다려준다.
_STARTUP_BUFFER_S = 300.0
_DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
_DEFAULT_MACHINE = "T4"
_OOM_MARKERS = ("out of memory", "oom", "cuda error: out of memory")


def _db_path() -> str:
    return os.environ.get("COSCIENTIST_LIGHTNING_DB") or str(
        project_root() / load_config()["paths"]["lightning_db"])


def _is_mock() -> bool:
    override = os.environ.get("COSCIENTIST_LIGHTNING_MOCK")
    if override is not None:
        return override == "1"
    return bool(load_config()["mock"]["lightning"])


def _require_teamspace() -> str:
    load_dotenv()
    api_key = os.environ.get("LIGHTNING_API_KEY")
    user_id = os.environ.get("LIGHTNING_USER_ID")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE")
    if not (api_key and user_id and teamspace):
        raise RuntimeError(
            "LIGHTNING_API_KEY/LIGHTNING_USER_ID/LIGHTNING_TEAMSPACE 필요 (.env 참고, "
            "README 'Lightning AI GPU 잡 제출 활성화')")
    return teamspace


@lru_cache(maxsize=1)
def _rest_client():
    from lightning_sdk.lightning_cloud.rest_client import LightningClient
    return LightningClient()


@lru_cache(maxsize=1)
def _username() -> str:
    return _rest_client().auth_service_get_user().username


@lru_cache(maxsize=8)
def _project_id(teamspace: str) -> str:
    resp = _rest_client().projects_service_list_memberships()
    for m in resp.memberships:
        if m.name == teamspace:
            return m.project_id
    raise RuntimeError(f"teamspace '{teamspace}' 를 찾을 수 없음 (LIGHTNING_TEAMSPACE 확인)")


def _job_name(job_id: str) -> str:
    return f"cosci-{job_id}"


def _real_submit_job(entrypoint_path: str, timeout_s: float) -> dict:
    """lightning_sdk Job으로 entrypoint 스크립트를 원격 T4(기본)에서 실행.

    로컬 파일 하나만 base64로 인라인해 이미지 안에서 복원 후 실행 — path_mappings는
    data connection 전용이라 임의 로컬 파일 업로드에는 쓸 수 없음. 데이터 의존성이 있는
    entrypoint는 아직 지원 밖(§ 검증은 사람 몫, 스펙 §7).
    """
    from lightning_sdk import Job, Machine

    teamspace = _require_teamspace()
    with open(entrypoint_path, encoding="utf-8") as f:
        script_b64 = base64.b64encode(f.read().encode("utf-8")).decode("ascii")

    machine_name = os.environ.get("COSCIENTIST_LIGHTNING_MACHINE", _DEFAULT_MACHINE)
    image = os.environ.get("COSCIENTIST_LIGHTNING_IMAGE", _DEFAULT_IMAGE)
    interruptible = os.environ.get("COSCIENTIST_LIGHTNING_INTERRUPTIBLE") == "1"

    job_id = uuid.uuid4().hex[:12]
    command = (
        f"echo {script_b64} | base64 -d > /tmp/entrypoint.py && "
        "python3 /tmp/entrypoint.py"
    )
    Job.run(
        name=_job_name(job_id),
        machine=Machine.from_str(machine_name),
        image=image,
        command=command,
        teamspace=teamspace,
        user=_username(),
        interruptible=interruptible,
        max_runtime=int(timeout_s + _STARTUP_BUFFER_S),
    )
    deadline = time.time() + timeout_s + _STARTUP_BUFFER_S
    db.save_running(_db_path(), job_id, deadline)
    return {"job_id": job_id}


def _classify_failure(logs: str) -> str:
    lowered = logs.lower()
    return "oom" if any(marker in lowered for marker in _OOM_MARKERS) else "failed"


def _real_poll_job(job_id: str) -> dict:
    """저장된 status가 아직 'running'일 때만 원격 상태를 실제로 조회(불필요한 API 호출 방지)."""
    from lightning_sdk import Job, Status

    path = _db_path()
    stored = db.get_job(path, job_id)
    if stored is None:
        return {"status": "unknown", "metrics": None, "artifacts": {}, "detail": "job 없음"}
    if stored["status"] != "running":
        return stored

    teamspace = _require_teamspace()
    job = Job(name=_job_name(job_id), teamspace=teamspace, user=_username())
    status = job.status

    if status in (Status.Pending, Status.Running):
        if time.time() > (stored["deadline"] or float("inf")):
            job.stop()
            db.save_job(path, job_id, "timeout", None, {}, "타임아웃(원격, deadline 초과)")
            return db.get_job(path, job_id)
        return stored

    logs = job.logs or ""
    if status == Status.Completed:
        metrics, artifacts = runner.parse_output(logs)
        if metrics is None:
            db.save_job(path, job_id, "failed", None, {},
                        "출력 계약 위반(마지막 줄 metrics JSON 없음)")
        else:
            db.save_job(path, job_id, "completed", metrics, artifacts, "")
    else:  # Failed / Stopped / Stopping — 원격에서 비정상 종료
        category = _classify_failure(logs)
        db.save_job(path, job_id, category, None, {}, logs[-500:])
    return db.get_job(path, job_id)


def _real_get_credits() -> dict:
    teamspace = _require_teamspace()
    balance = _rest_client().billing_service_get_project_balance(
        project_id=_project_id(teamspace))
    return {"remaining": balance.balance}


@mcp.tool()
def submit_job(entrypoint_path: str, timeout_s: float) -> dict:
    """잡 제출. mock: 크레딧 차감 후 동기 실행. real: 크레딧 잔액 확인 후 원격 T4에
    비동기 제출("running" 상태로 즉시 반환, 결과는 poll_job으로 조회)."""
    path = _db_path()
    if _is_mock():
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

    credits = _real_get_credits()
    if credits["remaining"] <= 0:
        return {"job_id": "", "rejected": "credit"}
    return _real_submit_job(entrypoint_path, timeout_s)


@mcp.tool()
def poll_job(job_id: str) -> dict:
    """잡 상태 조회 — {"status", "metrics", "artifacts", "detail"}. real은 논터미널일 때만 원격 조회."""
    if _is_mock():
        job = db.get_job(_db_path(), job_id)
        if job is None:
            return {"status": "unknown", "metrics": None, "artifacts": {}, "detail": "job 없음"}
        return job
    return _real_poll_job(job_id)


@mcp.tool()
def get_credits() -> dict:
    """잔여 크레딧 조회. mock: SQLite 카운터. real: teamspace 잔액($, 스펙 §7-② 리필 정책 미확인)."""
    if _is_mock():
        path = _db_path()
        db.init_credits(path, float(os.environ.get("COSCIENTIST_CREDITS", "100")))
        return {"remaining": db.get_credits(path)}
    return _real_get_credits()


if __name__ == "__main__":
    mcp.run()
