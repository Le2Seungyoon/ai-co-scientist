import os
import signal
import subprocess
import sys

import pytest

from ai_co_scientist.mcp_servers.shared_log import db


def test_runner_skeleton_e2e(tmp_path):
    """runner --skeleton 한 방: A2A 서버 spawn → PM 왕복 → MCP 로그 기록 → 정상 종료."""
    db_file = str(tmp_path / "log.sqlite3")
    proc = subprocess.Popen(
        [sys.executable, "-m", "ai_co_scientist.runner", "--skeleton"],
        env={**os.environ, "COSCIENTIST_LOG_DB": db_file},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # runner+자식들을 독립 프로세스 그룹으로 — 타임아웃 시 일괄 정리
    )
    try:
        stdout, stderr = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        stdout, stderr = proc.communicate()
        pytest.fail(f"runner가 120초 내에 끝나지 않아 프로세스 그룹을 강제 종료함.\nstdout:\n{stdout}\nstderr:\n{stderr}")

    assert proc.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "research_output" in stdout

    rows = db.query_ledger(db_file, record_type="hypothesis")
    assert len(rows) == 1
    assert rows[0]["owner"] == "research"
