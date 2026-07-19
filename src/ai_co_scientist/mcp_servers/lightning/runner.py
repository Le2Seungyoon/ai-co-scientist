"""잡 실행 유틸 — 잡 실행은 인프라(lightning) 소관 (M3 executor logic에서 이사)."""
import json
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class RunAttempt:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def execute_entrypoint(path: str, timeout_s: float) -> RunAttempt:
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, encoding="utf-8",
            timeout=timeout_s, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        return RunAttempt(-1, e.stdout or "", e.stderr or "", timed_out=True)
    return RunAttempt(proc.returncode, proc.stdout, proc.stderr, timed_out=False)


def parse_output(stdout: str) -> tuple[dict | None, dict]:
    """마지막 비공백 줄 JSON에서 (숫자 값 → metrics, 문자열 값 → artifacts) 분리.

    유효한 metrics가 하나도 없으면 (None, {}) — 출력 계약 위반.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None, {}
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None, {}
    if not isinstance(payload, dict):
        return None, {}
    metrics: dict = {}
    artifacts: dict = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
        elif isinstance(value, str):
            artifacts[key] = value
    if not metrics:
        return None, {}
    return metrics, artifacts
