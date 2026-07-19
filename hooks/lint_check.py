"""Coder 산출물 결정적 가드레일 — LLM 호출 없이 컴파일 + ruff 검사 (스펙 §5)."""
import py_compile
import subprocess
import sys


def lint_check(path: str) -> tuple[bool, str]:
    """(통과 여부, 사유). 실패 사유는 self-correct의 입력이 된다."""
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as e:
        return False, f"컴파일 실패: {e}"
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--quiet", path],
        capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return False, f"ruff 실패:\n{proc.stdout}{proc.stderr}"
    return True, "ok"
