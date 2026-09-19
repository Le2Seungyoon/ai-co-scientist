"""하니스 자체를 검사하는 층 — 훅이 못 보는 저작 경로까지 덮는다.

훅은 **이번 세션의 편집만** 본다. IDE에서 고친 파일, 다른 에이전트가 쓴 파일, 동료의 커밋은 어느
훅도 지나지 않는다. 그래서 이 파일이 보장이고 훅은 빠른 피드백이다
(`.agents/rules/enforcement.md` → Hooks only see this session's edits).

`.agent-hooks/` 밑의 테스트는 `__main__` 스크립트이고 그 디렉토리는 dot으로 시작해 pytest가
수집하지 않는다(`testpaths = ["tests"]`). 여기서 서브프로세스로 **실행만** 끌어온다 — 같은 판단을
두 번 구현하지 않기 위해서다.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".agent-hooks"

# 이름으로 찾지 않고 나열한다: glob이 0개를 매치해도 pytest는 조용히 초록이 되고,
# "검사 대상이 사라진 것"과 "통과한 것"이 구별되지 않는다.
HARNESS_TESTS = (
    "test-build-agents.py",
    "test_block_runtime_commands.py",
    "test_check_rule_links.py",
    "test_check_rules_size.py",
)


def _run(*args):
    return subprocess.run(
        [sys.executable] + [str(a) for a in args],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    )


@pytest.mark.parametrize("name", HARNESS_TESTS)
def test_harness_script_suite_passes(name):
    script = HOOKS / name
    assert script.is_file(), (
        f"{name} 이 `.agent-hooks/` 에 없다. 이름이 바뀌었다면 HARNESS_TESTS 도 같이 고칠 것 — "
        "목록에서 빠진 스위트는 아무 소리 없이 안 돌아간다."
    )
    proc = _run(script)
    assert proc.returncode == 0, f"{name} 실패\n{proc.stdout}\n{proc.stderr}"


def test_generated_lanes_and_skills_match_their_sources():
    proc = _run(HOOKS / "build-agents.py", "--check")
    marker = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
    assert marker.startswith("AGENTS_FRESH"), (
        "생성물이 소스와 어긋났거나 판정이 불가능하다.\n"
        "`uv run python .agent-hooks/build-agents.py` 로 재생성하고 함께 커밋할 것.\n"
        "AGENTS_STALE = 드리프트, AGENTS_UNKNOWN = 판정 불가. 둘은 다른 상태다.\n"
        f"marker={marker!r} rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )


def test_every_rule_and_lane_pointer_resolves():
    proc = _run(HOOKS / "check_rule_links.py")
    assert proc.returncode == 0, (
        "규칙·레인 파일이 가리키는 경로 중 존재하지 않는 것이 있다. 지시 전달 경로가 끊긴 것이므로 "
        f"산문이 옳게 읽히는 것과 무관하다.\n{proc.stdout}\n{proc.stderr}"
    )
