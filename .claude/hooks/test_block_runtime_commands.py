#!/usr/bin/env python3
"""Tests for block_runtime_commands.py — run: python .claude/hooks/test_block_runtime_commands.py

A hook is code, so it ships with a test (`enforcement.md` -> Hook contracts). The must-block
half proves the deny fires; the must-pass half is what keeps false positives from creeping in.
A test that only ever asserts "it blocked" cannot notice the day the hook blocks everything.

Each scenario needs its own tree because the verdict turns on whether
`<root>/runtime/registry.jsonl` exists. `CLAUDE_PROJECT_DIR` points the hook at that tree.

Stdlib only, no test runner: pytest does not collect this file (`testpaths = ["tests"]`).
"""
import json
import os
import subprocess
import sys
import tempfile

HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "block_runtime_commands.py")

failures = []


def run(root, command, env_extra=None, raw=None):
    """Feed the hook a Bash payload; return (stdout, returncode)."""
    env = dict(os.environ, CLAUDE_PROJECT_DIR=root)
    if env_extra:
        env.update(env_extra)
    payload = raw if raw is not None else json.dumps(
        {"hook_event_name": "PreToolUse", "tool_name": "Bash",
         "tool_input": {"command": command}}
    )
    proc = subprocess.run(
        [sys.executable, HOOK], input=payload, capture_output=True, text=True, env=env
    )
    return proc.stdout, proc.returncode


def denied(stdout):
    if not stdout.strip():
        return False
    try:
        out = json.loads(stdout)
    except ValueError:
        return False
    return out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"


def check(label, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + label + ("" if condition else f"  <- {detail}"))
    if not condition:
        failures.append(label)


def make_tree(with_registry):
    root = tempfile.mkdtemp()
    if with_registry:
        os.makedirs(os.path.join(root, "runtime"))
        with open(os.path.join(root, "runtime", "registry.jsonl"), "w") as f:
            f.write("{}\n")
    return root


def main():
    print("must-block")
    worktree = make_tree(with_registry=False)
    for cmd in (
        "uv run python scripts/train_structure.py --arch mlp",
        "uv run python scripts/train_level.py",
        "uv run python scripts/exp.py new --title x",
        "uv run python scripts/infer_decomposed.py --submit runtime/submissions/a.zip",
        "uv run python scripts/dacon_submit.py runtime/submissions/a.zip",
        "uv run python scripts/probe_level.py",
    ):
        out, rc = run(worktree, cmd)
        check(f"denies: {cmd.split()[3]}", denied(out), out[:120] or "silent")
        check("deny still exits 0", rc == 0, f"rc={rc}")

    out, _ = run(worktree, "uv run python scripts/exp.py new --title x")
    check("deny names the rule file", "architecture.md" in out, out[:120])
    check("deny names the escape hatch", "ACS_RUNTIME_EXEMPT" in out, out[:120])

    out, _ = run(worktree, "uv run python scripts/train_level.py",
                 env_extra={"ACS_RUNTIME_EXEMPT": "   "})
    check("empty exemption reason does NOT pass", denied(out), out[:120] or "silent")

    print("must-pass")
    main_tree = make_tree(with_registry=True)
    out, rc = run(main_tree, "uv run python scripts/train_structure.py --arch mlp")
    check("main worktree: experiment command passes", not denied(out), out[:120])
    check("main worktree: exits 0", rc == 0, f"rc={rc}")

    out, _ = run(worktree, "uv run pytest -q")
    check("worktree: unrelated command passes", not denied(out), out[:120])
    out, _ = run(worktree, "uv run ruff check src tests scripts")
    check("worktree: lint passes", not denied(out), out[:120])
    out, _ = run(worktree, "cat scripts/train_structure.py")
    check("worktree: reading a script passes", not denied(out), out[:120])

    out, _ = run(worktree, "uv run python scripts/train_level.py",
                 env_extra={"ACS_RUNTIME_EXEMPT": "restoring a ckpt the main tree lost"})
    check("stated exemption reason passes", not denied(out), out[:120])

    print("environment failure")
    out, rc = run(worktree, None, raw="not json at all")
    check("garbage payload exits 0", rc == 0, f"rc={rc}")
    check("garbage payload does not deny", not denied(out), out[:120])
    out, rc = run(worktree, None, raw="")
    check("empty payload exits 0", rc == 0, f"rc={rc}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
