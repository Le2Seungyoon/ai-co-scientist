#!/usr/bin/env python3
"""PreToolUse deny: experiment-execution commands only run where the registry lives.

See `.claude/rules/enforcement.md` -> Hook contracts and
`.claude/rules/architecture.md` -> Parallel execution contract.

  Event     PreToolUse (Bash) only. It DENIES; the advisory counterpart is check_rules_size.py.
  Governed  The commands in GUARDED, which all read or write `runtime/`.
  Verdict   Turns on whether `<project root>/runtime/registry.jsonl` exists. `runtime/` is
            gitignored, so a git worktree has none -- running `exp.py new` there would issue
            report_id 1 again and fork the registry silently. Registry divergence is worse
            than a lost race: `registry.locked()` locks a per-worktree path, so two worktrees
            never even contend.
  Failure   Unreadable or unparseable payload -> exit 0, note on stderr. A hook must never
            block an edit for a reason unrelated to what it checks.
  Escape    `ACS_RUNTIME_EXEMPT="<reason>"` in the environment. A blank reason does not pass:
            the point is to turn a silent bypass into a decision a reviewer can see.
  Tests     `test_block_runtime_commands.py`, beside this file.

KNOWN GAP (state it rather than let it pass quietly)
    This hook cannot see WHICH sub-agent issued the command -- the PreToolUse payload does not
    carry the sub-agent identity. So it does not stop an `engineer` from training inside the
    MAIN worktree; it only makes the boundary real in a registry-less tree. The consequence is
    a requirement, not a caveat: **the engineer lane must run in a worktree**, or its contract
    is prose only. And like every hook it sees this session's tool calls -- an IDE terminal
    bypasses it entirely.
"""
import json
import os
import re
import sys

GUARDED = (
    "scripts/exp.py",
    "scripts/train_level.py",
    "scripts/train_structure.py",
    "scripts/infer_decomposed.py",
    "scripts/dacon_submit.py",
    "scripts/probe_level.py",
)
EXEMPT_VAR = "ACS_RUNTIME_EXEMPT"
SENTINEL = os.path.join("runtime", "registry.jsonl")

REASON = (
    "This tree has no {sentinel} -- it is a git worktree, and `runtime/` is gitignored so it "
    "was never copied. Running `{hit}` here would fork the registry: report_id comes from "
    "len(records), so a second tree starts over at 1 and the two truths can never be merged. "
    "Run experiment commands in the MAIN worktree, where the registry lives. This lane "
    "(engineer / harness-manager) is for additive code and offline tests only. "
    "Escape hatch: set {var}=\"<reason>\" for this command. "
    "-- .claude/rules/architecture.md -> Parallel execution contract"
)


def project_root():
    """`__file__`-based, not cwd: this file is `<root>/.claude/hooks/`."""
    if os.environ.get("CLAUDE_PROJECT_DIR"):
        return os.environ["CLAUDE_PROJECT_DIR"]
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deny(message):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": message,
        }
    }))


def guarded_hit(command):
    """Return the guarded script the command *executes*, or None.

    Catches: the guarded path preceded by a `python` token on the same command segment
    (`uv run python scripts/train_level.py`, `python scripts/exp.py new`), with either path
    separator. Does NOT catch: the bare path with no preceding `python` -- so `cat
    scripts/train_structure.py`, `grep foo scripts/exp.py`, or an editor opening the file read
    it without tripping this hook. Reading a guarded script is legitimate work for the lane
    this hook governs; executing it is not. Not a shell parser: a caller who rewrites the
    command to dodge this has made a decision, which is what the escape hatch is for.
    """
    for script in GUARDED:
        escaped = re.escape(script).replace("/", "[/\\\\]")
        pattern = r"python[^|;&\n]*" + escaped
        if re.search(pattern, command):
            return script
    return None


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (ValueError, OSError) as exc:
        sys.stderr.write(
            "[runtime-lane] payload unreadable ({0}) -- command NOT checked.\n".format(exc)
        )
        return

    command = (payload.get("tool_input") or {}).get("command") or ""
    hit = guarded_hit(command)
    if not hit:
        return

    if os.path.exists(os.path.join(project_root(), SENTINEL)):
        return

    if os.environ.get(EXEMPT_VAR, "").strip():
        return

    deny(REASON.format(sentinel=SENTINEL.replace("\\", "/"), hit=hit, var=EXEMPT_VAR))


if __name__ == "__main__":
    main()
