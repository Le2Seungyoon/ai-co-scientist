#!/usr/bin/env python3
"""PreToolUse deny: experiment-execution commands only run where the registry lives.

See `.claude/rules/enforcement.md` -> Hook contracts and
`.claude/rules/architecture.md` -> Parallel execution contract.

  Event     PreToolUse (Bash) only. It DENIES; the advisory counterpart is check_rules_size.py.
  Governed  The commands in GUARDED, which all read or write `runtime/`. `scripts/legacy/*.py`
            also touch `runtime/` and were considered -- excluded because they are frozen,
            reproduction-only, and never run (`scripts/legacy/README.md`), not because they
            were overlooked.
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
    bypasses it entirely. Separately, the matcher itself only looks within one command segment
    (split on `;`, `&`, `|`, newline): `python -V; sh scripts/train_level.py` is not caught,
    because the invocation token and the guarded path sit in different segments. See
    `guarded_hit`'s docstring for the exact boundary.
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
    """`__file__`-based, not cwd: this file is `<root>/.agent-hooks/`."""
    if os.environ.get("CLAUDE_PROJECT_DIR"):
        return os.environ["CLAUDE_PROJECT_DIR"]
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    A match requires an invocation token -- `uv run` or a `python`/`pythonX.Y` token -- to START
    a command segment (a segment is the whole command, or the text following `;`, `&`, `|`, or a
    newline), with the guarded path appearing anywhere later in that same segment.

    Catches: `uv run python scripts/train_level.py`, `python scripts/exp.py new`, and the
    no-`python`-token forms `uv run scripts/exp.py` / `uv run ./scripts/exp.py` (the `./` is
    absorbed by the "anywhere later" match) -- with either path separator.

    Does NOT catch:
    - The guarded path with no invocation token opening its segment: `cat
      scripts/train_structure.py`, `grep foo scripts/exp.py`. Reading a guarded script is
      legitimate work for the lane this hook governs; executing it is not.
    - A MENTION of the invocation rather than its use: `echo "run python scripts/exp.py
      later"` -- the `python` token there sits inside the `echo` segment, not at its start, so
      it never anchors a match. Without this, a report-writing heredoc that merely quotes a
      command would be denied, and the only way past that denial is the same escape hatch that
      also unblocks real execution -- a worse failure than the false negative it would close.
    - An invocation token and the guarded path split across a segment boundary: `python -V; sh
      scripts/train_level.py` -- declared KNOWN GAP in the module docstring, left alone.

    Not a shell parser: a caller who restructures the command to dodge this has made a
    decision, which is what the escape hatch is for.
    """
    for script in GUARDED:
        escaped = re.escape(script).replace("/", r"[/\\]")
        pattern = r"(?:^|[;&|\n])\s*(?:uv\s+run|python[\w.]*)\b[^;&|\n]*" + escaped
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
