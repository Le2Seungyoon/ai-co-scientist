#!/usr/bin/env python3
"""PreToolUse deny: experiment-execution commands only run where the registry lives.

See `.agents/rules/enforcement.md` -> Hook contracts and
`.agents/rules/architecture.md` -> Parallel execution contract.

  Event     PreToolUse (Bash) only. It DENIES; the advisory counterpart is check_rules_size.py.
  Governed  The commands in GUARDED (REGISTRY_WRITERS + EXCLUSIVE), which all read or write
            `runtime/` -- graded by why they are guarded, not by one blanket reason.
            REGISTRY_WRITERS (`scripts/exp.py`) would fork the experiment registry.
            EXCLUSIVE (`train_level.py`, `train_structure.py`, `infer_decomposed.py`,
            `dacon_submit.py`) need a resource the main worktree owns: the single 8 GB GPU, or
            the finite DACON submission quota. `scripts/probe_level.py` (read-only, no
            `runtime/` writes) and `scripts/assemble_submission.py` (CPU-only, writes only its
            own tree) are deliberately NOT guarded. `scripts/legacy/*.py` also touch `runtime/`
            and were considered -- excluded because they are frozen, reproduction-only, and
            never run (`scripts/legacy/README.md`), not because they were overlooked.
  Verdict   Turns on whether `<project root>/runtime/registry.jsonl` exists. `runtime/` is
            gitignored, so a git worktree has none -- running `exp.py new` there would issue
            report_id 1 again and fork the registry silently. Registry divergence is worse
            than a lost race: `registry.locked()` locks a per-worktree path, so two worktrees
            never even contend.
  Failure   Unreadable or unparseable payload -> exit 0, note on stderr. A hook must never
            block an edit for a reason unrelated to what it checks.
  Escape    `ACS_RUNTIME_EXEMPT="<reason>"` in the environment, but ONLY for EXCLUSIVE. A blank
            reason does not pass: the point is to turn a silent bypass into a decision a
            reviewer can see. REGISTRY_WRITERS has NO escape hatch: writing here would itself
            create the sentinel and thereby unlock every other gate in this tree for good.
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

# The registry's single truth. `report_id` comes from `len(records)`, so a second tree starts
# over at 1 and the two truths can never be merged -- the escape hatch does NOT cover this:
# writing here CREATES the sentinel, which would permanently unlock the gate in that tree.
REGISTRY_WRITERS = ("scripts/exp.py",)

# Exclusive resources: the single 8 GB GPU, and the finite DACON submission quota.
EXCLUSIVE = (
    "scripts/train_level.py",
    "scripts/train_structure.py",
    "scripts/infer_decomposed.py",
    "scripts/dacon_submit.py",
)

# `scripts/probe_level.py` is deliberately absent: it reads with mmap_mode="r" and writes
# nothing under runtime/, so it has no registry-fork path. With no cache in the tree `np.load`
# fails loudly -- there is no silent-corruption route to guard against.
# `scripts/assemble_submission.py` is absent for the same reason: it writes only its own tree.
GUARDED = REGISTRY_WRITERS + EXCLUSIVE

EXEMPT_VAR = "ACS_RUNTIME_EXEMPT"
SENTINEL = os.path.join("runtime", "registry.jsonl")

REASON_REGISTRY = (
    "This tree has no {sentinel}. Two situations look identical to this gate -- tell them "
    "apart yourself: "
    "(1) This is a git worktree: `runtime/` is gitignored so it was never copied here. "
    "Running `{hit}` here would fork the registry -- report_id comes from len(records), so a "
    "second tree starts over at 1 and the two truths can never be merged. Run registry "
    "commands in the MAIN worktree, where the registry lives. "
    "(2) This already IS the main worktree -- a fresh clone, a re-imaged machine, or a "
    "`runtime/` lost to cleanup -- and there is no other tree to defer to. That is a "
    "bootstrap, not a fork: create the sentinel yourself, outside this gate, with an empty "
    "file (`mkdir -p runtime && touch {sentinel}`, no `{hit}` involved); the next `{hit}` run "
    "then sees zero existing records and starts at report_id 1, same as any other first run. "
    "There is NO escape hatch through this gate for either case: letting `{hit}` itself "
    "create {sentinel} would unlock every other gate in this tree for good. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
)

REASON_EXCLUSIVE = (
    "This tree has no {sentinel} -- it is a git worktree, and `runtime/` is gitignored so it "
    "was never copied. `{hit}` needs an exclusive resource the MAIN worktree owns: the single "
    "8 GB GPU, or the finite DACON submission quota. This lane is for additive code, offline "
    "tests, and CPU reassembly (`scripts/assemble_submission.py`, `scripts/probe_level.py`), "
    "which are not guarded here. "
    "Escape hatch: set {var}=\"<reason>\" for this command. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
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
    """Return (guarded script, kind) the command *executes*, or (None, None).

    `kind` is "registry" or "exclusive" -- the deny reason and whether the escape hatch
    applies both turn on it.

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
            return script, ("registry" if script in REGISTRY_WRITERS else "exclusive")
    return None, None


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (ValueError, OSError) as exc:
        sys.stderr.write(
            "[runtime-lane] payload unreadable ({0}) -- command NOT checked.\n".format(exc)
        )
        return

    command = (payload.get("tool_input") or {}).get("command") or ""
    hit, kind = guarded_hit(command)
    if not hit:
        return

    if os.path.exists(os.path.join(project_root(), SENTINEL)):
        return

    sentinel = SENTINEL.replace("\\", "/")
    if kind == "registry":
        deny(REASON_REGISTRY.format(sentinel=sentinel, hit=hit))
        return

    if os.environ.get(EXEMPT_VAR, "").strip():
        return

    deny(REASON_EXCLUSIVE.format(sentinel=sentinel, hit=hit, var=EXEMPT_VAR))


if __name__ == "__main__":
    main()
