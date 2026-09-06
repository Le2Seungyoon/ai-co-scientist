#!/usr/bin/env python3
"""PostToolUse nudge: warn when a governed instruction file exceeds the soft line budget.

Contract — this hook is the template every other hook in this project follows.
See `.claude/rules/enforcement.md` -> Hook contracts.

  Event     PostToolUse only. It ADVISES and never denies; denial belongs to PreToolUse.
  Governed  `.claude/rules/*.md` and the root `CLAUDE.md` (see GOVERNED).
  Input     None. It SCANS the governed set and never reads the payload — see below.
  Silence   Means exactly one thing: every governed file was read and is within budget.
            Anything that stopped the check from running says so in one line instead of
            passing quietly, so "all clear" and "never looked" stay distinguishable.
  Failure   Never blocks, never raises. An unreadable file exits 0 with a note.
  Advisory  Detection is deterministic; the fix is judgment, so this reports and lets the
            author choose — see `workflow.md` -> File size budget.
  Tests     `test_check_rules_size.py`, beside this file: must-block and must-pass halves.

WHY IT SCANS INSTEAD OF READING `tool_input.file_path`
    Keying on the payload path meant the nudge only ever saw Write / Edit / MultiEdit. A rules
    file changed through `sed`, a heredoc, or a `python3` one-liner slipped past in silence,
    and a session doing its edits through the shell tripped the hook exactly never. Recovering
    a path by parsing shell commands is the fragile alternative; scanning is not. The governed
    set is under twenty short files, so a full pass costs nothing, and a clean tree prints
    nothing — the nudge stays silent until some file is actually over budget, whoever put it
    there and however it was written. Wire the matcher to include `Bash` for the same reason.

KNOWN GAP (state it rather than let it pass quietly)
    GOVERNED does not walk the tree, so a nested `CLAUDE.md` in a subdirectory is not scanned.
    Add its pattern to GOVERNED if this project has one.

This file is project-agnostic; it ships verbatim in every project the harness scaffolds.
Do not hand-edit it here — reconcile with the `harness-spine:update` skill so the copies do not
drift. If this project deliberately diverges, say why in this docstring.
Tune BUDGET below if a project wants a different soft cap.

DELIBERATE DIVERGENCE (settings.json, not this file): this project wires the hook with
    `python`, not the skeleton's `python3` — Windows ships no `python3` on PATH. The
    interpreter that resolves here is 3.8, so keep this file free of 3.9+ syntax.
"""
import glob
import json
import os
import sys

BUDGET = 150  # soft line budget per instruction file
GOVERNED = (".claude/rules/*.md", "CLAUDE.md")

# A generated file is not hand-edited, so the four options below do not apply to it.
GENERATED_MARKERS = ("<!-- generated", "<!--generated")
HEAD_LINES = 5  # how far into a file to look for the marker

AUTHORED_ADVICE = (
    "REVIEW THE WHOLE FILE -- never just shave the line you added. Apply "
    "workflow.md -> 'File size budget', in order: (1) RELOCATE -- a section that's really "
    "another rules file's topic belongs there; move it and leave a one-line pointer. "
    "(2) SPLIT is your judgment -- if a distinct sub-topic can be gated by a paths: glob, "
    "move it to its own rules file (may drop this file under budget). (3) ABSTRACT -- if "
    "several concrete items are instances of one generative principle, state the principle "
    "and delete the examples it regenerates; keep only examples with a non-derivable why. "
    "(4) COMPRESS/dedupe/currency -- delegate: AskUserQuestion then invoke the "
    "claude-md-management:claude-md-improver skill on this file. Advisory: a tight "
    "single-topic file slightly over is fine."
)

GENERATED_ADVICE = (
    "GENERATED -- do NOT hand-edit it and do NOT run a compression skill over it; the next "
    "regeneration discards both. The only lever is the generator: narrow its scope, or cap "
    "how many entries it emits and have it state how many were truncated. See "
    "enforcement.md -> Keeping a generated artifact alive."
)


def emit(message: str) -> None:
    """Speak to the session. The only channel this hook has; it never changes the outcome."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": message,
                }
            }
        )
    )


def project_root() -> str:
    """`__file__`-based, not cwd: this file is `<root>/.claude/hooks/`, so the root is derivable
    wherever the hook is invoked from. cwd is wrong from any subdirectory."""
    if os.environ.get("CLAUDE_PROJECT_DIR"):
        return os.environ["CLAUDE_PROJECT_DIR"]
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_generated(head: "list[str]") -> bool:
    return any(line.lstrip().lower().startswith(GENERATED_MARKERS) for line in head)


def scan(root: str):
    """Return (over_budget, unreadable, n_examined). Every governed file is accounted for."""
    paths = set()
    for pattern in GOVERNED:
        paths.update(glob.glob(os.path.join(root, pattern)))

    over, unreadable = [], []
    for path in sorted(paths):
        rel = os.path.relpath(path, root).replace("\\", "/")
        try:
            with open(path, "r", encoding="utf-8") as f:
                head, n_lines = [], 0
                for line in f:
                    n_lines += 1
                    if n_lines <= HEAD_LINES:
                        head.append(line)
        except OSError as exc:
            unreadable.append((rel, exc.strerror))
            continue
        if n_lines > BUDGET:
            over.append((rel, n_lines, is_generated(head)))
    return over, unreadable, len(paths)


def main() -> None:
    # The payload is drained and deliberately NOT parsed: what was edited, and by which tool,
    # no longer decides whether this check runs. Removing that coupling is the whole point.
    try:
        sys.stdin.read()
    except Exception:  # noqa: BLE001 - a hook never raises into the session
        pass

    root = project_root()
    over, unreadable, n_examined = scan(root)

    if n_examined == 0:
        emit(
            f"[rules-size] the governed set is EMPTY ({', '.join(GOVERNED)} under {root}) -- "
            "size budget NOT checked. A glob that stopped matching looks exactly like a clean "
            "tree; that is why this says so instead of passing."
        )
        return

    notes = []
    if unreadable:
        listed = ", ".join(f"{rel} ({why})" for rel, why in unreadable)
        notes.append(f"could not read {listed} -- those were NOT checked.")

    if over:
        listed = ", ".join(f"{rel} ({n} lines)" for rel, n, _ in over)
        notes.append(f"over the {BUDGET}-line soft budget: {listed}.")
        if any(not gen for _, _, gen in over):
            notes.append(AUTHORED_ADVICE)
        if any(gen for _, _, gen in over):
            gen_list = ", ".join(rel for rel, _, gen in over if gen)
            notes.append(f"{gen_list}: {GENERATED_ADVICE}")

    if notes:
        emit("[rules-size] " + " ".join(notes))
    # else: every governed file was read and is within budget -- the one thing silence may mean


if __name__ == "__main__":
    main()
