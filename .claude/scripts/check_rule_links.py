#!/usr/bin/env python3
"""Scanner: every file a rules file points at must exist.

Run: python3 .claude/scripts/check_rule_links.py     (exit 1 on findings)

The CLAUDE.md rules table and the cross-references between rules files are the only delivery
mechanism the harness has, so **a stale path there is a rule nobody is told to read**. Nothing
else notices: the prose still reads correctly, and the reader simply never arrives.

`workflow.md` -> File size budget actively creates this risk — option (1) Relocate says to move
a section and "leave a one-line pointer behind", and option (2) Split creates a new file others
must now reference. This is the check that keeps those pointers honest.

Why a script and not a hook (`enforcement.md` -> Four layers): a rules file is routinely edited
in an IDE, and a pointer breaks when a file is *renamed or deleted* — which is not an edit any
PostToolUse hook observes on the file that names it. This belongs in the test/CI layer; wire it
into the project's suite (`testing.md` -> Invariant tests) so it runs on every authoring path.

FALSE-POSITIVE SCOPING (`enforcement.md` -> Before promoting a check to deny). Deliberately
narrow, because a checker that flags prose gets deleted:
  - Only tokens ending in .md / .py / .json / .sh — harness-internal references. A directory
    (`docs/`), a config file the project may not have yet (`.env.template`), and a bare English
    word are all out of scope by construction.
  - Anything containing a glob, a placeholder, or an angle-bracket is a PATTERN, not a path.
  - A bare filename resolves against the places harness files live, so `settings.json` and
    `workflow.md` are found without every reference having to spell out a full path.
  - A backticked path WRAPPED ACROSS A LINE BREAK is not matched at all (the regex is
    newline-free), so it reads as a pointer and is checked by nothing. Judge a new pointer by the
    `N references` count moving, never by the CLEAN verdict alone.
Measured on the shipped skeleton: 0 findings. Re-measure before tightening any of the above.

This file ships in every project the harness scaffolds. Its LOGIC is project-agnostic, with one
deliberate exception: the source roots in `SEARCH_DIRS` are filled per project. That line is
expected to differ and is NOT drift — `harness-spine:update` reconciles around it, never onto it.
Everything else: do not hand-edit here; reconcile with that skill so the copies do not drift. If
this project diverges anywhere beyond the source roots, say why in this docstring.
"""
import os
import re
import sys

GOVERNED = ("CLAUDE.md", ".claude/rules")
EXTENSIONS = (".md", ".py", ".json", ".sh")
# Where a bare filename is allowed to live: the harness dirs, plus this project's SOURCE ROOTS.
# Source roots, filled for this project. Rules files name modules the way the code imports them
# (`features/series.py`, `_common.py`), not by repo-relative path, so a project whose source is
# not at the repo root resolves almost nothing without them. Measured on one such repo: 13 of 19
# findings were real files under an unlisted `include/custflow`, a 95% false-positive rate — and
# `enforcement.md` -> Before promoting a check to deny is exactly about that being fatal.
SEARCH_DIRS = (
    "", ".claude", ".claude/rules", ".claude/hooks", ".claude/scripts",
    "src", "src/ai_co_scientist", "scripts", "scripts/legacy", "tests", "docs",
)
# A token carrying any of these describes a shape, not a file.
PATTERN_CHARS = ("*", "?", "{{", "<", ">", " ", "|")

TOKEN_RE = re.compile(r"`([^`\n]+)`|\]\(([^)\s]+)\)")


def candidates(text):
    for backticked, linked in TOKEN_RE.findall(text):
        tok = (backticked or linked).strip().rstrip(".,;:")
        if not tok.endswith(EXTENSIONS):
            continue
        if tok.startswith(("http://", "https://", "#")):
            continue
        if any(c in tok for c in PATTERN_CHARS):
            continue
        yield tok


def resolves(root, referrer_dir, token):
    if os.path.exists(os.path.join(referrer_dir, token)):
        return True
    return any(os.path.exists(os.path.join(root, d, token)) for d in SEARCH_DIRS)


def governed_files(root):
    for entry in GOVERNED:
        path = os.path.join(root, entry)
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.endswith(".md"):
                    yield os.path.join(path, name)


def main():
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    files = list(governed_files(root))

    # An empty target set is a failure, not a pass (`enforcement.md` -> Hook contracts).
    if not files:
        print(f"RULE_LINKS_NO_TARGETS  no governed files under {root} ({', '.join(GOVERNED)})")
        return 1

    findings, checked = [], 0
    for path in files:
        rel = os.path.relpath(path, root).replace("\\", "/")
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                for token in candidates(line):
                    checked += 1
                    if not resolves(root, os.path.dirname(path), token):
                        findings.append(f"{rel}:{lineno}  -> {token}")

    if findings:
        print(f"RULE_LINKS_BROKEN  {len(findings)} of {checked} references do not resolve:")
        for f in findings:
            print("  " + f)
        print("Fix the pointer or restore the file; a rule nobody is told to read is not a rule.")
        return 1

    print(f"RULE_LINKS_CLEAN  {checked} references across {len(files)} files all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
