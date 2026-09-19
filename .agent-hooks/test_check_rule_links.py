#!/usr/bin/env python3
"""Tests for check_rule_links.py — run: python3 test_check_rule_links.py

A scanner is code, so it ships with a test (`enforcement.md` -> Hook contracts). The must-pass
half carries the weight here: this check reads prose, so its whole risk is flagging something
that was never a path. Every scoping decision in the scanner's docstring has a case below.

Stdlib only, no test runner — it must be verifiable in a freshly scaffolded repo.
"""
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_rule_links.py")
failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def run(root):
    proc = subprocess.run(
        [sys.executable, SCRIPT], capture_output=True, text=True, timeout=10,
        env=dict(os.environ, CLAUDE_PROJECT_DIR=root),
    )
    return proc.stdout.strip(), proc.returncode


def build(root, rules=None, extra=None, claude_md="# Project\n"):
    os.makedirs(os.path.join(root, ".agents", "rules"), exist_ok=True)
    os.makedirs(os.path.join(root, ".agent-hooks"), exist_ok=True)
    with open(os.path.join(root, "AGENTS.md"), "w", encoding="utf-8") as f:
        f.write(claude_md)
    for name, body in (rules or {}).items():
        with open(os.path.join(root, ".agents", "rules", name), "w", encoding="utf-8") as f:
            f.write(body)
    for rel in extra or []:
        path = os.path.join(root, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w", encoding="utf-8").close()


def main():
    # --- must-block half ---
    with tempfile.TemporaryDirectory() as root:
        build(root, rules={"a.md": "See `moved-away.md` for details.\n"})
        out, rc = run(root)
        check("broken sibling pointer is caught", "RULE_LINKS_BROKEN" in out, out[:70])
        check("finding names the referrer and line", "a.md:1" in out, out[:70])
        check("finding names the token", "moved-away.md" in out, out[:70])
        check("broken pointer exits 1", rc == 1, f"rc={rc}")

    with tempfile.TemporaryDirectory() as root:
        build(root, rules={"a.md": "Wired in `.agent-hooks/gone.py`.\n"})
        check("broken full path is caught", "RULE_LINKS_BROKEN" in run(root)[0])

    with tempfile.TemporaryDirectory() as root:
        build(root, rules={"a.md": "Read [the design](docs/2026-01-01-gone.md).\n"})
        check("broken markdown link is caught", "RULE_LINKS_BROKEN" in run(root)[0])

    with tempfile.TemporaryDirectory() as root:
        build(root, claude_md="| `nope.md` | when to read |\n")
        check("broken pointer in AGENTS.md is caught", "RULE_LINKS_BROKEN" in run(root)[0])

    with tempfile.TemporaryDirectory() as root:
        build(root)
        os.makedirs(os.path.join(root, ".agents", "agents"), exist_ok=True)
        with open(os.path.join(root, ".agents", "agents", "gardener.md"), "w", encoding="utf-8") as f:
            f.write("---\nname: gardener\n---\nSee `.agent-hooks/gone.py` for the guard.\n")
        out, rc = run(root)
        check("broken pointer in .agents/agents/*.md is caught", "RULE_LINKS_BROKEN" in out, out[:70])
        check(".agents/agents finding names its file", "gardener.md:4" in out, out[:70])

    # --- must-pass half: none of these is a broken path ---
    with tempfile.TemporaryDirectory() as root:
        build(
            root,
            rules={
                "a.md": (
                    "Sibling by bare name: `b.md`.\n"
                    "Bare harness file: `settings.json`.\n"
                    "Full path: `.agent-hooks/check_rules_size.py`.\n"
                    "Root file by name: `AGENTS.md`.\n"
                    "A glob: `.claude/rules/*.md`.\n"
                    "A placeholder: `origin/{{DEFAULT_BRANCH}}/notes.md`.\n"
                    "A shape: `test_<name>.py` and `tests/<module>/test_*.py`.\n"
                    "A URL: `https://example.com/spec.md`.\n"
                    "A command: `git push` and `uv run pytest`.\n"
                    "A directory: `docs/` and `src/`.\n"
                    "A file with no listed extension: `.env.template`.\n"
                    "A skill id: `claude-md-management:claude-md-improver`.\n"
                ),
                "b.md": "# B\n",
            },
            extra=[".claude/settings.json", ".agent-hooks/check_rules_size.py"],
        )
        out, rc = run(root)
        check("clean tree reports clean", "RULE_LINKS_CLEAN" in out, out[:90])
        check("clean tree exits 0", rc == 0, f"rc={rc}")
        check("clean tree says how many it examined", "references across" in out, out[:90])

    with tempfile.TemporaryDirectory() as root:
        build(root, rules={"a.md": "# A\n"})
        os.makedirs(os.path.join(root, ".agents", "agents"), exist_ok=True)
        with open(os.path.join(root, ".agents", "agents", "gardener.md"), "w", encoding="utf-8") as f:
            f.write("---\nname: gardener\n---\nRead `.agents/rules/a.md` first.\n")
        out, rc = run(root)
        check(".agents/agents is in the governed scope", "RULE_LINKS_CLEAN" in out, out[:90])
        check(".agents/agents reference is counted", "1 references across 3 files" in out, out)

    # --- an empty target set is a failure, not a pass ---
    with tempfile.TemporaryDirectory() as root:
        out, rc = run(root)
        check("empty governed set says so", "RULE_LINKS_NO_TARGETS" in out, out[:70] or "silent")
        check("empty governed set exits 1", rc == 1, f"rc={rc}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
