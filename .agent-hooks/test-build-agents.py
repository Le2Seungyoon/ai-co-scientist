#!/usr/bin/env python3
"""Tests for `.agent-hooks/build-agents.py` — run: uv run python .agent-hooks/test-build-agents.py

Every case runs the script as a subprocess against a **synthetic tree** under `--root`, never the
real one, so a failing test cannot dirty `.claude/agents/**` or `.codex/agents/**`. The fixtures
invent their own lane and skill names on purpose: a test keyed on this repo's six lanes would go
quiet the day one is renamed, and quiet is exactly what must not happen here.

The properties, and what breaks when each is lost:

  round-trip      Building an unmodified tree changes nothing. Without it the PostToolUse hook
                  rewrites files on every unrelated edit, and the drift it exists to prevent
                  shows up in every diff instead.
  --check         Fails, and names the file, on a hand-edit of a generated file. Generated files
                  are committed and therefore editable; without this, editing the artifact
                  instead of the source is a silent no-op that survives review.
  key routing     `tools.claude` and `model.claude` reach Claude only; `model.codex` and
                  `nickname_candidates` reach Codex only. A key that leaks does not merely add
                  noise — Codex has no per-lane tool allowlist, so a `tools` line there would
                  read as a restriction that is not enforced.
  hard errors     An unknown frontmatter key, and a skill whose directory name and `name:`
                  disagree, stop the build. Both are silent misconfigurations otherwise: the
                  first is an edit that appears to work and generates nothing, the second
                  registers one skill under two names with no error from either harness.
  three states    AGENTS_FRESH / AGENTS_STALE / AGENTS_UNKNOWN are distinguishable, so a caller
                  cannot mistake "cannot judge" for "drifted" or a traceback for either.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, ".agent-hooks", "build-agents.py")

failures = []


def check(label, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + label + ("" if ok else "  (%s)" % detail))
    if not ok:
        failures.append(label)


def run(root, *args):
    return subprocess.run(
        [sys.executable, SCRIPT, "--root", root] + list(args),
        capture_output=True, text=True, encoding="utf-8",
    )


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as fh:
        return fh.read()


def exists(root, rel):
    return os.path.exists(os.path.join(root, rel))


LANE = """---
name: gardener
description: Prunes things.
tools.claude: Read, Grep
model.claude: opus
model.codex: gpt-5.6-sol
nickname_candidates: ["gardener", "pruner"]
---

Prune the thing. Read `.agents/rules/harness.md` first.
"""

SKILL = """---
name: watering
description: How to water.
---

Water it.
"""


def tree(**files):
    root = tempfile.mkdtemp()
    write(root, ".agents/agents/gardener.md", LANE)
    for rel, text in files.items():
        write(root, rel.replace("__", "/"), text)
    return root


def marker(proc):
    return (proc.stdout or "").splitlines()[0] if proc.stdout else ""


def main():
    # --- generation, and where each key lands -------------------------------------------
    root = tree()
    proc = run(root)
    check("build exits 0", proc.returncode == 0, proc.stdout + proc.stderr)
    check("Claude lane is generated", exists(root, ".claude/agents/gardener.md"))
    check("Codex lane is generated", exists(root, ".codex/agents/gardener.toml"))

    claude = read(root, ".claude/agents/gardener.md")
    codex = read(root, ".codex/agents/gardener.toml")

    check("tools.claude becomes tools: on Claude", "\ntools: Read, Grep\n" in claude, claude[:120])
    check("model.claude becomes model: on Claude", "\nmodel: opus\n" in claude, claude[:120])
    check("no dotted key survives into Claude output", ".claude:" not in claude, claude[:120])
    check("model.codex does NOT reach Claude", "gpt-5.6-sol" not in claude, claude[:120])
    check("nickname_candidates does NOT reach Claude", "nickname" not in claude, claude[:120])

    check("model.codex becomes model = on Codex", 'model = "gpt-5.6-sol"' in codex, codex[:200])
    check("nickname_candidates reaches Codex", "nickname_candidates = " in codex, codex[:200])
    check("tools does NOT reach Codex", "tools" not in codex, codex[:200])
    check("model.claude does NOT reach Codex", '"opus"' not in codex, codex[:200])
    check("body reaches Codex as developer_instructions",
          "developer_instructions" in codex and "Prune the thing." in codex, codex[:200])
    check("body reaches Claude", "Prune the thing." in claude, claude[:200])

    # --- round-trip ---------------------------------------------------------------------
    before = read(root, ".claude/agents/gardener.md"), read(root, ".codex/agents/gardener.toml")
    run(root)
    after = read(root, ".claude/agents/gardener.md"), read(root, ".codex/agents/gardener.toml")
    check("rebuilding an unchanged tree changes nothing", before == after)
    check("--check on a fresh tree says FRESH", marker(run(root, "--check")).startswith("AGENTS_FRESH"),
          marker(run(root, "--check")))
    check("--check on a fresh tree exits 0", run(root, "--check").returncode == 0)

    # --- --check catches a hand-edited generated file ------------------------------------
    write(root, ".claude/agents/gardener.md", claude + "\nHAND EDIT\n")
    proc = run(root, "--check")
    check("--check on a hand-edited artifact says STALE", marker(proc).startswith("AGENTS_STALE"), marker(proc))
    check("--check names the drifted file", ".claude/agents/gardener.md" in proc.stdout, proc.stdout[:200])
    check("--check exits 1 on drift", proc.returncode == 1, "rc=%s" % proc.returncode)
    check("--check wrote nothing", "HAND EDIT" in read(root, ".claude/agents/gardener.md"))

    # --- a missing generated file is drift, not a pass ------------------------------------
    root = tree()
    run(root)
    os.remove(os.path.join(root, ".codex/agents/gardener.toml"))
    proc = run(root, "--check")
    check("a deleted generated file is STALE", marker(proc).startswith("AGENTS_STALE"), marker(proc))

    # --- hard errors ----------------------------------------------------------------------
    root = tree()
    write(root, ".agents/agents/gardener.md", LANE.replace("tools.claude:", "tools:"))
    proc = run(root)
    check("an unknown frontmatter key is UNKNOWN", marker(proc).startswith("AGENTS_UNKNOWN"), marker(proc))
    check("an unknown frontmatter key exits 2", proc.returncode == 2, "rc=%s" % proc.returncode)
    check("the unknown key is named", "tools" in proc.stdout, proc.stdout[:200])

    root = tree()
    write(root, ".agents/agents/gardener.md", LANE.replace("name: gardener", "name: mower"))
    proc = run(root)
    check("frontmatter name must match the filename", proc.returncode == 2, proc.stdout[:200])

    root = tree()
    write(root, ".agents/agents/gardener.md", "no frontmatter here\n")
    proc = run(root)
    check("a missing frontmatter fence is UNKNOWN", marker(proc).startswith("AGENTS_UNKNOWN"), marker(proc))

    # --- skills ---------------------------------------------------------------------------
    root = tree(**{".agents__skills__watering__SKILL.md": SKILL})
    proc = run(root)
    check("skill is copied to Claude", exists(root, ".claude/skills/watering/SKILL.md"), proc.stdout)
    check("skill is copied to Codex", exists(root, ".codex/skills/watering/SKILL.md"), proc.stdout)
    check("skill body is copied verbatim",
          read(root, ".claude/skills/watering/SKILL.md") == SKILL)

    root = tree(**{".agents__skills__watering__SKILL.md": SKILL.replace("name: watering", "name: sprinkling")})
    proc = run(root)
    check("a skill whose dir and name disagree is refused", proc.returncode == 2, proc.stdout[:200])
    check("the refusal names both identities",
          "watering" in proc.stdout and "sprinkling" in proc.stdout, proc.stdout[:250])

    root = tree(**{".agents__skills__watering__SKILL.md": SKILL.replace(
        "description: How to water.", "description: How to water.\nallowed-tools: Read")})
    proc = run(root)
    check("a skill frontmatter key Codex rejects is refused", proc.returncode == 2, proc.stdout[:200])

    # A generated skill directory whose source is gone stays registered until someone removes it.
    root = tree(**{".agents__skills__watering__SKILL.md": SKILL})
    run(root)
    import shutil
    shutil.rmtree(os.path.join(root, ".agents/skills/watering"))
    proc = run(root, "--check")
    check("an orphaned generated skill dir is reported", "watering" in proc.stdout, proc.stdout[:250])
    check("an orphaned generated skill dir is not FRESH",
          not marker(proc).startswith("AGENTS_FRESH"), marker(proc))

    # --- --hook: only rebuild when a SOURCE was edited --------------------------------------
    root = tree()
    run(root)
    write(root, ".agents/agents/gardener.md", LANE.replace("Prune the thing.", "Prune it twice."))

    proc = subprocess.run(
        [sys.executable, SCRIPT, "--root", root, "--hook"],
        input=json.dumps({"tool_input": {"file_path": "docs/unrelated.md"}}),
        capture_output=True, text=True, encoding="utf-8",
    )
    check("--hook ignores an edit outside the sources",
          "Prune it twice." not in read(root, ".claude/agents/gardener.md"), proc.stdout)

    proc = subprocess.run(
        [sys.executable, SCRIPT, "--root", root, "--hook"],
        input=json.dumps({"tool_input": {"file_path": ".agents/agents/gardener.md"}}),
        capture_output=True, text=True, encoding="utf-8",
    )
    check("--hook rebuilds after a source edit (file_path)",
          "Prune it twice." in read(root, ".claude/agents/gardener.md"), proc.stdout)
    check("--hook exits 0", proc.returncode == 0, "rc=%s" % proc.returncode)

    # Codex delivers an edit as apply_patch with the paths inside `command` and no file_path at
    # all. Reading only file_path would make every Codex edit invisible to this hook.
    write(root, ".agents/agents/gardener.md", LANE.replace("Prune the thing.", "Prune it thrice."))
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--root", root, "--hook"],
        input=json.dumps({"tool_input": {"command":
            "apply_patch <<'EOF'\n*** Update File: .agents/agents/gardener.md\n@@\n-a\n+b\nEOF"}}),
        capture_output=True, text=True, encoding="utf-8",
    )
    check("--hook rebuilds after an apply_patch source edit",
          "Prune it thrice." in read(root, ".claude/agents/gardener.md"), proc.stdout)

    proc = subprocess.run(
        [sys.executable, SCRIPT, "--root", root, "--hook"],
        input="not json at all",
        capture_output=True, text=True, encoding="utf-8",
    )
    check("--hook on an unreadable payload exits 0", proc.returncode == 0, proc.stdout + proc.stderr)

    # --- an unusable root is UNKNOWN, never FRESH -------------------------------------------
    empty = tempfile.mkdtemp()
    proc = run(empty, "--check")
    check("a root with no sources is UNKNOWN", marker(proc).startswith("AGENTS_UNKNOWN"), marker(proc))
    check("a root with no sources exits 2", proc.returncode == 2, "rc=%s" % proc.returncode)

    print()
    if failures:
        print("%d FAILED: %s" % (len(failures), ", ".join(failures)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
