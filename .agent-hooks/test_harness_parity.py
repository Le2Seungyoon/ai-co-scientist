#!/usr/bin/env python3
"""Fail when the two harnesses' registrations drift apart.

Run: uv run python .agent-hooks/test_harness_parity.py   (needs 3.11+ for tomllib -- the project
venv is 3.12; the bare `python` on this machine is 3.8, so do not wire this into a hook.)

`.claude/settings.json` and `.codex/config.toml` cannot be one file: the schemas are unrelated
and neither derives from the other. So a hook added to one and forgotten in the other leaves that
harness silently unprotected, and nothing about the repo looks wrong. `.agents/rules/harness.md`
has asked for both registrations since it was written; this turns the asking into a check.

What it catches:

  1. A shared script registered on one harness only, per event.
  2. A registered script that does not exist.
  3. A generated Codex role file that no `[agents.*]` entry registers -- it exists, it is fresh,
     and it is inert.
  4. A `[agents.*]` entry pointing at a role file that was never generated.
  5. A skill live on one harness only, or whose directory name and `name:` disagree.

What it deliberately does NOT do: re-check that rule pointers resolve. `check_rule_links.py`
already makes that judgment, and running a second implementation of one judgment proves nothing
while doubling what can rot (`.agents/rules/self-review.md` -> Gates). `tests/` runs both.

UNMEASURED: whether Codex reads any of this. Trust state, hook execution and role registration
are ledger items 1-3 in `docs/codex-verification-ledger.md`. This file proves the two
registrations AGREE; it cannot prove either one is honoured.
"""
import json
import os
import re
import sys
import tomllib

ROOT = os.environ.get(
    "CLAUDE_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
CODEX_HOME = os.environ.get("CODEX_HOME", os.path.expanduser("~/.codex"))
SHARED = re.compile(r"[\w./$-]*\.agent-hooks/([\w.-]+)")


def claude_hooks():
    with open(os.path.join(ROOT, ".claude", "settings.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    out = {}
    for event, entries in (cfg.get("hooks") or {}).items():
        for entry in entries:
            for hook in entry.get("hooks", []):
                for name in SHARED.findall(hook.get("command", "")):
                    out.setdefault(event, set()).add(name)
    return out


def codex_hooks():
    path = os.path.join(ROOT, ".codex", "config.toml")
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    out, total = {}, 0
    for event, entries in (cfg.get("hooks") or {}).items():
        for entry in entries:
            for hook in entry.get("hooks", []):
                total += 1
                for name in SHARED.findall(hook.get("command", "")):
                    out.setdefault(event, set()).add(name)
    return out, total


def trusted_count():
    """How many of this project's hook entries Codex has a trust record for.

    A project with hook entries and NO trust record has never been trusted -- a real, actionable
    state, and the one that makes every hook here inert. Anything stronger (reading `trusted_hash`
    as proof that hooks still run) has been refuted elsewhere and is not attempted.
    """
    path = os.path.join(CODEX_HOME, "config.toml")
    if not os.path.exists(path):
        return None
    marker = os.path.join(ROOT, ".codex", "config.toml")
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.startswith("[hooks.state.") and marker in line)


def main():
    problems = []

    claude = claude_hooks()
    codex, codex_total = codex_hooks()

    if not claude or not codex:
        print("FAIL  one side registered NO shared hooks at all -- that is a broken parse, "
              "not parity")
        print("  claude: %r\n  codex: %r" % (claude, codex))
        return 1

    for event in sorted(set(claude) | set(codex)):
        a, b = claude.get(event, set()), codex.get(event, set())
        for name in sorted(a - b):
            problems.append("%s: `%s` is registered on Claude Code but not on Codex" % (event, name))
        for name in sorted(b - a):
            problems.append("%s: `%s` is registered on Codex but not on Claude Code" % (event, name))

    for event in sorted(set(claude) | set(codex)):
        for name in sorted(claude.get(event, set()) | codex.get(event, set())):
            if not os.path.exists(os.path.join(ROOT, ".agent-hooks", name)):
                problems.append("%s: `%s` is registered but does not exist" % (event, name))

    # Generated role files that nothing registers, and registrations pointing at nothing.
    roles_dir = os.path.join(ROOT, ".codex", "agents")
    cfg_path = os.path.join(ROOT, ".codex", "config.toml")
    with open(cfg_path, "rb") as fh:
        agents = tomllib.load(fh).get("agents") or {}
    registered = {}
    for key, entry in agents.items():
        if isinstance(entry, dict):
            registered[os.path.basename(str(entry.get("config_file", "")))] = key
    if not registered:
        problems.append(".codex/config.toml registers no [agents.*] entry at all")
    on_disk = sorted(n for n in os.listdir(roles_dir) if n.endswith(".toml")) \
        if os.path.isdir(roles_dir) else []
    for name in on_disk:
        if name not in registered:
            problems.append(
                "`.codex/agents/%s` is generated but no [agents.*] entry registers it -- the "
                "role is inert. Register it or stop generating it." % name)
    for name, key in sorted(registered.items()):
        if name not in on_disk:
            problems.append(
                "[agents.%s] points at `agents/%s`, which was not generated. Add the source "
                "under .agents/agents/ or drop the entry." % (key, name))

    # Skills ship by copy, and the two harnesses read a skill's identity from different places --
    # Claude Code from the DIRECTORY name, Codex from the frontmatter `name:`. Symmetry is this
    # file's question; whether the copies are CURRENT is `build-agents.py --check`'s, and the two
    # are not the same: a skill can be byte-identical in one harness and absent from the other.
    skills_src = os.path.join(ROOT, ".agents", "skills")
    skill_names = []
    if os.path.isdir(skills_src):
        for name in sorted(os.listdir(skills_src)):
            entry = os.path.join(skills_src, name, "SKILL.md")
            if not os.path.isfile(entry):
                continue
            skill_names.append(name)
            for target in (".claude/skills", ".codex/skills"):
                if not os.path.isfile(os.path.join(ROOT, target, name, "SKILL.md")):
                    problems.append(
                        "`%s/%s/SKILL.md` is missing -- the skill is live on one harness only. "
                        "Run `uv run python .agent-hooks/build-agents.py`." % (target, name))
            with open(entry, encoding="utf-8") as fh:
                declared = next(
                    (ln.split(":", 1)[1].strip()
                     for ln in fh.read().split("\n---", 1)[0].splitlines()
                     if ln.startswith("name:")),
                    None,
                )
            if declared != name:
                problems.append(
                    "`.agents/skills/%s/SKILL.md` declares `name: %s` but sits in `%s/`. Claude "
                    "Code would register it as `%s` and Codex as `%s` -- one skill, two names, "
                    "no error from either harness." % (name, declared, name, name, declared))

    trusted = trusted_count()
    if trusted is None:
        print("NOTE  no %s/config.toml -- Codex trust NOT checked (ledger item 1)" % CODEX_HOME)
    elif trusted == 0 and codex_total:
        print("NOTE  Codex has no trust record for this project: %d hook entries in "
              ".codex/config.toml, 0 trusted in %s/config.toml. Those hooks are NOT running. "
              "Run the Codex CLI once from this directory and accept 'Trust all and continue' "
              "-- the IDE never shows that prompt. (ledger item 1)" % (codex_total, CODEX_HOME))

    if problems:
        print("FAIL  harness registrations have drifted\n")
        for p in problems:
            print("  - %s" % p)
        return 1

    shared = sorted({n for names in claude.values() for n in names})
    print("PASS  both harnesses register the same %d shared hook(s), all present; %d Codex "
          "role(s) generated and registered; %d skill(s) live on both with matching "
          "directory/name" % (len(shared), len(on_disk), len(skill_names)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
