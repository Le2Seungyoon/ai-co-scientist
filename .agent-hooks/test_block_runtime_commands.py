#!/usr/bin/env python3
"""Tests for block_runtime_commands.py — run: uv run python .agent-hooks/test_block_runtime_commands.py

A hook is code, so it ships with a test (`enforcement.md` -> Hook contracts). The must-block
half proves the deny fires; the must-pass half is what keeps false positives from creeping in.
A test that only ever asserts "it blocked" cannot notice the day the hook blocks everything.

Each scenario needs its own tree because the verdict turns on whether
`<root>/runtime/registry.jsonl` exists. `CLAUDE_PROJECT_DIR` points the hook at that tree.

Stdlib only, no test runner: pytest does not collect this file (`testpaths = ["tests"]`).
"""
import importlib.util
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


def run_no_project_dir(command, env_extra=None):
    """Same as run(), but with CLAUDE_PROJECT_DIR unset -- exercises the __file__ fallback in
    project_root(). The real repo root may be the main checkout (registry present) or a linked
    worktree (registry absent), so the expected decision follows that sentinel."""
    env = dict(os.environ)
    env.pop("CLAUDE_PROJECT_DIR", None)
    if env_extra:
        env.update(env_extra)
    payload = json.dumps(
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


def load_hook_module():
    spec = importlib.util.spec_from_file_location("block_runtime_commands", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    print("must-block")
    worktree = make_tree(with_registry=False)
    for cmd in (
        "uv run python scripts/train_structure.py --arch mlp",
        "uv run python scripts/train_level.py",
        "uv run python scripts/exp.py new --title x",
        "uv run python scripts/infer_decomposed.py --submit runtime/submissions/a.zip",
        "uv run python scripts/dacon_submit.py runtime/submissions/a.zip",
        "uv run python scripts/train_dann.py --arm B --lambda-max 1.0",
    ):
        out, rc = run(worktree, cmd)
        check(f"denies: {cmd.split()[3]}", denied(out), out[:120] or "silent")
        check("deny still exits 0", rc == 0, f"rc={rc}")

    out, _ = run(worktree, "uv run python scripts/exp.py new --title x")
    check("deny names the rule file", "architecture.md" in out, out[:120])
    check("registry deny does NOT name the escape hatch -- there is none for it",
          "ACS_RUNTIME_EXEMPT" not in out, out[:120])
    check("registry deny names the bootstrap path (fresh clone / re-imaged / lost runtime/)",
          "bootstrap" in out and "mkdir -p runtime" in out, out[:300])

    out, _ = run(worktree, "uv run python scripts/train_level.py")
    check("exclusive deny names the rule file", "architecture.md" in out, out[:120])
    check("exclusive deny names the escape hatch", "ACS_RUNTIME_EXEMPT" in out, out[:120])

    out, _ = run(worktree, "uv run python scripts/train_level.py",
                 env_extra={"ACS_RUNTIME_EXEMPT": "   "})
    check("empty exemption reason does NOT pass", denied(out), out[:120] or "silent")

    print("must-pass")
    main_tree = make_tree(with_registry=True)
    out, rc = run(main_tree, "uv run python scripts/train_structure.py --arch mlp")
    check("main worktree: experiment command passes", not denied(out), out[:120])
    check("main worktree: exits 0", rc == 0, f"rc={rc}")

    out, rc = run(main_tree, "uv run python scripts/exp.py new --title x")
    check("main worktree: registry-tier command (exp.py) passes -- sentinel covers both tiers",
          not denied(out), out[:120])
    check("main worktree: exp.py exits 0", rc == 0, f"rc={rc}")

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

    print("ruling 4 -- invocation must start a command segment (fix round 1)")
    # Finding 1: `uv run <script>` with no `python` token is a real invocation and must deny.
    out, _ = run(worktree, "uv run scripts/exp.py new")
    check("denies: uv run with no python token", denied(out), out[:120] or "silent")
    out, _ = run(worktree, "uv run ./scripts/exp.py new")
    check("denies: uv run ./relative-path with no python token", denied(out), out[:120] or "silent")
    # Finding 2: a MENTION of the invocation (inside another command's arguments) must pass.
    out, _ = run(worktree, 'echo "run python scripts/exp.py later"')
    check("passes: echo merely quoting the invocation", not denied(out), out[:120])
    # Finding 5b: a ";"-separated segment still gets caught when the token opens ITS segment.
    out, _ = run(worktree, "cd /tmp; uv run python scripts/train_level.py")
    check("denies: guarded invocation in the second ;-separated segment", denied(out),
          out[:120] or "silent")
    # Known-gap class (declared, not fixed): token and guarded path in different segments.
    out, _ = run(worktree, "python -V; sh scripts/train_level.py")
    check("passes: token and guarded path split across segments (declared known gap)",
          not denied(out), out[:120])

    print("finding 5a -- project_root() __file__ fallback")
    out, rc = run_no_project_dir("uv run python scripts/train_structure.py --arch mlp")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(HOOK)))
    fallback_has_registry = os.path.isfile(
        os.path.join(repo_root, "runtime", "registry.jsonl")
    )
    check("no CLAUDE_PROJECT_DIR: decision follows fallback root registry sentinel",
          denied(out) is not fallback_has_registry, out[:120])
    check("no CLAUDE_PROJECT_DIR: exits 0", rc == 0, f"rc={rc}")

    print("finding 3 -- GUARDED stays tied to reality")
    hook_mod = load_hook_module()
    guarded = hook_mod.GUARDED
    check("GUARDED is non-empty", bool(guarded), "GUARDED is empty")
    for script in guarded:
        path = os.path.join(repo_root, *script.split("/"))
        check(f"GUARDED entry exists on disk: {script}", os.path.isfile(path), path)

    print("grade split -- what unlocks in a registry-less tree, and what stays guarded")
    with tempfile.TemporaryDirectory() as root:  # no registry.jsonl = a worktree
        out, _ = run(root, "uv run python scripts/probe_level.py --cache-dir ../../x/runtime/cache")
        if out.strip():
            failures.append("probe_level is still denied -- it is read-only and should be released")

        out, _ = run(root, "uv run python scripts/assemble_submission.py --submit a.zip")
        if out.strip():
            failures.append("assemble_submission is denied -- it is not a GUARDED target")

        out, _ = run(root, "uv run python scripts/exp.py new --title x")
        if not out.strip():
            failures.append("exp.py passed -- it must fork the registry and be denied")

        out, _ = run(root, "uv run python scripts/exp.py new --title x",
                     env_extra={"ACS_RUNTIME_EXEMPT": "because I said so"})
        if not out.strip():
            failures.append(
                "exp.py was let through by the escape hatch -- that would create the sentinel "
                "and permanently unlock the gate in this tree"
            )

        for script in ("train_level.py", "train_structure.py", "infer_decomposed.py",
                       "dacon_submit.py", "train_dann.py"):
            out, _ = run(root, "uv run python scripts/{0} --submit a.zip".format(script),
                         env_extra={"ACS_RUNTIME_EXEMPT": "measured one-off"})
            if out.strip():
                failures.append(
                    "{0} was still denied with the escape hatch set -- narrowing spread too "
                    "far".format(script)
                )

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
