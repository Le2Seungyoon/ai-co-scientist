#!/usr/bin/env python3
"""PreToolUse deny: experiment-execution commands only run where the registry lives.

See `.agents/rules/enforcement.md` -> Hook contracts and
`.agents/rules/architecture.md` -> Parallel execution contract.

  Event     PreToolUse (Bash) only. It DENIES; the advisory counterpart is check_rules_size.py.
  Governed  The commands in GUARDED (REGISTRY_WRITERS + EXCLUSIVE + MAIN_ONLY), which all read or write
            `runtime/` -- graded by why they are guarded, not by one blanket reason.
            REGISTRY_WRITERS (`scripts/exp.py`) would fork the experiment registry.
            EXCLUSIVE (`train_level.py`, `train_structure.py`, `infer_decomposed.py`,
            `train_dann.py`, `build_pseudo_labels.py`, `train_self_training.py`,
            `train_cyclegan.py`, `translate_sim.py`, `train_two_head.py`, `infer_two_head.py`,
            `dump_level_proba.py`) need a resource the main worktree owns:
            the single 8 GB GPU (serialized across trees by the gpu-0 resource lock).
            MAIN_ONLY (`dacon_submit.py`) spends the DACON quota -- a user decision, so it runs
            only in the main checkout, and neither a link nor the escape hatch unlocks it.
            `scripts/probe_level.py` (read-only, no
            `runtime/` writes) and `scripts/assemble_submission.py` (CPU-only, writes only its
            own tree) are deliberately NOT guarded. `scripts/legacy/*.py` also touch `runtime/`
            and were considered -- excluded because they are frozen, reproduction-only, and
            never run (`scripts/legacy/README.md`), not because they were overlooked.
  Verdict   Turns on two facts about `<project root>`: whether `runtime/registry.jsonl` exists
            (the sentinel), and whether `.git` is a DIRECTORY (the main checkout) or a FILE (a
            git worktree). `runtime/` is gitignored, so a worktree has none -- running
            `exp.py new` there would issue report_id 1 again and fork the registry silently.
            A worktree holding its own registry is denied outright, whatever the command: it
            shadows the link and forks ids. MAIN_ONLY additionally requires `.git` to be a
            directory.
            A lane whose plan the orchestrator approved holds `runtime/registry.link` (one
            line: the main registry's absolute path, written by `scripts/exp.py link` from the
            main checkout). A valid link unlocks EXCLUSIVE and the lane's own `exp.py`
            subcommands (`result`, `show`, `list`, `verdict`) -- registry writes follow the link
            and `registry.locked()` locks the resolved registry path, so every linked lane
            contends on main's lock and report_ids never fork. `new`, `lb`, `render` and `link`
            stay the orchestrator's. A dangling link is denied with its own reason.
  Failure   Unreadable or unparseable payload -> exit 0, note on stderr. A hook must never
            block an edit for a reason unrelated to what it checks.
  Escape    `ACS_RUNTIME_EXEMPT="<reason>"` in the environment, but ONLY for EXCLUSIVE. A blank
            reason does not pass: the point is to turn a silent bypass into a decision a
            reviewer can see. Setting it is the user's or the orchestrator's decision; a lane
            never sets it. REGISTRY_WRITERS has NO escape hatch: writing here would itself
            create the sentinel and thereby unlock every other gate in this tree for good.
            MAIN_ONLY, a shadowing local registry and a broken link have none either.
  Tests     `test_block_runtime_commands.py`, beside this file.

KNOWN GAP (state it rather than let it pass quietly)
    This hook cannot see WHICH sub-agent issued the command -- the PreToolUse payload does not
    carry the sub-agent identity. So it does not stop an agent from training inside the MAIN
    checkout; it only makes the boundary real in a lane's worktree. The consequence is a
    requirement, not a caveat: **every lane runs in its own worktree** and the orchestrator
    runs no sub-agent in main, or the contract is prose only. And like every hook it sees
    this session's tool calls -- an IDE terminal
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

# What a valid link unlocks of REGISTRY_WRITERS: the lane records its own results. Registering a
# report (`new`), recording the leaderboard (`lb`), rendering (`render`) and issuing links
# (`link`) are the orchestrator's, run from the main checkout.
LANE_EXP_SUBCOMMANDS = ("result", "show", "list", "verdict")

# Exclusive resources: the single 8 GB GPU.
EXCLUSIVE = (
    "scripts/train_level.py",
    "scripts/train_structure.py",
    "scripts/infer_decomposed.py",
    "scripts/dump_level_proba.py",
    "scripts/train_dann.py",
    "scripts/build_pseudo_labels.py",
    "scripts/train_self_training.py",
    "scripts/train_cyclegan.py",
    "scripts/translate_sim.py",
    "scripts/train_two_head.py",
    "scripts/infer_two_head.py",
)

# The DACON submission quota is finite and spending it is the USER's decision, relayed by the
# orchestrator -- not a resource a lane may take by holding a lock. So it runs only where the real
# registry lives (the main checkout), a registry link does not unlock it, and there is no escape
# hatch: a lane that could submit would be taking a user decision for itself.
MAIN_ONLY = ("scripts/dacon_submit.py",)

# `scripts/probe_level.py` is deliberately absent: it reads with mmap_mode="r" and writes
# nothing under runtime/, so it has no registry-fork path. With no cache in the tree `np.load`
# fails loudly -- there is no silent-corruption route to guard against.
# `scripts/assemble_submission.py` is absent for the same reason: it writes only its own tree.
GUARDED = REGISTRY_WRITERS + EXCLUSIVE + MAIN_ONLY

EXEMPT_VAR = "ACS_RUNTIME_EXEMPT"
SENTINEL = os.path.join("runtime", "registry.jsonl")
GIT = ".git"
LINK = os.path.join("runtime", "registry.link")

REASON_REGISTRY = (
    "This tree has no {sentinel} and no runtime/registry.link. "
    "If this is a git worktree (a lane): `runtime/` is gitignored so it was never copied here, "
    "and running `{hit}` here would fork the registry -- report_id comes from len(records), so "
    "a second tree starts over at 1 and the two truths can never be merged. The registry lives "
    "in the main checkout; a lane reaches it only through the orchestrator's approval link. "
    "{bootstrap}"
    "There is NO escape hatch through this gate: letting `{hit}` itself create {sentinel} "
    "would unlock every other gate in this tree for good. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
)

# The bootstrap case of REASON_REGISTRY -- shown only where `.git` is a directory. In a worktree
# (or a tree whose kind is unknown) the only path forward is approval; advising a lane to create
# the sentinel would hand it a registry of its own.
REASON_BOOTSTRAP = (
    "If instead this IS the main checkout (`.git` is a directory here) -- a fresh clone, a "
    "re-imaged machine, or a `runtime/` lost to cleanup -- and there is no other tree to defer "
    "to. That is a bootstrap, not a fork: create the sentinel yourself, outside this gate, with "
    "an empty file (`mkdir -p runtime && touch {sentinel}`, no `{hit}` involved); the next "
    "`{hit}` run then sees zero existing records and starts at report_id 1, same as any other "
    "first run. "
)

REASON_EXCLUSIVE = (
    "This tree has no {sentinel} and no runtime/registry.link -- a lane whose plan is not "
    "approved yet. `{hit}` needs the single 8 GB GPU, which a lane uses only after approval: "
    "the orchestrator merges the plan, registers it from the main checkout and links this "
    "worktree (`scripts/exp.py link`). Ask the orchestrator for that approval. "
    "Escape hatch: {var}=\"<reason>\" exists, but setting it is the user's or the "
    "orchestrator's decision -- a lane never sets it. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
)

REASON_SHADOW = (
    "This tree's `.git` is a file -- it is a git worktree -- yet it holds its own {sentinel}. "
    "A worktree must not hold its own registry: it shadows runtime/registry.link (the registry "
    "code reads the local file first) and forks report_ids, which can never be merged back. "
    "`{hit}` is denied here. Delete {sentinel} from this worktree and ask the orchestrator for "
    "a link (`scripts/exp.py link <this worktree>`, run from the main checkout). "
    "There is no escape hatch. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
)

REASON_ORCHESTRATOR_EXP = (
    "`{hit} {sub}` is an orchestrator command. A linked lane may run only `exp.py "
    "result|show|list|verdict`; `new` (registration at approval), `lb` (leaderboard after "
    "submission), `render` and `link` are run by the orchestrator from the main checkout. Report "
    "what you need to the orchestrator instead. There is no escape hatch. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
)

REASON_MAIN_ONLY = (
    "`{hit}` spends the finite DACON submission quota, and submitting is the user's decision. "
    "It runs only in the MAIN checkout, by the orchestrator, after the user approves. A lane "
    "builds and verifies the zip (`verify_submission()`), reports it, and stops there. There is "
    "no escape hatch: a lane that could submit would be taking a user decision for itself. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
)

REASON_BROKEN_LINK = (
    "This tree's {link} points at `{target}`, which is not a file. The link is how an approved "
    "lane reaches the main registry, so `{hit}` would run against nothing -- or start a fresh "
    "registry at report_id 1. Ask the orchestrator to re-issue it from the main checkout: "
    "`uv run python scripts/exp.py link <this worktree>`. "
    "There is no escape hatch: a dangling permit is not a permit. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
)

LINK_HINT = (
    " If this is a lane whose plan the orchestrator approved, the approval is incomplete: the "
    "orchestrator writes runtime/registry.link into this worktree (`scripts/exp.py link`, run "
    "from the main checkout), which unlocks the lane's commands here. Do not write it yourself."
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

    `kind` is "registry", "exclusive" or "main_only" -- the deny reason and whether the escape hatch
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
            if script in REGISTRY_WRITERS:
                return script, "registry"
            if script in MAIN_ONLY:
                return script, "main_only"
            return script, "exclusive"
    return None, None


def git_kind(root):
    """"dir" (main checkout), "file" (git worktree: `.git` holds `gitdir: ...`) or "absent"."""
    path = os.path.join(root, GIT)
    if os.path.isdir(path):
        return "dir"
    if os.path.isfile(path):
        return "file"
    return "absent"


def exp_subcommands(command):
    """The subcommand token after each `scripts/exp.py` invocation in `command`.

    Same segment anchoring as `guarded_hit`, one entry per segment-anchored invocation. The
    token read is the one after the FIRST `scripts/exp.py` following the invocation token (lazy
    quantifier) -- the script actually run, not a later mention of the path inside an argument
    such as `--title "... scripts/exp.py result"`. A token that cannot be determined (nothing
    follows, an option or a quote comes first, the path runs on into other characters) is
    returned as None, and the caller denies it: fail closed."""
    escaped = re.escape(REGISTRY_WRITERS[0]).replace("/", r"[/\\]")
    pattern = (r"(?:^|[;&|\n])\s*(?:uv\s+run|python[\w.]*)\b[^;&|\n]*?" + escaped
               + r"(?:\s+([A-Za-z_]\w*)(?=$|[\s;&|]))?")
    return [m.group(1) for m in re.finditer(pattern, command)]


def linked_registry(root):
    """(state, target) of `<root>/runtime/registry.link`: absent, valid, or broken.

    Broken is its own state, not "absent": a lane whose link dangles was approved once, and
    falling back to "no registry here" would print the wrong reason for the denial."""
    path = os.path.join(root, LINK)
    if not os.path.exists(path):
        return "absent", None
    try:
        with open(path, encoding="utf-8") as f:
            target = f.read().strip()
    except (OSError, ValueError):  # UnicodeDecodeError is a ValueError: undecodable = broken
        return "broken", "<unreadable>"
    # Same rule as registry._default_path: an absolute path to a file named registry.jsonl.
    # A relative target would resolve against the hook's cwd, not this tree.
    if (target and os.path.isabs(target) and os.path.basename(target) == "registry.jsonl"
            and os.path.isfile(target)):
        return "valid", target
    return "broken", target or "<empty>"


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

    root = project_root()
    git = git_kind(root)
    sentinel = SENTINEL.replace("\\", "/")
    local = os.path.exists(os.path.join(root, SENTINEL))

    # A worktree's own registry is never trusted: it would shadow the link and fork ids.
    if local and git == "file":
        deny(REASON_SHADOW.format(sentinel=sentinel, hit=hit))
        return

    if kind == "main_only":
        if local and git == "dir":
            return
        deny(REASON_MAIN_ONLY.format(hit=hit))
        return

    if local:
        return

    state, target = linked_registry(root)
    if state == "valid":
        if kind == "registry":
            for sub in exp_subcommands(command):
                if sub not in LANE_EXP_SUBCOMMANDS:
                    deny(REASON_ORCHESTRATOR_EXP.format(hit=hit, sub=sub or "<undetermined>"))
                    return
        return
    if state == "broken":
        deny(REASON_BROKEN_LINK.format(link=LINK.replace("\\", "/"), target=target, hit=hit))
        return

    if kind == "registry":
        bootstrap = REASON_BOOTSTRAP.format(sentinel=sentinel, hit=hit) if git == "dir" else ""
        deny(REASON_REGISTRY.format(sentinel=sentinel, hit=hit, bootstrap=bootstrap) + LINK_HINT)
        return

    if os.environ.get(EXEMPT_VAR, "").strip():
        return

    deny(REASON_EXCLUSIVE.format(sentinel=sentinel, hit=hit, var=EXEMPT_VAR) + LINK_HINT)


if __name__ == "__main__":
    main()
