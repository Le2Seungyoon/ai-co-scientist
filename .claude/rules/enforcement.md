# Enforcement

Where a rule lives is a design decision, not a formatting one. This file is the routing
function — prose vs hook vs test vs `permissions.deny` — plus the contracts every hook in
this project honors. Route a rule **before** writing it: a rule in the wrong layer is not a
weaker rule, it is a rule that never runs.

## Four layers

| Layer | Use when | How it fails |
|-------|----------|--------------|
| **Prose** (`.claude/rules/*.md`) | context-dependent advice; the *why* behind a ban | quietly ignored under deadline |
| **Hook** (`.agent-hooks/*.py`) | deterministic check, fast feedback, this session's edits | one false positive and it gets switched off |
| **Test / CI invariant** | must hold no matter who authored the code | slow feedback; needs a real assertion |
| **`permissions.deny`** | absolute in every context (person · solo/team · time) | blocks legitimate work with no escape |

## Hooks only see this session's edits

A hook fires on this session's tool calls. Code authored in an IDE, by another agent, or by a
teammate bypasses every hook here. **A rule that must hold on every authoring path needs a test
layer** — the hook is fast feedback, the test is the guarantee.

A hook can widen its own reach: **scan the governed set instead of reading the payload's file
path.** Keying on the path means edits made through the shell (`sed`, a heredoc, a one-liner)
are invisible, and no matcher list fixes that. Scanning closes the shell hole and costs nothing
on a small set — but it still does not see the IDE, so it does not replace the test layer.

## Hook contracts

Breaking one of these is how a hook stops being trusted, and an untrusted hook gets removed.

- **PreToolUse may deny; PostToolUse never does.** One script may serve both events (branch on
  `hook_event_name`), but the PostToolUse path only advises.
- **Environment failure exits 0.** Tool missing, timeout, unreadable payload → pass. A hook must
  never block an edit for a reason unrelated to what it checks.
- **Silence means exactly one thing: "checked, and it passes."** If the check could not run,
  emit one line saying so. Otherwise "all clear" and "never looked" are indistinguishable.
- **Timeouts nest**: any subprocess timeout stays below the hook's own `timeout` in
  `settings.json`, so the hook controls its own outcome instead of being killed mid-write.
- **A hook is code — it ships with `test_<name>.py` beside it.** Cover the must-block and the
  must-pass halves; the must-pass half is what keeps false positives from creeping in.
- **A deny message says what to do instead**: why it fired, the canonical alternative, the
  escape hatch, and the rule file it enforces.
- **Account for every item in scope.** Compare it, or list it as unresolved with a stated
  reason. An item that is neither fails. Silently skipping what it cannot parse is how a
  checker ships inert while reading as protection.
- **An empty target set is a failure, not a pass.** A glob that stops matching — a moved
  directory, a pinned path depth — takes a check from N files to zero with no error and no
  output. Assert the set is non-empty, and say how many things were examined.

## Before promoting a check to deny

- **Measure the false-positive rate first.** Run the candidate rule over the existing tree and
  classify every hit. A rule that blocks legitimate work trains everyone to route around it.
- **Set the gate one occurrence later than the rule asks for.** If the convention says "extract
  on the second copy", gate the third. The gap is deliberate: it is where judgment lives.
- **Every deny needs an escape hatch that costs one sentence** — a marker comment carrying a
  *reason* (`<gate-name>-exempt: <reason>`); an empty or near-empty reason must not pass. The
  goal is not prohibition, it is surfacing: turn a silent copy into an explicit decision a
  reviewer can see.
- **When a hook takes over a rule, delete the prose it replaced.** Keep only what the hook
  cannot check — it blocks the import, but it cannot say why the ban exists.

## Answers, not lookup procedures

A rule that says *"check X first"* makes the reader perform a lookup, and every point where
that lookup fails is an incident. Where the answer is derivable from the tree, **generate it**:

- The generated file carries a `do not edit by hand — regenerate with <cmd>` header and is
  committed, so it is readable without running anything.
- Generate from the **default branch**, not the working tree. A long-lived branch otherwise
  reports "this does not exist yet" about something that was shared while it was away.
- A **freshness test** fails when the committed artifact is stale. Hooks cannot cover this:
  the artifact goes stale through edits they never see.
- A hand-maintained list of the same information rots. Do not keep one beside the generated one.
- **Scope it with `paths:` front-matter.** A generated file grows with the codebase and is the
  first rules file to cross the budget; path-gating keeps it out of unrelated sessions.
- **Its headings are an interface, not prose.** The generator re-parses its own output, and gates
  key on section titles. Rename one and the reader silently finds zero entries — move the anchor
  in the same change.

## Keeping a generated artifact alive

Generating the answer moves the problem, it does not end it: **triggering the regeneration is
still a procedure someone has to remember** — the same defect one level up.

- **A freshness test is the only lever that covers every authoring path.** A PostToolUse
  staleness warning is a complement, never a substitute (it sees this session's edits only);
  re-checking where a merge hook already fires is nearly free.
- **Never regenerate as a side effect of an unrelated edit.** Cost is not the objection —
  measured at 0.13s and idempotent. The objection is that a tracked file modified by a write
  that had nothing to do with it leaves a diff whose origin nobody can trace.
- **Regenerate in the same change that invalidates it.** Otherwise the default branch goes stale
  the moment the PR merges, with the PR that caused it already closed.
- **A gate whose input is a generated file must distinguish three states**: nothing registered
  yet · stale · unreadable. Failing open on all three is right for *blocking* and wrong for
  *reporting* — "no violations" and "never looked" collapse into one silence, and the defect
  stays dormant until the first real entry arrives.
- **Prove staleness is caught by causing it**: delete an entry, delete a section, add a real
  occurrence without regenerating. Then name the authoring paths that still bypass the check.

## An empty result is not proof

A checker that reports nothing has three ways to be lying: the count fell below its threshold
when all but one instance was fixed; a promotion masked an unrelated repetition sharing the
same signature; near-matches were never grouped at all. **Judge completion by searching the
source, not by the checker being quiet** — and remember a detector only sees copies *before*
they drift apart.

There is a fourth, and it is the quietest: **the thing was never in scope.** One repo's
duplication ledger drew its canon from two directories and scanned two others, so a third
directory sat outside *both* sides — ten shadowed names accumulated there while the ledger
stayed green. Green means "not in scope", never "not there".

## Where harness code lives

`.agent-hooks/` — one directory, one copy of every hook and scanner, because more than one
harness runs them and a per-harness home would fork the logic. What kind of script it is stays
visible in **who calls it**, never in a directory name:

- **Event-driven** — registered in `.claude/settings.json` and `.codex/config.toml`, whose
  schemas are unrelated. Registered on one and not the other = that harness is unprotected.
- **Generators and repo-wide scanners** — not event-driven: a person runs them, and so does the
  freshness/invariant test that covers the paths hooks cannot see.

Both ship with `test_<name>.py` beside them. The dividing line is **not** "is it agent-only" —
both get referenced from rules files and both may be run by hand. It is **would this exist if
the harness didn't?** A load-test monitor would; a ledger generator would not. A directory name
says who *owns* a script, never who may run it.

## This project's gates

| Gate | Layer | Checks | Escape hatch |
|---|---|---|---|
| PR gate | PreToolUse hook (`settings.json`) | `origin/main` is merged into the branch before `git push` | none — deny |
| Commit-attribution gate | PreToolUse hook | no `Co-Authored-By` / `Generated with` trailer in `git commit` | none — deny |
| Runtime-command gate | PreToolUse hook (`block_runtime_commands.py`) | the six experiment scripts run only where `runtime/registry.jsonl` exists | `ACS_RUNTIME_EXEMPT="<reason>"` |
| File-size budget | PostToolUse hook (`check_rules_size.py`) | instruction files stay under the ~150-line soft budget | advisory, never blocks |
| Rule-link scan | script (`check_rule_links.py`) | every path a rules file or agent definition names still exists | advisory; run it by hand until its false-positive rate on this tree is measured |
| Registry write lock | code (`registry.locked()`) | concurrent pre-report writes cannot drop an entry | none — the lock wraps read+write |
| Manifest contract | test (`tests/test_train_manifest.py`) | training scripts keep declaring the (X, y) domains | none |
| Dependency management | `permissions.deny` | `uv pip install` and hand-edits to `uv.lock` | none — absolute |
