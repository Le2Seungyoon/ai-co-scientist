# Workflow

## Planning

- Use plan mode for work that changes module/layer boundaries or affects 3+ files.
- If you hit an unexpected blocker mid-implementation, stop and redesign — don't force it through.
- Before a non-trivial change, ask once "is there a more elegant way?". If it's hacky, do it properly.
- In pre-finalized stages (config values still undecided), only make values easy to change via
  config — don't hard-couple logic to a specific value.

## Experiment Pre-Report (MANDATORY before any ML experiment)

Before running ANY training / validation / submission experiment, **write a pre-report and get it
confirmed first**. No experiment runs without these five stated explicitly:

1. **X** — the exact input. State the domain: **sim** SEM or **real** SEM, which subset (case? fold?).
2. **y** — the exact target/label. **sim depth GT? real `average_depth`? model-generated pseudo-label?**
3. **Model + hyperparameters** — arch, lr, batch, epochs, data amount.
4. **Methodology** — the training scheme: plain supervised / pseudo-labeling / additional training on
   derived data / conditioning on extra inputs / etc.
5. **Purpose** — what question it answers, and **how it's judged** — name the metric AND verify the
   metric's own (X, y) matches the target you care about (usually real→real / the leaderboard).

**Why (the failure this prevents)**: a whole line of "case-holdout validation" work silently measured
**sim SEM → sim depth** (X and y both sim, `real_proxy_rmse: null` — `average_depth` never used)
while it was believed to be building a *real* validation. Nobody stated X and y, so the domain
mismatch (validation ≠ leaderboard's real→real) went unnoticed for many runs. Stating X and y up
front catches this instantly. A sim-domain metric cannot validate real-domain performance.

## Superpowers (High-Impact Tasks)

For a **new feature** or a **3+ file patch**, before writing code use `AskUserQuestion` to ask
whether to apply a superpowers workflow: `brainstorming` (lock intent/design) /
`test-driven-development` (failing test first) / `verification-before-completion` (gather evidence
before done). If selected, actually invoke it with the `Skill` tool. Skip for trivial edits /
1–2 lines / doc-only changes.

## Bug Fixing

- Investigate → fix → verify. Reproduce first; point at logs/errors directly rather than
  guessing from symptoms.
- **Before "fixing" a missing decorator/guard/line, read the file around it — not a grep window.**
  `grep -A` hides *preceding* lines, so an already-present decorator looks absent. This cost us a
  duplicated `@torch.no_grad()` written up as a bug fix in the EXP-012 registry entry, then
  retracted. **A claimed fix that changes no behavior is a signal you misread, not a win.**
- If a fix feels hacky, implement the proper solution instead. Skip this for simple, obvious
  one-liners.

## Task Completion

- Don't mark done without proving behavior.
- Bugs: reproduce → fix → confirm it's gone.
- After a structural change, compare behavior against the previous state (tests + a real run).
- Type checks and test suites verify code correctness, not feature correctness. When the change
  has a runtime surface (a UI, a CLI, an endpoint, a running service), exercise it directly — start
  it and drive the actual path — before declaring done, not just its automated tests.
- **A quiet checker is not evidence.** A hook or linter can be silent because it passed, because it
  never ran, or because the last instance fell below its threshold. Confirm by searching the source
  (`enforcement.md` → An empty result is not proof).

## Self-review

At the end of every task, before declaring done: work `self-review.md` — ① this project's gates,
and ② the judgments no gate can make (escape hatches, allowlist lines, re-invented equivalents,
both halves of a mirror). Write the answers where the task is reported.

## Capturing Learnings

At the end of every task, before declaring done: **did anything reusable/recurring emerge this
session?** If so, don't leave it in chat — capture it.

Route first — the layer decides whether the rule ever runs (`enforcement.md` → Four layers):
- Anyone touching this repo (convention · contract · gotcha) → a committed `.claude/rules/` file.
- This machine/session only (local path, personal taste, one-off setup) → auto memory.
- Deterministic, and checkable on this session's edits → a **hook** (+ its test).
- Must hold on every authoring path (IDE · teammate · another agent) → a **test/CI invariant**;
  absolute in every context → `permissions.deny`. Contracts and gate promotion: `enforcement.md`.

Qualifies for a rules file: a convention/pattern decided this time · non-obvious design rationale ·
a contract other modules depend on (or a change to one) · a gotcha that bit us and will bite again ·
a missing step in an existing "how to add …" checklist.

Do not capture: one-off facts specific to this task (already in code/tests/commit), or anything
code/git already makes self-evident.

Format:
- Write instruction files (CLAUDE.md, `.claude/rules/*`) in **English** — clarity + tokens. Domain
  string literals (column names, error constants) stay in their original language: they are data.
- Pick the file by topic; **read the target file first** and match its existing style/format —
  update the relevant section, don't blindly append a duplicate.
- Keep it terse and actionable — rules, not prose narrative. Stage it with the code change.
- **Prune as you add**: when adding a rule, check whether an existing item is now dead — absorbed
  into a default, or promoted to enforcement (test / `settings.json` / `permissions.deny`) — and
  delete it in the same change. A gotcha that code/config already blocks is noise.

## File size budget (keep each instruction file dense)

Gotchas accumulate; a bloated rules file loads in full every session and dilutes signal. Soft
budget: **~150 lines per file** (CLAUDE.md and each `.claude/rules/*.md`). A PostToolUse hook
(`.agent-hooks/check_rules_size.py`) scans the governed set and nudges.

Detection is deterministic; the response is judgment. **Invoke the `refactor-agent-rules`
skill** — it holds the four remedies (relocate / split / abstract / compress, in that order),
the parallel prune-what-enforcement-covers check, and the two deletion tests. Do not improvise
a shortening pass: compression is the weakest of the four and the untrained reflex.

**A generated file takes none of the four** — `docs/experiment-registry.md` above all. Hand
edits and compression are both discarded by the next `scripts/exp.py render`; the only lever is
the generator (`enforcement.md` -> Keeping a generated artifact alive).

A tight single-topic file slightly over budget is fine — these are levers, not a mandate.

## Rule Conflicts & Harness Improvement

The harness (CLAUDE.md · `.claude/rules/` · settings) is not a static document — it's a device
that keeps growing and getting corrected.

- **Rule ↔ request conflict**: don't silently follow the rule and ignore the request, and don't
  silently break the rule. **Surface the conflict**: name which item in which file it conflicts with
  and why, and confirm which takes precedence. User instructions override rules, but present the
  rationale so the reason the rule exists isn't lost.
- **Rule wrong or stale**: if a rule doesn't match reality (code · convention), fixing it is part of
  the task. Propose/apply the update immediately and tell the user.
- **Screen every rule revision with one question**: *was the rule wrong to begin with, or did this
  change just make it inconvenient?* Only the first justifies rewriting it.
- **An unsupported claim is an assumption.** If a statement in the harness lives only in the sentence
  that asserts it — no test, no generated artifact, no command that re-checks it — mark it as an
  assumption or delete it.
- **Improving the harness itself**: a missing trigger, a dead rule, a wrong path-gate, a bloated
  CLAUDE.md — refine the harness alongside Capturing Learnings. Reconciling against the shared
  skeleton is the `harness-spine:update` skill's job, not a hand-diff.

## Verification Commands

```bash
uv run pytest -q                       # must pass before declaring done (offline, no API keys)
uv run ruff check src tests scripts    # lint (line-length 100)
python .agent-hooks/check_rule_links.py   # every path a rules file / agent names exists
```
