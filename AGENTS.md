# ai-co-scientist

Follow-up to the 2025 Samsung AI Challenge — AI Co-Scientist. The point of the competition is
**several autonomous agents collaborating to solve a deep-learning task**; the task itself is SEM
image → depth map regression. The human is the project lead, the main Claude is PM/orchestrator,
and the work is done by the sub-agents defined in `.agents/agents/`.

The A2A multi-server / MCP-server layout was removed on 2026-07-30 — over-engineered at this size,
and what the competition asks for is autonomous collaboration, not a particular protocol. Sub-agents
now call the `scripts/` CLI directly.

## Core invariants

- **No experiment without a pre-report** — X / y / model+hyperparameters / methodology / purpose go
  into the registry first, and that is what issues a `report_id`.
  `.agents/rules/workflow.md` → Experiment Pre-Report.
- **The registry is the single truth** — `docs/experiment-registry.md` (rendered) ←
  `runtime/registry.jsonl`. Everything before the 2026-07-29 reset is void and must not be cited.
- **The leaderboard judges** — if a metric's (X, y) differs from the target (real SEM → real depth),
  it cannot be used to claim real performance.
- **Data facts come first** — the depth-map structure (background level L, 1 of 4, plus normalized
  structure) and the meaning of `average_depth` are fixed in `docs/data-facts.md`. Code or docs that
  disagree with it are bugs.
- **Tests are offline** — `uv run pytest -q` passes in full with no API keys.
- **Logic in `src/`, `scripts/` thin** — the standalone constraint was abolished on 2026-08-17.
  The migration is unfinished, so **write new code in `src/` and do not grow logic in a script**.
  `.agents/rules/architecture.md`.
- **Only `executor` is exclusive** — one `executor` at a time (GPU + submissions). The other
  five run in parallel; the three that write are kept apart by file domain, not by luck.
  `.agents/rules/architecture.md` → Parallel execution contract.

## Commands

```bash
uv sync                                   # reproduce the environment
uv run pytest -q                          # tests (offline)
uv run ruff check src tests scripts       # lint (line-length 100)

uv run python scripts/exp.py new --title ... # pre-report → report_id
uv run python scripts/exp.py list | render   # query the registry / regenerate the doc

# Current pipeline — depth = L (level, 1 of 4) × (1 − s (normalized structure)). docs/data-facts.md §2
uv run python scripts/train_level.py                              # level classifier (real→real)
uv run python scripts/train_structure.py --arch mlp               # structure regressor (swappable backbone)
uv run python scripts/dacon_submit.py runtime/submissions/EXP-0NN-arm.zip --report-id EXP-0NN
```

### Reproducing the current best — LB **3.0493** / private 2.9961 (EXP-019)

```bash
uv run python scripts/infer_decomposed.py \
    --ckpt runtime/ckpt/EXP-005-structure.pt \
    --level-source cnn --level-ckpt runtime/ckpt/EXP-013-level-cnn.pt \
    --adabn real --adabn-shuffle 42 --tau 0.0 --level-smooth 9 \
    --submit runtime/submissions/<report_id>.zip
```

No training needed — both checkpoints are in `runtime/ckpt/`. **All four flags were won by
experiments**, so dropping any one costs score: `--adabn real` (EXP-010, −0.52) ·
`--adabn-shuffle 42` (EXP-016, **−0.51**) · `--level-source cnn` (EXP-014, −0.47) ·
`--level-smooth 9` (EXP-019, −0.07). `--tau 0.0` (no clamp) was optimal across EXP-005 and 007.

**Always verify the submission**: 25,988 files · every image's max in {140,150,160,170} · 0.00 %
outside.

Pre-decomposition scripts live in `scripts/legacy/` — reproduction-only, **frozen**, not the current
path (`scripts/legacy/README.md`). `--arch smp:*` needs `uv run --group baseline`. If training dies
on the 8 GB card, **find the cause before cutting the batch size**: never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on Windows (it triggers the 0x10E bugcheck), and
a whole-PC reboot is a driver bugcheck, not an OOM. `.agents/rules/coding-patterns.md` → Gotchas.

## Configuration

| What | Where |
|------|-------|
| Secrets (`DACON_API_TOKEN`, `DACON_CPT_ID`, `DACON_TEAM_NAME`, `WANDB_*`) | `.env` (copy `.env.example`) |
| Tunables (paths, target, train defaults) | `config.yaml` via `ai_co_scientist.config.load_config()` |
| Dependencies | `pyproject.toml` + `uv.lock` (`uv` only — `uv pip install` is denied) |

## Writing rules in this file

**Name the action, never the harness.** This file is read verbatim by every agent that works
here, so a statement true of one harness and false of another does more damage than no statement
— an agent told its rules arrive automatically will not go and read them. Anything that depends
on which agent is running belongs in the rule it modifies, as a **Claude Code —** or **Codex —**
paragraph beside the neutral statement. `.agents/rules/harness.md` has the convention.

## Docs convention

- **User-facing docs** — `docs/`, written in Korean. All three are **living documents, overwritten
  in place**, so they carry no date prefix: current state is rewritten, traps and prerequisites
  accumulate, and per-experiment decisions belong in the registry, not here.
- `docs/experiment-registry.md` is **generated** by `scripts/exp.py render`. Never hand-edit it, and
  never run a compression skill over it — the next render discards both.
- **Agent-facing instruction files** (this file, the one-line `CLAUDE.md` that imports it, and
  `.agents/**/*.md`) — written in **English**. Korean
  docstrings and comments in `src/` stay as they are: they are source, not instructions.

## Rules

| File | When to read |
|------|--------------|
| `.agents/rules/workflow.md` | **before every experiment** (Experiment Pre-Report) · planning · capturing learnings |
| `.agents/rules/git-workflow.md` | before any task that changes files |
| `.agents/rules/architecture.md` | harness structure · layer boundaries · **sub-agent parallel contract** |
| `.agents/rules/coding-patterns.md` | editing `src/**`, `scripts/**` |
| `.agents/rules/testing.md` | writing or changing tests, hooks, scanners |
| `.agents/rules/enforcement.md` | turning a rule into a hook / test / deny — and before promoting any check |
| `.agents/rules/harness.md` | **before changing anything that configures an agent** — instructions, rules, hooks, registrations |
| `.agents/rules/self-review.md` | at the end of every task, before declaring done |
| `.agents/rules/orca-parallel.md` | dispatching work to a **second Orca session** (not sub-agents) |

## Sub-agent roster

| Agent | Owns | Class |
|---|---|---|
| `researcher` | hypotheses, pre-report drafts | parallel (read) |
| `reviewer` | audits of designs and conclusions | parallel (read) |
| `engineer` | `src/` `scripts/` `tests/` | parallel (write) |
| `harness-manager` | `AGENTS.md` `.agents/**` `.agent-hooks/**` + both registrations | parallel (write) |
| `analyst` | `docs/` (not the generated registry) | parallel (write) |
| `executor` | `runtime/` — runs and records | **exclusive** |

`README.md` ownership: `.agents/rules/architecture.md` → Parallel execution contract.

The orchestrator (main session) owns the queue, the assignment, the ranking and the
integration, and:

- **does not execute experiments** — that bypasses the exclusivity contract and the registry path;
- **escalates irreversible actions to the human** — submission, `git push`/`checkout`/`branch`,
  registry corrections;
- **does not skip `reviewer`** — judging a pre-report sound is not the same as auditing it;
- **does not invent conclusions** — only what `analyst` and `reviewer` support;
- **does not rank without stated criteria** — write the criteria and their application down
  (`docs/hypotheses.md` → 순위 기준).

## Enforcement hooks

Hook scripts live in `.agent-hooks/`, **one copy**, registered by each harness in its own file —
`.claude/settings.json` and `.codex/config.toml`, whose schemas are unrelated. Wired: a PR gate
(blocks `git push` when `origin/main` is not merged in), a commit-attribution deny hook, a
runtime-command deny hook (`block_runtime_commands.py` — experiment scripts only run where
`runtime/registry.jsonl` lives), a PostToolUse nudge on the ~150-line instruction budget
(`check_rules_size.py`), and a PostToolUse rebuild of the generated lanes (`build-agents.py
--hook`). Repo-wide scanners sit beside them, each with its test —
`.agents/rules/enforcement.md` → Where harness code lives.

Lane definitions live in `.agents/agents/` and skills in `.agents/skills/`; `build-agents.py`
generates the `.claude/` and `.codex/` copies both harnesses read — **never hand-edit those**.
`uv run pytest -q` fails when they drift, when the two registrations disagree, or when a rule
pointer stops resolving.

**Hooks only see this session's edits** — `.agents/rules/enforcement.md` → Hooks only see this
session's edits. What has not been measured on the Codex side is in
`docs/codex-verification-ledger.md`; do not build a rule on an entry that is still empty.

## References

- `docs/data-facts.md` — confirmed data structure (measurements + the organizer's official answers).
  **Required reading before designing an experiment.**
- `docs/experiment-registry.md` — the experiment registry (the only admissible evidence).
- `docs/hypotheses.md` — hypothesis backlog + **error budget**. Pick the next experiment here and
  compute its expected gain; rejected entries carry their reasons, so check before re-proposing.
  The three budget components are nearly flat (4.3 / 2.5 / 2.4), so there is no dominant lever —
  and **all four** submissions that improved the leaderboard by 0.4 or more came with **zero
  training** (EXP-009 −2.353 · EXP-010 armC −0.522 · EXP-016 −0.513 · EXP-014 −0.469; the only
  training-bearing improvement is EXP-003, −0.396), which is why the cheap axes are swept first.
  Deltas are re-derivable from `runtime/registry.jsonl`; `docs/hypotheses.md` → 순위 기준 holds
  the same claim — fix both or neither.
- `docs/codex-verification-ledger.md` — what has NOT been measured on the Codex side.
  **Do not build a rule on an entry that is still empty.**
- README — data preparation / baseline reproduction / enabling the real backends.
