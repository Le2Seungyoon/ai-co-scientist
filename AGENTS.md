# ai-co-scientist

Follow-up to the 2025 Samsung AI Challenge — AI Co-Scientist. The point of the competition is
**several autonomous agents collaborating to solve a deep-learning task**; the task itself is SEM
image → depth map regression. The human is the project lead, the main Claude is PM/orchestrator,
and the work is done by the sub-agents defined in `.agents/agents/`.

Sub-agents call the `scripts/` CLI directly — no servers, no protocols.

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
- **Logic in `src/`, `scripts/` thin** — the migration is unfinished, so **write new code in
  `src/` and do not grow logic in a script**. `.agents/rules/architecture.md`.
- **Only `executor` is exclusive** — one at a time (GPU + submissions); the other five run in
  parallel. `.agents/rules/architecture.md` → Parallel execution contract.
- **The harness has one copy of everything but its registration** — rules, hooks and lane sources
  under `.agents/` and `.agent-hooks/`; the `.claude/` and `.codex/` copies are **generated, never
  hand-edited**. `uv run pytest -q` gates that, and hooks only see this session's edits.
  `.agents/rules/harness.md` · `enforcement.md` → This project's gates.

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

No training needed — both checkpoints are in `runtime/ckpt/`. **Every flag here was won by an
experiment**, so dropping one costs score: `--adabn real` (EXP-010) · `--adabn-shuffle 42`
(EXP-016) · `--level-source cnn` (EXP-014) · `--level-smooth 9` (EXP-019) · `--tau 0.0`, no clamp,
optimal across EXP-005 and 007. The registry holds what each is worth.

**Never submit an unverified zip** — the check and its numbers are in
`.agents/rules/self-review.md` → Gates.

Pre-decomposition scripts live in `scripts/legacy/` — reproduction-only, **frozen**, not the current
path (`scripts/legacy/README.md`). `--arch smp:*` needs `uv run --group baseline`. If training dies
on the 8 GB card, **find the cause before cutting the batch size** — the two real causes, the
diagnosis command, and the trap that is never the fix are in
`.agents/rules/coding-patterns.md` → Gotchas.

## Configuration

| What | Where |
|------|-------|
| Secrets (`DACON_API_TOKEN`, `DACON_CPT_ID`, `DACON_TEAM_NAME`, `WANDB_*`) | `.env` (copy `.env.example`) |
| Tunables (paths, target, train defaults) | `config.yaml` via `ai_co_scientist.config.load_config()` |
| Dependencies | `pyproject.toml` + `uv.lock` (`uv` only — `uv pip install` is denied) |

## Writing rules in this file

**Name the action, never the harness.** This file is read verbatim by every agent that works
here, so a statement true of one harness and false of another does more damage than no statement.
Harness-dependent text goes in the rule it modifies as a labelled paragraph, never here —
`.agents/rules/harness.md` → Where a thing lives.

## Rules

| File | When to read |
|------|--------------|
| `.agents/rules/workflow.md` | **before every experiment** (Experiment Pre-Report) · planning · capturing learnings |
| `.agents/rules/git-workflow.md` | before any task that changes files |
| `.agents/rules/architecture.md` | harness structure · layer boundaries · **sub-agent parallel contract** |
| `.agents/rules/coding-patterns.md` | editing `src/**`, `scripts/**` |
| `.agents/rules/testing.md` | writing or changing tests, hooks, scanners |
| `.agents/rules/docs.md` | writing anything anyone reads — which language, which lifetime, what may not be hand-edited |
| `.agents/rules/enforcement.md` | turning a rule into a hook / test / deny — and before promoting any check |
| `.agents/rules/harness.md` | **before changing anything that configures an agent** — instructions, rules, hooks, registrations |
| `.agents/rules/self-review.md` | at the end of every task, before declaring done |
| `.agents/rules/orca-parallel.md` | dispatching work to a **second Orca session** (not sub-agents) |

## Sub-agent roster

| Agent | Use it for | Class |
|---|---|---|
| `researcher` | hypotheses, pre-report drafts | parallel (read) |
| `reviewer` | audits of designs and conclusions | parallel (read) |
| `engineer` | pipeline code an approved pre-report needs | parallel (write) |
| `harness-manager` | rules, hooks, gates, lane definitions | parallel (write) |
| `analyst` | interpreting results, keeping docs current | parallel (write) |
| `executor` | running one approved experiment and recording it | **exclusive** |

**File domains are deliberately not listed here** — two lists of one boundary drift apart.
`.agents/rules/architecture.md` → Parallel execution contract owns them, plus `README.md`'s
ownership and what the worktree hook can actually enforce.

The orchestrator (main session) owns the queue, the assignment, the ranking and the
integration, and:

- **does not execute experiments** — that bypasses the exclusivity contract and the registry path;
- **escalates irreversible actions to the human** — submission, `git push`/`checkout`/`branch`,
  registry corrections;
- **does not skip `reviewer`** — judging a pre-report sound is not the same as auditing it;
- **does not invent conclusions** — only what `analyst` and `reviewer` support;
- **does not rank without stated criteria** — write the criteria and their application down
  (`docs/hypotheses.md` → 순위 기준).

## References

- `docs/data-facts.md` — confirmed data structure (measurements + the organizer's official answers).
  **Required reading before designing an experiment.**
- `docs/experiment-registry.md` — the experiment registry (the only admissible evidence).
- `docs/hypotheses.md` — hypothesis backlog + **error budget** + the ranking criteria. Pick the
  next experiment here and compute its expected gain; rejected entries carry their reasons, so
  check before re-proposing. No budget component dominates, and the big leaderboard gains have
  come from the cheap axes rather than from training — which is why those are swept first.
  → 순위 기준 states that with its numbers, re-derivable from `runtime/registry.jsonl`.
- `docs/codex-verification-ledger.md` — what has NOT been measured on the Codex side.
  **Do not build a rule on an entry that is still empty.**
- README — data preparation / baseline reproduction / enabling the real backends.
