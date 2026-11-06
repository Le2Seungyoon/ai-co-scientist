# Architecture

User → orchestrator → lane main agent → sub-agent → `scripts/` CLI → data / GPU / submission. No servers, no protocols. The agents share exactly one piece of state: the experiment
registry.

## Layers & dependency direction

| Layer | Location | Rule |
|---|---|---|
| Lane definitions | `.agents/agents/*.md` (source) → `.claude/agents/*.md` · `.codex/agents/*.toml` (**generated**) | Role prompts only. No code. Edit the source, never a generated copy |
| Harness package | `src/ai_co_scientist/` | `config.py` (settings) · `registry.py` (registry) · `sem.py` (pure SEM/depth logic: split, reparameterization, smoothing, QDA) · `backends/` (external I/O). This is what tests cover |
| Execution CLI | `scripts/*.py` | Sub-agent entry points. Keep them thin |
| State | `runtime/registry.jsonl` → `docs/experiment-registry.md` | The single truth of every experiment. **`runtime/` is gitignored: the jsonl exists on this machine only, with no backup** — the rendered doc is the only copy in git |

**Two layers of markdown, never mixed** (relocated here from `workflow.md`, 2026-09-01): the
co-scientist's own agents are the role prompts in `.agents/agents/*.md`; `.agents/rules/*.md` are the
rules for whoever works *on* this repo. A runtime instruction belongs in the first, a working
convention in the second.

## CLI / logic separation (migration in progress)

The standalone-script boundary was abolished on 2026-08-17: nothing ever ran remotely, and the
cost was duplication no gate could see (`coding-patterns.md` → When to extract carries what it
left behind).

- **Restore it only if** remote GPU becomes a real need (training beyond the local 8 GB). The
  Lightning Studio CLI and its backend are in git history, not at any path in the tree.
- **Migration is unfinished**: move logic into `src/` and keep `scripts/` a thin CLI. New code goes
  in `src/`; do not grow logic in a script. The one known remaining item: the smp model zoo still
  lives only in the frozen legacy trainer and has not been absorbed into `src/`.
- **Because the migration is unfinished** — not because of any standalone rule — the training
  scripts still do not call `load_config()`, so their path defaults are hand-matched to
  `config.yaml` `paths.*` (`--cache-dir`=`runtime/cache`, `--data-dir`=`data`,
  `--output-dir`=`runtime/ckpt`). This has drifted once: a `--cache-dir` default of `cache` started
  re-baking a 350k-image cache in the wrong place. Change `paths.*` and you must change the scripts
  in the same commit. Only `scripts/exp.py` uses `load_config()` today.

## Module boundary contracts

- **Backends stay thin**: `backends/` only calls external APIs. State lives in the registry —
  submission records in two places means neither is the truth.
- **Config through `load_config()` only**: never open `config.yaml` directly.
- **Domain labels travel with the data**: training scripts write `x_domain` / `y_source` into the
  manifest and the registry checks them against the target (real→real). Do not cut this wiring —
  it is what stops a sim metric from being mistaken for real validation.

## Parallel execution contract (lanes)

**Hierarchical Parallel-Lane.** One hypothesis is one lane: one branch off `develop`, one
worktree, one main agent. The orchestrator talks only to lane main agents; each main agent runs
its own sub-agents in three phases.

| Phase | Sub-agents | Output |
|---|---|---|
| Before Execution | `researcher`, `engineer`; `reviewer` audits `researcher` | a committed plan: design, pre-report, code, recipe |
| — approval — | orchestrator | plan merged into `develop`, `report_id` issued, `runtime/registry.link` written |
| Execution | `executor` | results recorded through the link, a verified zip |
| — submission — | orchestrator, after the user approves | zip submitted from main, `exp.py lb` recorded, scores replied to the lane |
| After Execution | `analyst`, `harness-manager`; `reviewer` audits `analyst` | hypothesis file, rule changes, one `worker_done` |

**Approval is the pre-registration.** The plan is fixed when the orchestrator merges the lane's
plan commit into `develop` and issues the `report_id` — before any run, so commit order proves
it. `scripts/exp.py new` runs on the orchestrator's side, in the main checkout; an id issued once
the outcome is known would be registry data pretending to be pre-registration.

**The link is the execution permit.** `uv run python scripts/exp.py link <worktree>` writes
`runtime/registry.link` — the main registry's absolute path. `registry.py` follows it, so writes
share the main registry's lock and ids never fork. `block_runtime_commands.py` unlocks
experiment commands only where the real registry or a valid link exists; a dangling link is
denied with its own reason. **Never create `runtime/registry.jsonl` in a worktree** — it would
shadow the link.

**File domains inside a lane.** `engineer`: `src/ scripts/ tests/`. `harness-manager`:
`AGENTS.md`, `.agents/**`, `.agent-hooks/**`, both registrations. `analyst`: the lane's hypothesis
file and `docs/hypotheses.md`. `executor`: `runtime/` (through the link, and its own
outputs). `researcher`, `reviewer`: read-only by contract. Two lanes may touch the same shared
file (`docs/hypotheses.md`, a rule); the orchestrator resolves that at the merge into `develop`.

**Resources are exclusive, lanes are not.** The GPU is serialized by `locks.resource_lock("gpu-0")`,
whose lock directory is machine-wide, so executors in several lanes queue on one card. A script
that cannot take the lock exits before doing work. Lightning is the second GPU and needs the
user's approval. DACON submission is `MAIN_ONLY` and sits between Execution and After Execution:
a lane builds the zip, runs `verify_submission()`, and sends the zip path to the orchestrator
through the preamble's `ask`, then waits. The orchestrator asks the user, submits from the main
checkout, records the leaderboard with `exp.py lb`, and replies with the public/private scores;
only then does the lane's `analyst` decompose with the leaderboard. If the user declines or
defers, the analyst reports on validation/holdout only and marks the leaderboard component
"pending".

**Execution environment.** A lane that executes needs `uv sync --group baseline` in its own
worktree (never a shared `.venv`). Data, caches and checkpoints stay in the main checkout; the
recipe names their absolute paths. `docs/experiment-registry.md` is rendered (`scripts/exp.py
render`) on `develop` by the orchestrator, never in a lane — two lanes rendering it would
conflict on a generated file.

**Shared-state race conditions** — both were measured and fixed:

- **Registry write race**: `report_id` comes from `len(records)` and `_write_all` rewrites the whole
  file, so without a lock two agents take the same id and the later write erases the earlier
  pre-report. Under 8 concurrent registrations, **only 2 of 8 survived**. → `registry.locked()` wraps
  read+write together, and the link makes every lane take the same lock.
- **Shared submission work dir**: parallel inference overwrote each other's PNGs and a zip ended
  up mixing two models' output — **while still scoring normally**. → split into
  `submission_work/<zip stem>/`.

**CPU reassembly** (`scripts/assemble_submission.py`) stays unguarded: numpy and stdlib only, so
N level-post-processing sweeps run in lanes off one GPU inference without a link.

Panes in one worktree **share its branch** — split the worktree, not the pane, when experiments
need separate branches. Mechanics of dispatching and watching a lane: `orca-parallel.md`.
