# Architecture

Human (project lead) → main Claude (PM) → sub-agent (execution) → `scripts/` CLI → data / GPU /
submission. No servers, no protocols. The agents share exactly one piece of state: the experiment
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

## Parallel execution contract (sub-agents)

| Class | Agents | Domain | Why it is safe |
|---|---|---|---|
| **Parallel (read)** | `researcher` · `reviewer` | — | read-only by contract; Bash is not withheld |
| **Parallel (write)** | `engineer` · `harness-manager` · `analyst` | `src/scripts/tests` · `AGENTS.md` + `.agents/**` + `.agent-hooks/**` + both registrations · `docs/` | disjoint file domains by contract; the hook covers the worktree case only |
| **Exclusive (one)** | `executor` | `runtime/` | one 8 GB GPU · DACON quota · checkpoint writes |

`README.md` is human-facing and owned by no agent — it is outside every domain above, not folded
into `harness-manager`'s. An agent may still edit it on an explicit instruction; that is not the
same as it being anyone's standing domain.

Five agents may run alongside one `executor`. With a single GPU there is no way to parallelize
training, so the parallel gain is in analysis, criticism, proposals, and preparing the code for
experiments still queued.

`engineer` and `executor` hold the SAME tools. What separates them is
`.agent-hooks/block_runtime_commands.py`, which guards five scripts in two tiers — `exp.py` as
a registry writer, and `train_level.py` / `train_structure.py` / `infer_decomposed.py` /
`dacon_submit.py` as exclusive-resource scripts — wherever `runtime/registry.jsonl` is absent
(`probe_level.py` was released: read-only, no `runtime/` writes). That hook cannot see which
sub-agent issued a command, so it only makes the boundary real in a worktree — **the engineer
lane must run in a worktree**, or its contract is prose alone.

**CPU reassembly is a fourth class.** `scripts/assemble_submission.py` rebuilds a submission from
a dumped ŝ and level posterior with numpy and stdlib only -- no torch, no cv2, so it runs in a
worktree, whose `uv sync` brings the dev group alone. Level post-processing (`--level-smooth`,
`--level-hmm`, `--tau`) moves no structure component, so N of these run in parallel off ONE GPU
inference. They write into their own tree and never call `scripts/exp.py`: the coordinator issues
the pre-report, dispatches the `report_id`, and records the result, so `report_id = len(records)`
has no fork path across trees. A worker reports its own `git rev-parse HEAD` and branch, and the
coordinator records THOSE -- never its own.

**Never create `runtime/registry.jsonl` in a worktree.** That file's existence IS the runtime
gate's sentinel; once it exists the gate is unlocked in that tree for good. This is why the
escape hatch does not cover `scripts/exp.py` (`enforcement.md` -> This project's gates).

**Workers build submissions; they never submit one.** The leaderboard is the only verdict and its
slots are finite, so N parallel zips cannot all be spent. A worker runs `verify_submission()` in
its own tree -- that one is not guarded, and it catches a broken zip before a slot pays for it --
and reports its pre-selection numbers. Ranking for the level axis is real train + `site_split`
holdout accuracy, which is real->real and is how `k=9` was chosen; it **ranks, it does not
judge** (real has 2,836 runs against test's 1,046, so the optimum does not transfer). The
coordinator submits the top one. One sweep is ONE pre-report: `exp.py result` merges into the
existing `val` by default, so per-arm numbers stack without a new schema.

**Shared-state race conditions** — both were measured and fixed:

- **Registry write race**: `report_id` comes from `len(records)` and `_write_all` rewrites the whole
  file, so without a lock two agents take the same id and the later write erases the earlier
  pre-report. Under 8 concurrent registrations, **only 2 of 8 survived**. → `registry.locked()` wraps
  read+write together. New code that touches the registry must go through it.
- **Shared submission work dir**: every run derived its output names from the cached test_names.json
  list, so parallel inference overwrote each other's PNGs and a zip ended up mixing two models'
  output — **while still scoring normally**, the worst kind of failure. → split into
  `submission_work/<zip stem>/`.

### Extending the contract to a second Orca session

A dispatched Orca session is a third execution class, and the contract above governs it unchanged:
its file domain must be disjoint from every other writer, `executor` stays exclusive (Orca will
dispatch two GPU tasks at once — nothing in its lifecycle knows about the 8 GB card), and a
dispatched session is no exemption from the pre-report. `registry.locked()` already covers
concurrent pre-report writes. Mechanics: `orca-parallel.md`.

Panes in one worktree **share its branch** — a second session cannot be on a different one. Split
the worktree, not the pane, when experiments need separate branches.
