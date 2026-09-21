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
| State | `runtime/registry.jsonl` → `docs/experiment-registry.md` | The single truth of every experiment |

**Two layers of markdown, never mixed** (relocated here from `workflow.md`, 2026-09-01): the
co-scientist's own agents are the role prompts in `.agents/agents/*.md`; `.agents/rules/*.md` are the
rules for whoever works *on* this repo. A runtime instruction belongs in the first, a working
convention in the second. The old top-level `rules/` directory went away in the 2026-07-30 A2A strip.

## CLI / logic separation (migration in progress)

The standalone-script boundary was **abolished on 2026-08-17**, together with the provisional
Lightning Studio plan. Its premise — "upload one file and run it remotely" — was never once
realized: every recorded experiment ran on the local GPU, and the registry mentions lightning zero
times. The cost was real: `seed_everything` ×5, `ensure_utf8_console` ×3, `PlainMLP` / `UNetSmall` /
`SmpModel` ×2 each, and, because scripts could not be imported, a `tests/` suite stuck on
source-text contract checks.

- **Restore it only if** remote GPU becomes a real need (training beyond the local 8 GB). The
  deletion commit still holds the Lightning Studio CLI and its backend module; neither is in the
  current tree, so look them up in git history rather than by path.
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
`.agent-hooks/block_runtime_commands.py`, which denies the six experiment scripts wherever
`runtime/registry.jsonl` is absent. That hook cannot see which sub-agent issued a command, so
it only makes the boundary real in a worktree — **the engineer lane must run in a worktree**,
or its contract is prose alone.

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

## Removed structure (2026-07-30)

The A2A 7-server layout (`a2a/`, `agents/`), five MCP servers (`mcp_servers/`), the LLM router
(`llm/`) and the toy task. They are in git history. Do not revive them — the competition asks for
autonomous collaboration, not a protocol.
