# Documentation

**The reader decides the language, the lifetime, and whether the file may be hand-edited** — not
the directory it sits in.

## Audience → language

| What | Read by | Language |
|---|---|---|
| `AGENTS.md`, the one-line `CLAUDE.md`, `.agents/**/*.md` | agents, every session | **English** |
| `docs/**`, `README.md` | people | **Korean** |
| docstrings and comments in `src/` | whoever edits that code | **Korean** — source, not instructions |

Hook output and deny messages are agent-facing: English. **Quoted domain literals keep their own
language everywhere** — column names, error constants, `docs/hypotheses.md` → 순위 기준 as a
section title. They are data, not prose.

## Lifetime

- **Living, no date prefix** — `docs/data-facts.md`, `docs/hypotheses.md`,
  `docs/codex-verification-ledger.md`. Rewritten in place; traps and prerequisites accumulate. A
  date prefix would imply a series where there is one current answer.
- **Dated, never rewritten** — `docs/superpowers/specs/` and `docs/superpowers/plans/`. They age
  on purpose: a spec rewritten to match today's tree stops being a record (`harness.md` →
  `.claude/` paths in the record are historical).

## Not in `docs/`

- **Per-experiment numbers** — the registry is the only admissible evidence, and a copy here
  drifts from what the leaderboard judged.
- **Rules for working in this repo** — `.agents/rules/`. `docs/` describes the problem;
  `.agents/rules/` describes how we work on it (`architecture.md` → Layers & dependency
  direction).

## Generated documents take no hand edits

`docs/experiment-registry.md` comes from `scripts/exp.py render`. Hand edits **and** compression
passes are discarded by the next render; the only lever is the generator (`enforcement.md` →
Keeping a generated artifact alive).

## Unmeasured is a status, not a hedge

Where an area is unverified, give it a ledger with per-item status instead of hedging in prose —
`docs/codex-verification-ledger.md`. An entry still marked 미측정 is not something to build a
rule on.
