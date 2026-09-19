---
paths:
  - tests/**
  - .agent-hooks/**
  - .agent-hooks/**
---
# Testing

## Conventions

- Location: `tests/test_*.py`, and `tests/<pkg>/test_*.py` for package modules — mirroring
  `src/ai_co_scientist/<pkg>/`. Run: `uv run pytest -q` (all) · `uv run pytest tests/test_registry.py -q`
  (one file) · `-k <pattern>` (one test).
- Deterministic, **no network access** — the whole suite must pass with no API keys and no `.env`.
  The DACON backend swaps HTTP out through `COSCIENTIST_DACON_FAKE_HTTP`; file state is isolated with
  `tmp_path`. Registry tests always pass `path=tmp_path/...` — touching the real
  `runtime/registry.jsonl` corrupts the experiment record. Copy `tests/test_registry.py`.
- **No weak asserts**: a range check lets a wrong implementation pass. Verify exact values and
  relationships.
- Regression tests come **with rationale** — a comment naming the bug they pin, so the assert is not
  "mysteriously specific" to a future reader.

## Invariant tests (the layer hooks cannot reach)

A hook only sees edits made in a Claude session. Code written in an IDE, by a teammate, or by another
agent never passes one. **A rule that must hold no matter who authored the code belongs here**, as a
test over the tree itself (`enforcement.md` → Four layers).

- Assert the property, not the sample: walk the source and fail with the offending paths listed.
- Give every invariant test an escape hatch the source can carry (a marker comment with a reason), or
  the first legitimate exception gets the test deleted.
- **`tests/test_train_manifest.py`** — training scripts need GPU and data, so they are not unit-run;
  what is pinned instead is the contract that survives only as source text: the manifest's (X, y)
  domain declaration, `infer_decomposed`'s `cnn` level arm, AdaBN staying off the level classifier,
  no per-image normalization in `train_level`, a swappable structure backbone, ckpt
  backward-compatibility with EXP-005's bare `state_dict`, and no wandb import chain reaching
  inference. **This is a stand-in that cannot check behavior** — as logic moves into `src/`, replace
  each check with a real unit test (`architecture.md` → CLI / logic separation). `scripts/legacy/` is
  not checked: frozen, reproduction-only, cannot regress.
- **`.agent-hooks/check_rule_links.py`** pins that every file a rules file or an agent
  definition points at still exists — the pointers `workflow.md` → File size budget tells you to
  leave behind. Not yet wired into the
  suite; run it by hand (`self-review.md` → Gates) until its false-positive rate here is measured.

## Freshness tests for generated artifacts

`docs/experiment-registry.md` is generated from `runtime/registry.jsonl` by `scripts/exp.py render`,
and it is the file every agent actually reads. Nothing currently fails when the two drift apart —
**the render is a step someone has to remember, which is the defect one level up.**

- Regenerate in the test and compare; the failure message names the regeneration command.
- **Prove it catches staleness by causing it**: delete one entry, delete a section, add a record
  without rendering.
- Distinguish three states — nothing registered yet · stale · unreadable. Collapsing them into one
  silence means "no drift" and "never looked" read identically.

## Verifying the defenses themselves

- **Delete the defense and re-run.** A green suite proves nothing currently violates a guard, never
  that the guard works. Remove the check, or feed it a violating input, and confirm something goes
  red. An untested guard is indistinguishable from a comment.
- Harness code ships with its test beside it: `.agent-hooks/test_check_rules_size.py`,
  `.agent-hooks/test_check_rule_links.py`. Both halves matter — the must-block half proves it
  fires, the must-pass half is what keeps false positives out. They are stdlib-only and run directly
  (`python .agent-hooks/test_check_rules_size.py`), because a hook must be verifiable before
  dev dependencies are installed.
- **Fixtures must be distinguishable.** If two code paths coincidentally produce the same value, one
  output collapses both and a broken path still passes.
