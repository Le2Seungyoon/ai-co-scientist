# Experiment ledger (docs/experiment/)

Where a hypothesis lives while it is being tested, and how parallel lanes write results without
fighting over one file. `runtime/registry.jsonl` owns the numbers; this file owns the document
side and the join between them.

## One hypothesis, one file, one writer at a time

`docs/experiment/H<id>-<slug>.md`, Korean, from `docs/experiment/_TEMPLATE.md`. **An id prefix,
not a date** — a hypothesis is revisited, and a date would lie about when it was last true
(specs keep dates: they are decisions made at one moment and never edited again).

| Section / front-matter field | Written by | When |
|---|---|---|
| 질문 · 조건 · 무엇이 답인가, `status: 계획` | orchestrator | before dispatch, on `main` |
| 결과 · 관찰 · 판정 · 미검증, `status: 진행중 → 측정됨 → 판정`, `verdict` | the lane | during/after its run |
| 이관 범위, `registry` (append each issued id) | orchestrator | after merge |

**`registry: []` is appended, not drafted.** `EXP-0NN` ids are issued by `new_report` against
`runtime/registry.jsonl` — the same single-writer reasoning that keeps the registry the original
(below) puts landing each confirmed id in front-matter on the orchestrator, after merge. The
field is a list because a hypothesis may collect several reports. The lane's own 결과 table may
already name the ids it expects; this field is the reconciled copy.

**A lane touches only its own hypothesis file.** Not another lane's, not the backlog. Two lanes
therefore never edit the same path, so git merges them without a conflict — the property depends
on the single-writer rule, not on the directory.

**The branch point is the snapshot.** A lane cut from commit X already holds every hypothesis
file as of X; copying the ledger into the task spec would recreate the divergence this layout
removes.

## No shared index

Status lives in each file's front-matter; `ls docs/experiment/` plus that is the view. A
hand-maintained index puts every lane back on one hunk — the exact conflict the split removes.
When a table is genuinely wanted, **generate it** and give it a freshness test, never hand-write
it (`enforcement.md`).

`docs/hypotheses.md` is the **error budget and the backlog** — what has not been tried and what
was rejected. A dispatched hypothesis moves to its own file and is closed there.

## The registry is the original; the file quotes it

- **`runtime/registry.jsonl` is an append-only list of report records.** Each report has one
  unique `report_id`; recording more arms or results updates that report rather than issuing
  another id.
- **Numbers, conditions, the code commit: `runtime/registry.jsonl`.** The file copies only the
  headline figures a reader needs to follow the verdict.
- **Judgment, plan, transfer scope: this file.**
- **If it is not in the registry, it is not a result.**

## The join key

Every pre-report carries `hypothesis=<id>` — one id, a plain string — so the link is queryable
both ways: which runs tested H12, and why a run exists at all. **One report names exactly one
hypothesis; one hypothesis may have many reports** — report → hypothesis is many-to-one,
hypothesis → reports is one-to-many. Arms stay inside one report: EXP-010 recorded a 3-arm sweep
under one `report_id`. `registry.new_report` refuses a missing or blank id, and
`tests/test_registry.py` pins both refusals and that two reports may share one hypothesis id —
an untagged record is invisible to the join and nothing else notices, which is why it is a
refusal and not a convention. The registry does not validate the id's shape: `H12,H13` would be
stored as one string, so "exactly one" is a lane contract, not a registry gate.

**`new_report` runs before dispatch, not after the lane reports** — an id issued once the run's
outcome is already known is registry data pretending to be pre-registration.

## Write the "don't run" condition too

A hypothesis that pins a symptom on something the optimiser or the model can absorb is usually
already answered. 「무엇이 답인가」 states the condition for running **and** the condition for
closing it unrun — the second is what makes a hypothesis cost nothing.

## One lane, one hypothesis

Two lanes on one hypothesis is a dispatch bug, not a merge bug: the single-writer property above
is what the automatic merge depends on. Assign one lane per hypothesis, even when that
hypothesis accumulates several reports. The orchestrator writes 질문 · 조건 · 무엇이 답인가
**before** dispatch, on `main`, so pre-registration is proved by commit order. How a lane is
dispatched and driven: `orca-parallel.md`.
