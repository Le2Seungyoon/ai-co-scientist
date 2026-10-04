# Experiment ledger (docs/experiment/)

Where a hypothesis lives while it is being tested, and how parallel lanes write results without
fighting over one file. `runtime/registry.jsonl` owns the numbers; this file owns the document
side and the join between them.

## One hypothesis, one file, one writer at a time

`docs/experiment/H<id>-<slug>.md`, Korean, from `docs/experiment/_TEMPLATE.md`. **An id prefix,
not a date** — a hypothesis is revisited, and a date would lie about when it was last true
(specs keep dates: they are decisions made at one moment and never edited again).

| Section | Written by | When |
|---|---|---|
| 질문 · 조건 · 무엇이 답인가 | orchestrator | before dispatch, on `main` |
| 결과 · 관찰 · 판정 · 미검증 | the lane | during/after its run |
| 이관 범위 | orchestrator | after merge |

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

- **Numbers, conditions, the code commit: `runtime/registry.jsonl`.** The file copies only the
  headline figures a reader needs to follow the verdict.
- **Judgment, plan, transfer scope: this file.**
- **If it is not in the registry, it is not a result.**

## The join key

Every pre-report carries `hypothesis=<id>`, so the link is queryable both ways: which runs tested
H12, and why a run exists at all. The relation is many-to-many — one execution can answer several
hypotheses (EXP-010 recorded a 3-arm sweep as one entry). `registry.new_report` refuses without
it, and `tests/test_registry.py` pins that — an untagged record is invisible to the join and
nothing else notices, which is why it is a refusal and not a convention.

## Write the "don't run" condition too

A hypothesis that pins a symptom on something the optimiser or the model can absorb is usually
already answered. 「무엇이 답인가」 states the condition for running **and** the condition for
closing it unrun — the second is what makes a hypothesis cost nothing.

## One lane, one hypothesis

Two lanes on one hypothesis is a dispatch bug, not a merge bug: the single-writer property above
is what the automatic merge depends on. The orchestrator writes 질문 · 조건 · 무엇이 답인가
**before** dispatch, on `main`, so pre-registration is proved by commit order. How a lane is
dispatched and driven: `orca-parallel.md`.
