# Self-review

**Run this at the end of every task, before declaring done** — the same trigger as
`workflow.md` → Capturing Learnings, and the pass right after `workflow.md` → Task Completion.
Task Completion asks *did it work*; this asks *what did I decide that nothing can check*;
Capturing Learnings asks *what should outlive this task*.

**The reviewer should be checking your ruling, not discovering it.** So the output is written
down, not just thought: into the PR description when the task becomes a PR, into the completion
report when it does not. Work that never opens a PR still needs ②.

Where a PR is involved, run it **after** merging the default branch in and **before** you push
(`git-workflow.md` → Merge main before opening a PR) — gates run on the merged state or they
prove nothing.

## ① Gates — run them, paste the output

| Run | What it proves |
|---|---|
| `uv run pytest -q` | the offline suite passes with no API keys and no `.env` |
| `uv run ruff check src tests scripts` | lint clean (line-length 100) |
| `python .claude/scripts/check_rule_links.py` | every file a rules file or agent definition points at exists |
| zip check before `dacon_submit.py` | 25,988 files, every image max in {140,150,160,170}, 0.00 % outside |

There is no single command that runs all four — run them separately. The zip check applies only
when the task produced a submission.

Red here is not a review comment; fix it before you declare done. Where the same judgment exists
as both a hook and a repo-wide scan, **the scan is not redundant** — a hook sees this session's
edits only, so IDE-authored code arrives unchecked. But never run two *implementations* of one
judgment: if a test already calls the checker, running the checker separately proves nothing.

## ② Judgments — no gate can make these

**A gate proves a string exists, never that it is *true*.** These are the places the rules ask a
reviewer to judge something you already decided — so decide it in writing:

- **Reuse.** For each thing you added: does a shared home already hold one? If this is the
  *second* occurrence, promoting it was your job in *this* change (`coding-patterns.md` → When to
  extract). A counting gate denies the *third*; the second is yours alone.
- **Every escape-hatch marker your diff adds.** Quote each and say why the canonical route
  fails. The hook checks that a reason is *present*, never that it is *right*.
- **Every line you added to a list that makes a gate pass, and why** — allowlists,
  exempt-by-name sets, route/permission tables, stub lists. Such a line turns a red check green
  **without fixing anything**, and unlike a marker comment it reads as ordinary code in the
  diff. Fix by using the canon, never by an allowlist entry.
- **The calls the gates refuse to make.** Duplication checks match *copies*; something
  re-invented under another name, with a different guard, stays invisible — and that is the more
  dangerous kind. Name the nearest existing helper and why this is not it. Merging two? Diff the
  **guards** across the empty/zero/null grid, not the happy path.
- **Both halves of every deliberate mirror** — a contract and its test/doc/counterpart; a moved
  path and every glob that matched it; a README section and the rules file it mirrors.
- **What the green suite does not prove** (`testing.md`): name the environment gaps, and the
  mutation showing any fixture you added or widened actually discriminates.
- **`refactor/` only** — the prefix claims behavior does not change (`git-workflow.md` → Branch
  Naming). Name what proves it.

**A checklist answered "n/a" across the board is a smell, not a clean bill.** Say which items the
change could not have triggered and why, so the reader can tell *considered* from *skipped*.
