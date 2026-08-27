# Git Workflow

> Remote: `https://github.com/Le2Seungyoon/ai-co-scientist.git` — host `github.com` (default
> branch `main`). `.gitignore` is in place. Solo repo; `gh` CLI is not installed, so PRs (if any)
> go through the web UI and the CLI gate lives on `git push`.

## Branch-First Rule

For any request that changes files, **before writing the first file**:

1. `git branch --show-current`
2. Ask the user: "You're on `<branch>`. Continue here? / new branch name? / update
   `main` first?"

Don't skip this even for "small" changes or doc edits.

## Branch Naming

- New feature: `feature/<name>` · bug fix: `fix/<name>`

## Protected Commands

Do not run without explicit confirmation: `git checkout`/`switch`, `git pull`/`push`, branch
creation (`checkout -b`, `branch`), `git reset`/`rebase`.

**Why prose and not `permissions.deny`**: this is a collaboration convention that protects a shared
repo — it doesn't hold in every context. A solo admin who authorizes direct git on their own
credentials for one session overrides this list, and that authorization wins. `permissions.deny` is
only for things absolute across every context (person · solo/team · time) — e.g. "never commit
secrets", "never force-push `main`". Context-dependent conventions stay here as prose.

## Commits

- Message style: **English**, **single-line subject** with the core change only — no
  bullet body, no verbose explanation. Goal: the change is **scannable at a glance** in history.
  (Matches existing history: "Fix depth clamping in baseline validation".)
- **No AI attribution** in commit messages — do NOT append `Co-Authored-By: …` or
  `🤖 Generated with …` trailers. This **overrides** the harness default.
  **Enforced:** a PreToolUse hook in `settings.json` denies `git commit` when the command contains
  a `Co-Authored-By` / `Generated with` trailer.
- Default: **stage + diff only, the developer runs `git commit`**. Exception: in a solo session
  where the admin has explicitly authorized direct git, Claude may commit during that session.

## Syncing main into a feature branch

`git fetch origin main && git merge origin/main`. On conflict, check
**which side deleted vs. modified** the file before resolving automatically:
- Deleted on `main`, modified on the branch → the modification usually wins; decide
  intent, then `git add <path>` to keep it (or `git rm <path>` if the deletion should stand).
- Deleted on the branch, modified on `main` → same judgment, reversed.
Don't resolve by reflex (`git checkout --ours/--theirs`) — a delete/modify conflict is almost
always an intent decision, not a textual one.

## Merge main before opening a PR (required)

1. Sync `main` into the feature branch (see "Syncing main" above) — resolve conflicts
   locally, not at PR time.
2. Run the verification suite (`uv run pytest -q` + `uv run ruff check src tests scripts`) and
   confirm it passes on the merged tree.
3. Confirm a clean working tree (`git status`) before opening the PR.

## Pull Requests

- **PR title/description: concise, no AI attribution**, no lengthy narrative — same reason as
  commits (scan the diff, not prose). A short summary + key changes + how it was verified is enough.

**PR-gate hook**: since `gh` is absent, the gate lives on `git push` — it blocks push when
`origin/main` isn't merged into the current branch (a `main` push passes since it's its own
ancestor). The hook fetches first, is a no-op before git init, and can't gate a web-UI PR — so
keep "merge `main` first" as a **convention** too.
