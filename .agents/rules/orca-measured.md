# Orca — what was measured, and against which build

The rules in `orca-parallel.md` are short because their evidence is here. Read this when a
dispatch misbehaves in a way the rules do not name, or **before trusting any of them against a new
Orca build** — every entry carries the version it was taken on, and a version bump invalidates
nothing automatically but re-opens everything.

## Placing a worker — 1.4.206, Windows, 2026-09-21

**The repo must be registered; the container does not count.** With only the orchestrator
container open as a workspace, `orca worktree list` showed one entry — the container — with
`branch: ""`, because it is not a git repository. Four calls, four failures, all `repo_not_found`:

| Call | Selector |
|---|---|
| `orchestration worker-start` | `--repo path:<abs, forward slashes>` |
| `orchestration worker-start` | `--repo path:<abs, backslashes>` |
| `orchestration worker-start` | `--repo name:ai-co-scientist` |
| `worktree create` | `--repo path:<abs, forward slashes>` |

So it is not a selector-format problem and the dispatch never reaches the lifecycle. Opening a
workspace on the child registered it (`repoId 50b59abe…`), after which `--repo id:<repoId>` placed
a worker on the first try.

**Hand-made worktrees are dispatchable but unlisted.** `git worktree add
../.worktrees/ai-co-scientist/lane-h12 -b feature/h12-smooth feature/lane-harness` produced a
working checkout that never appeared in `orca worktree list` — before or after use — yet
`worker-start --worktree "path:<it>"` reported `action: "reused"`, opened a terminal there, and the
round trip completed first try. The sibling repo's rule says a hand-made checkout leaves Orca with
no record and is therefore invisible to dispatch; on 1.4.206 that is **half true**: invisible to
`worktree list`, dispatchable by path.

**Orca-created worktrees land under `workspaceDir`.** The path is
`<base>/<repo dir basename>/<name>` and only `<base>` is configurable — Orca's `workspaceDir`
setting, here still the default `C:/Users/user/orca/workspaces`. The sibling repo points its at
`.worktrees`, which is why its rule describes `.worktrees/custflow-pipeline/<name>`. Nothing in
the repository controls this.

**Naming.** A worker started `--name lane-probe --display-name feature/lane-probe` reported its
git branch as `lane-probe`: the prefix reaches the pane label, not git.

**Setup.** `--setup run` with no hook configured reports `hookFound: false,
state: not_configured` and proceeds. The worktree then has no `.venv`; `uv run` built one on
demand in **484 ms** (18 packages, CPython 3.12.8), and the lane's pytest run finished in 1.86 s.

## The trust prompt eats the first injection — 1.4.206, 2026-09-21

A worktree created under `workspaceDir` was a folder Claude Code had never seen, so it opened
*"Is this a project you created or one you trust?"*. The start result carried
`turnStart: "permission"`; Orca injected the spec while the prompt was up; once a person answered,
the agent sat at an **empty prompt with no task**. The launch used
`--dangerously-skip-permissions`, which therefore does **not** cover workspace trust.

A worktree under the already-trusted projects tree (`.worktrees/…`) opened **no** prompt and ran
first try. The likely reason is an already-trusted parent; **unverified** — trust bookkeeping was
not inspected.

After `worker-abandon` and a re-dispatch into the same terminal, the retry reported
`turnStart: observed` and `worker_done` arrived in **~37 s**, carrying the worker's outcome
verbatim, collected by one `check --wait`.

**Unacknowledged batches replay.** While waiting on a second lane, `check` returned the *previous*
lane's `worker_done` again, because `--ack <deliveryId>` had never been passed. A coordinator that
skips the ack reads stale mail and concludes nothing new arrived.

## Codex cannot reach the Orca CLI — 1.4.206, Windows, 2026-09-21

Why a Codex lane must report through a file, and what the rule in `harness.md` rests on:

| Probe | Result |
|---|---|
| `$env:PATH` filtered for orca | the bin directory **is** there, twice, behind a Codex `arg0` shim dir |
| `where.exe orca` | `Could not find files for the given pattern(s).` |
| `Test-Path '<abs>\orca.exe'` | **`Access is denied`** → `False` |
| `& '<abs>\orca.exe' --version` | `CommandNotFoundException` |

Codex's config carries `[windows] sandbox = "elevated"` with `trust_level = "trusted"` on the
project roots and no entry for Orca's install directory. A first probe spent **5 min** retrying
the send while the coordinator saw `count: 0` on every bounded wait; a second, told not to attempt
it and given a path inside a trusted root, wrote a complete JSON report on the first try with no
human intervention.

## Measured on 1.4.197, not re-verified on 1.4.206

These cost a probe to learn and nothing since has contradicted them; they sit below a lifecycle
that has changed. Treat as likely-true, not confirmed.

- **The empty-inbox trap.** `inbox --terminal <handle>` read `count: 0` while `worker_done` mail
  already existed — it is addressed to `run:<id>`, not the terminal. Poll with `check` or
  `inbox --full`; never conclude silence from a `--terminal`-scoped query.
- **Delivery is pull-only.** A plain `send` to a running session sat unread (`delivered_at: null`)
  for 20 s; only injection reached the pane with zero keystrokes typed.
- **Screen vs stream.** For a Claude Code pane the **bare** `terminal read` had `source: screen`
  and returned 40 lines — the live TUI including the agent's answer; adding `--cursor`/`--limit`
  switched to `source: stream` and returned 3 lines of pre-TUI shell output, empty by
  construction. The bare read is the useful one. `worker-read --source terminal` exists on
  1.4.206; whether the same split holds under it is unmeasured.
- **Non-ASCII is mangled** (`?��`) in terminal reads — write a detector's match target in ASCII.
  This bites harder here than in the sibling repo, since this repo's specs are Korean.
- **`agent_prompt_blocked` is hook state, not a modal.** It persisted across a full agent restart
  with an idle composer, on a pane whose agent hook was failing.
- **`agentIdentity` in `terminal list` lags the pane** in both directions — do not branch on it.
- **Two dormant symptoms, causes unexplained:** calls returning `EPIPE`, or exiting 0 after only a
  handshake line (validate the JSON body, never the exit code); a terminal-handle variable no
  longer matching after a reconnect (resolve it via `terminal list` instead).
- **`ask` ↔ `reply`.** A timed-out `ask` leaves the question pending — resume with
  `ask --resume <message_id>`, never a fresh question. Do not paste literal CLI commands into a
  spec; the preamble supplies tokens a spec cannot know.
- **Mutations are idempotent.** A mutating call returns `mutation: {requestId, replayed}`;
  re-issue with `--retry-request <id>` instead of hand-rolled create-then-verify.
