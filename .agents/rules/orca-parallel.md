# Orca — running experiments across parallel sessions

> **Orca-only**, CLI surface read on **1.4.206 / Windows, 2026-09-21**. Behaviour claims carry
> their own measurement date; anything unmeasured says so. Sub-agents stay the default — reach
> for a second *session* only when the work must outlive a turn, hold its own approval gate, or
> hold the GPU while this session keeps planning.

## The lifecycle

```bash
orca orchestration run-create --objective "<what this batch is for>" --json   # binds THIS terminal
orca orchestration worker-start --spec "<task spec>" --agent claude \
    --worktree new-top-level --repo id:<repoId> --base-branch <ref> \
    --name lane-<x> --display-name <branch> --setup run --json
orca orchestration check --wait --timeout-ms 45000 --json                     # collect
orca orchestration worker-release --dispatch <dispatch_id> --json             # after it settles
```

`worker-start` creates the worktree, launches the agent and injects the spec in one action —
there is no separate "open a terminal, then dispatch into it" step. `coordinator-start` is
retired; the worker contract now arrives as an Orca skill.

**The coordinator is a terminal, not a person.** `run-create` binds the terminal that runs it,
so every later `check` reads that Run. `$ORCA_TERMINAL_HANDLE` names it; after a reconnect,
resolve it from `orca terminal list` instead of trusting the variable.

## Reading the waiter

`check --wait` emits JSON keepalive lines on **stderr** every 15 s (`_keepalive`), which is how a
caller tells a live wait from a hung one. Filter them out when merging streams. `_heartbeat` is a
deprecated alias.

**Read `ok` before the payload.** Measured in the sibling repo (custflow-pipeline, 2026-09-16): a
Run holds one active waiter, and a second `check --wait` is refused — a parser that reaches
straight for the count renders that refusal as "nothing has arrived yet".

## A silent wait is the coordinator's failure

While the coordinator blocks, the person who asked sees nothing and cannot tell a running lane
from a stuck one. **Bound every wait and report at each expiry** — elapsed time, the lane's
liveness verdict, and, when it needs something, the one action that unblocks it. Never re-enter
a wait silently. An unverifiable verdict means *unknown*, never *running*.

**A permission prompt is the one thing waiting cannot resolve.** Measured in the sibling repo
(custflow-pipeline, 2026-09-16): while one is open the session is not merely unwatched but
**blocked** — messages queue until its next tool round, which does not come until a person
answers. Surface it the moment it appears.

## What goes in a spec

**Open with the approval scope.** State that the user approved this lane, list what is approved
and what is not, and tell the lane to route new questions through the preamble's `ask`. Measured
in the sibling repo (custflow, 2026-09-21, same worker-start and one variable): without the scope
both lanes **asked at their own window and sat idle**; with it both started within a minute.

**Approval relayed after the fact does not work, and the lane is right to refuse it** — it is a
quote the lane cannot verify, and a coordinator rule that makes its own relays authoritative is
an agent granting itself authority. The scope arrives as part of the task, never as a correction
to it.

Put in the spec only the task, the domain limits, the file domain it owns, and the hypothesis
file it writes (`experiment-ledger.md`). **Do not copy the ledger in** — the branch point is
already the snapshot.

## A Codex lane: injection and work land, the return leg does not — measured 2026-09-21

Dispatched into an **already-trusted** Codex pane on the main checkout (`--worktree path:<repo>`
plus `--terminal <handle>`), the start reported `turnStart: observed` on the first try — no trust
prompt, because that path was already trusted. The preamble and the spec both rendered in the
pane, and Codex **did the work**: it ran the three approved read-only commands and composed a
correct `worker_done` with the right `--dispatch-capability`, task id and dispatch id.

**It could not send it.** The call failed in Codex's own shell with
`'orca' ... is not recognized as a cmdlet, function, script file, or executable program`. The
lane then spun — `Working (4m 40s)` — with no way to report, and the coordinator saw only
`count: 0, timedOut: true` on every bounded `check --wait`. Silence at the coordinator meant a
**finished** lane, not a stalled one.

**The cause was measured, and it is not PATH.** Three probes, read back off the pane:

| Probe | Result |
|---|---|
| `$env:PATH -split ';'` filtered for orca | the bin directory **is** there, twice, behind a Codex `arg0` shim dir |
| `where.exe orca` | `INFO: Could not find files for the given pattern(s).` |
| `Test-Path '<abs>\orca.exe'` | **`Access is denied`** → `False` |
| `& '<abs>\orca.exe' --version` | `CommandNotFoundException` |

The file is on disk — the coordinator lists it, and a plain PowerShell started from the
coordinator's environment resolves it. Codex's own process cannot even `Test-Path` it. So **the
Orca CLI sits outside Codex's execution sandbox on this host**, and neither PATH nor an absolute
path reaches it. This is the Windows form of what the sibling repo hit on Linux, where Codex's
sandbox could not start at all: *a dispatched agent needs the coordinator CLI itself reachable,
and it lives outside the worktree.*

**Consequence: a Codex lane on this host is receive-only.** It reads its spec, does the work, and
composes a correct `worker_done` it can never send. Do not dispatch one expecting a report until
the sandbox grants that path. The operational rules:

- **Before trusting a Codex lane to report, prove `orca` runs in its shell.** One read-only probe
  whose entire task is `orca orchestration send --type heartbeat` is enough, and it costs seconds.
- **A worker that never reports is indistinguishable from one still working.** Bound the wait,
  and when it expires read the pane (`terminal read`) instead of waiting again — the failure was
  visible there minutes before any timeout would have suggested it.

Separately, that pane logged `Hook failed — hook exited with code 1` on every turn. The sibling
repo measured the same class of failure for untrusted Codex project hooks. It did **not** stop the
injection here: the spec arrived and was acted on. Hook state and dispatch delivery are
independent.

## Measured on 1.4.197 — not re-verified on 1.4.206

These cost a probe to learn and nothing on 1.4.206 contradicts them; they just sit below a
lifecycle that has since changed. Treat as likely-true, not confirmed.

- **The empty-inbox trap.** `inbox --terminal <handle>` read `count: 0` while `worker_done` mail
  already existed — it's addressed to `run:<id>`, not the terminal. Poll with `check` or
  `inbox --full`; never conclude silence from a `--terminal`-scoped query.
- **Delivery is pull-only.** A plain `send` to a running session sat unread
  (`delivered_at: null`) for 20 s; only injection reached the pane with zero keystrokes typed.
- **Screen vs stream.** For a Claude Code pane, the **bare** `terminal read` had `source: screen`
  and returned 40 lines — the live TUI, including the agent's answer; adding `--cursor`/`--limit`
  switched to `source: stream` and returned 3 lines of pre-TUI shell output only, empty by
  construction. The bare read is the useful one. `worker-read --source terminal` exists on
  1.4.206; whether this same split holds under it is unmeasured.
- **Non-ASCII is mangled** (`?��`) in terminal reads — write a detector's match target in ASCII;
  this matters more here than in the sibling repo, since this repo's specs are Korean.
- **`agent_prompt_blocked` is hook state, not a modal.** It persisted across a full agent restart
  with an idle composer, on a pane whose agent hook was failing.
- **`agentIdentity` in `terminal list` lags the pane** in both directions — do not branch on it.
- **Two dormant symptoms, causes unexplained:** calls returning `EPIPE`, or exiting 0 after only
  a handshake line (validate the JSON body, never the exit code); a terminal-handle variable no
  longer matching after a reconnect (resolve it via `terminal list` instead).
- **`ask` ↔ `reply`.** A timed-out `ask` leaves the question pending — resume with
  `ask --resume <message_id>`, never a fresh question. Do not paste literal CLI commands into a
  spec; the preamble supplies tokens a spec cannot know.
- **Mutations are idempotent.** A mutating call returns `mutation: {requestId, replayed}`;
  re-issue with `--retry-request <id>` instead of hand-rolled create-then-verify.

## Placing a worker — measured 2026-09-21, 1.4.206, Windows

**The repo must be registered with Orca, and the orchestrator container does not count.** Orca's
repo registry is keyed to opened workspaces. With only the container open, `orca worktree list`
showed one entry — the container — with `branch: ""`, because it is not a git repository. Every
attempt to name the child failed with `repo_not_found`: `--repo path:<abs>` in forward- and
backslash form, `--repo name:ai-co-scientist`, and `worktree create` with the same. Not a selector
format problem; the dispatch never reaches the lifecycle. Opening a workspace on the child
registered it, after which `--repo id:<repoId>` placed a worker first try. `run-create` binds and
succeeds either way, so a bound Run proves nothing about placement.

**The worktree path is `<base>/<repo dir basename>/<name>`, and only `<base>` is configurable.**
It is Orca's `workspaceDir` setting, which on this host is still the default
`C:/Users/user/orca/workspaces` — so a lane lands in Orca's own tree, not beside the repository.
The sibling repo points its `workspaceDir` at `.worktrees`, which is why its rule describes
`.worktrees/custflow-pipeline/<name>`. Nothing in the repo controls this; it is an app setting.
Read the path out of the `worker-start` result rather than assuming either layout.

**The git branch is `--name`; `--display-name` only labels the pane.** A worker started with
`--name lane-probe --display-name feature/lane-probe` reported its branch as `lane-probe` — the
prefix does not reach git on its own.

**`--setup run` with no hook configured is not a failure**: it reports
`hookFound: false, state: not_configured` and proceeds. The worktree then has no `.venv`, and
`uv run` builds one on demand — measured 484 ms for 18 packages, CPython 3.12.8. A CPU-only lane
needs no setup hook; a torch lane would.

## The first dispatch into a new worktree is eaten by the trust prompt

**Measured 2026-09-21.** A new worktree is a folder Claude Code has never seen, so it opens its
workspace-trust prompt ("Is this a project you created or one you trust?"). Orca injects the spec
while that prompt is up and **the injection is lost**: the start result carried
`turnStart: "permission"`, and once a person answered, the agent sat at an **empty prompt** with
no task. `--dangerously-skip-permissions` does **not** cover workspace trust. This is the same
shape the sibling repo measured for Codex hook trust (custflow-pipeline, its runtime rules),
now confirmed for
Claude Code — and it is why **a lane keeps its worktree**: the same path is not asked again.

Recovering from it is a specific sequence, because the obvious commands refuse:

| Call | Result |
|---|---|
| `worker-release --dispatch <id>` | `dispatch_inactive` — only a **settled** worker can be released |
| `worker-stop --dispatch <id>` | `stop_unknown`, `processAction: none` — the terminal became `user_owned` once a person answered the prompt, and Orca will not close it |
| `worker-start … --terminal <handle>` (dispatch still active) | fails at `agent_readiness`: *terminal already has an active dispatch* |
| **`worker-abandon --dispatch <id>`** | `abandoned` — fences it, warns that live resources were retained |

So: **abandon, then re-dispatch into the same terminal.** And when the coordinator is bound to a
different worktree, `--terminal` alone is refused with `terminal_worktree_mismatch` — pass
`--worktree` alongside it. The retry then reported `turnStart: observed` and the round trip
completed: **`worker_done` in ~37 s**, carrying the worker's outcome verbatim, collected by a
single `check --wait`.

