# Orca — running experiments across parallel sessions

> **Orca-only**, CLI surface read on **1.4.206 / Windows, 2026-09-21**. Behaviour claims carry
> their own measurement date; anything unmeasured says so. Sub-agents stay the default — reach
> for a second *session* only when the work must outlive a turn, hold its own approval gate, or
> hold the GPU while this session keeps planning.

## The lifecycle

```bash
orca orchestration run-create --objective "<what this batch is for>" --json   # binds THIS terminal
orca orchestration worker-start --spec "<task spec>" --agent claude \
    --worktree new-top-level --repo path:<repo> --base-branch <ref> \
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

## Measured on 1.4.197 — not re-verified on 1.4.206

These cost a probe to learn and nothing on 1.4.206 contradicts them; they just sit below a
lifecycle that has since changed. Treat as likely-true, not confirmed.

- **The empty-inbox trap.** `inbox --terminal <handle>` read `count: 0` while `worker_done` mail
  already existed — it's addressed to `run:<id>`, not the terminal. Poll with `check` or
  `inbox --full`; never conclude silence from a `--terminal`-scoped query.
- **Delivery is pull-only.** A plain `send` to a running session sat unread
  (`delivered_at: null`) for 20 s; only injection reached the pane with zero keystrokes typed.
- **Screen vs stream.** A Claude Code pane's bare terminal read returned only pre-TUI shell
  output (3 lines); the screen-rendered read returned the live TUI (40 lines, including the
  answer). Bears directly on `worker-read --source terminal` today.
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

## Not measured — treat as open

Everything above is the CLI surface plus the sibling repo's measurements. **This repo has not yet
measured a round trip on 1.4.206.** Until it has, no claim here about delivery, injection or
worker lifetime is this repo's own.
