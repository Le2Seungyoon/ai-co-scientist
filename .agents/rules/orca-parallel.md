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

**Read `ok` before the payload.** A Run holds one active waiter; a second `check --wait` is
refused, and a parser that reaches straight for the count renders that refusal as "nothing has
arrived yet".

## A silent wait is the coordinator's failure

While the coordinator blocks, the person who asked sees nothing and cannot tell a running lane
from a stuck one. **Bound every wait and report at each expiry** — elapsed time, the lane's
liveness verdict, and, when it needs something, the one action that unblocks it. Never re-enter
a wait silently. An unverifiable verdict means *unknown*, never *running*.

**A permission prompt is the one thing waiting cannot resolve.** While one is open the session is
not merely unwatched but **blocked**: messages queue until its next tool round, which does not
come until a person answers. Surface it the moment it appears.

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

## Not measured — treat as open

Everything above is the CLI surface plus the sibling repo's measurements. **This repo has not yet
measured a round trip on 1.4.206.** Until it has, no claim here about delivery, injection or
worker lifetime is this repo's own.
