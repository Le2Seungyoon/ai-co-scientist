# Orca — running experiments across parallel sessions

> **Orca-only**, measured on **1.4.206 / Windows, 2026-09-21**. Behaviour claims carry their own
> date; anything unmeasured says so, and the evidence behind the rules here — probe by probe —
> lives in `orca-measured.md`. Sub-agents stay the default: reach for a second *session* only when
> the work must outlive a turn, hold its own approval gate, or hold the GPU while this session
> keeps planning. Which execution class a dispatched session is, and what it may write:
> `architecture.md` → Extending the contract to a second Orca session.

## The lifecycle

```bash
git -C <repo> worktree add ../.worktrees/<repo>/<name> -b <branch> <base>    # prefer this
orca orchestration run-create --objective "<what this batch is for>" --json  # binds THIS terminal
orca orchestration worker-start --spec "<task spec>" --agent claude \
    --worktree "path:<that worktree>" --task-title "<title>" --json
orca orchestration check --wait --timeout-ms 45000 --ack <prior deliveryId> --json
orca orchestration worker-release --dispatch <dispatch_id> --json            # once it settles
```

`worker-start` launches the agent and injects the spec in one action — there is no separate "open
a terminal, then dispatch into it" step. `coordinator-start` is retired; the worker contract now
arrives as an Orca skill.

**The coordinator is a terminal, not a person.** `run-create` binds the terminal that runs it, so
every later `check` reads that Run. `$ORCA_TERMINAL_HANDLE` names it; after a reconnect, resolve
it from `orca terminal list` rather than trusting the variable.

**The repo must be registered with Orca** — a workspace opened on the repository itself. The
orchestrator container is not a git repository and does not count; until a workspace is open on
the child, every selector fails `repo_not_found`. **A bound Run proves nothing about placement**:
`run-create` succeeds either way.

**Prefer a hand-made worktree** under `.worktrees/<repo>/<name>`. Orca does not list it, but
`worker-start --worktree "path:<it>"` adopts it (`action: "reused"`) — and unlike a path under
Orca's own `workspaceDir`, it opens no workspace-trust prompt. Let Orca create one instead and it
lands at `<workspaceDir>/<repo dir basename>/<name>`; read the path out of the result rather than
assuming either layout. The git branch comes from `--name`; `--display-name` only labels the pane.
A worktree needs no setup hook for CPU work — `uv run` builds its venv on demand.

**Acknowledge each batch.** `check` without `--ack <deliveryId>` replays the previous one, so a
coordinator that skips it re-reads an old message and concludes nothing new arrived.

## When a new worktree eats the first injection

A folder the agent has never seen opens a workspace-trust prompt; the spec is injected while it is
up and **lost**; once a person answers, the agent sits at an empty prompt with no task.
`--dangerously-skip-permissions` does not cover workspace trust. This is why **a lane keeps its
worktree** — the same path is never asked twice — and why the hand-made path above earns its extra
command.

Recovery is a specific sequence, because the obvious calls refuse:

| Call | Result |
|---|---|
| `worker-release --dispatch <id>` | `dispatch_inactive` — only a **settled** worker can be released |
| `worker-stop --dispatch <id>` | `stop_unknown`, `processAction: none` — the terminal is `user_owned` once a person answered, and Orca will not close it |
| `worker-start … --terminal <handle>` (dispatch still active) | `agent_readiness`: *terminal already has an active dispatch* |
| **`worker-abandon --dispatch <id>`** | `abandoned` — fences it, warns that live resources were retained |

**Abandon, then re-dispatch into the same terminal.** From a coordinator bound to a different
worktree, `--terminal` alone is refused with `terminal_worktree_mismatch` — pass `--worktree`
alongside it.

## A Codex lane reports through the mailbox

A Codex lane here cannot execute `orca` at all (`harness.md` → Codex cannot reach the Orca CLI),
so it writes its message to a file and a relay outside the sandbox sends it:

```bash
uv run python .agent-hooks/orca_mailbox_relay.py --mailbox <dir> --interval 3   # coordinator side
```

**`--mailbox` has no default, deliberately.** One directory has to be shared by every lane and the
coordinator, so a path derived from `__file__` would break it per worktree — the flaw
`architecture.md` names for `registry.locked()`. It must also sit under a root the lane's sandbox
trusts, which is why a machine-level temp directory is wrong here and right in `locks.py`.
`runtime/lane-mailbox/` works: gitignored, so lane traffic never dirties the tree.

The lane writes one JSON object — `from`, `dispatch_capability`, `type`, `task_id`, `dispatch_id`
required, `subject`/`body`/`outcome`/`phase` optional — **atomically**, `.tmp` then rename, or the
relay reads a half-written message. All four ids come from the injected preamble; a spec must tell
the lane to copy them rather than invent them, and must say plainly **not** to call `orca`, since
the preamble instructs otherwise (→ What goes in a spec).

**Delivery is at-least-once and says so.** A message moves to `sending/` before the send and
`sent/` after; a crash in between leaves it visible in `sending/` rather than lost, and the relay
never retries on its own — a silent re-send would put a second `worker_done` on the bus and settle
a dispatch that had not finished. Recovering one is a person's call.

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
liveness verdict, and, when it needs something, the one action that unblocks it. Never re-enter a
wait silently. An unverifiable verdict means *unknown*, never *running*.

**When a bound wait expires, read the pane before waiting again.** Silence at the coordinator does
not mean a lane is still working: a Codex lane that had already finished could not send its report
at all, and the failure was visible on its screen minutes before any timeout suggested it
(`harness.md` → Codex cannot reach the Orca CLI).

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
quote the lane cannot verify, and a coordinator rule that makes its own relays authoritative is an
agent granting itself authority. The scope arrives as part of the task, never as a correction to it.

**Contradict the preamble where this machine contradicts it.** The injected contract tells every
lane to report with `orca orchestration send`; a Codex lane cannot reach that binary here, and
without a line in the spec saying so it will spend minutes obeying the preamble
(`harness.md` → Codex cannot reach the Orca CLI).

Put in the spec only the task, the domain limits, the file domain it owns, and the hypothesis file
it writes (`experiment-ledger.md`). **Do not copy the ledger in** — the branch point is already
the snapshot.
