# Orca — running experiments across parallel sessions

> **Orca-only**, and measured on **Orca 1.4.197 / Windows, 2026-09-08** against a second live Claude
> Code session in this repo. Unmeasured claims say so — do not promote one without re-running it.
> Sub-agents stay the default (no Orca in the path); reach for a second *session* only when the work
> must outlive a turn, hold its own approval gate, or hold the GPU while this session keeps planning.

## The only lifecycle that works

```bash
orca orchestration run-create --objective "<what this batch of experiments is for>" --json
orca orchestration task-create --run <run_id> --task-title EXP-0NN-arm --spec "<spec>" --json
orca orchestration dispatch --task <task_id> --to <worker handle> --run <run_id> --inject --json
# worker runs; then, in the coordinator:
orca orchestration check --terminal <coordinator handle> --json     # worker_done lands here
```

Measured: dispatch→`worker_done` round trip **~40 s**, `injected: true`, and `worker_done`
**auto-completes both the dispatch and the task** — no `task-update` needed.

**Do not address a worker by its `term_…` handle.** That mailbox still accepts `send` and shows the
message in `inbox`, but `reply` is refused (`Legacy orchestration messages are inspect-only; no
reply was applied`) and the send itself warns `legacy_terminal_recipient` — not durable past the
terminal's life. A round trip cannot be built on it. Address `run:<id>` / `dispatch:<id>`.

## The empty-inbox trap — the one that will cost you a run

`worker_done` is addressed to `run:<run_id>`, **not** to the coordinator's terminal. So:

```
orca orchestration inbox --terminal <coordinator handle>   → count: 0    # mail exists, unseen
orca orchestration inbox --full        /  orchestration check           → count: 2    # here it is
```

Both were true at the same instant in the probe. `inbox` has **no `--run` flag**; the Run is taken
from this terminal's binding, and `--terminal` *narrows* it to a mailbox the lifecycle never uses.

**Rule: poll with `check` or `inbox --full`. Never conclude "the worker is silent" from a
`--terminal` query.** Cross-check with `dispatch-show --task <id>` — a `status: completed` there
with an empty inbox means you queried the wrong scope, not that the worker died.

## Delivery is pull-only; `--inject` is the only push

A plain `orchestration send` to a running session is **stored and never announced**: measured 30
polls over ~20 s with `read=0`, `delivered_at=null`, and the recipient's TUI cursor frozen. The
other session only acted once text was typed into it.

`dispatch --inject` is the exception — it delivers the preamble as input with **zero keystrokes**
(the worker reported so itself), and it reaches a Codex pane as readily as a Claude one. So a worker
never notices mid-task mail on its own: anything it must react to belongs in the **spec**, or
arrives as a fresh `--inject`. Silence stays unreadable — no `worker_done` means unknown, never
"fine". Preview what will be injected with `dispatch --dry-run --return-preamble`.

## Two-way mid-task: `ask` ↔ `reply`

Measured end to end. The worker blocks on `ask`; the question reaches the coordinator's Run mailbox
in ~3 s; `reply --id <msg_id>` unblocks it and the body arrives verbatim.

```bash
# worker (the preamble hands it the exact command — see below)
orca orchestration ask --from <worker handle> --question "<q>" --timeout-ms 300000
# coordinator
orca orchestration reply --id <question msg_id> --from <coordinator handle> --body "<answer>"
```

The reply is sent **from `run:<id>` to `dispatch:<id>`**, threaded on the question's id — another
reason coordinator polling must be `check` / `inbox --full`.

- **A timed-out `ask` leaves the question pending**; the worker resumes with
  `ask --resume <message_id>`, never a fresh question. An unanswered gate still means the worker
  finishes wrong — treat gate latency as urgent.
- **Do not paste literal CLI commands into a spec.** The probe's spec spelled out the `ask` line and
  the worker's first attempt **failed for a missing `--dispatch-capability`** — a token the preamble
  supplies and a spec cannot know. It recovered by using the preamble's version. Specs describe
  *what to ask*, never *how to call the CLI*.

## What `--inject` already tells the worker

The preamble is generated, so **do not restate it in the spec**: `worker_done` exactly once with an
`--outcome`; a heartbeat every 5 minutes; `ask` for questions; stop at idle once done. It also
carries the ban that matters most here — **a worker must never call `AskUserQuestion`**, because
that opens a local TUI prompt the coordinator cannot see or answer, and the session hangs forever.

Put in the spec only the task, the domain limits, and the file domain it owns.

## Mutations are idempotent now — do not hand-roll create-then-verify

Every mutating call returns `mutation: {requestId, replayed}` and accepts `--retry-request <id>`,
"only for exact recovery after an unknown mutation result". Re-issue with the returned id instead of
the old read-back-and-match-by-name dance. Blind re-running without `--retry-request` still
double-creates.

## Reading a worker's screen

Claude Code runs in the alternate screen buffer, so the two reads are **the opposite** of what a
plain terminal would give:

| Call | `source` | Measured |
|---|---|---|
| `terminal read --terminal <h>` | `screen` | 40 lines — the live TUI, including the agent's answer |
| `terminal read … --cursor 0 --limit 400` | `stream` | 3 lines — pre-TUI shell output only |

For a Claude Code worker the **bare read is the useful one**; the cursor/scrollback path is empty by
construction. Any prompt- or state-detector built on the scrollback finds nothing.

Non-ASCII is mangled to `?��` in both reads. **Write anything a detector must match in ASCII** —
Korean is fine for human-facing prose in the spec, never for a match target.

Before typing into a session (`terminal send`), confirm it is idle: `status: running`, no
`Do you want to` on screen, and `latestCursor` unchanged across two reads. A keystroke sent to a
working agent interrupts its command.

## When the agent-hook path is blocked (`agent_prompt_blocked`)

Submitting a prompt to an agent pane goes through Orca's agent hook; raw pty writes do not. Measured
on a Codex pane whose hook was failing (`Hook failed — hook exited with code 1` on every submit):

| Call | Result |
|---|---|
| `dispatch --inject` | `agent_prompt_blocked` |
| `terminal send --text <t> --enter` | `agent_prompt_blocked` |
| `terminal send --text <t>` then `terminal send --enter` (two calls) | both accepted; the prompt ran |

So **`agent_prompt_blocked` does not mean "a modal is on screen"** — it persisted across a full agent
restart with an idle composer. It is hook state. The two-call split is the fallback when a worker is
otherwise unreachable; it lost the leading token of the text once, so put nothing load-bearing first.

`agentIdentity` in `terminal list` lags the pane in both directions (showed `claude` for a running
Codex, and again after a Codex restart). Do not branch on it.

## Fitting this repo's invariants

Exclusivity, file domains, the pre-report and the shared branch are one contract, owned by
`architecture.md` → **Extending the contract to a second Orca session**. Ordering between dispatched
tasks would be `task-create --deps <json_array>` — **unmeasured**; until it is, serialize by not
dispatching the second task.

## Not measured — treat as open

Everything above was measured in one sitting on a healthy app; a fault that needs hours or a
reconnect to appear could not be. Open: `--deps`, heartbeat visibility, `terminal create` for a
*new* session, anything across worktrees or hosts, and a **Codex round trip** (injection reached it;
its account was rate-limited, so no reply was ever observed).

Two predecessors of this file's claims were historical failures that may simply be dormant:

- **A latching relay.** No `node relay.js` daemon exists in 1.4.197 and reads measured 20/20 at
  ~0.6 s — but the app had been up 5 minutes. If calls start returning `EPIPE`, or exiting 0 having
  printed only a handshake line, that is it returning: **validate the JSON body, never the exit
  code**, and reconnect the workspace between runs.
- **A stale `$ORCA_TERMINAL_HANDLE`.** It matched `terminal list` throughout, and rows now carry an
  `incarnationId` — but not across a reconnect. Resolve the handle from `terminal list`; one call.
