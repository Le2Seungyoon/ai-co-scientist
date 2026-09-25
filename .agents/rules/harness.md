# Changing the harness

Read this before touching anything that configures an agent: `AGENTS.md`, `CLAUDE.md`, this
directory, `.agent-hooks/`, `.claude/settings.json`, `.codex/config.toml`.

**More than one agent works in this repo.** Claude Code and Codex read the same instructions and
run the same hooks. A change that works in the harness you happen to be running is not finished —
it is half done, and the half you did not check is where the failures hide.

## The invariant

**One copy of the content and the logic; per-harness registration only.**

| Shared — never fork it | Per-harness — necessarily different |
|---|---|
| `AGENTS.md`, everything in `.agents/` | `CLAUDE.md` (one line, imports AGENTS.md) |
| `.agents/agents/**` — lane sources | `.claude/agents/`, `.codex/agents/` — **generated** |
| `.agents/skills/**` — skill sources | `.claude/skills/`, `.codex/skills/` — **generated** |
| the scripts in `.agent-hooks/` | `.claude/settings.json`, `.codex/config.toml` |
| the tests beside them | — |

## Where a thing lives

**Prose lives under `.agents/`. Harness directories hold registration and build output only.**
Name a rule file by its topic, never by a harness, and write the variants as adjacent paragraphs:

```markdown
**Claude Code —** `/context` lists the memory files that actually loaded.
**Codex —** `codex debug prompt-input` renders the same thing, with no model call.
```

Inline emphasis, not headings: a heading level collides with the file's own structure and with
anything that parses it. Both harnesses read both labels, which costs a little context and buys
the only thing that matters — **a rule and its variant cannot be read apart.**

If a whole file would be one harness's paragraphs, that is a signal rather than a layout: either
it belongs beside the neutral rule it modifies, or it is operational fact about launching and
debugging a harness.

**No `paths:` front-matter on a rules file.** Codex cannot read it and nothing here consumes it;
removed 2026-09-08. Path-scoping returns when the rules set outgrows one read, as a key the
generator reads and emits the `AGENTS.md` table from.

## `.claude/` paths in the record are historical

`.superpowers/**` and `docs/superpowers/plans/2026-09-04-agent-roster.md` name `.claude/rules/`
and `.claude/hooks/` because that is where those files were when that work happened. **They are
not fixed.** A record rewritten to match today's tree stops being a record. Everything a *live*
pointer names must resolve, and `check_rule_links.py` governs exactly that set — the record is
outside it on purpose.

## This machine has no `python3`

Measured 2026-09-08: `python` on PATH is 3.8.10; `python3` does not exist. Hook registrations
therefore resolve the interpreter (`command -v python3 || command -v python`) instead of naming
one, and say so on stderr when neither resolves rather than passing in silence.

Two consequences:

- **Scripts under `.agent-hooks/` must stay Python 3.8 compatible** — they run under the bare
  interpreter. No `tomllib`, no match statements.
- **Their tests may use 3.12** — run them with `uv run python`, which is the project venv.

## Codex cannot reach the Orca CLI on this machine

Measured 2026-09-21. Codex runs under `[windows] sandbox = "elevated"` with `trust_level`
granted to the project roots and **nothing for Orca's install directory**, so a dispatched Codex
lane cannot execute `orca` — by PATH or by absolute path. The file is on disk and a plain
PowerShell resolves it; Codex's own process cannot stat it. The four probes:
`orca-measured.md` → Codex cannot reach the Orca CLI.

**The consequence is not that a Codex lane is useless — it is that it cannot report the usual
way.** One read its spec, ran its commands, composed a correct `worker_done`, and spent five
minutes failing to send it while the coordinator saw only `count: 0`.

**So route it through the mailbox relay rather than lowering anyone's sandbox.** The lane writes
its message into a directory its sandbox already trusts and a relay outside the sandbox puts it on
the bus — measured 2026-09-22, and a relayed `worker_done` auto-completes its dispatch exactly as
a self-sent one would. Running it: `orca-parallel.md` → A Codex lane reports through the mailbox.
`sandbox_mode = "danger-full-access"` also works (`orca-measured.md`) but removes the sandbox from
**every** command that lane runs to buy the one thing the relay buys alone.

## `tools:` restricts a lane on Claude Code only

`tools.claude` is emitted for Claude Code and generates **nothing** for Codex: measured from the
installed binary, `[agents.*]` carries exactly `description` / `config_file` /
`nickname_candidates`, and there is no per-lane tool allowlist anywhere. So `reviewer` and
`researcher` are read-only *by mechanism* in one harness and *by instruction* in the other.

- **Do not invent a Codex spelling for it.** A `tools` line there would read as a restriction
  nothing enforces — worse than none.
- **Write every lane's domain limit into its body**, where both harnesses read it.
- The nearest real equivalent is `sandbox_mode = "read-only"` in the role file, deliberately
  unset because a sandbox that cannot start kills every shell command and nobody has measured
  whether it starts here. Ledger item 5.

**Until that item is closed, a Codex lane's domain compliance is not checked by anything.** Say
so rather than implying parity.

## Before you call a harness change done

```bash
uv run pytest -q                                     # runs the four below, plus the repo suite
uv run python .agent-hooks/build-agents.py --check    # AGENTS_FRESH — generated lanes and skills
uv run python .agent-hooks/test_harness_parity.py     # the two registrations agree
uv run python .agent-hooks/check_rule_links.py        # every pointer resolves
uv run python .agent-hooks/test-build-agents.py       # the generator itself
```

`uv run pytest -q` already calls all four (`tests/test_harness_generated.py`), so running them
separately is for a faster loop, never for extra assurance — two implementations of one judgment
prove nothing (`self-review.md` → Gates).

Three questions no command answers:

1. **Both registrations updated?** The schemas are unrelated and neither derives from the other.
   Changing one and not the other leaves that harness silently unprotected. The parity test
   catches a *name* mismatch, not a semantic one.
2. **Does the shared logic still handle the other harness's payload?** They differ — a Codex file
   edit arrives as `apply_patch` with the whole patch in `tool_input.command` and no `file_path`
   at all. Adding a branch for one is fine; replacing the other's is not.
3. **Did you verify, or assume?** Registration syntax cannot be tested offline. An unverified
   change is reported as unverified and goes in the ledger.

## Harness work fails silently — so verify the effect, not the input

**A thing that is configured but does nothing looks exactly like a thing that works.** The
distinction is *accepted* versus *resolved*: a key the parser tolerates but never acts on, a hook
registered but untrusted. Never conclude a layer works because nothing complained — make it *do*
something and watch that happen.

**Prove the check can fail before trusting it to pass.** A probe that cannot discriminate is
worse than none, because it manufactures confidence. Run the negative case first, and make sure
it is genuinely negative: a guard that passes because the condition it guards was absent has told
you nothing.

**Verify one step past where the change lives.** Extracting the right path is not the hook firing;
the hook firing is not the harness honouring its verdict.

**A registration is read at session start.** Editing `.claude/settings.json` mid-session does take
effect, but on the *next* tool call — so a hook that fails once right after an edit is not
necessarily broken. Re-run before diagnosing.

**"Harness-specific" and "unportable" are claims.** A label is what stops anyone reopening a
question, so check before applying one, and when a claim survives, write down what would falsify
it. What has not been measured on the Codex side lives in `docs/codex-verification-ledger.md`;
**do not build a rule on an entry that is still empty.**
