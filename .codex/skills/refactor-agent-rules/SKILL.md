---
name: refactor-agent-rules
description: Decide how to restructure an instruction file that has grown too long or too repetitive — an AGENTS.md, CLAUDE.md, or any rules file a size-budget check flagged. Judges each section against four remedies (relocate, split, abstract, compress) plus a check for rules that deterministic enforcement already covers, and reports its verdict before changing anything. Use this whenever a budget hook nudges about an instruction or rules file, and whenever someone asks to trim, tidy, deduplicate, reorganize, or clean up agent instructions — including when they only ask to "make it shorter", because shortening is the weakest of the four remedies and is usually the wrong one.
---

# Refactor Agent Rules

An instruction file is read in full every session it applies to. Length is therefore not a style
problem but a signal-dilution one, and the fix is a judgment about what the file is *for* — not
word-trimming.

## What this is not

Do not score the file against a template of sections a good document "should" have. Instruction
files differ by purpose — a routing index, a method, a checklist, a runbook — and one with no
"Commands" or "Architecture" section is usually correct, not deficient.

**The primary outputs are deletion and relocation.** A pass that produces mostly additions has
misread the task: the file was flagged for being long.

## Read the file whole, and read its neighbours

Two of the four remedies cannot be judged from the file alone. Before judging, gather four things
and name them in the report:

1. The target file, read end to end — not the lines just added. A nudge names a file, not a diff.
2. The sibling files and the index that routes readers to them — ① turns on what each is titled for.
3. Other instances of the same pattern, wherever they live — ③ cannot be judged from one copy.
4. The deterministic enforcement already in place — hooks, deny lists, CI checks, tests.

A verdict reached without these is guesswork, and it will land on ④ every time.

## The four remedies, in order

The order runs from *preserving* information to *destroying* it. Try them in it.

| | What happens to the content | |
|---|---|---|
| ① Relocate | Nothing lost — it moves | Safest |
| ② Split | Nothing lost — a boundary is added | Safe, but adds a file to route to |
| ③ Abstract | Examples are deleted deliberately | Highest leverage: the principle regenerates them |
| ④ Compress | Wording only | Buys least — these files are already dense |

The order matters because the untrained reflex is ④. Left alone, a reader shortens sentences —
the weakest move — while the section that belonged in another file stays exactly where it was.

**Each remedy has its own test. Ask the test, not "is this too long?"**

- **① Relocate — is this section's topic another existing file's title?**
  Move it there and leave a one-line pointer. Content ownership beats file convenience.
  **Pick the target by ownership, not by the index's one-line label** — prefer the file that already
  has a section on this sub-topic, or that an existing pointer names. An index describes a file's
  centre of gravity; what you are moving often belongs to an edge of it. Measured: of two readers,
  the one who followed a pointer the file already carried beat the one who matched the label.

- **② Split — is this a distinct sub-topic with its own trigger?**
  Give it its own file and register it in whatever index routes readers to files. Do not split to
  hit a number; a tight single-topic file slightly over budget is fine.

- **③ Abstract — are these N entries instances of one principle?**
  State the principle, then apply the deletion tests below to each entry it claims to cover.

- **④ Compress — whatever the first three did not claim.**
  Tighten prose, cut stale content.

## The parallel check: prune what enforcement already covers

Alongside the four, ask of every rule: **is this already enforced deterministically?** A rule that
a hook, a deny list, or a test already blocks is noise in prose, and two statements of one rule are
how contradictions start.

Delete the prose it replaced, keeping only what enforcement cannot express — the design intent
behind the ban. A hook can refuse an import; it cannot say why the import was banned.

## Before deleting, run two tests — then delete

Deletion is the point of this work, so a vague caution against it is worse than none: read as "do
not delete", it is how an instruction file grows without limit. Content that passes either test goes.

- **Does a stated principle regenerate it?** A concrete case that is one instance of a general form
  you have written down is already carried by that form.
- **Is the claim still evidenced elsewhere?** A date, a count, a measured case is the evidence for a
  rule — but only where that rule lives. If another file states the rule and carries its own
  evidence, this copy is duplication, and duplication is how two versions of one rule drift apart.

What fails both is a claim whose only support is the sentence asserting it. That stays.

## Report before changing anything

Produce the report, get approval, then apply. Splitting a file in particular is a person's call: a
report they can decline costs one turn, an unwanted reorganization costs a review.

```
## <file> — <current length> vs <budget>

Neighbours read: <sibling files>
Enforcement read: <hooks / deny lists / tests>

### <section> (<length>) → ① Relocate → <target file>
<one line: why that file owns this topic>

### <section> (<length>) → ③ Abstract
<the principle, stated>
<which entries it regenerates; which single example survives, and why>

### <rule> → Prune
<the hook or deny rule that already enforces it>

### Leave as is
<sections that are fine, and why — an over-budget file may still be mostly correct>
```

Then, per approved change, one diff and one reason:

```
File: <path>
Section: <where>

<diff>

Why: <one line — what a reader gains, or what the deletion stops duplicating>
```

The reason line is not decoration. An edit to an instruction file is itself a rule change, and a
rule asserted without evidence is an assumption.

**When a relocation also rewrote or shortened what it moved, say so on that line.** Compressing on
the way is usually right; a silent one is not reviewable — the diff shows text leaving one file, and
nobody can see that it never arrived in the other.

## What not to do

**Shortening in place when the section belongs elsewhere.**
Bad: squeezing a 12-line runtime section inside a review-method file down to 6.
Good: move all 12 lines to the runtime file, leaving `Launching and debugging: see <runtime file>.`
The review file gets shorter *and* the runtime content stops living where nobody looks for it.

**Blocking a deletion you cannot fault under either test.**
Bad: keeping a measured case because removing it "loses information", when the rule it evidences
lives in another file with its own evidence.
Every deletion looks like loss from inside the file. The tests are what separate loss from
duplication.

**Adding a section because the file lacks it.**
The file was flagged for being long.

## Generated files take none of the four

A file with a `generated by <cmd> — do not edit by hand` header is out of scope for every
remedy above, compression included: the next regeneration discards hand edits and compression
alike. The only lever is the generator — narrow its scope, or cap what it emits and have it
say how many entries were truncated.

In this repository that file is `docs/experiment-registry.md`, generated by
`scripts/exp.py render`. See `.agents/rules/enforcement.md` -> Keeping a generated artifact
alive.

## Language

Write the report and the edits in the language the file itself is written in.
