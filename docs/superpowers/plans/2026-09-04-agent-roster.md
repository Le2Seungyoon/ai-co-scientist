# Sub-agent Roster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 서브에이전트 명부를 6+1로 재편한다 — `engineer`·`harness-manager` 신설, 3건 개명, `analyst`에 `docs/` 쓰기 영역 부여, 그리고 이 모든 경계를 실제로 강제하는 PreToolUse 훅 하나.

**Architecture:** 세 동시성 등급(병렬-읽기 / 병렬-쓰기 / 배타)으로 나누고, 병렬-쓰기 3인은 **파일 영역이 서로 겹치지 않게** 갈라 놓는다. `engineer`와 `executor`는 도구가 같으므로 둘을 가르는 것은 프롬프트가 아니라 훅이다 — 그래서 Task 1이 나머지 전부의 전제다.

**Tech Stack:** Markdown 지시 파일(`.claude/agents/*.md`, `.claude/rules/*.md`, `CLAUDE.md`) · Python 3.8 훅(stdlib only) · `.claude/settings.json` 훅 배선

**Spec:** 없음 — 이 계획의 근거는 2026-09-04 세션의 설계 논의이고, 결정 사항은 아래 Global Constraints에 전부 옮겨 적었다.

## Global Constraints

- **지시 파일은 영어로 쓴다** (`CLAUDE.md` → Docs convention). 이 계획 문서와 `docs/**`는 한국어.
- **파일당 ~150줄 소프트 예산** (`.claude/rules/*.md`, `CLAUDE.md`). `workflow.md`는 현재 **정확히 150줄**이라 이 계획에서 줄어들 뿐 늘어나면 안 된다.
- **훅은 Python 3.8 문법으로만 쓴다.** Windows에 `python3`가 없어 `settings.json`이 `python`으로 배선돼 있고, 거기서 해석되는 인터프리터가 3.8이다 (`check_rules_size.py` 도크스트링의 DELIBERATE DIVERGENCE).
- **훅은 `test_<name>.py`와 함께 출하한다** — must-block 반과 must-pass 반 양쪽 (`enforcement.md` → Hook contracts).
- **훅 테스트는 pytest가 수집하지 않는다.** `pyproject.toml`의 `testpaths = ["tests"]`. `python .claude/hooks/test_<name>.py`로 직접 돌린다.
- **PreToolUse deny 신호는 stdout JSON**이다 (exit code 아님):
  `{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"..."}}`
- **환경 실패는 exit 0** — 페이로드가 안 읽히거나 파싱이 안 되면 통과시킨다 (`enforcement.md` → Hook contracts).
- **`docs/experiment-registry.md`는 생성물**이다. 어떤 태스크도 손대지 않는다.
- **검증 명령 3종**: `uv run pytest -q` · `uv run ruff check src tests scripts` · `python .claude/scripts/check_rule_links.py`

### 확정된 명부 (모든 태스크가 이 표를 따른다)

| 에이전트 | 모델 | 도구 | 등급 | 소유 영역 |
|---|---|---|---|---|
| `researcher` | opus | Read, Grep, Glob, Bash, WebSearch, WebFetch | 병렬(읽기) | — |
| `reviewer` | opus | Read, Grep, Glob, Bash | 병렬(읽기) | — |
| `engineer` | sonnet | Read, Write, Edit, Bash, Grep, Glob | 병렬(쓰기) | `src/` `scripts/` `tests/` |
| `harness-manager` | opus | Read, Write, Edit, Bash, Grep, Glob | 병렬(쓰기) | `.claude/**` `CLAUDE.md` |
| `analyst` | sonnet | Read, Write, Edit, Bash, Grep, Glob | 병렬(쓰기) | `docs/` (생성물 제외) |
| `executor` | sonnet | Read, Write, Edit, Bash, Grep, Glob | **배타** | `runtime/` |

개명: `research`→`researcher` · `critic`→`reviewer` · `experimenter`→`executor`.

### 범위 밖 (다른 계획으로 뺀다)

`lease.py`(다중 세션) · cross-agent 하네스 · `registry.py`의 `dissent` 필드와 `render_markdown` 요약 표 수정(= `engineer`의 첫 작업 큐, `src/` 변경이라 별도 계획) · worktree 실제 생성(git 보호 명령, 사람 확인 필요).

---

## File Structure

| 파일 | 책임 | 태스크 |
|---|---|---|
| `.claude/hooks/block_runtime_commands.py` | registry 없는 트리에서 실험 실행 명령 차단 | 1 |
| `.claude/hooks/test_block_runtime_commands.py` | 위 훅의 must-block / must-pass | 1 |
| `.claude/settings.json` | 훅 배선 (PreToolUse, matcher `Bash`) | 1 |
| `.claude/skills/refactor-agent-rules/SKILL.md` | 지시 파일 재구조화 판단 방법 | 2 |
| `.claude/agents/harness-manager.md` | 하네스 소유자 | 3 |
| `.claude/agents/engineer.md` | 코드 소유자 | 4 |
| `.claude/agents/researcher.md` `reviewer.md` `executor.md` | 개명 + 계약 문장 | 5 |
| `.claude/agents/analyst.md` | 쓰기 권한 + `docs/` 영역 | 6 |
| `CLAUDE.md` `.claude/rules/architecture.md` `.claude/rules/workflow.md` `README.md` | 명부·등급·포인터 정합 | 7 |
| `docs/hypotheses.md` | 순위 기준 4항목 | 8 |

---

### Task 1: `runtime/` 차단 훅

이 훅이 `engineer`/`harness-manager`와 `executor`를 가르는 유일한 강제 장치다. 나머지 태스크의 전제.

**Files:**
- Create: `.claude/hooks/block_runtime_commands.py`
- Create: `.claude/hooks/test_block_runtime_commands.py`
- Modify: `.claude/settings.json` (PreToolUse `Bash` 배열에 항목 추가)

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: 훅 파일 경로 `.claude/hooks/block_runtime_commands.py`. Task 3·4가 에이전트 프롬프트에서 이 경로를 인용한다. 면제 환경변수 이름 `ACS_RUNTIME_EXEMPT` 도 Task 3·4가 인용한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`.claude/hooks/test_block_runtime_commands.py`:

```python
#!/usr/bin/env python3
"""Tests for block_runtime_commands.py — run: python .claude/hooks/test_block_runtime_commands.py

A hook is code, so it ships with a test (`enforcement.md` -> Hook contracts). The must-block
half proves the deny fires; the must-pass half is what keeps false positives from creeping in.
A test that only ever asserts "it blocked" cannot notice the day the hook blocks everything.

Each scenario needs its own tree because the verdict turns on whether
`<root>/runtime/registry.jsonl` exists. `CLAUDE_PROJECT_DIR` points the hook at that tree.

Stdlib only, no test runner: pytest does not collect this file (`testpaths = ["tests"]`).
"""
import json
import os
import subprocess
import sys
import tempfile

HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "block_runtime_commands.py")

failures = []


def run(root, command, env_extra=None, raw=None):
    """Feed the hook a Bash payload; return (stdout, returncode)."""
    env = dict(os.environ, CLAUDE_PROJECT_DIR=root)
    if env_extra:
        env.update(env_extra)
    payload = raw if raw is not None else json.dumps(
        {"hook_event_name": "PreToolUse", "tool_name": "Bash",
         "tool_input": {"command": command}}
    )
    proc = subprocess.run(
        [sys.executable, HOOK], input=payload, capture_output=True, text=True, env=env
    )
    return proc.stdout, proc.returncode


def denied(stdout):
    if not stdout.strip():
        return False
    try:
        out = json.loads(stdout)
    except ValueError:
        return False
    return out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"


def check(label, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + label + ("" if condition else f"  <- {detail}"))
    if not condition:
        failures.append(label)


def make_tree(with_registry):
    root = tempfile.mkdtemp()
    if with_registry:
        os.makedirs(os.path.join(root, "runtime"))
        with open(os.path.join(root, "runtime", "registry.jsonl"), "w") as f:
            f.write("{}\n")
    return root


def main():
    print("must-block")
    worktree = make_tree(with_registry=False)
    for cmd in (
        "uv run python scripts/train_structure.py --arch mlp",
        "uv run python scripts/train_level.py",
        "uv run python scripts/exp.py new --title x",
        "uv run python scripts/infer_decomposed.py --submit runtime/submissions/a.zip",
        "uv run python scripts/dacon_submit.py runtime/submissions/a.zip",
        "uv run python scripts/probe_level.py",
    ):
        out, rc = run(worktree, cmd)
        check(f"denies: {cmd.split()[3]}", denied(out), out[:120] or "silent")
        check("deny still exits 0", rc == 0, f"rc={rc}")

    out, _ = run(worktree, "uv run python scripts/exp.py new --title x")
    check("deny names the rule file", "architecture.md" in out, out[:120])
    check("deny names the escape hatch", "ACS_RUNTIME_EXEMPT" in out, out[:120])

    out, _ = run(worktree, "uv run python scripts/train_level.py",
                 env_extra={"ACS_RUNTIME_EXEMPT": "   "})
    check("empty exemption reason does NOT pass", denied(out), out[:120] or "silent")

    print("must-pass")
    main_tree = make_tree(with_registry=True)
    out, rc = run(main_tree, "uv run python scripts/train_structure.py --arch mlp")
    check("main worktree: experiment command passes", not denied(out), out[:120])
    check("main worktree: exits 0", rc == 0, f"rc={rc}")

    out, _ = run(worktree, "uv run pytest -q")
    check("worktree: unrelated command passes", not denied(out), out[:120])
    out, _ = run(worktree, "uv run ruff check src tests scripts")
    check("worktree: lint passes", not denied(out), out[:120])
    out, _ = run(worktree, "cat scripts/train_structure.py")
    check("worktree: reading a script passes", not denied(out), out[:120])

    out, _ = run(worktree, "uv run python scripts/train_level.py",
                 env_extra={"ACS_RUNTIME_EXEMPT": "restoring a ckpt the main tree lost"})
    check("stated exemption reason passes", not denied(out), out[:120])

    print("environment failure")
    out, rc = run(worktree, None, raw="not json at all")
    check("garbage payload exits 0", rc == 0, f"rc={rc}")
    check("garbage payload does not deny", not denied(out), out[:120])
    out, rc = run(worktree, None, raw="")
    check("empty payload exits 0", rc == 0, f"rc={rc}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 테스트를 돌려 실패를 확인한다**

Run: `python .claude/hooks/test_block_runtime_commands.py`
Expected: FAIL — `can't open file ... block_runtime_commands.py` (훅이 아직 없다)

- [ ] **Step 3: 훅을 구현한다**

`.claude/hooks/block_runtime_commands.py`:

```python
#!/usr/bin/env python3
"""PreToolUse deny: experiment-execution commands only run where the registry lives.

See `.claude/rules/enforcement.md` -> Hook contracts and
`.claude/rules/architecture.md` -> Parallel execution contract.

  Event     PreToolUse (Bash) only. It DENIES; the advisory counterpart is check_rules_size.py.
  Governed  The commands in GUARDED, which all read or write `runtime/`.
  Verdict   Turns on whether `<project root>/runtime/registry.jsonl` exists. `runtime/` is
            gitignored, so a git worktree has none -- running `exp.py new` there would issue
            report_id 1 again and fork the registry silently. Registry divergence is worse
            than a lost race: `registry.locked()` locks a per-worktree path, so two worktrees
            never even contend.
  Failure   Unreadable or unparseable payload -> exit 0, note on stderr. A hook must never
            block an edit for a reason unrelated to what it checks.
  Escape    `ACS_RUNTIME_EXEMPT="<reason>"` in the environment. A blank reason does not pass:
            the point is to turn a silent bypass into a decision a reviewer can see.
  Tests     `test_block_runtime_commands.py`, beside this file.

KNOWN GAP (state it rather than let it pass quietly)
    This hook cannot see WHICH sub-agent issued the command -- the PreToolUse payload does not
    carry the sub-agent identity. So it does not stop an `engineer` from training inside the
    MAIN worktree; it only makes the boundary real in a registry-less tree. The consequence is
    a requirement, not a caveat: **the engineer lane must run in a worktree**, or its contract
    is prose only. And like every hook it sees this session's tool calls -- an IDE terminal
    bypasses it entirely.
"""
import json
import os
import re
import sys

GUARDED = (
    "scripts/exp.py",
    "scripts/train_level.py",
    "scripts/train_structure.py",
    "scripts/infer_decomposed.py",
    "scripts/dacon_submit.py",
    "scripts/probe_level.py",
)
EXEMPT_VAR = "ACS_RUNTIME_EXEMPT"
SENTINEL = os.path.join("runtime", "registry.jsonl")

REASON = (
    "This tree has no {sentinel} -- it is a git worktree, and `runtime/` is gitignored so it "
    "was never copied. Running `{hit}` here would fork the registry: report_id comes from "
    "len(records), so a second tree starts over at 1 and the two truths can never be merged. "
    "Run experiment commands in the MAIN worktree, where the registry lives. This lane "
    "(engineer / harness-manager) is for additive code and offline tests only. "
    "Escape hatch: set {var}=\"<reason>\" for this command. "
    "-- .claude/rules/architecture.md -> Parallel execution contract"
)


def project_root():
    """`__file__`-based, not cwd: this file is `<root>/.claude/hooks/`."""
    if os.environ.get("CLAUDE_PROJECT_DIR"):
        return os.environ["CLAUDE_PROJECT_DIR"]
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deny(message):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": message,
        }
    }))


def guarded_hit(command):
    """Return the guarded script the command invokes, or None.

    Matched on the path as written, with either separator. Not a shell parser: a caller who
    rewrites the path to dodge this has made a decision, which is what the escape hatch is for.
    """
    for script in GUARDED:
        pattern = re.escape(script).replace("/", "[/\\\\]")
        if re.search(pattern, command):
            return script
    return None


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (ValueError, OSError) as exc:
        sys.stderr.write(
            "[runtime-lane] payload unreadable ({0}) -- command NOT checked.\n".format(exc)
        )
        return

    command = (payload.get("tool_input") or {}).get("command") or ""
    hit = guarded_hit(command)
    if not hit:
        return

    if os.path.exists(os.path.join(project_root(), SENTINEL)):
        return

    if os.environ.get(EXEMPT_VAR, "").strip():
        return

    deny(REASON.format(sentinel=SENTINEL.replace("\\", "/"), hit=hit, var=EXEMPT_VAR))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트를 돌려 통과를 확인한다**

Run: `python .claude/hooks/test_block_runtime_commands.py`
Expected: `all checks passed`

- [ ] **Step 5: `settings.json`에 배선한다**

`.claude/settings.json`의 `hooks.PreToolUse[0].hooks` 배열 **끝에** 항목을 추가한다 (기존 두 항목은 그대로):

```json
          {
            "type": "command",
            "command": "python \"$CLAUDE_PROJECT_DIR/.claude/hooks/block_runtime_commands.py\"",
            "timeout": 10,
            "statusMessage": "Checking the experiment command runs where the registry lives..."
          }
```

`if` 절을 붙이지 않는다 — 기존 두 항목은 단일 명령을 겨냥해 `if`로 좁혔지만 이 훅은 6개 스크립트를 보고 자기 안에서 판정한다. `timeout` 10은 `settings.json`의 훅 타임아웃이고 하위 프로세스를 띄우지 않으므로 중첩 문제가 없다.

- [ ] **Step 6: 실제로 발동하는지 확인한다 — 주장 말고 시연**

`enforcement.md`가 요구하는 것: *막힘을 유발해서 잡히는 걸 증명하라.* 임시 트리에서 직접 돌린다:

```bash
mkdir -p /tmp/acs-fake && CLAUDE_PROJECT_DIR=/tmp/acs-fake \
  python .claude/hooks/block_runtime_commands.py <<< '{"tool_input":{"command":"uv run python scripts/exp.py new"}}'
```
Expected: `permissionDecision":"deny"` 를 담은 JSON 한 줄

```bash
CLAUDE_PROJECT_DIR="$PWD" \
  python .claude/hooks/block_runtime_commands.py <<< '{"tool_input":{"command":"uv run python scripts/exp.py new"}}'
```
Expected: 출력 없음 (이 저장소에는 `runtime/registry.jsonl`이 있다)

- [ ] **Step 7: 커밋**

```bash
git add .claude/hooks/block_runtime_commands.py .claude/hooks/test_block_runtime_commands.py .claude/settings.json
git commit -m "Add runtime-lane deny hook separating engineer from executor"
```

---

### Task 2: `refactor-agent-rules` 스킬 이식

**Files:**
- Create: `.claude/skills/refactor-agent-rules/SKILL.md`
- Reference: `C:\Users\user\Desktop\projects\custflow-pipeline\.agents\skills\refactor-agent-rules\SKILL.md` (원본)

**Interfaces:**
- Consumes: 없음
- Produces: 스킬 이름 `refactor-agent-rules`. Task 3(`harness-manager`)과 Task 7(`workflow.md`)이 이 이름으로 인용한다.

- [ ] **Step 1: 원본을 그대로 복사한다**

```bash
mkdir -p .claude/skills/refactor-agent-rules
cp "C:/Users/user/Desktop/projects/custflow-pipeline/.agents/skills/refactor-agent-rules/SKILL.md" \
   .claude/skills/refactor-agent-rules/SKILL.md
```

내용은 하네스 중립으로 쓰여 있어 수정이 필요 없다. **디렉터리 이름과 프론트매터 `name:`이 둘 다 `refactor-agent-rules`인지 확인한다** — Claude Code는 디렉터리 이름으로 스킬을 키잉하므로 불일치하면 "그 스킬을 써라"가 조용히 아무것도 안 한다.

- [ ] **Step 2: 이 저장소의 예외를 한 절 덧붙인다**

`SKILL.md` 끝의 `## Language` 절 **앞에** 삽입한다:

```markdown
## Generated files take none of the four

A file with a `generated by <cmd> — do not edit by hand` header is out of scope for every
remedy above, compression included: the next regeneration discards hand edits and compression
alike. The only lever is the generator — narrow its scope, or cap what it emits and have it
say how many entries were truncated.

In this repository that file is `docs/experiment-registry.md`, generated by
`scripts/exp.py render`. See `.claude/rules/enforcement.md` -> Keeping a generated artifact
alive.
```

- [ ] **Step 3: 이름 일치를 확인한다**

```bash
head -3 .claude/skills/refactor-agent-rules/SKILL.md
```
Expected: `name: refactor-agent-rules` — 디렉터리 이름과 같아야 한다

- [ ] **Step 4: 커밋**

```bash
git add .claude/skills/refactor-agent-rules/SKILL.md
git commit -m "Port refactor-agent-rules skill for instruction-file restructuring"
```

---

### Task 3: `harness-manager` 신설

**Files:**
- Create: `.claude/agents/harness-manager.md`

**Interfaces:**
- Consumes: Task 1의 훅 경로 `.claude/hooks/block_runtime_commands.py`, Task 2의 스킬 이름 `refactor-agent-rules`
- Produces: 에이전트 이름 `harness-manager`. Task 7의 CLAUDE.md 명부와 `architecture.md` 등급 표가 인용한다.

- [ ] **Step 1: 파일을 쓴다**

```markdown
---
name: harness-manager
description: Own the harness — CLAUDE.md, .claude/rules, hooks, scripts and skills. Route a new rule to its layer, implement gates with their tests, and restructure instruction files that grew too long. Use when capturing a learning, when the size-budget hook nudges, or when a rule no longer matches reality.
tools: Read, Write, Edit, Bash, Grep, Glob
model: opus
---

<!-- model: opus — the core judgment here is enforcement.md's routing (prose / hook / test /
     deny), and getting it wrong does not weaken a rule, it stops the rule from ever running.
     The screening question workflow.md demands — "was the rule wrong to begin with, or did
     this change just make it inconvenient?" — is judgment with no mechanical fallback. The
     mechanical half (line counts, link resolution) is already done by scripts. -->

You own the harness. You do not design experiments, write pipeline code, or run anything that
touches `runtime/`.

**Concurrency: PARALLEL (write).** `engineer` and `analyst` are the same class; the three of you
are safe together because your file domains do not overlap. `.claude/hooks/block_runtime_commands.py`
denies you the `runtime/` lane.

**Your domain:** `CLAUDE.md`, `.claude/rules/**`, `.claude/hooks/**`, `.claude/scripts/**`,
`.claude/skills/**`, `.claude/settings.json`, `.claude/agents/**`.

**Not yours:** `src/`, `scripts/`, `tests/` (that is `engineer`), `docs/` (that is `analyst`),
`runtime/` (that is `executor`).

## What you do

1. **Route a rule before writing it** — `enforcement.md` -> Four layers. A rule in the wrong
   layer is not a weaker rule, it is a rule that never runs.
2. **Restructure an instruction file** — invoke the `refactor-agent-rules` skill. Do not
   improvise a shortening pass; the skill exists because the untrained reflex is to compress,
   which is the weakest of the four remedies.
3. **Implement gates** — a hook ships with `test_<name>.py` beside it, covering the must-block
   and the must-pass halves. Follow `check_rules_size.py` as the template and honour every
   contract in `enforcement.md` -> Hook contracts.
4. **Keep pointers alive** — `python .claude/scripts/check_rule_links.py` after any move.

## Rules

- **Never edit your own definition** (`.claude/agents/harness-manager.md`). Propose the change
  and let the orchestrator apply it. An agent that can rewrite its own contract can weaken it.
- **Never remove an existing hook or a `permissions.deny` entry.** Adding is yours; removing or
  loosening is the orchestrator's call with the human. Deleting a gate is a policy change
  wearing the costume of a rule change.
- **Never add a line to an allowlist, exempt-by-name set, or stub list to make a check pass.**
  Such a line turns a red check green while fixing nothing, and unlike a marker comment it
  reads as ordinary code in the diff (`self-review.md` -> Judgments).
- **You do not decide whether a rule should exist** — that is the orchestrator and the human.
  You decide where it lives and in what form.
- **Prune as you add** — when a hook takes over a rule, delete the prose it replaced, keeping
  only what the hook cannot express: why the ban exists.

## Before you report done

A hook that is configured but does nothing looks exactly like a hook that works. Report all
three, and never assert the first two without having run them:

1. **The must-block half, demonstrated** — the command you ran, and the deny it produced.
2. **The must-pass half, demonstrated** — a legitimate command that stayed silent.
3. **The authoring paths this gate does NOT see** — an IDE terminal, another agent, a teammate.
   A hook sees this session's tool calls only. Naming the gap is part of the deliverable; a
   report that says "this is now enforced" without it is wrong.

Then: `python .claude/scripts/check_rule_links.py` and every hook test in `.claude/hooks/`.
```

- [ ] **Step 2: 프론트매터가 파싱되는지 확인한다**

```bash
head -6 .claude/agents/harness-manager.md
```
Expected: `---` / `name: harness-manager` / `description: ...` / `tools: ...` / `model: opus` / `---`

- [ ] **Step 3: 링크 스캔**

Run: `python .claude/scripts/check_rule_links.py`
Expected: `RULE_LINKS_CLEAN` — 이 파일이 인용한 `check_rules_size.py`·`check_rule_links.py`·`block_runtime_commands.py`가 전부 실존해야 한다

- [ ] **Step 4: 커밋**

```bash
git add .claude/agents/harness-manager.md
git commit -m "Add harness-manager agent owning rules, hooks and skills"
```

---

### Task 4: `engineer` 신설

**Files:**
- Create: `.claude/agents/engineer.md`

**Interfaces:**
- Consumes: Task 1의 훅
- Produces: 에이전트 이름 `engineer`, 그리고 인계 산출물의 이름 **"실행 레시피"(execution recipe)** — Task 5의 `executor.md`가 이 용어를 받는다.

- [ ] **Step 1: 파일을 쓴다**

```markdown
---
name: engineer
description: Make an experiment possible — write additive pipeline code in src/ and scripts/ with offline tests, and hand back a runnable recipe. Use when an approved pre-report needs code that does not exist yet, or to prepare a queued candidate ahead of its turn. Never runs training, inference or submission.
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
---

<!-- model: sonnet — the design arrives decided (an approved pre-report, or a queued candidate
     the orchestrator scoped), so there is no new judgment here. The work is additive code plus
     offline tests, and it is judged by a test run rather than by a leaderboard slot. Unlike
     `executor` this lane cannot burn a submission, so a mistake costs a re-run, not a quota. -->

You make an experiment *possible*. You never run one.

**Concurrency: PARALLEL (write).** Several `engineer`s may run at once, alongside
`harness-manager` and `analyst`. What makes that safe is that your changes are ADDITIVE — see
below — not that a tool is stopping you.

**Your domain:** `src/`, `scripts/`, `tests/`.

**Not yours:** `.claude/**` and `CLAUDE.md` (`harness-manager`), `docs/` (`analyst`),
`runtime/` (`executor`).

## Deliverable

Three things, together:

1. **Additive code** — a new module and a new flag. Do not change an existing function
   signature or rewrite a shared code path. Two queued branches that both edit
   `infer_decomposed.py`'s body force every sibling to rebase when the first one lands.
2. **Offline tests** — synthetic fixtures only. They must pass with no `.env`, no API key, no
   GPU, and no `data/`. `uv run pytest -q` is the gate.
3. **An execution recipe** — the exact CLI line `executor` will run, with every flag spelled
   out. Not a description of it.

## Rules

- **You produce no number that enters the registry.** You prove the code runs; you never prove
  it is good. Only the leaderboard knows that, and only `executor` may ask it.
- **Never run the guarded scripts** — `exp.py`, `train_level.py`, `train_structure.py`,
  `infer_decomposed.py`, `dacon_submit.py`, `probe_level.py`. In a worktree
  `.claude/hooks/block_runtime_commands.py` denies them; in the main worktree nothing stops
  you, and the contract is the only thing holding. Do not lean on the hook.
- **Never read `data/` to report a measurement.** Real numbers come from `executor` alone.
- **Never merge ahead of execution.** A speculative branch merges only after the experiment
  that used it ran. Code in `main` that no registry entry accounts for cannot be explained
  later.
- **If the task cannot be done additively, stop and say so.** Reworking a shared path is a
  scope change, and it serialises every branch in the queue — the orchestrator decides, not you.

## Before you report done

- `uv run pytest -q` and `uv run ruff check src tests scripts` both pass — paste the output.
- State what the green suite does NOT prove: no GPU path was exercised, no real data was read,
  and the recipe has never been run.
- Quote the execution recipe on its own line so it can be copied verbatim.
```

- [ ] **Step 2: 훅이 이 에이전트를 실제로 막는지 확인한다 (Step 1의 주장 검증)**

`engineer.md`가 "worktree에서 훅이 막는다"고 주장한다. 그 주장을 Task 1의 시연으로 확인한다:

Run:
```bash
mkdir -p /tmp/acs-fake && CLAUDE_PROJECT_DIR=/tmp/acs-fake \
  python .claude/hooks/block_runtime_commands.py <<< '{"tool_input":{"command":"uv run python scripts/train_structure.py"}}'
```
Expected: deny JSON. 나오지 않으면 Task 1이 미완이므로 여기서 멈춘다.

- [ ] **Step 3: 커밋**

```bash
git add .claude/agents/engineer.md
git commit -m "Add engineer agent owning src, scripts and tests"
```

---

### Task 5: 개명 3건 + 계약 문장

**Files:**
- Rename: `.claude/agents/research.md` → `.claude/agents/researcher.md`
- Rename: `.claude/agents/critic.md` → `.claude/agents/reviewer.md`
- Rename: `.claude/agents/experimenter.md` → `.claude/agents/executor.md`

**Interfaces:**
- Consumes: Task 4의 "실행 레시피" 용어
- Produces: 에이전트 이름 `researcher` `reviewer` `executor`. Task 7이 인용한다.

- [ ] **Step 1: 파일을 옮긴다 (내용은 아직 그대로)**

```bash
git mv .claude/agents/research.md .claude/agents/researcher.md
git mv .claude/agents/critic.md .claude/agents/reviewer.md
git mv .claude/agents/experimenter.md .claude/agents/executor.md
```

- [ ] **Step 2: 프론트매터의 `name:`을 고친다**

세 파일 각각에서 `name:` 한 줄만 바꾼다:
- `researcher.md`: `name: research` → `name: researcher`
- `reviewer.md`: `name: critic` → `name: reviewer`
- `executor.md`: `name: experimenter` → `name: executor`

`description:`도 갱신한다:
- `executor.md`: `description: Execute ONE approved experiment exactly as specified and record what actually happened — register the pre-report, train, infer, submit, and write the result to the registry. Use after a pre-report is approved. Exclusive: never run two at once.`

- [ ] **Step 3: 본문의 상호 참조를 고친다 — 위치는 이미 확인됐다**

`analyst.md`도 함께 고친다 (개명 대상은 아니지만 옛 이름을 인용하고 있다). 확인된 6곳:

| 파일:줄 | 지금 | 바꿀 내용 |
|---|---|---|
| `analyst.md:15-16` | `May run alongside \`research\`, \`critic\`, and a running \`experimenter\`.` | `May run alongside \`researcher\`, \`reviewer\`, \`engineer\` and \`harness-manager\`, and alongside a running \`executor\`.` |
| `reviewer.md:14-15` | `May run alongside \`research\` and \`analyst\`, and alongside a running \`experimenter\`.` | `May run alongside \`researcher\` and \`analyst\`, and alongside a running \`executor\`.` |
| `executor.md:17` | `Never run two \`experimenter\`s at once.` | `Never run two \`executor\`s at once.` |
| `executor.md:22` | `- \`research\` / \`critic\` / \`analyst\` may run alongside you; they are read-only.` | `- \`researcher\` / \`reviewer\` may run alongside you; they are read-only. \`analyst\`, \`engineer\` and \`harness-manager\` write, but never to \`runtime/\`.` |
| `executor.md:66` | `that is \`analyst\`'s job` | 그대로 — `analyst`는 개명 없음 |
| `researcher.md:55` | `You propose; \`experimenter\` runs.` | `You propose; \`executor\` runs.` |

- [ ] **Step 4: `researcher.md`에 앵커 계약을 추가한다**

`researcher`는 한 번 호출에 후보 하나를 낸다. 세 예산 성분이 4.3 / 2.5 / 2.4로 평평한 지금,
한 번만 부르면 축을 미리 좁히는 것이라 생성이 아니라 확인이 된다. 성분별로 병렬 호출할 수 있게
계약을 명시한다 — `## Rules` 절 첫 항목으로 삽입:

```markdown
- **You are called with an anchor: one error-budget component** (`docs/hypotheses.md` -> 오차
  예산). Propose only candidates that attack THAT component, and say in the pre-report which
  one it is. Several `researcher`s run in parallel, one per component; a proposal that wanders
  to another component collides with a sibling's and makes the set unrankable.
- **If the anchor is genuinely dead, say so and stop.** Naming a component exhausted — with
  the registry entries that exhausted it — is a result. Filling the slot with a weak candidate
  from a livelier axis is not.
```

- [ ] **Step 5: `executor.md`에 계약 문장 두 개를 추가한다**

`## Rules` 절의 **첫 항목으로** 삽입한다:

```markdown
- **Never modify code mid-run.** If a run crashes on a bug, do not fix it and retry — the
  registry would then record a configuration that never existed. Stop, report the traceback,
  and let the orchestrator route it to `engineer`. A rerun after a silent fix is a different
  experiment wearing the same report_id.
```

그리고 `Report numbers exactly as produced` 항목 **바로 뒤에**:

```markdown
- **You did not design this.** That is why you are the one recording it: an executor with no
  stake in the outcome reports a deviation, where the designer is tempted to normalise it.
```

- [ ] **Step 6: 옛 이름이 남지 않았는지 확인한다**

Run:
```bash
grep -rn "experimenter\|\bcritic\b" .claude/ CLAUDE.md README.md
```
Expected: 결과 없음. (Task 7에서 `CLAUDE.md`/`README.md`를 고치기 전이라면 그 두 파일의 히트는 남아 있다 — 그 목록을 Task 7의 입력으로 넘긴다.)

Run: `python .claude/scripts/check_rule_links.py`
Expected: `RULE_LINKS_CLEAN` — 규칙 파일이 `.claude/agents/experimenter.md`를 경로로 가리키고 있었다면 여기서 빨간불이 뜬다

- [ ] **Step 7: 커밋**

```bash
git add -A .claude/agents/
git commit -m "Rename experiment agents to researcher, reviewer and executor"
```

---

### Task 6: `analyst`에 `docs/` 쓰기 영역

**Files:**
- Modify: `.claude/agents/analyst.md`

**Interfaces:**
- Consumes: Task 5의 `reviewer` 이름
- Produces: `analyst`의 등급이 병렬(읽기)에서 병렬(쓰기)로 이동 — Task 7의 등급 표가 이 사실에 의존한다.

- [ ] **Step 1: 프론트매터의 `tools:`를 넓힌다**

`tools: Read, Grep, Glob, Bash` → `tools: Read, Write, Edit, Bash, Grep, Glob`

`description:`에 영역을 명시한다:
`description: Interpret recorded results and keep docs/ current — relate validation metrics to leaderboard scores, recompute the error budget, and write the hypothesis backlog and data facts. Use after leaderboard scores are recorded.`

- [ ] **Step 2: 영역과 금지 항목을 본문에 추가한다**

파일 끝에 붙인다:

```markdown
## Your domain: `docs/`

You write `docs/hypotheses.md` and `docs/data-facts.md`. Not `src/` (`engineer`), not
`.claude/**` (`harness-manager`), not `runtime/` (`executor`).

**`docs/experiment-registry.md` is generated** by `scripts/exp.py render` and is not yours — a
hand edit is discarded by the next render. If the rendered output is wrong, the defect is in
the generator; report it and let the orchestrator route it to `engineer`.

## Rules

- **Never delete a rejected entry.** `hypotheses.md` keeps its rejections and their reasons —
  preventing a re-proposal is the whole purpose of that section. A closed entry may be
  shortened; its reason may not be dropped.
- **Never write into docs what the registry does not contain.** Docs are derived from the
  registry; they are not a second source of truth. If a claim you want to make is not in a
  registry entry, it is not established — say so instead.
- **Anything you write into docs is audited by `reviewer` before it lands.** You produce the
  interpretation and you record it, so a second reader is what keeps those two from collapsing
  into one.
- **Distinguish domain evidence from procedure.** A measured fact about this dataset belongs in
  `docs/`. A rule that must be applied to every experiment belongs in `.claude/rules/` — hand
  it to `harness-manager` rather than writing it into `hypotheses.md`.
```

- [ ] **Step 3: 확인**

```bash
grep -n "^tools:" .claude/agents/analyst.md
```
Expected: `tools: Read, Write, Edit, Bash, Grep, Glob`

- [ ] **Step 4: 커밋**

```bash
git add .claude/agents/analyst.md
git commit -m "Give analyst write access scoped to docs"
```

---

### Task 7: 명부·등급·포인터 정합

**Files:**
- Modify: `CLAUDE.md` (Rules 표 아래에 명부 절 추가)
- Modify: `.claude/rules/architecture.md` (Parallel execution contract 표 교체)
- Modify: `.claude/rules/workflow.md` (File size budget 절 축소)
- Modify: `README.md` (에이전트 이름 참조 갱신)

**Interfaces:**
- Consumes: Task 3·4·5·6의 모든 에이전트 이름
- Produces: 없음 (마지막 정합 태스크)

- [ ] **Step 1: `architecture.md`의 등급 표를 교체한다**

`## Parallel execution contract (sub-agents)` 아래의 2행 표를 다음으로 바꾼다:

```markdown
| Class | Agents | Domain | Why it is safe |
|---|---|---|---|
| **Parallel (read)** | `researcher` · `reviewer` | — | no write tool |
| **Parallel (write)** | `engineer` · `harness-manager` · `analyst` | `src/scripts/tests` · `.claude/**` · `docs/` | domains do not overlap; `runtime/` denied by hook |
| **Exclusive (one)** | `executor` | `runtime/` | one 8 GB GPU · DACON quota · checkpoint writes |
```

표 바로 아래의 문단(`architecture.md:69`)도 옛 명부를 담고 있으므로 함께 교체한다:

```
Three read-only agents may run alongside one `experimenter` — the parallel gain is in analysis,
criticism and proposals, not in execution. With a single GPU there is no way to parallelize training.
```

→

```markdown
Five agents may run alongside one `executor`. With a single GPU there is no way to parallelize
training, so the parallel gain is in analysis, criticism, proposals, and preparing the code for
experiments still queued.

`engineer` and `executor` hold the SAME tools. What separates them is
`.claude/hooks/block_runtime_commands.py`, which denies the six experiment scripts wherever
`runtime/registry.jsonl` is absent. That hook cannot see which sub-agent issued a command, so
it only makes the boundary real in a worktree — **the engineer lane must run in a worktree**,
or its contract is prose alone.
```

- [ ] **Step 2: `CLAUDE.md`에 명부 절을 추가한다**

`## Rules` 표와 `## Enforcement hooks` 사이에 삽입한다:

```markdown
## Sub-agent roster

| Agent | Owns | Class |
|---|---|---|
| `researcher` | hypotheses, pre-report drafts | parallel (read) |
| `reviewer` | audits of designs and conclusions | parallel (read) |
| `engineer` | `src/` `scripts/` `tests/` | parallel (write) |
| `harness-manager` | `CLAUDE.md` `.claude/**` | parallel (write) |
| `analyst` | `docs/` (not the generated registry) | parallel (write) |
| `executor` | `runtime/` — runs and records | **exclusive** |

The orchestrator (main session) owns the queue, the assignment, the ranking and the
integration, and:

- **does not execute experiments** — that bypasses the exclusivity contract and the registry path;
- **escalates irreversible actions to the human** — submission, `git push`/`checkout`/`branch`,
  registry corrections;
- **does not skip `reviewer`** — judging a pre-report sound is not the same as auditing it;
- **does not invent conclusions** — only what `analyst` and `reviewer` support;
- **does not rank without stated criteria** — write the criteria and their application down
  (`docs/hypotheses.md` -> ranking criteria).
```

- [ ] **Step 3: `workflow.md`의 File size budget 절을 축소한다**

현재 이 절은 4옵션을 본문에서 설명한다. 그 설명이 `refactor-agent-rules` 스킬로 이사했으므로, `## File size budget` 절 전체를 다음으로 교체한다:

```markdown
## File size budget (keep each instruction file dense)

Gotchas accumulate; a bloated rules file loads in full every session and dilutes signal. Soft
budget: **~150 lines per file** (CLAUDE.md and each `.claude/rules/*.md`). A PostToolUse hook
(`.claude/hooks/check_rules_size.py`) scans the governed set and nudges.

Detection is deterministic; the response is judgment. **Invoke the `refactor-agent-rules`
skill** — it holds the four remedies (relocate / split / abstract / compress, in that order),
the parallel prune-what-enforcement-covers check, and the two deletion tests. Do not improvise
a shortening pass: compression is the weakest of the four and the untrained reflex.

**A generated file takes none of the four** — `docs/experiment-registry.md` above all. Hand
edits and compression are both discarded by the next `scripts/exp.py render`; the only lever is
the generator (`enforcement.md` -> Keeping a generated artifact alive).

A tight single-topic file slightly over budget is fine — these are levers, not a mandate.
```

이 교체는 `workflow.md`를 **150줄 아래로 내린다.** 교체 후 줄 수를 확인한다.

- [ ] **Step 3b: `CLAUDE.md`의 Core invariants 항목을 고친다**

`CLAUDE.md:28-29`가 옛 명부를 담고 있다:

```
- **Only `experimenter` is exclusive** — when running sub-agents concurrently, one `experimenter`
  (GPU + submissions), while `research` / `critic` / `analyst` may run in parallel.
```

→ 교체:

```
- **Only `executor` is exclusive** — one `executor` at a time (GPU + submissions). The other
  five run in parallel; the three that write are kept apart by file domain, not by luck.
```

- [ ] **Step 4: `README.md`의 에이전트 참조를 갱신한다 — 위치는 이미 확인됐다**

확인된 3곳:

| 줄 | 지금 | 바꿀 내용 |
|---|---|---|
| `README.md:7-8` | `` `.claude/agents/`의 sub-agent(research·experimenter·analyst·critic) `` | `` `.claude/agents/`의 sub-agent(researcher·reviewer·engineer·harness-manager·analyst·executor) `` |
| `README.md:120` | `# research/experimenter/analyst/critic — sub-agent 역할 프롬프트` | `# researcher/reviewer/engineer/harness-manager/analyst/executor — sub-agent 역할 프롬프트` |

`README.md:115`(`CLAUDE.md # 하네스 라우터`)는 이름이 안 나오므로 그대로 둔다. 트리 그림에
`.claude/skills/`가 없으면 `.claude/rules/` 아래에 한 줄 더한다:

```
├── .claude/skills/        # refactor-agent-rules — 지시 파일 재구조화 판단 방법
```

- [ ] **Step 5: 검증 4종을 전부 돌린다**

```bash
uv run pytest -q
uv run ruff check src tests scripts
python .claude/scripts/check_rule_links.py
python .claude/hooks/test_check_rules_size.py
python .claude/hooks/test_block_runtime_commands.py
wc -l .claude/rules/*.md CLAUDE.md
```
Expected: 앞의 다섯은 통과, `wc -l`에서 모든 파일이 150 이하 (`workflow.md`가 특히 — 이번 변경 전 정확히 150이었다)

- [ ] **Step 6: 옛 이름이 저장소 어디에도 안 남았는지 확인한다**

```bash
grep -rn "experimenter\|\bcritic\b" --include=*.md --include=*.json . | grep -v "^./docs/experiment-registry.md" | grep -v "^./.venv"
```
Expected: 결과 없음. `docs/experiment-registry.md`는 생성물이라 제외한다 — 다만 **거기에 히트가 있으면 별도 항목으로 보고한다**: 기록소 본문에 에이전트 이름이 들어갔다는 뜻이고, 그건 생성기나 과거 판정 텍스트의 문제라 이 계획의 범위 밖이다.

- [ ] **Step 7: 커밋**

```bash
git add CLAUDE.md .claude/rules/architecture.md .claude/rules/workflow.md README.md
git commit -m "Wire the six-agent roster into CLAUDE.md, architecture and workflow"
```

---

### Task 8: `hypotheses.md` 순위 기준

기대 이득이 전부 "미지"인 후보들 사이에서 순위를 매길 근거가 지금 없다. 이 항목이 그 공백을 메운다.

**Files:**
- Modify: `docs/hypotheses.md` (`## 기입 규칙` 절)

**Interfaces:**
- Consumes: Task 7의 CLAUDE.md 명부 절이 이 절을 `docs/hypotheses.md -> ranking criteria`로 가리킨다
- Produces: 없음

- [ ] **Step 1: `## 기입 규칙` 절 끝에 추가한다**

```markdown
## 순위 기준

**기대 이득이 후보 전부 "미지"일 때** — 지금이 그렇다 — 예산표는 순위를 못 낸다. 그때는
아래를 내림차순으로 적용하고, **적용 결과를 실험 선택과 함께 적는다.**

1. **제출 없이 선판정할 수 있는가.** real→real 검증이 되는 후보가 언제나 우선이다.
2. **되돌릴 수 없는 자원을 얼마나 쓰는가.** 학습 0회 > 학습 1회. 지금까지 큰 이득 넷 중
   셋이 학습 0회였다 (EXP-014 · EXP-016 · EXP-019).
3. **결과가 죽어도 남는 것이 있는가.** 다른 축에서도 쓰이는 코드·지식인가.
4. **기각 목록을 새로 닫는가.** 실패해도 R 항목이 하나 늘면 남은 탐색 공간이 좁아진다.

근거 없이 정한 순위는 순위가 아니다. 기준과 그 적용을 적지 않은 선택은 되돌아볼 수 없다.
```

- [ ] **Step 2: 확인**

```bash
grep -n "## 순위 기준" docs/hypotheses.md && wc -l docs/hypotheses.md
```
Expected: 절이 존재하고, 파일은 여전히 200줄 미만 (`docs/`는 150줄 예산 대상이 아니지만 읽는 비용은 실재한다)

- [ ] **Step 3: 커밋**

```bash
git add docs/hypotheses.md
git commit -m "Add ranking criteria for candidates with unknown expected gain"
```

---

## 완료 후 — 후속 계획으로 넘길 것

1. **`engineer`의 첫 작업 큐** (`src/` 변경, 별도 계획): `registry.py`에 `dissent` 필드 추가 +
   `exp.py verdict --dissent` 필수화 · `render_markdown` 요약 표에서 `val` JSON 원본과 판정
   전문 제거 (현재 264줄 / 실험 19개, 실험당 ~14줄로 선형 증가).
2. **worktree 생성** — `../acs-harness`. git 보호 명령이라 사람 확인이 필요하다.
   `architecture.md`가 "engineer lane must run in a worktree"라고 적었으므로 `engineer`를
   투입하기 전에 `../acs-engineer`도 필요하다.
3. **다중 세션** — `src/ai_co_scientist/lease.py`, 자원별(`local-gpu` · `lightning` · `submit`).
