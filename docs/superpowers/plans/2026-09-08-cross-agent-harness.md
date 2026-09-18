# Cross-agent harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 지시·규칙·레인 정의·훅을 `.claude/` 밖의 하니스 중립 위치로 옮기고, Claude Code와 Codex가 같은 내용·같은 로직을 읽되 등록만 각자 하게 만든다.

**Architecture:** `.agents/`(규칙·레인 소스·스킬 소스)와 `.agent-hooks/`(스크립트 1벌 + 테스트)가 단일 진실원이 된다. `AGENTS.md`가 공용 라우터이고 `CLAUDE.md`는 한 줄 import다. `.agent-hooks/build-agents.py`가 레인 소스에서 `.claude/agents/*.md`와 `.codex/agents/*.toml`을, 스킬 소스에서 양쪽 복사본을 생성하며, 생성물은 커밋된다. 신선도는 `uv run pytest -q`가 수집하는 테스트로 보장한다.

**Tech Stack:** Python (훅·스캐너는 **3.8 호환**, bare `python`으로 실행 / 테스트는 `uv run python` = 3.12), TOML, JSON, git

**Spec:** `docs/superpowers/specs/2026-09-08-cross-agent-harness-design.md`

## Global Constraints

- **훅·스캐너 스크립트는 Python 3.8 호환**으로 쓴다. 이 머신의 bare `python`이 3.8.10이고 훅은 그것으로 돈다. `tomllib`(3.11+), walrus 이후 문법, `dict |` 병합 금지.
- **테스트는 `uv run python` / `uv run pytest`(3.12)로 돌린다.** `tomllib`을 쓰는 테스트는 이쪽에서만 돈다.
- **`python3`는 이 머신에 없다.** 훅 등록 커맨드는 `command -v python3 || command -v python`으로 해석하고, 둘 다 없으면 조용히 통과하지 말고 한 줄 출력한다.
- **생성물을 손으로 고치지 않는다**: `.claude/agents/**`, `.codex/agents/**`, `.claude/skills/**`, `.codex/skills/**`.
- **지시 파일은 영어**(`AGENTS.md`, `.agents/**/*.md`), **`docs/`는 한국어**. `src/`의 한국어 주석·도크스트링은 그대로 둔다.
- **커밋 메시지는 영어 한 줄 제목**, AI attribution trailer 금지(`Co-Authored-By` / `Generated with` — PreToolUse 훅이 deny).
- **이동은 `git mv`**로 한다. rename으로 추적돼야 diff가 읽힌다.
- 린트: `uv run ruff check src tests scripts` (line-length 100).
- `.superpowers/**`와 `docs/superpowers/plans/2026-09-04-agent-roster.md` 안의 `.claude/` 경로는 **고치지 않는다**(과거 기록).

---

## File Structure

**신규**
- `AGENTS.md` — 공용 라우터 (현 `CLAUDE.md` 본문)
- `.agents/rules/harness.md` — 하니스 변경 규칙 + 불변식
- `.agent-hooks/build-agents.py` — 생성기
- `.agent-hooks/test-build-agents.py` — 생성기 테스트
- `.agent-hooks/test_harness_parity.py` — 두 등록의 드리프트 테스트
- `.codex/config.toml` — Codex 등록
- `tests/test_harness_generated.py` — 수집되는 신선도/패리티 게이트
- `docs/codex-verification-ledger.md` — 미측정 항목 원장

**이동**
- `CLAUDE.md` → `AGENTS.md` (그 자리에 한 줄 `@AGENTS.md`)
- `.claude/rules/*.md` → `.agents/rules/*.md`
- `.claude/agents/*.md` → `.agents/agents/*.md` (소스; 같은 경로에 생성물이 다시 생김)
- `.claude/skills/refactor-agent-rules/` → `.agents/skills/refactor-agent-rules/`
- `.claude/hooks/*.py`, `.claude/scripts/*.py` → `.agent-hooks/`

**수정**
- `.claude/settings.json` — 훅 경로 + 생성기 훅
- `.agent-hooks/check_rules_size.py` — `GOVERNED`, `DIVERGENCE`
- `.agent-hooks/check_rule_links.py` — 소스 루트, `DIVERGENCE`
- `.agents/rules/enforcement.md` · `self-review.md` · `architecture.md`
- `README.md`, `src/ai_co_scientist/registry.py`, `src/ai_co_scientist/submission.py`,
  `tests/test_registry.py`, `tests/test_train_manifest.py`, `tests/test_viterbi_levels.py`,
  `scripts/legacy/README.md`

---

### Task 1: 훅·스캐너를 `.agent-hooks/`로 옮기고 두 등록이 아닌 한 등록부터 고친다

레인·규칙 이동보다 먼저 한다. 스크립트 위치가 확정돼야 이후 모든 등록 문자열이 한 번만 쓰인다.

**Files:**
- Move: `.claude/hooks/block_runtime_commands.py`, `.claude/hooks/test_block_runtime_commands.py`, `.claude/hooks/check_rules_size.py`, `.claude/hooks/test_check_rules_size.py`, `.claude/scripts/check_rule_links.py`, `.claude/scripts/test_check_rule_links.py` → `.agent-hooks/`
- Modify: `.claude/settings.json`

**Interfaces:**
- Consumes: —
- Produces: 이후 모든 태스크가 훅 스크립트를 `.agent-hooks/<name>.py`로 참조한다. 등록 커맨드의 인터프리터 관용구는 이 태스크가 정한 것을 그대로 쓴다.

- [ ] **Step 1: 이동 전 상태를 기록한다**

```bash
cd "c:/Users/user/Desktop/projects/ai-co-scientist"
find .claude/hooks .claude/scripts -name '__pycache__' -type d -exec rm -rf {} +
uv run python .claude/scripts/check_rule_links.py > /tmp/before-links.txt; echo "exit=$?"
cat /tmp/before-links.txt
```

이 출력이 이동 후에도 같아야 한다(파일 경로만 다르게). 기대: 0 findings.

- [ ] **Step 2: `git mv`로 옮긴다**

```bash
mkdir -p .agent-hooks
git mv .claude/hooks/block_runtime_commands.py      .agent-hooks/
git mv .claude/hooks/test_block_runtime_commands.py .agent-hooks/
git mv .claude/hooks/check_rules_size.py            .agent-hooks/
git mv .claude/hooks/test_check_rules_size.py       .agent-hooks/
git mv .claude/scripts/check_rule_links.py          .agent-hooks/
git mv .claude/scripts/test_check_rule_links.py     .agent-hooks/
rmdir .claude/hooks .claude/scripts 2>/dev/null || true
git status --short
```

- [ ] **Step 3: 옮긴 테스트가 새 위치에서 도는지 먼저 실패를 본다**

```bash
uv run python -m pytest .agent-hooks/test_check_rules_size.py .agent-hooks/test_block_runtime_commands.py .agent-hooks/test_check_rule_links.py -q
```

기대: 스크립트 안에 `.claude/hooks/...` 자기 경로 문자열이 남아 있으면 여기서 깨진다. 깨지는 항목을 적어 둔다. 안 깨지더라도 다음 스텝은 건너뛰지 않는다 — 도크스트링의 경로는 테스트가 보지 않는다.

- [ ] **Step 4: 옮긴 6개 파일 안의 자기 경로 표기를 갱신한다**

각 파일의 `Run:` 줄, 도크스트링, 테스트 픽스처에서 `.claude/hooks/` → `.agent-hooks/`, `.claude/scripts/` → `.agent-hooks/`. `GOVERNED`와 소스 루트는 **아직 건드리지 않는다**(규칙은 아직 `.claude/rules/`에 있다 — Task 2에서 같이 옮긴다).

```bash
grep -rn "\.claude/hooks\|\.claude/scripts" .agent-hooks/
```

기대: 이 grep이 빈 출력이 될 때까지 고친다.

- [ ] **Step 5: `.claude/settings.json`의 훅 경로를 고친다**

`python "$CLAUDE_PROJECT_DIR/.claude/hooks/<name>.py"` 두 곳을 인터프리터 해석형으로 바꾼다. 두 훅 모두 같은 관용구를 쓴다:

```
p=$(command -v python3 || command -v python); if [ -z "$p" ]; then echo "[hook] no python interpreter on PATH -- <name> NOT run" >&2; exit 0; fi; "$p" "$CLAUDE_PROJECT_DIR/.agent-hooks/<name>.py"
```

`<name>`은 `block_runtime_commands`(PreToolUse, Bash matcher)와 `check_rules_size`(PostToolUse). `permissions`·`enabledPlugins`·`if`·`timeout`·`statusMessage`는 그대로 둔다.

- [ ] **Step 6: 훅이 실제로 도는지 음성 케이스로 확인한다**

막혀야 할 것을 먼저 시도한다 — 통과를 보고 "동작한다"고 결론내지 않는다.

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"uv run python scripts/train_level.py"}}' \
  | uv run python .agent-hooks/block_runtime_commands.py; echo "exit=$?"
```

기대: deny JSON이 나오거나 exit≠0 (스크립트의 계약대로). 그 다음 통과 케이스:

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"ls"}}' \
  | uv run python .agent-hooks/block_runtime_commands.py; echo "exit=$?"
```

기대: 조용히 exit 0.

- [ ] **Step 7: 게이트를 돌린다**

```bash
uv run python -m pytest .agent-hooks/ -q
uv run python .agent-hooks/check_rule_links.py; echo "exit=$?"
uv run pytest -q
uv run ruff check src tests scripts
```

기대: 전부 통과. `check_rule_links.py`의 findings 수가 Step 1과 같아야 한다.

- [ ] **Step 8: 커밋**

```bash
git add -A
git commit -m "Move hook and scanner scripts to harness-neutral .agent-hooks/"
```

---

### Task 2: `AGENTS.md`와 `.agents/rules/`로 옮기고 모든 포인터를 따라 고친다

**Files:**
- Move: `CLAUDE.md` → `AGENTS.md`; `.claude/rules/*.md` (8개) → `.agents/rules/`
- Create: `CLAUDE.md` (한 줄)
- Modify: `.agent-hooks/check_rules_size.py`, `.agent-hooks/check_rule_links.py`, `.agent-hooks/test_check_rules_size.py`, `.agent-hooks/test_check_rule_links.py`, `README.md`, `src/ai_co_scientist/registry.py`, `src/ai_co_scientist/submission.py`, `tests/test_registry.py`, `tests/test_train_manifest.py`, `tests/test_viterbi_levels.py`, `scripts/legacy/README.md`

**Interfaces:**
- Consumes: Task 1의 `.agent-hooks/` 배치
- Produces: `.agents/rules/<topic>.md` 경로 규약. 이후 모든 문서·레인 본문이 이 경로로 규칙을 가리킨다. `AGENTS.md`가 규칙 표를 들고 있는 유일한 파일이다.

- [ ] **Step 1: `git mv`**

```bash
mkdir -p .agents/rules
git mv CLAUDE.md AGENTS.md
for f in architecture coding-patterns enforcement git-workflow orca-parallel self-review testing workflow; do
  git mv ".claude/rules/$f.md" ".agents/rules/$f.md"
done
rmdir .claude/rules 2>/dev/null || true
printf '@AGENTS.md\n' > CLAUDE.md
git status --short
```

- [ ] **Step 2: 스캐너의 대상 집합을 먼저 실패시킨다**

```bash
uv run python .agent-hooks/check_rule_links.py; echo "exit=$?"
echo '{"tool_name":"Write","tool_input":{"file_path":"AGENTS.md"}}' \
  | uv run python .agent-hooks/check_rules_size.py; echo "exit=$?"
```

기대: `check_rules_size.py`가 **"the governed set is EMPTY"** 를 낸다 — `GOVERNED = (".claude/rules/*.md", "CLAUDE.md")`의 첫 패턴이 0개를 매치하고 `CLAUDE.md`는 이제 한 줄이다. 빈 대상 집합은 통과가 아니라 실패라는 그 파일의 계약이 여기서 증명된다.

- [ ] **Step 3: `check_rules_size.py`의 `GOVERNED`를 고친다**

```python
GOVERNED = (".agents/rules/*.md", "AGENTS.md")
```

그리고 도크스트링의 `Governed` 줄을 같이 고친다. 이 파일은 `harness-spine` 스켈레톤 산물이므로 도크스트링 하단 `DIVERGENCE` 블록에 항목을 추가한다:

```
DIVERGENCE (2026-09-08): GOVERNED moved from (".claude/rules/*.md", "CLAUDE.md") to
(".agents/rules/*.md", "AGENTS.md"). This project serves Claude Code and Codex from one set of
instructions, so the governed prose is no longer under `.claude/` — `CLAUDE.md` is a one-line
import of AGENTS.md and has nothing to budget. The script itself is unchanged; only the per-
project target set moved. Do not reconcile this line back onto the skeleton.
```

- [ ] **Step 4: `check_rule_links.py`의 소스 루트를 고친다**

`SEARCH_DIRS` / `GOVERNED`에 해당하는 per-project 줄에서 `.claude/rules` → `.agents/rules`, `.claude/agents` → `.agents/agents`, `.claude/hooks`·`.claude/scripts` → `.agent-hooks`, `CLAUDE.md` → `AGENTS.md`. 도크스트링의 2026-09-04 `DIVERGENCE` 항목은 지우지 말고 그 아래에 추가한다:

```
DIVERGENCE (2026-09-08): the source roots moved with the cross-agent port —
`.agents/rules`, `.agents/agents`, `.agent-hooks`, and `AGENTS.md` in place of their `.claude/`
predecessors. The logic is untouched. `harness-spine:update` must reconcile around these roots,
never onto them.
```

- [ ] **Step 5: 두 스캐너의 테스트를 고치고 통과시킨다**

`test_check_rules_size.py` / `test_check_rule_links.py`의 픽스처가 만드는 디렉토리 이름을 새 배치에 맞춘다. **must-block 절반과 must-pass 절반을 둘 다 유지**한다 — must-pass 쪽이 오탐을 막는 부분이다.

```bash
uv run python -m pytest .agent-hooks/test_check_rules_size.py .agent-hooks/test_check_rule_links.py -q
```

기대: PASS.

- [ ] **Step 6: 규칙 파일 안의 상호 참조를 고친다**

```bash
grep -rn "\.claude/rules\|\.claude/hooks\|\.claude/scripts\|\.claude/skills\|CLAUDE\.md" .agents/rules/ AGENTS.md
```

`.claude/rules/x.md` → `.agents/rules/x.md`, `.claude/hooks/` · `.claude/scripts/` → `.agent-hooks/`, "CLAUDE.md" → "AGENTS.md". **`.claude/settings.json`과 `.claude/agents/`는 그대로 둔다** — 전자는 여전히 Claude 등록 파일이고 후자는 Task 3에서 생성물이 된다.

- [ ] **Step 7: 저장소 나머지의 포인터를 고친다**

```bash
grep -rn "\.claude/rules\|\.claude/hooks\|\.claude/scripts" \
  README.md src/ tests/ scripts/ docs/ --include='*.md' --include='*.py' \
  | grep -v "docs/superpowers/plans/2026-09-04-agent-roster.md"
```

나온 곳을 전부 고친다: `README.md`(본문 + 구조 트리), `src/ai_co_scientist/registry.py:3`, `src/ai_co_scientist/submission.py:47`, `tests/test_registry.py:275`, `tests/test_train_manifest.py:10`, `tests/test_viterbi_levels.py:100`, `scripts/legacy/README.md:57`. **`.superpowers/**`와 `docs/superpowers/plans/2026-09-04-agent-roster.md`는 건드리지 않는다.**

`README.md` 구조 트리는 이렇게 된다:

```
├── AGENTS.md              # 공용 라우터 (CLAUDE.md는 이걸 import하는 한 줄)
├── .agents/agents/        # researcher/reviewer/engineer/harness-manager/analyst/executor — 역할 프롬프트 소스
├── .agents/rules/         # 이 저장소에서 일하는 규칙 (agents/와 다른 층)
├── .agents/skills/        # refactor-agent-rules — 지시 파일 재구조화 판단 방법
├── .agent-hooks/          # 훅·스캐너 스크립트 1벌 + 각각의 테스트
├── .claude/settings.json  # Claude Code 등록
├── .codex/config.toml     # Codex 등록
```

- [ ] **Step 8: `AGENTS.md`에 "이 파일에 규칙을 쓰는 법" 절을 넣는다**

`## Docs convention` 바로 앞에 넣는다. 영어로:

```markdown
## Writing rules in this file

**Name the action, never the harness.** This file is read verbatim by every agent that works
here, so a statement true of one harness and false of another does more damage than no statement
— an agent told its rules arrive automatically will not go and read them. Anything that depends
on which agent is running belongs in the rule it modifies, as a **Claude Code —** or **Codex —**
paragraph beside the neutral statement. `.agents/rules/harness.md` has the convention.
```

`## Rules` 표의 경로 열은 이미 Step 6에서 `.agents/rules/`로 바뀌었다. `harness.md` 행은 Task 7에서 추가한다(파일이 아직 없으므로 지금 넣으면 링크 스캐너가 깨진다).

- [ ] **Step 9: 게이트**

```bash
uv run python -m pytest .agent-hooks/ -q
uv run python .agent-hooks/check_rule_links.py; echo "exit=$?"
uv run pytest -q
uv run ruff check src tests scripts
```

기대: 전부 통과, 링크 findings 0.

- [ ] **Step 10: 커밋**

```bash
git add -A
git commit -m "Move rules to .agents/ and split AGENTS.md from a one-line CLAUDE.md"
```

---

### Task 3: 생성기 — 레인 소스 하나에서 두 하니스 파일을 만든다

**Files:**
- Create: `.agent-hooks/build-agents.py`, `.agent-hooks/test-build-agents.py`, `tests/test_harness_generated.py`
- Move: `.claude/agents/*.md` (6개) → `.agents/agents/`
- Generate (커밋됨): `.claude/agents/*.md`, `.codex/agents/*.toml`
- Modify: `.claude/settings.json`

**Interfaces:**
- Consumes: Task 1의 `.agent-hooks/` 배치
- Produces:
  - `build-agents.py` CLI: 인자 없음(생성) · `--check`(비교만, 쓰지 않음) · `--hook`(PostToolUse) · `--root <path>`
  - 종료 계약: `AGENTS_FRESH` exit 0 / `AGENTS_STALE` exit 1 / `AGENTS_UNKNOWN: <why>` exit 2 — 첫 stdout 줄이 마커
  - 소스 프론트매터 키: `name`, `description`, `tools.claude`, `model.claude`, `model.codex`, `nickname_candidates`. 그 외는 하드 에러
  - Task 5의 parity 테스트가 `.codex/agents/<name>.toml` 파일명을 등록과 대조한다

- [ ] **Step 1: 이동 전 6개 파일의 해시를 박제한다**

이 태스크의 수용 기준이다 — 이동이 내용을 바꾸지 않았음을 diff가 아니라 해시로 증명한다.

```bash
mkdir -p /tmp/agents-before
cp .claude/agents/*.md /tmp/agents-before/
(cd .claude/agents && sha256sum *.md) | sort > /tmp/agents-before.sha256
cat /tmp/agents-before.sha256
```

- [ ] **Step 2: 소스로 옮기고 프론트매터를 중립 키로 바꾼다**

```bash
mkdir -p .agents/agents
for f in analyst engineer executor harness-manager researcher reviewer; do
  git mv ".claude/agents/$f.md" ".agents/agents/$f.md"
done
```

각 파일에서 `tools:` → `tools.claude:`, `model:` → `model.claude:`로 바꾸고 `model.claude` 바로 뒤에 `model.codex: gpt-5.6-sol`를 넣는다(custflow 실측값). **키 순서는 `name, description, tools.claude, model.claude, model.codex`** — Claude 출력이 원본과 바이트 동일하려면 `tools`가 `model` 앞이어야 한다.

본문 첫 주석의 모델 근거는 Claude 기준이므로 라벨을 붙인다. 예: `<!-- model: sonnet — ...` → `<!-- **Claude Code —** model: sonnet — ...`.

- [ ] **Step 3: 생성기 테스트를 먼저 놓고 실패를 본다**

`C:\Users\user\Desktop\projects\custflow-pipeline\.agent-hooks\test-build-agents.py`를 `.agent-hooks/test-build-agents.py`로 복사한 뒤 이 저장소에 맞게 고친다:
- `python3` 호출 → `sys.executable`
- 픽스처 프론트매터에 `tools.claude: Read, Grep` 케이스를 추가하고, 생성된 Claude 파일에 `tools: Read, Grep`이 나오고 **Codex TOML에는 `tools`가 나오지 않는지** 단언한다
- `model.codex`가 없는 소스에서 Codex TOML에 `model =` 줄이 안 나오는 케이스를 유지한다
- 미지 프론트매터 키가 `AGENTS_UNKNOWN` + exit 2를 내는 케이스를 유지한다
- 스킬 디렉토리명 ≠ `name:` 이 빌드를 거부하는 케이스를 유지한다

```bash
uv run python -m pytest .agent-hooks/test-build-agents.py -q
```

기대: FAIL — `build-agents.py`가 없다.

- [ ] **Step 4: `build-agents.py`를 쓴다**

`C:\Users\user\Desktop\projects\custflow-pipeline\.agent-hooks\build-agents.py`를 옮기되:

1. **도크스트링을 다시 쓴다.** 원본 상단은 이미 삭제된 `.codex.md` 사이드카와 "Codex fragment"를 설명하는 스테일 상태다("Harness differences are frontmatter keys, not sidecar files. A name with no Codex output at all." 문장은 중간에 끊겨 있기까지 하다). 실제 코드가 하는 일 — 소스 1개 → Claude `.md` + Codex `.toml`, 스킬은 양쪽 복사 — 만 적는다.
2. `SRC_KEYS`에 `"tools.claude"`를 추가하고 `CLAUDE_ONLY`를 `("model.claude", "tools.claude")`로 바꾼다.
3. `render_claude`의 키 매핑에 `tools.claude` → `tools`를 추가한다:

```python
    for k, v in fields:
        if k in CODEX_ONLY:
            continue
        if k == "model.claude":
            lines.append(f"model: {v}")
        elif k == "tools.claude":
            lines.append(f"tools: {v}")
        else:
            lines.append(f"{k}: {v}")
```

4. `render_all`의 `if codex is not None:` 는 항상 참이다(`load_source`가 늘 dict를 돌려준다). 조건을 지우고 두 출력을 무조건 낸다 — 죽은 분기를 이식하지 않는다.
5. **Python 3.8 호환**을 확인한다: f-string, `os.walk`, `difflib`만 쓰므로 그대로 통과하지만, 옮긴 뒤 `python -c "import ast,sys; ast.parse(open('.agent-hooks/build-agents.py',encoding='utf-8').read())"` 를 3.8로 한 번 돌려 확인한다.

```bash
python -c "import ast; ast.parse(open('.agent-hooks/build-agents.py',encoding='utf-8').read())" && echo "3.8 parse OK"
uv run python -m pytest .agent-hooks/test-build-agents.py -q
```

기대: 둘 다 통과.

- [ ] **Step 5: 생성하고, 바이트 동일성을 증명한다**

```bash
uv run python .agent-hooks/build-agents.py
(cd .claude/agents && sha256sum *.md) | sort > /tmp/agents-after.sha256
diff /tmp/agents-before.sha256 /tmp/agents-after.sha256 && echo "BYTE-IDENTICAL"
```

기대: `BYTE-IDENTICAL`. 다르면 **여기서 멈추고** 어느 파일이 왜 다른지 밝힌다 — 이동이 내용을 바꿨다는 뜻이고, 그건 이 태스크의 전제가 깨진 것이다.

```bash
ls .codex/agents/
uv run python .agent-hooks/build-agents.py --check; echo "exit=$?"
```

기대: 6개 `.toml` 생성, `AGENTS_FRESH` + exit 0.

- [ ] **Step 6: `--check`를 수집되는 테스트로 만든다**

`pyproject.toml`의 `testpaths = ["tests"]`라서 `.agent-hooks/` 밑 테스트는 `uv run pytest -q`가 수집하지 않는다. 신선도는 모든 저작 경로를 덮어야 하므로 `tests/`에 얇은 호출을 놓는다.

```python
"""생성물이 소스와 맞는지 — 훅이 못 보는 저작 경로(IDE·다른 에이전트·동료)까지 덮는 층.

`.agent-hooks/build-agents.py --hook`은 이번 세션의 편집만 본다. 이 파일이 보장이고 훅은
빠른 피드백이다 (`.agents/rules/enforcement.md` → Hooks only see this session's edits).
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_generated_agent_files_are_current():
    proc = subprocess.run(
        [sys.executable, str(ROOT / ".agent-hooks" / "build-agents.py"), "--check"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    )
    marker = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
    assert marker.startswith("AGENTS_FRESH"), (
        "생성물이 소스와 어긋났거나 판정이 불가능하다.\n"
        "`uv run python .agent-hooks/build-agents.py` 로 재생성하고 함께 커밋할 것.\n"
        f"marker={marker!r} rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )
```

- [ ] **Step 7: 그 테스트가 실제로 실패할 수 있는지 증명한다**

통과를 신뢰하기 전에 음성 케이스를 먼저 돌린다.

```bash
printf '\nDRIFT\n' >> .claude/agents/reviewer.md
uv run pytest tests/test_harness_generated.py -q; echo "exit=$?"
```

기대: FAIL, 메시지에 `.claude/agents/reviewer.md`가 나온다.

```bash
uv run python .agent-hooks/build-agents.py
uv run pytest tests/test_harness_generated.py -q
```

기대: PASS로 복귀.

- [ ] **Step 8: PostToolUse에 생성기 훅을 등록한다**

`.claude/settings.json`의 `PostToolUse`에 항목을 추가한다. matcher `Write|Edit|MultiEdit`, Task 1의 인터프리터 관용구를 그대로 쓰고 `<name>.py` 자리에 `build-agents.py --hook`, `timeout` 20.

- [ ] **Step 9: 게이트**

```bash
uv run python -m pytest .agent-hooks/ -q
uv run python .agent-hooks/check_rule_links.py; echo "exit=$?"
uv run pytest -q
uv run ruff check src tests scripts
```

- [ ] **Step 10: 커밋**

```bash
git add -A
git commit -m "Generate per-harness agent lanes from a single source in .agents/agents"
```

---

### Task 4: 스킬을 소스로 옮기고 양쪽에 생성한다

**Files:**
- Move: `.claude/skills/refactor-agent-rules/SKILL.md` → `.agents/skills/refactor-agent-rules/SKILL.md`
- Generate: `.claude/skills/refactor-agent-rules/SKILL.md`, `.codex/skills/refactor-agent-rules/SKILL.md`

**Interfaces:**
- Consumes: Task 3의 `build-agents.py` (스킬 복사와 이름 일치 검사를 이미 구현하고 있다)
- Produces: `refactor-agent-rules`가 두 하니스에 같은 이름으로 등록된다 — `AGENTS.md`와 `check_rules_size.py`의 nudge 문구가 이 이름으로 지목한다

- [ ] **Step 1: 이동 전 해시**

```bash
sha256sum .claude/skills/refactor-agent-rules/SKILL.md
```

- [ ] **Step 2: 옮긴다**

```bash
mkdir -p .agents/skills
git mv .claude/skills/refactor-agent-rules .agents/skills/refactor-agent-rules
git status --short
```

- [ ] **Step 3: 프론트매터를 Codex가 받는 교집합으로 줄인다**

```bash
sed -n '1,8p' .agents/skills/refactor-agent-rules/SKILL.md
```

`name`과 `description` 외의 키가 있으면 지운다(Codex가 미지의 키에 파일을 통째로 거부한다). `name:`이 디렉토리명 `refactor-agent-rules`와 **정확히 같은지** 확인한다 — 다르면 Claude는 디렉토리명으로, Codex는 `name:`으로 등록해 한 스킬이 두 이름을 갖는다.

- [ ] **Step 4: 이름 불일치가 실제로 빌드를 막는지 먼저 본다**

```bash
sed -i 's/^name: refactor-agent-rules$/name: refactor-rules/' .agents/skills/refactor-agent-rules/SKILL.md
uv run python .agent-hooks/build-agents.py; echo "exit=$?"
```

기대: `AGENTS_UNKNOWN` + exit 2, 메시지가 두 이름을 모두 보여준다. 되돌린다:

```bash
sed -i 's/^name: refactor-rules$/name: refactor-agent-rules/' .agents/skills/refactor-agent-rules/SKILL.md
```

- [ ] **Step 5: 생성하고 원본과 대조한다**

```bash
uv run python .agent-hooks/build-agents.py
sha256sum .claude/skills/refactor-agent-rules/SKILL.md .codex/skills/refactor-agent-rules/SKILL.md .agents/skills/refactor-agent-rules/SKILL.md
```

기대: 세 해시가 모두 같다(Step 3에서 프론트매터를 줄였다면 그 값으로 셋이 같다).

- [ ] **Step 6: 게이트**

```bash
uv run pytest -q
uv run python .agent-hooks/check_rule_links.py; echo "exit=$?"
```

- [ ] **Step 7: 커밋**

```bash
git add -A
git commit -m "Move refactor-agent-rules skill to .agents/skills and copy to both harnesses"
```

---

### Task 5: Codex 등록과 패리티 테스트

**Files:**
- Create: `.codex/config.toml`, `.agent-hooks/test_harness_parity.py`
- Modify: `tests/test_harness_generated.py`

**Interfaces:**
- Consumes: Task 1의 훅 파일명, Task 3이 만든 `.codex/agents/*.toml`, Task 4의 `.codex/skills/`
- Produces: `[agents.<snake_name>]` 6개. `config_file`은 `agents/<name>.toml`(`.codex/` 상대 경로)

- [ ] **Step 1: `.codex/config.toml`을 쓴다**

```toml
# Project config for ai-co-scientist. Codex ignores this until the project is trusted
# (`[projects."<abs path>"] trust_level = "trusted"` in ~/.codex/config.toml, or the trust
# prompt on first run here — the IDE never shows that prompt).
#
# No sandbox_mode / approval_policy: whether the sandbox starts on this Windows machine has not
# been measured, and a sandbox that cannot start kills every shell command.
# docs/codex-verification-ledger.md item 5.
#
# Hook commands resolve the repo through `git rev-parse --show-toplevel` (Codex may start in a
# subdirectory) and the interpreter through `command -v` (this machine has `python`, not
# `python3`). The advisory hooks say so and pass when they cannot resolve; the PreToolUse guard
# denies instead, because a guard that cannot find itself must not let the call through.

project_doc_max_bytes = 65536
project_root_markers = [".git"]

[[hooks.PostToolUse]]
[[hooks.PostToolUse.hooks]]
type = "command"
command = 'r=$(git rev-parse --show-toplevel 2>/dev/null); p=$(command -v python3 || command -v python); if [ -n "$r" ] && [ -n "$p" ] && [ -f "$r/.agent-hooks/check_rules_size.py" ]; then "$p" "$r/.agent-hooks/check_rules_size.py"; else echo "[rules-size] could not resolve repo or python -- size budget NOT checked" >&2; cat >/dev/null 2>&1 || :; fi'
timeout = 15

[[hooks.PostToolUse]]
[[hooks.PostToolUse.hooks]]
type = "command"
command = 'r=$(git rev-parse --show-toplevel 2>/dev/null); p=$(command -v python3 || command -v python); if [ -n "$r" ] && [ -n "$p" ] && [ -f "$r/.agent-hooks/build-agents.py" ]; then "$p" "$r/.agent-hooks/build-agents.py" --hook; else echo "[build-agents] could not resolve repo or python -- generated lanes NOT rebuilt" >&2; cat >/dev/null 2>&1 || :; fi'
timeout = 20

[[hooks.PreToolUse]]
[[hooks.PreToolUse.hooks]]
type = "command"
command = 'r=$(git rev-parse --show-toplevel 2>/dev/null); p=$(command -v python3 || command -v python); if [ -n "$r" ] && [ -n "$p" ] && [ -f "$r/.agent-hooks/block_runtime_commands.py" ]; then "$p" "$r/.agent-hooks/block_runtime_commands.py"; else printf "{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"block_runtime_commands.py could not be resolved; refusing rather than running unguarded.\"}}\n"; exit 2; fi'
timeout = 10

[agents.researcher]
description = "Propose the next experiment hypothesis and draft its pre-report."
config_file = "agents/researcher.toml"
nickname_candidates = ["researcher"]

[agents.reviewer]
description = "Audit a pre-report before it runs or a conclusion before it is accepted."
config_file = "agents/reviewer.toml"
nickname_candidates = ["reviewer"]

[agents.engineer]
description = "Write additive pipeline code in src/ and scripts/ with offline tests."
config_file = "agents/engineer.toml"
nickname_candidates = ["engineer"]

[agents.harness_manager]
description = "Own the harness — AGENTS.md, .agents/, hooks, scripts and skills."
config_file = "agents/harness-manager.toml"
nickname_candidates = ["harness manager"]

[agents.analyst]
description = "Interpret recorded results and keep docs/ current."
config_file = "agents/analyst.toml"
nickname_candidates = ["analyst"]

[agents.executor]
description = "Execute ONE approved experiment and record what happened. Exclusive."
config_file = "agents/executor.toml"
nickname_candidates = ["executor"]
```

- [ ] **Step 2: TOML이 파싱되는지 확인한다**

```bash
uv run python -c "import tomllib;d=tomllib.load(open('.codex/config.toml','rb'));print(sorted(d['agents']));print(sum(len(e['hooks']) for v in d['hooks'].values() for e in v),'hook entries')"
```

기대: 6개 에이전트, 3 hook entries.

- [ ] **Step 3: 패리티 테스트를 놓고 실패를 본다**

`C:\Users\user\Desktop\projects\custflow-pipeline\.agent-hooks\test_harness_parity.py`를 복사한 뒤 이 저장소에 맞게 고친다:
- `ROOT`의 환경변수는 `CLAUDE_PROJECT_DIR` 그대로 두되 기본값은 `.agent-hooks`의 부모
- 규칙 링크 검사가 도는 파일 목록: `["AGENTS.md", "CLAUDE.md"]` + `.agents/rules/*.md` — 이 저장소는 이미 `check_rule_links.py`가 같은 판단을 하므로 **중복 구현을 남기지 않는다**: parity 테스트에서 링크 블록을 통째로 지우고, 대신 `check_rule_links.py`를 서브프로세스로 호출해 exit code를 단언한다(한 판단, 한 구현)
- `guard-sql.sh` 같은 `.sh` 실행권한 검사는 이 저장소에 `.sh` 훅이 없으므로 그대로 두면 무해하다 — 남긴다
- `trust_note()`와 `trusted_count()`는 그대로 이식한다. 이 저장소에서 아직 측정되지 않았다는 사실을 도크스트링에 명시한다

```bash
uv run python .agent-hooks/test_harness_parity.py; echo "exit=$?"
```

기대: 이 시점에 **FAIL** 이어야 한다 — Claude에는 3개 훅, Codex에는 3개인데 `check_rules_size`가 Claude PostToolUse에도 있으니 이름 집합이 맞아야 한다. 불일치가 나오면 그것이 이 테스트가 잡으라고 있는 바로 그 상태다. 맞을 때까지 등록을 고친다.

- [ ] **Step 4: 패리티가 실제로 실패할 수 있는지 증명한다**

```bash
uv run python - <<'PY'
import io
p = ".codex/config.toml"
s = io.open(p, encoding="utf-8").read()
io.open(p+".bak","w",encoding="utf-8").write(s)
io.open(p,"w",encoding="utf-8").write(s.replace("build-agents.py", "build-agents-typo.py"))
PY
uv run python .agent-hooks/test_harness_parity.py; echo "exit=$?"
```

기대: FAIL, "registered on Claude Code but not on Codex" 또는 "registered but does not exist". 되돌린다:

```bash
mv .codex/config.toml.bak .codex/config.toml
uv run python .agent-hooks/test_harness_parity.py; echo "exit=$?"
```

기대: PASS.

- [ ] **Step 5: 패리티를 수집되는 테스트에 편입한다**

`tests/test_harness_generated.py`에 추가한다:

```python
def test_harness_registrations_have_not_drifted():
    proc = subprocess.run(
        [sys.executable, str(ROOT / ".agent-hooks" / "test_harness_parity.py")],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, (
        "두 하니스의 등록이 벌어졌다 — 한쪽에만 걸린 훅, 없는 스크립트, 아무도 등록하지 않는 "
        f"Codex 레인 중 하나다.\n{proc.stdout}\n{proc.stderr}"
    )
```

```bash
uv run pytest tests/test_harness_generated.py -q
```

- [ ] **Step 6: 게이트**

```bash
uv run python -m pytest .agent-hooks/ -q
uv run pytest -q
uv run ruff check src tests scripts
uv run python .agent-hooks/check_rule_links.py; echo "exit=$?"
```

- [ ] **Step 7: 커밋**

```bash
git add -A
git commit -m "Register hooks and lanes for Codex and gate the two registrations against drift"
```

---

### Task 6: `harness.md`와 그 규칙을 가리키는 곳들

**Files:**
- Create: `.agents/rules/harness.md`
- Modify: `AGENTS.md`, `.agents/rules/enforcement.md`, `.agents/rules/self-review.md`, `.agents/rules/architecture.md`

**Interfaces:**
- Consumes: Task 1~5가 만든 배치 전부 — 이 파일은 그것을 설명한다
- Produces: `AGENTS.md` 규칙 표의 `harness.md` 행. 이후 하니스를 건드리는 모든 작업의 진입점

- [ ] **Step 1: `.agents/rules/harness.md`를 쓴다**

영어로. 담을 것(레퍼런스의 같은 파일을 뼈대로 하되 이 저장소의 사실로 채운다):

1. **The invariant** — 공유/하니스별 표(스펙 §1 그대로)
2. **Where a thing lives** — 산문은 `.agents/` 아래. 규칙 파일은 주제로 이름 짓고, 하니스 차이는 `**Claude Code —**` / `**Codex —**` **인접 문단**으로 쓴다(제목이 아니라 인라인 강조 — 제목 레벨은 파일 구조와 충돌한다). 규칙과 그 변형은 떨어져 읽힐 수 없어야 한다.
3. **스킬의 함정** — Claude는 디렉토리명, Codex는 `name:`. 빌드가 거부하는 유일한 지점.
4. **`tools:`에는 Codex 대응 키가 없다** — 스펙 §6의 측정 결과를 요약하고, `sandbox_mode`는 **미측정**이라 비워 뒀음을 명시. 현재 Codex 레인의 도메인 준수는 **검사되지 않는다**고 그대로 적는다.
5. **이 머신에 `python3`가 없다** — 등록 커맨드가 인터프리터를 해석하는 이유.
6. **기록물의 `.claude/` 경로는 역사적 표기다** — `.superpowers/**`, `docs/superpowers/plans/2026-09-04-agent-roster.md`는 고치지 않는다.
7. **Before you call a harness change done** — 실행할 명령:

```bash
uv run python .agent-hooks/test_harness_parity.py    # 두 등록 · 레인 등록 · 스킬 대칭
uv run python .agent-hooks/build-agents.py --check   # AGENTS_FRESH
uv run python .agent-hooks/check_rule_links.py       # 모든 포인터가 해석되는가
uv run pytest -q                                     # 위 둘을 수집하는 층 포함
```

8. **Harness work fails silently** — *accepted* ≠ *resolved*. 통과를 보고 동작한다고 결론내지 말고, 막혀야 할 것을 먼저 시도한다. 미측정 항목은 `docs/codex-verification-ledger.md`에 있고, **비어 있는 항목 위에 규칙을 세우지 않는다**.

150줄 예산 안에 둔다.

- [ ] **Step 2: `AGENTS.md` 규칙 표에 행을 추가한다**

`enforcement.md` 행 바로 아래:

```markdown
| `.agents/rules/harness.md` | **before changing anything that configures an agent** — instructions, rules, hooks, registrations |
```

그리고 `## Enforcement hooks` 절에 한 문단을 추가한다: 훅 스크립트는 `.agent-hooks/`에 1벌이고 각 하니스가 자기 파일(`.claude/settings.json` · `.codex/config.toml`, 스키마는 무관)에 등록한다. 레인 소스는 `.agents/agents/`, 스킬 소스는 `.agents/skills/`이며 `build-agents.py`가 양쪽 복사본을 만든다 — **생성물을 손으로 고치지 않는다**.

- [ ] **Step 3: `enforcement.md`의 "Where harness code lives"를 고친다**

`.claude/hooks/` / `.claude/scripts/` 2분할 서술을 `.agent-hooks/` 단일 디렉토리로 바꾼다. 구분은 사라지지 않고 **파일명과 등록**으로 남는다는 점을 적는다: `settings.json`/`config.toml`이 부르면 이벤트 구동, 사람과 테스트가 부르면 스캐너. 마지막 문단("would this exist if the harness didn't?")은 유지한다.

같은 절의 `This project's gates` 표에 3행을 추가한다:

```markdown
| Generated-lane freshness | test (`tests/test_harness_generated.py`) | `.claude/agents/**`·`.codex/agents/**`·양쪽 skills가 `.agents/` 소스와 일치 | none — regenerate |
| Harness parity | test (`tests/test_harness_generated.py`) | 두 등록이 같은 훅을 걸고, 모든 Codex 레인이 등록돼 있고, 스킬이 양쪽에 같은 이름으로 산다 | none |
| Generated-lane rebuild | PostToolUse hook (`build-agents.py --hook`) | 레인/스킬 소스 편집 후 재생성 | advisory, never blocks |
```

- [ ] **Step 4: `self-review.md` ① 게이트 표에 두 줄을 추가한다**

```markdown
| `uv run python .agent-hooks/build-agents.py --check` | 생성된 레인·스킬이 `.agents/` 소스와 일치 (`AGENTS_FRESH`) |
| `uv run python .agent-hooks/test_harness_parity.py` | 두 하니스의 등록이 벌어지지 않았다 |
```

바로 아래 문장("There is no single command that runs all four")의 **four**를 실제 개수로 고친다. 그리고 한 줄을 덧붙인다: `uv run pytest -q`가 이 둘을 이미 호출하므로, **하니스를 건드리지 않은 작업에서는 따로 돌릴 필요가 없다** — 같은 판단의 두 구현을 돌리지 말라는 이 파일 자신의 규칙이다.

- [ ] **Step 5: `architecture.md`의 하니스 설명을 고친다**

`.claude/**` 배치를 서술하는 부분을 새 배치로 바꾼다. `harness-manager`가 소유하는 도메인이 `AGENTS.md` · `.agents/**` · `.agent-hooks/**` · 두 등록 파일임을 명시한다(생성물은 소유가 아니라 산출이다).

- [ ] **Step 6: 게이트 — 링크와 크기를 둘 다 본다**

```bash
uv run python .agent-hooks/check_rule_links.py; echo "exit=$?"
uv run pytest -q
echo '{"tool_name":"Write","tool_input":{"file_path":"AGENTS.md"}}' \
  | uv run python .agent-hooks/check_rules_size.py
```

기대: 링크 findings 0. 크기 훅은 `AGENTS.md`(~165줄)와 `orca-parallel.md`(152줄)를 nudge한다 — 스펙 §2에서 이번엔 줄이지 않기로 했으므로 **예상된 출력**이다.

- [ ] **Step 7: 커밋**

```bash
git add -A
git commit -m "Add harness.md and update the rules that describe where harness code lives"
```

---

### Task 7: Codex 확인 원장

**Files:**
- Create: `docs/codex-verification-ledger.md`
- Modify: `AGENTS.md` (References 절)

**Interfaces:**
- Consumes: Task 5의 `.codex/config.toml`, Task 6의 `harness.md`
- Produces: 사람이 Codex에서 채우는 원장. 채워진 항목은 결론과 고칠 파일을 가리키고, 빈 항목은 `미측정`으로 남아 그 위에 규칙을 세우지 못하게 한다

- [ ] **Step 1: 원장을 쓴다**

`docs/` 규약대로 **한국어**, living 문서(날짜 접두사 없음, 제자리 덮어쓰기). 머리말에 적을 것:

> 이 문서는 **확인 절차가 아니라 답**을 담는다. 이 세션(Claude Code)에서 확인할 수 없는 것만 여기 있고, Codex를 직접 띄우는 사람이 채운다. **비어 있는 항목 위에 규칙을 세우지 않는다** — `미측정`은 "아마 된다"가 아니다.
> 각 항목은 **막혀야 할 것을 먼저 시도**해서 확인한다. 통과만 보고 "동작한다"고 결론내면, 아무것도 안 하는 설정과 동작하는 설정을 구별하지 못한다.

항목 7개를 같은 틀로 쓴다:

```markdown
### 1. Codex가 이 저장소를 trusted로 잡고 `.codex/config.toml`을 읽는가

- **상태**: 미측정
- **왜 중요한가**: 이게 아니면 아래 전부가 무의미하다. 훅도 레인도 등록되지 않은 채
  "설정돼 있다"고 보인다.
- **확인**: 이 디렉토리에서 Codex CLI를 한 번 띄우고 트러스트 프롬프트에서
  'Trust all and continue'를 받는다(IDE는 이 프롬프트를 띄우지 않는다). 그 다음
  `~/.codex/config.toml`에 이 저장소 경로의 `[hooks.state.…]` 항목이 생겼는지 본다.
- **관측**: (비어 있음)
- **결론 → 고칠 파일**: (비어 있음)
```

나머지 6개의 제목·확인 방법:

2. **훅 3개가 실제로 실행되는가** — 음성 케이스 먼저: Codex에서 `uv run python scripts/train_level.py`를 시도해 `block_runtime_commands`가 **막는지** 본다. 그 다음 `.agents/rules/` 아무 파일을 편집해 `check_rules_size`의 nudge가 뜨는지, `.agents/agents/reviewer.md`를 편집해 `build-agents --hook`이 재생성하는지 본다. → 고칠 파일: `.codex/config.toml`의 커맨드 문자열
3. **agent 파일의 `name` / `nickname_candidates`가 사는가 죽는가** — 바이너리 문자열상 agent의 `config_file`은 `ConfigToml`로 파싱되고 거기에 `name`도 `nickname_candidates`도 없다(`AgentRoleToml`에만 있다). 닉네임으로 레인을 부를 수 있는지, `name`을 바꿔도 아무 일이 없는지 확인한다. → 고칠 파일: `.agent-hooks/build-agents.py`의 `render_codex` (죽어 있으면 그 줄을 찍지 않는다)
4. **`model = "gpt-5.6-sol"`가 해석되는가** — custflow에서 가져온 값이다. 레인을 띄워 실제 모델을 확인한다. → 고칠 파일: `.agents/agents/*.md`의 `model.codex`
5. **`sandbox_mode = "read-only"`가 이 Windows에서 뜨는가** — 못 뜨는 샌드박스는 모든 셸 명령을 죽이므로 지금은 비워 뒀다. 뜨면 `reviewer`·`researcher`의 쓰기 차단을 기구로 얻는다. → 고칠 파일: `.codex/agents/*.toml` 생성 규칙 + `harness.md`의 "도메인 준수는 검사되지 않는다" 문장
6. **`block_runtime_commands`가 Codex 셸 페이로드를 읽는가** — 이 훅은 `tool_input.command`만 읽는다. Codex의 셸 호출 페이로드가 같은 모양인지는 미측정이다. 5번과 달리 이건 **이미 등록된 게이트가 무력할 수 있다**는 뜻이라 우선순위가 높다. → 고칠 파일: `.agent-hooks/block_runtime_commands.py`의 페이로드 파싱
7. **6개 레인이 `AGENTS.md`·`.agents/rules/`를 실제로 열고 일하는가** — 레인 하나에 규칙을 인용해야 답할 수 있는 질문을 준다. → 고칠 파일: `.agents/agents/*.md` 본문의 "먼저 읽어라" 절

- [ ] **Step 2: `AGENTS.md` References에 한 줄 추가**

```markdown
- `docs/codex-verification-ledger.md` — Codex 쪽에서 아직 측정되지 않은 것들. **비어 있는 항목
  위에 규칙을 세우지 않는다.**
```

- [ ] **Step 3: 게이트**

```bash
uv run python .agent-hooks/check_rule_links.py; echo "exit=$?"
uv run pytest -q
```

- [ ] **Step 4: 커밋**

```bash
git add -A
git commit -m "Add the Codex verification ledger for what this session cannot measure"
```

---

### Task 8: 최종 검증과 self-review

**Files:**
- Modify: 없음 (발견된 것이 있으면 그것)

**Interfaces:**
- Consumes: Task 1~7 전부
- Produces: PR 설명에 들어갈 ①게이트 출력과 ②판단 기록

- [ ] **Step 1: 남은 `.claude/` 참조를 소스에서 직접 찾는다**

조용한 체커를 증거로 삼지 않는다.

```bash
grep -rn "\.claude/rules\|\.claude/hooks\|\.claude/scripts\|\.claude/skills\|\.claude/agents" \
  AGENTS.md README.md .agents/ .agent-hooks/ src/ tests/ scripts/ docs/ config.yaml \
  --include='*.md' --include='*.py' --include='*.yaml' --include='*.toml' \
  | grep -v "docs/superpowers/plans/2026-09-04-agent-roster.md"
```

남아 있어야 하는 것만 남아야 한다: 생성물임을 설명하는 문장(`.claude/agents/`는 생성물이다 등). 그 외는 고친다.

- [ ] **Step 2: `.claude/` 아래에 남은 것이 등록뿐인지 확인한다**

```bash
find .claude -type f | sort
```

기대: `settings.json`, `settings.local.json`, 생성된 `agents/*.md` 6개, 생성된 `skills/refactor-agent-rules/SKILL.md`. 그 외에 무엇이 있다면 소스가 잘못된 곳에 남은 것이다.

- [ ] **Step 3: ① 게이트를 전부 돌리고 출력을 보관한다**

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run python .agent-hooks/check_rule_links.py; echo "exit=$?"
uv run python .agent-hooks/build-agents.py --check; echo "exit=$?"
uv run python .agent-hooks/test_harness_parity.py; echo "exit=$?"
uv run python -m pytest .agent-hooks/ -q
```

제출물이 없는 작업이므로 zip 검사는 해당 없음. 빨간 것은 리뷰 코멘트가 아니라 **여기서 고칠 것**이다.

- [ ] **Step 4: ② 판단을 글로 쓴다**

`self-review.md` ②의 항목별로. 최소한 다음은 반드시 답한다:

- **재사용**: parity 테스트에서 규칙 링크 검사를 지우고 `check_rule_links.py`를 호출하게 한 것 — 한 판단에 두 구현을 남기지 않았다.
- **게이트를 통과시키려고 추가한 목록 줄**: `GOVERNED`와 `check_rule_links.py`의 소스 루트. 각각 왜 그 값인지, 그리고 그것이 **대상 집합을 넓혔는지 좁혔는지**.
- **거울의 두 짝**: `AGENTS.md` 규칙 표 ↔ `.agents/rules/*.md` 실제 파일, `enforcement.md` 게이트 표 ↔ 실제 게이트, `README.md` 구조 트리 ↔ 실제 트리, 스펙 §7 원장 항목 ↔ `docs/codex-verification-ledger.md`.
- **초록 스위트가 증명하지 못하는 것**: Codex 등록 문법은 오프라인에서 검증되지 않는다. 원장 7개 항목이 그 목록이다.
- **탈출구 마커**: 이번 변경이 추가한 것이 있는가(없으면 없다고 쓴다).

- [ ] **Step 5: `main`을 병합하고 게이트를 병합 상태에서 다시 돌린다**

```bash
git fetch origin main && git merge origin/main
uv run pytest -q
uv run ruff check src tests scripts
git status
```

충돌이 나면 delete/modify는 반사적으로 `--ours/--theirs`로 풀지 않는다 — 의도 판단이다.

- [ ] **Step 6: 사람에게 넘긴다**

`git push`와 PR 생성은 **사람의 승인이 필요한 되돌리기 어려운 동작**이다. ①의 출력과 ②의 판단을 정리해 보고하고, push 여부를 묻는다.

---

## Self-Review (이 계획에 대한)

**스펙 커버리지**: §1 불변식 → Task 6 Step 1 · §2 이동표 → Task 1·2·3·4 (기록물 미갱신 → Task 2 Step 7, Task 8 Step 1) · §2 AGENTS.md 크기 비축소 → Task 6 Step 6이 nudge를 예상 출력로 명시 · §3 생성기 전부 → Task 3 · §3 스킬 → Task 4 · §4 등록 → Task 1 Step 5, Task 3 Step 8, Task 5 Step 1 · §5 게이트 → Task 3 Step 6, Task 5 Step 5, Task 6 Step 3~4 · §6 tools/Codex 사실 → Task 3 Step 2, Task 6 Step 1 · §7 원장 → Task 7 · §8 증명 못 하는 것 → Task 8 Step 4 · §9 범위 밖 → 태스크 없음(의도).

**이름 일관성**: `build-agents.py`의 `--check`/`--hook`/`--root`, 마커 `AGENTS_FRESH`/`AGENTS_STALE`/`AGENTS_UNKNOWN`, 소스 키 `tools.claude`/`model.claude`/`model.codex`, 테스트 파일 `tests/test_harness_generated.py`의 두 함수명이 Task 3·5·6에서 같은 철자로 쓰였다.

**남은 위험**: Task 3 Step 5의 바이트 동일성은 프론트매터 키 순서에 달려 있다. `tools`가 `model` 앞이라는 것은 현 6개 파일에서 확인된 사실이고, Step 2가 그 순서를 유지하라고 명시한다. 어긋나면 Step 5에서 멈춘다 — 조용히 넘어가지 않는다.
