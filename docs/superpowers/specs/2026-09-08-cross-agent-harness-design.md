# Cross-agent harness — 설계

> 2026-09-08 · 브랜치 `feature/cross-agent-harness`
> 레퍼런스: `C:\Users\user\Desktop\projects\custflow-pipeline` (Claude Code + Codex 병행 운용 중)

## 1. 무엇을 하려는가

지금 이 저장소의 지시·규칙·레인 정의·훅은 전부 `.claude/` 안에 있다. 즉 **Claude Code에서만
전달된다.** 다른 하니스로 이 저장소를 열면 규칙도, 게이트도, 레인도 없다.

이번 변경의 목표는 **지시의 이식성** 하나다. 실험 운영 방식(선보고 → 레지스트리 → 제출)은 바꾸지
않는다. Codex 레인을 실제 워커로 돌리는 것은 이 범위 밖이고, 사람이 직접 확인한다(§7).

**불변식**: *내용과 로직은 1벌, 하니스별로는 등록만.*

| 공유 — 절대 분기시키지 않는다 | 하니스별 — 필연적으로 다르다 |
|---|---|
| `AGENTS.md`, `.agents/**` 전부 | `CLAUDE.md` (한 줄, AGENTS.md를 import) |
| `.agents/agents/**` — 레인 소스 | `.claude/agents/`, `.codex/agents/` — **생성물** |
| `.agents/skills/**` — 스킬 소스 | `.claude/skills/`, `.codex/skills/` — **생성물** |
| `.agent-hooks/` 의 스크립트와 그 테스트 | `.claude/settings.json`, `.codex/config.toml` |

## 2. 디렉토리 이동

`git mv`로 옮긴다 — rename으로 추적돼야 이 크기의 diff가 리뷰 가능하다.

| 이동 전 | 이동 후 | 성격 |
|---|---|---|
| `CLAUDE.md` (151줄) | `AGENTS.md` | 공용 라우터, 본문 1벌 |
| — | `CLAUDE.md` = `@AGENTS.md` | Claude 등록 |
| `.claude/rules/*.md` (8) | `.agents/rules/*.md` | 소스 |
| — | `.agents/rules/harness.md` (신규) | 하니스 변경 규칙 |
| `.claude/agents/*.md` (6) | `.agents/agents/*.md` | 소스 |
| — | `.claude/agents/*.md` · `.codex/agents/*.toml` | 생성물, 커밋됨 |
| `.claude/skills/refactor-agent-rules/` | `.agents/skills/refactor-agent-rules/` | 소스 → 양쪽 복사 |
| `.claude/hooks/*.py` · `.claude/scripts/*.py` (+테스트) | `.agent-hooks/` | 로직 1벌 |
| `.claude/settings.json` · `.local.json` | 제자리 | Claude 등록 전용 |
| — | `.codex/config.toml` (신규) | Codex 등록 전용 |
| — | `.agent-hooks/build-agents.py` (+테스트) | 생성기 |

**경로 참조를 갱신하는 곳**: `README.md`(구조 트리 포함), `src/ai_co_scientist/registry.py`·
`submission.py` 도크스트링, `tests/test_registry.py`·`test_train_manifest.py`·
`test_viterbi_levels.py` 도크스트링, `scripts/legacy/README.md`, `docs/**`.

**갱신하지 않는 곳**: `.superpowers/**`, `docs/superpowers/plans/2026-09-04-agent-roster.md`.
그때 그 경로에서 벌어진 일의 **기록**이고, 고치면 기록이 거짓이 된다. `harness.md`에
"기록물 안의 `.claude/` 경로는 역사적 표기"라고 한 줄 남긴다.

**AGENTS.md의 크기**: 지금도 151줄로 예산(~150) 초과이며 `harness.md` 행과 "이 파일에 규칙을
쓰는 법" 절이 붙어 ~165줄이 된다. **이번에는 줄이지 않는다.** 이 변경은 *이동*이지 *내용 개편*이
아니며, 한 diff에 섞으면 "이 문장이 왜 바뀌었나"를 리뷰할 수 없게 된다. 예산 초과는 이 변경이
만든 문제가 아니라 기존 상태다. 개편은 별도 작업으로 남긴다.

## 3. 생성기 — `.agent-hooks/build-agents.py`

레퍼런스 구현을 이식하되 **도크스트링은 다시 쓴다**. 레퍼런스 상단은 이미 삭제된 `.codex.md`
사이드카를 설명하는 스테일 상태라, 그대로 복사하면 거짓말을 이식하게 된다.

```
.agents/agents/<name>.md    ← 유일한 소스: 하니스 중립 프론트매터 + 공용 본문
   ├→ .claude/agents/<name>.md     프론트매터 + 본문
   └→ .codex/agents/<name>.toml    developer_instructions = 본문
```

**왜 라우팅이 아니라 생성인가.** 두 하니스 모두 레인 정의를 시스템 프롬프트에 넣어 전달하고,
어느 쪽도 경로를 역참조해 주지 않는다 — Claude 서브에이전트 파일의 본문은 *그 자체가* 시스템
프롬프트이고 `@path` import는 확장되지 않는다. Codex의 `developer_instructions`는 리터럴 TOML
문자열이다. "한 파일, 두 포인터" 배치는 보장된 주입을 "모델이 파일을 열어보기를 기대함"으로
바꾼다. 그래서 소스는 하나로 두고 하니스별 파일을 그 **빌드 산출물**로 두되, 하니스가 직접 읽을
수 있도록 커밋한다.

**프론트매터 키.** 이외의 키는 하드 에러다 — 아무것도 생성하지 않는 키는 "고친 것처럼 보이는
수정"이고, 이 파일의 존재 이유가 "소스가 유일한 편집 지점"이기 때문이다.

| 키 | Claude 출력 | Codex 출력 |
|---|---|---|
| `name`, `description` | 그대로 | 그대로 |
| `tools.claude` | `tools:` | 대응 키 없음 (§6) |
| `model.claude` | `model:` | — |
| `model.codex` | — | `model =` |
| `nickname_candidates` | — | 그대로 |

소스에 쓰는 키 이름은 `tools.claude`다(`model.claude`와 같은 규약). 레퍼런스의 `SRC_KEYS`에는
`tools` 계열이 아예 없는데, 그 저장소의 유일한 레인에 `tools:`가 없어서 이 질문을 만난 적이 없을
뿐이며 키 하나를 추가하면 끝난다.

**값은 custflow의 실측값을 그대로 쓴다** — 6개 레인 전부 `model.codex: gpt-5.6-sol`.
`model.claude`는 현재 값(opus/sonnet)을 유지한다. 각 레인 본문 상단의 모델 선택 근거 주석은
Claude 기준이므로 `**Claude Code —**` 라벨을 붙여 양쪽에서 옳게 읽히게 한다.

**이동이 내용을 바꾸지 않았음을 해시로 증명한다.** `tools:`/`model:`을 소스 순서 그대로 되돌려
쓰므로, 지금의 6개 `.claude/agents/*.md`는 생성 후 **바이트 단위로 동일**해야 한다. 이동 전
파일을 보관해 두고 생성 결과와 대조하는 것이 이 단계의 수용 기준이다.

**세 상태 마커** (호출자가 파이썬 트레이스백을 드리프트로 오독하지 못하게):

```
AGENTS_FRESH             exit 0
AGENTS_STALE             exit 1   드리프트한 경로를 전부 나열
AGENTS_UNKNOWN: <why>    exit 2   판정 불가 — 드리프트와 다른 상태다
```

`--check`는 아무것도 쓰지 않는다. `--hook`은 PostToolUse에서 **편집된 파일이 소스일 때만**
재빌드하고, 실패해도 편집을 막지 않고 보고만 한다 (PostToolUse는 deny하지 않는다 —
`enforcement.md` 훅 계약). 페이로드에서 경로를 뽑을 때 `file_path`와 `apply_patch` 형식을 모두
읽는다 — 한쪽만 읽으면 다른 하니스의 편집이 보이지 않는다.

**스킬.** `.agents/skills/<name>/SKILL.md`가 소스이고 양쪽에 그대로 복사된다. 프론트매터는
`name`/`description`만 허용한다(Codex가 미지의 키를 거부). **디렉토리명 ≠ `name:`이면 빌드를
거부한다** — Claude는 디렉토리명으로, Codex는 `name:`으로 스킬을 식별하는데 둘 다 불일치에
에러를 내지 않아 "한 스킬, 두 이름"이 조용히 생긴다. `refactor-agent-rules`는 규칙 파일과 훅
메시지가 이름으로 지목하므로, 깨지면 그 지시가 한쪽에서만 동작한다. 소스 없는 생성 스킬
디렉토리(rename 잔해)도 경고한다.

**Python 3.8 호환으로 작성한다** — 이 머신의 `python`이 3.8이다.

## 4. 등록

**인터프리터를 해석해서 쓴다.** 이 머신에 `python3`는 **없다**(PATH에는 `python`=3.8만).
custflow의 `python3 ...` 커맨드를 그대로 복사하면 훅이 조용히 안 도는 상태가 된다. 양쪽 등록
모두 `command -v python3 || command -v python`으로 해석하고, **둘 다 없으면 조용히 통과하지 않고
그 사실을 한 줄 출력한다** — 침묵은 "검사했고 통과"라는 뜻 하나여야 한다.

**`.claude/settings.json`** — 경로를 `.claude/hooks/` → `.agent-hooks/`로, PostToolUse에
`build-agents.py --hook` 추가. `permissions`·`enabledPlugins`는 그대로.

**`.codex/config.toml`** (신규) — `project_doc_max_bytes`, `project_root_markers = [".git"]`,
훅 3개(PostToolUse: `check_rules_size`, `build-agents --hook` / PreToolUse:
`block_runtime_commands`)를 `git rev-parse --show-toplevel` 기준으로 해석, `[agents.*]` 6개
(`description` + `config_file` + `nickname_candidates`).

`sandbox_mode` / `approval_policy`는 **넣지 않는다.** 이 머신에서 샌드박스가 뜨는지 측정한 적이
없고, 못 뜨는 샌드박스는 모든 셸 명령을 죽인다. §7의 원장 항목으로 넘긴다.

## 5. 게이트와 테스트

| 무엇 | 층 | 왜 그 층인가 |
|---|---|---|
| `test-build-agents.py` | 테스트 | 생성기는 코드다 |
| `test_harness_parity.py` | 테스트 | 두 등록이 벌어지는 것은 어느 훅도 못 본다 |
| `build-agents.py --check`를 pytest에 편입 | 테스트 | 훅은 이번 세션 편집만 본다. 생성물 신선도는 모든 저작 경로를 덮어야 한다 |
| `build-agents.py --hook` | PostToolUse | 빠른 피드백. 테스트의 대체가 아니라 보완 |

- `check_rules_size.py`의 `GOVERNED` → `.agents/rules/*.md` + `AGENTS.md`
- `check_rule_links.py`의 소스 루트 → `.agents/**` + `.agent-hooks/`
- 이 두 파일은 **`harness-spine` 스켈레톤 산물**이고 도크스트링에
  *"do not hand-edit here; reconcile with `harness-spine:update`"* 가 박혀 있다. 이번 이동을
  각 파일의 `DIVERGENCE` 블록에 기록한다 — 안 적으면 다음 `harness-spine:update`가 되돌린다.
- `self-review.md` ① 게이트 표에 `build-agents.py --check`와 parity 테스트를 추가한다.
- `enforcement.md` → "Where harness code lives"를 `.agent-hooks/` 단일 디렉토리로 갱신한다.
  이벤트 구동 / 리포지토리 전역 스캔의 구분은 **파일명과 등록**으로만 남는다.
- `README.md`의 구조 트리와 `.agents/rules/architecture.md`의 하니스 설명은 **한 변경 안에서**
  같이 고친다(양쪽 거울의 두 짝).

## 6. `tools`에는 Codex 대응 키가 없다 — 무엇이 사실인가

설치된 Codex 바이너리에서 확인한 것(문자열 추출이므로 **정황이지 증명이 아니다**):

```
struct AgentRoleToml with 3 elements   → description, config_file, nickname_candidates
AgentRoleOverrides                     → developer_instructions, model_reasoning_effort,
                                         model_reasoning_summary, model_verbosity,
                                         personality, features, skills
struct ToolsToml with 3 elements       → 전역 도구 토글이며 레인별 허용목록이 아니다
```

레인별 `Read, Write, Edit` 같은 도구 허용목록은 없다. 다만 agent의 `config_file`은
`ConfigToml`(99필드) 전체로 파싱되고 거기에 `sandbox_mode` · `approval_policy` ·
`default_permissions`가 있다. 즉 `sandbox_mode = "read-only"`가 `reviewer`/`researcher`의 실질적
대응물이며, **대응물이 없는 것이 아니라 이 머신에서 미측정**이다.

같은 조사에서 나온 것 하나 더: `AgentRoleToml`의 필드는 정확히 3개이고 agent 자기 파일은
`ConfigToml`인데 거기에는 `name`도 `nickname_candidates`도 없다. 그런데 custflow가 생성하는
agent TOML은 `name = / description = / nickname_candidates =` 세 줄을 찍는다. **무시되고 있을
가능성이 크다** — 실제로 사는 것은 `.codex/config.toml`의 `[agents.*]` 쪽이다. 이것이
*accepted vs resolved* 그 자체이므로, 확정 전까지 결론을 내리지 않고 §7 원장 3번으로 넘긴다.

이번 변경에서 이 갭은 **규칙으로 적어 봉인하지 않는다.** 원장이 채워지면 없앤다.

## 7. 원장 — `docs/codex-verification-ledger.md`

Codex 레인은 사람이 직접 돌린다. 이 세션이 확인할 수 없는 것은 규칙이 아니라 **원장**으로 남긴다.
`docs/` 규약대로 한국어, living 문서(날짜 접두사 없음). 항목마다
`상태 / 확인 명령 / 관측 / 결론 → 고칠 파일`.

| # | 확인할 것 | 결론이 바꾸는 것 |
|---|---|---|
| 1 | Codex가 이 repo를 trusted로 잡고 `.codex/config.toml`을 읽는가 | 전체 등록의 전제 |
| 2 | 훅 3개가 실제로 실행되는가 — **막혀야 할 것을 먼저 시도**한다 | 훅 커맨드 문자열 |
| 3 | agent 파일의 `name` / `nickname_candidates`가 사는가 죽는가 | `build-agents.py` 렌더러 |
| 4 | `model = "gpt-5.6-sol"`가 실제로 해석되는가 | `model.codex` 값 |
| 5 | `sandbox_mode = "read-only"`가 이 Windows에서 뜨는가 | 읽기 전용 레인의 쓰기 차단 |
| 6 | `block_runtime_commands`가 Codex 셸 페이로드를 읽는가 | 그 훅의 페이로드 파싱 |
| 7 | 6개 레인이 `AGENTS.md`·`.agents/rules/`를 실제로 열고 일하는가 | 레인 본문의 "먼저 읽어라" 절 |

원장은 **확인 절차가 아니라 답**을 담는다. 채워진 항목은 결론과 그것이 반영된 파일을 가리키고,
비어 있는 항목은 `미측정`으로 남아 **그 위에 규칙을 세우지 못하게 한다**.

## 8. 이 변경이 증명하지 못하는 것

- **등록 문법은 오프라인에서 검증되지 않는다.** `pytest`가 초록이어도 그것은 생성기와 스캐너가
  옳다는 뜻이지, Codex가 `.codex/config.toml`을 받아들인다는 뜻이 아니다. §7이 그 자리다.
- **Codex 레인의 도메인 준수는 검사되지 않는다.** 원장 5번이 열려 있는 동안, 읽기 전용 레인의
  읽기 전용성은 Claude 쪽에서만 기구로 보장된다.
- **훅은 이 세션의 편집만 본다.** IDE·다른 에이전트·동료가 쓴 코드는 어느 훅도 지나지 않는다.
  그래서 생성물 신선도가 테스트 층에 있다.

## 9. 범위 밖

- `AGENTS.md`의 내용 개편·압축 (§2)
- Codex를 실제 실험 워커로 운용하는 것 — `orca-parallel.md`의 Codex 왕복은 여전히 미측정
- 세 번째 하니스
- 파일 경로 단위 쓰기 차단 게이트 — Codex 쪽 쓰기를 PreToolUse 훅으로 막는 안. §6의 원장 5번이
  닫히면 필요 없을 수도 있으므로 그 결과를 보고 판단한다.
