# Codex 확인 원장

이 문서는 **확인 절차가 아니라 답**을 담는다. Claude Code 세션에서 확인할 수 없는 것만 여기 있고,
Codex를 직접 띄우는 사람이 채운다.

**비어 있는 항목 위에 규칙을 세우지 않는다.** `미측정`은 "아마 된다"가 아니다. 어떤 규칙이나 문서가
아래 항목 하나에 기대고 있다면, 그 항목이 채워지기 전까지 그 규칙은 가정이다.

각 항목은 **막혀야 할 것을 먼저 시도**해서 확인한다. 통과만 보고 "동작한다"고 결론내면, 아무것도
하지 않는 설정과 실제로 동작하는 설정을 구별하지 못한다 (`.agents/rules/harness.md` → *accepted*
vs *resolved*).

살아 있는 문서다 — 제자리에서 덮어쓰며, 날짜 접두사를 붙이지 않는다.

---

### 1. Codex가 이 저장소를 trusted로 잡고 `.codex/config.toml`을 읽는가

- **상태**: **확인됨** (2026-09-24, 유지 중인 Orca Codex lane에서 측정)
- **왜 중요한가**: 이게 아니면 아래 전부가 무의미하다. 훅도 레인도 등록되지 않은 채 "설정돼 있다"고
  보인다 — 정확히 이 원장이 막으려는 상태다.
- **확인**: Codex의 `/hooks` 화면에서 project config가 로드됐는지와 각 hook의 Active/Trust를 본다.
  프로젝트 trust와 개별 hook-definition hash trust는 서로 다른 상태이므로, 사용자 config의
  `[hooks.state.…]` 개수로 프로젝트 trust를 추론하지 않는다.
- **관측**: `/hooks`가 `Project config - …/ai-co-scientist/.codex/config.toml`의 PreToolUse를
  `Active`, `Trusted`로 표시했고, 전체 표는 PreToolUse 3/3·PostToolUse 4/4였다. 사용자 config에도
  이 저장소 경로의 `trust_level = "trusted"`가 있다. 반면 child hook hash 항목은 0개였으므로,
  이전 parity NOTE는 서로 다른 trust 상태를 한 값으로 취급한 오진이었다.
- **결론 → 고칠 파일**: `.agent-hooks/test_harness_parity.py`에서 환경 의존적인 trust 추론을 제거했다.
  parity gate는 이제 등록 대칭성과 플랫폼 shell 실행 가능성만 검사하고, live trust는 이 원장과
  `/hooks`가 소유한다.

### 2. 훅 3개가 실제로 실행되는가

- **상태**: **등록 실행 확인, deny 효과 live 미측정** (2026-09-24)
- **확인** — 음성 케이스 먼저:
  1. Codex에서 레지스트리가 없는 트리(예: worktree)로 `uv run python scripts/train_level.py`를
     시도해 `block_runtime_commands`가 **막는지** 본다. 막지 않으면 6번이 원인일 수 있다.
  2. `.agents/rules/` 아무 파일을 예산 초과로 만들어 편집하고 `check_rules_size`의 nudge가 뜨는지 본다.
  3. `.agents/agents/reviewer.md`를 편집하고 `build-agents --hook`이 생성물을 다시 쓰는지 본다.
- **관측**: `/hooks`에서 세 project hook이 active/trusted인 상태에서도 모든 tool call 뒤
  `Hook failed — hook exited with code 1`이 반복됐다. 상세 화면의 명령은 POSIX `$()`와
  `command -v`를 사용했고, 같은 문자열을 Windows 플랫폼 shell에서 실행하자 세 개 모두 exit 1로
  재현됐다. `python -c`로 shell 의존성을 제거한 뒤 parity의 실제 실행 probe는 세 명령 모두
  exit 0이다. 새 hash를 trust한 뒤 `/hooks`는 다시 PreToolUse 3/3·PostToolUse 4/4 active를
  표시했고, 같은 유지 세션에서 새로 실행한 `git status --short --branch` 전후에는 `Hook failed`가
  없었다. 즉 세 등록 명령의 Windows live launch는 확인됐다. 현재 registry 없는 worktree는 모두
  정리돼 있어 1번 음성 케이스는 live로 실행하지 않았다. worktree를 새로 만들지 않는 한 deny
  효과와 Codex payload shape는 6번과 함께 미측정으로 남는다.
- **결론 → 고칠 파일**: `.codex/config.toml`을 플랫폼 중립 Python wrapper로 교체했고,
  `.agent-hooks/test_harness_parity.py`가 각 등록 명령을 실제 플랫폼 shell에서 실행한다.

### 3. agent 파일의 `name` / `nickname_candidates`가 사는가 죽는가

- **상태**: 미측정 (정황만 있음)
- **정황**: 설치된 Codex 바이너리에서 추출한 문자열상 `[agents.*]` 항목은 `AgentRoleToml`이고 필드가
  정확히 3개(`description`, `config_file`, `nickname_candidates`)다. agent 자기 파일은
  `ConfigToml`(99필드)로 파싱되는데 거기에는 `name`도 `nickname_candidates`도 없다. 즉 생성된
  `.codex/agents/*.toml`의 그 두 줄은 **무시되고 있을 가능성이 크다**. 바이너리 문자열은 정황이지
  증명이 아니다.
- **확인**: `nickname_candidates`에 적은 별명으로 레인을 부를 수 있는지. agent 파일의 `name`을 다른
  값으로 바꿨을 때 아무 일도 일어나지 않는지.
- **관측**: (비어 있음)
- **결론 → 고칠 파일**: `.agent-hooks/build-agents.py`의 `render_codex` — 죽어 있으면 그 줄을 찍지
  않는다(찍히지만 아무것도 하지 않는 키가 남아 있는 것이 이 저장소가 가장 싫어하는 상태다)

### 4. `model = "gpt-5.6-sol"`가 실제로 해석되는가

- **상태**: 미측정 (custflow-pipeline에서 가져온 값)
- **확인**: 레인을 하나 띄워 실제로 어떤 모델이 응답하는지 확인한다. 미지의 모델명이 조용히 기본값으로
  떨어지는지, 에러가 나는지도 함께 본다.
- **관측**: (비어 있음)
- **결론 → 고칠 파일**: `.agents/agents/*.md`의 `model.codex`

### 5. `sandbox_mode = "read-only"`가 이 Windows에서 뜨는가

- **상태**: 미측정 — 그래서 `.codex/config.toml`에 **넣지 않았다**
- **왜 비워 뒀는가**: 못 뜨는 샌드박스는 모든 셸 명령을 죽인다. 반대로 뜨기만 하면 `reviewer`와
  `researcher`의 쓰기 차단을 프롬프트 준수가 아니라 **기구**로 얻는다. Claude Code의 `tools:`에
  대응하는 레인별 도구 허용목록은 Codex에 없고, 이것이 가장 가까운 대응물이다.
- **확인**: 레인 하나에만 `sandbox_mode = "read-only"`를 넣고 띄워, ① 셸이 살아 있는지 ② 쓰기가
  실제로 거부되는지 순서로 본다.
- **관측**: (비어 있음)
- **결론 → 고칠 파일**: `.codex/agents/*.toml` 생성 규칙 + `.agents/rules/harness.md`의
  "Codex 레인의 도메인 준수는 검사되지 않는다" 문장

### 6. `block_runtime_commands`가 Codex 셸 페이로드를 읽는가

- **상태**: 미측정 — **우선순위 높음**
- **왜 높은가**: 다른 항목은 "아직 안 켰다"지만 이건 **이미 등록된 게이트가 무력할 수 있다**는 뜻이다.
  이 훅은 `tool_input.command` 하나만 읽는다. Codex의 셸 호출 페이로드가 다른 모양이면 훅은 조용히
  통과하고, 침묵은 "검사했고 통과"와 구별되지 않는다.
- **확인**: 2번의 음성 케이스가 막지 않으면, Codex 쪽 페이로드를 덤프해 `tool_input.command`가 있는지
  직접 본다.
- **관측**: (비어 있음)
- **결론 → 고칠 파일**: `.agent-hooks/block_runtime_commands.py`의 페이로드 파싱 (두 모양을 모두
  읽되, 한쪽을 다른 쪽으로 대체하지 않는다)

### 7. 6개 레인이 `AGENTS.md`·`.agents/rules/`를 실제로 열고 일하는가

- **상태**: 미측정
- **왜 중요한가**: 레인 본문은 "이것들을 먼저 읽어라"라고 말할 뿐이고, 어느 하니스도 그 경로를 대신
  열어 주지 않는다. 규칙이 전달되지 않는 레인은 규칙이 없는 레인이다.
- **확인**: 레인 하나에 규칙을 인용해야만 답할 수 있는 질문을 준다(예: reviewer에게 "선보고에 X와 y를
  적어야 하는 이유가 무엇이고 어느 파일이 그렇게 정하는가").
- **관측**: (비어 있음)
- **결론 → 고칠 파일**: `.agents/agents/*.md` 본문의 "먼저 읽어라" 절
