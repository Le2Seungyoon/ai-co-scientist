---
paths:
  - src/**
---
# Architecture

사람(프로젝트 리더) → 메인 Claude(PM) → sub-agent(실무) → `scripts/` CLI → 데이터/GPU/제출.
서버도 프로토콜도 없다. 에이전트 간 상태 공유는 **실험 기록소** 하나로 한다.

## 레이어

| 레이어 | 위치 | 규칙 |
|---|---|---|
| 에이전트 정의 | `.claude/agents/*.md` | 역할 프롬프트만. 코드 없음 |
| 하네스 패키지 | `src/ai_co_scientist/` | `config.py`(설정) · `registry.py`(기록소) · `backends/`(외부 연동). 테스트 대상 |
| 실행 CLI | `scripts/*.py` | sub-agent의 진입점. 얇게 유지 |
| 상태 | `runtime/registry.jsonl` → `docs/experiment-registry.md` | 실험의 단일 진실 |

## 불변식

- **standalone 경계는 2026-08-17 폐기됐다** (Lightning Studio 잠정 폐기와 함께). 근거였던
  "파일 하나만 올려 원격 실행"이 **한 번도 실현되지 않았다** — 기록된 실험 전부가 로컬 GPU에서
  돌았고 기록소의 lightning 언급은 0건이다. 대가는 실재했다: `seed_everything` ×5,
  `ensure_utf8_console` ×3, `PlainMLP`/`UNetSmall`/`SmpModel` 각 ×2 중복, 그리고 스크립트를
  import할 수 없어 `tests/`가 **소스 텍스트 grep** 계약 검사에 머물렀다.
  - 되살릴 조건: 원격 GPU가 실제로 필요해질 때(예: 로컬 8GB를 넘는 학습). git 히스토리에
    삭제 커밋에 scripts/lightning_studio.py 와 src/ai_co_scientist/backends/lightning.py 가 남아 있다
    (현재 트리에는 없다).
  - **이행 중**: 로직은 `src/`로 옮기고 `scripts/`는 얇은 CLI로 되돌린다. 아직 옮기지 않은
    스크립트가 남아 있으므로, 새 코드는 `src/`에 쓰고 스크립트에 로직을 늘리지 말 것.
  - 그래서 이 스크립트들은 `load_config()`를 못 쓰고 **경로 기본값을 `config.yaml` `paths.*`와
    손으로 맞춰야 한다**(`--cache-dir`=`runtime/cache`, `--data-dir`=`data`,
    `--output-dir`=`runtime/ckpt`). 실제로 어긋난 적 있음 — `--cache-dir` 기본값이 `cache`라
    엉뚱한 위치에 35만 장 캐시를 다시 굽기 시작했다. `paths.*`를 바꾸면 두 스크립트도 같이 고칠 것.
- **백엔드는 얇게**: `backends/`는 외부 API 호출만 한다. 상태 저장은 기록소가 맡는다 — 제출
  기록이 두 곳에 흩어지면 어느 쪽이 진실인지 알 수 없게 된다.
- **설정은 `load_config()`로만**: `config.yaml`을 직접 열지 않는다.
- **도메인 라벨은 데이터와 함께 흐른다**: 학습 스크립트는 매니페스트에 `x_domain`/`y_source`를
  남기고, 기록소가 그걸 타깃(real→real)과 대조한다. 이 배선을 끊지 말 것 — sim 지표를 real
  validation으로 착각한 실패가 여기서 막힌다.

## 병렬 실행 계약 (sub-agent)

| 클래스 | 에이전트 | 근거 |
|---|---|---|
| **병렬 가능** | `research` · `critic` · `analyst` | read-only. GPU·제출·쓰기 없음 |
| **배타 (동시 1개)** | `experimenter` | GPU 1장(8GB) · DACON 할당량 · 체크포인트 쓰기 |

`experimenter` 하나가 도는 동안 나머지 셋은 붙여 돌려도 된다 — 실제 병렬 이득은 **실행이 아니라
분석·비평·제안**에서 나온다. GPU가 하나라 학습을 병렬화할 방법은 없다.

**공유 상태의 경쟁 조건** — 둘 다 실측으로 확인하고 고쳤다:

- **기록소 write race**: `report_id`가 `len(records)`에서 나오고 `_write_all`이 파일 전체를 다시
  쓰므로, 락이 없으면 두 에이전트가 같은 ID를 발급받고 뒤 write가 앞 선보고를 덮어쓴다.
  8스레드 동시 등록 시 **8건 중 2건만 생존**했다. → `registry.locked()`가 읽기+쓰기를 함께 감싼다.
  기록소를 바꾸는 새 코드는 반드시 이 안에서 해야 한다.
- **제출 작업 디렉터리 공유**: 파일명이 `test_names.json`에서 오므로 모든 실행이 동일하다.
  병렬 추론 시 서로의 PNG를 덮어써 zip에 다른 모델 출력이 섞이는데, **점수는 정상적으로 나온다**
  — 무엇의 점수인지 알 수 없게 되는 최악의 실패다. → `submission_work/<zip stem>/`로 분리.

## 폐기된 구조 (2026-07-30)

A2A 7-서버(`a2a/`, `agents/`), MCP 서버 5종(`mcp_servers/`), LLM 라우터(`llm/`), toy task.
git 히스토리에 있다. 되살리지 말 것 — 대회 목적은 프로토콜이 아니라 자율 협업이다.
