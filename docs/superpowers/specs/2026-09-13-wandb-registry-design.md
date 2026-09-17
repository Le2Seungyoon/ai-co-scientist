# 실험 기록소를 wandb로 — 전환 제약과 단계 계획

> 2026-09-13 · 브랜치 `feature/split-infer-stages`에서 **조사만** 함
> 이 문서는 **완결된 설계가 아니다.** 오늘 측정된 사실과 반드시 풀어야 할 제약을 기록하고,
> 설계되지 않은 부분을 §5에 명시한다. 실제 설계는 별도 사이클의 브레인스토밍에서 한다.
> 함께 쓰인 문서: `2026-09-13-split-infer-stages-design.md` (추론 단계 분리 — 이번 사이클)

## 1. 출발점의 오해를 먼저 정정한다

레퍼런스로 삼은 `custflow-pipeline`은 수치 근거를 **MLflow**에 의존한다
(`include/custflow/registry/client.py`가 유일한 touchpoint, `.agents/rules/mlflow.md`가 규칙을
소유). 그래서 "mlflow를 wandb로 대체"라는 표현이 나왔지만, **이 저장소에는 대체할 MLflow가
없다.** 여기서 수치 근거는 두 개다:

- `runtime/registry.jsonl` — 선보고·결과·리더보드 점수·판정의 단일 진실
- DACON 리더보드 — 유일한 최종 판정 (`config.yaml` → `target`: real→real GT는 우리에게 없다)

따라서 실제 작업은 **JSONL 레지스트리 → wandb** 다. custflow의 MLflow 경험은 그대로 옮겨오는
설계가 아니라, "기록소가 서버에 있을 때 무엇이 달라지는가"의 참고 사례로만 쓴다.

## 2. wandb는 이미 절반 들어와 있고, 나머지 절반은 의도적으로 막혀 있다

| 사실 | 위치 |
|---|---|
| `wandb>=0.18`이 이미 의존성 | `pyproject.toml` |
| `WANDB_*` 시크릿이 `.env` 관례에 등재 | `AGENTS.md` → Secrets 표 |
| `WANDB_API_KEY` 없으면 `mode="disabled"` 로 도는 전례 | `scripts/legacy/baseline_sem_depth.py` |
| **현행 경로는 wandb를 import하지 않는다** | `scripts/legacy/` 만 import |
| 그 배제가 **테스트로 고정**돼 있다 | `tests/test_train_manifest.py` → "no wandb import chain reaching inference" |
| legacy가 wandb를 최상단 import하는 것이 현행 경로가 그 파일들을 import하지 않는 이유 중 하나 | `scripts/legacy/README.md` → 주의 |

즉 이것은 신규 도입이 아니라 **과거에 내린 배제 결정을 되돌리는 일**이다. `testing.md`는
"모든 invariant 테스트에 소스가 들 수 있는 탈출구를 주라, 그러지 않으면 첫 정당한 예외에서
테스트가 삭제된다"고 정해두었다 — 그 첫 정당한 예외가 이 전환이다. 테스트를 조용히 지우지 말고
**명시적으로 바꾸는 것**이 이 조항의 요구다.

## 3. 왜 옮기려 하는가 — 병렬 실험이 만드는 압력

`2026-09-13-split-infer-stages-design.md`가 CPU 후처리 가설을 워크트리에서 병렬 실행 가능하게
만든다. 그 설계는 기록 측면에서 다음을 **손으로** 푼다:

- `report_id = f"EXP-{len(records)+1:03d}"` 포크 위험 → "레지스트리는 코디네이터 독점"이라는 규율
- 레코드에 git provenance가 전혀 없음 → 선보고에 브랜치/커밋 필드를 새로 추가
- 워커가 결과를 직접 못 씀 → `worker_done` outcome으로 회수해 코디네이터가 대신 기록

서버 기반 기록소는 이 셋을 **구조적으로** 없앤다: run id를 서버가 발급하므로 포크가 불가능하고,
provenance가 내장이며, 워커가 각자 직접 기록해도 안전하다. sweep은 우리가 하려는 arm 스윕과
같은 개념이다. 이것이 전환의 동기다 — 이번 사이클에서 손으로 푼 것들이 다음 사이클에서
사라진다는 뜻이기도 하다.

## 4. 반드시 풀어야 할 제약 (측정됨)

### 4.1 게이트 센티넬이 무효화된다 — 가장 큰 것

`.agent-hooks/block_runtime_commands.py`의 판정은 **`<project root>/runtime/registry.jsonl`의
존재 여부** 하나다. `runtime/`이 gitignore이므로 워크트리에는 그 파일이 없고, 그것이 곧
"여기는 실험을 돌릴 트리가 아니다"의 신호로 쓰인다.

기록소가 wandb로 가면 이 파일은 사라지거나 의미를 잃고, **게이트 전체가 조용히 무효가 된다.**
훅이 deny를 멈춘 것과 통과시킨 것이 구분되지 않는 상태 — `enforcement.md`가
"Silence means exactly one thing"으로 금지하는 실패 양식이다. 대체 센티넬을 먼저 정하지 않고
기록소를 옮기면 안 된다.

### 4.2 레지스트리는 저장소이자 강제 게이트다

`registry.new_report()`는 선보고 5항목(X / y / 모델·하이퍼 / 방법론 / 목적)과 지표의 (X, y)가
모두 없으면 **report_id 발급을 거부**한다. `workflow.md`의 "MANDATORY before any ML experiment"가
여기서 기계적으로 집행된다. 그 실패(도메인 불일치를 아무도 진술하지 않아 여러 런을 헛돌린 사건)의
재발 방지선이므로, 저장소만 옮기고 강제력을 흘리면 규칙이 프로스로 되돌아간다.

`metric_matches_target()`의 경고 박아두기도 같은 층에 있다.

### 4.3 오프라인 테스트 원칙과의 긴장

`testing.md`는 테스트가 GPU·캐시·네트워크 없이 돌 것을 요구하고, 실제로 기존 테스트들이
합성 데이터로 계약을 고정한다. wandb는 네트워크 서비스다. legacy의
`mode="online" if os.environ.get("WANDB_API_KEY") else "disabled"` 패턴이 전례로 쓸 만하지만,
"disabled일 때 무엇이 보장되는가"를 계약으로 못 박아야 한다 — 기록이 조용히 사라지는 모드는
`registry.jsonl`이 주던 보장보다 약하다.

### 4.4 생성 문서

`docs/experiment-registry.md`는 `scripts/exp.py render`의 생성물이고 손편집이 금지돼 있다
(`docs.md` → 생성 문서는 손대지 않는다). 기록소가 바뀌면 **생성기가 바뀌어야** 하며, 이 문서는
사람이 읽는 유일한 실험 일람이므로 전환 중에도 끊기면 안 된다.

### 4.5 기존 20건

현재 `runtime/registry.jsonl`에 EXP-001~020이 있고, 그중 다수가 채택 판정과 리더보드 점수를
들고 있다(EXP-019 LB 3.0493 등). 이관할지, 동결하고 신규만 wandb로 갈지는 결정 사항이다.
`registry.py`의 `RESET_NOTICE`가 "2026-07-29 이전은 폐기"라는 선례를 이미 만들어 두었다.

## 5. 아직 설계되지 않은 것 — 다음 사이클의 브레인스토밍 대상

- **대체 센티넬의 형태.** 무엇이 "여기는 실험 트리다"를 말하는가. runtime/ 하위의 다른 파일인가,
  워크트리 여부의 직접 판정인가, 환경변수인가. §4.1 없이는 전환 착수 불가.
- **선보고 강제력의 이식처.** wandb run을 여는 얇은 래퍼가 5항목을 요구하는 형태가 유력하나
  미설계. custflow의 `start_tagged_run`이 참고 사례.
- **disabled 모드 계약.** 네트워크 없이 돌 때 무엇이 보장되고 무엇이 유실되는가.
- **이관 범위.** 기존 20건 / 신규만 / 동결.
- **리더보드 점수의 자리.** 점수는 API가 주지 않고 사람이 리더보드에서 읽어 `exp.py lb`로 넣는다.
  이 수동 경로가 wandb에서 어떤 모양이 되는가.

## 6. 단계 계획 (초안 — 설계 후 확정)

1. 대체 센티넬 설계·구현·테스트 (§4.1). **기록소는 아직 건드리지 않는다.**
2. wandb run 래퍼 — 선보고 5항목 강제 + provenance + disabled 모드 계약 (§4.2, §4.3).
3. 이중 기록 기간: JSONL과 wandb에 동시 기록해 두 기록이 일치하는지 실측.
4. `render` 생성기를 wandb 소스로 전환 (§4.4). 문서가 끊기지 않는지 확인.
5. `test_train_manifest.py`의 wandb 배제 항목을 명시적으로 교체 (§2).
6. JSONL을 읽기 전용 기록으로 동결하거나 이관 (§4.5).

## 7. 범위 밖

- custflow-pipeline의 MLflow → wandb 마이그레이션. 별개 저장소, 별개 결정.
- 학습 지표의 실시간 스트리밍. 지금 필요가 확인되지 않았다.
