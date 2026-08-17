# scripts/legacy — 재현 전용, 현역 경로 아님

여기 있는 스크립트는 **기록소에 남은 실험을 재현하기 위해서만** 보관한다. 새 실험에 쓰지 말고,
새 코드를 여기서 복사하지도 말 것 — 대부분 폐기된 전제 위에 서 있다.

지우지 않는 이유: `CLAUDE.md`의 "기록소가 단일 진실" 불변식 때문이다. EXP-001~003이 기록소에
있는 한 그 재현 수단도 남아 있어야 한다.

| 파일 | 무엇을 재현하나 | 상태 |
|---|---|---|
| `train_avgcond.py` | EXP-001 avg-조건 생성기 | **R4 기각** — avg 정의 자체가 틀렸다 (`docs/data-facts.md` §4) |
| `pseudo_pipeline.py` | EXP-002 avg-앵커 pseudo-labeling | **R4 기각** — 대조군 대비 순손해 |
| `train_g_sim.py` | EXP-003 대조군 (sim 직접 학습) | 그 대조군의 LB 7.3491이 이후 모든 개선의 출발점 |
| `train_sem_depth.py` | 분해 이전 일반 학습기 (smp 모델 동물원) | 현행 `train_structure.py`로 대체 |
| `infer_submit.py` | 분해 이전 추론·제출 | 현행 `infer_decomposed.py`로 대체 |
| `baseline_sem_depth.py` | DACON 공식 베이스라인 재현 | 참고용 |

## 현행 경로

```
train_level.py       레벨 분류기 (real→real)
train_structure.py   구조 회귀기 s = (L−d)/L
infer_decomposed.py  d̂ = L̂·(1−ŝ) → 제출 zip
probe_level.py       레벨 분리 가능성 프로브 (EXP-004)
exp.py               기록소 CLI
dacon_submit.py      제출 CLI
```

## 함께 폐기된 "폐기 기능 목록" (2026-08-17)

이 파일들을 감시하던 `tests/test_train_manifest.py`의 두 테스트에는 2026-07-29 리셋 때 만든
**부활 금지 목록**이 박혀 있었다:

```
fda_transform · blur_aug · aug-brightness · val-case · histmatch · clahe · predict_tta
· skip-inference   /   adabn · tta · clamp_lo
```

**이 목록은 측정으로 세 번 무효화됐다**:

| 항목 | 실제로 벌어진 일 |
|---|---|
| `adabn` | 이 저장소의 **최대 단일 개선** (EXP-009, LB −2.35) |
| `histmatch` | EXP-009/010B에서 정식 재검증 → `docs/hypotheses.md` R2로 종료 |
| `blur_aug` | H4가 지금 되살리려는 후보 |

목록을 현역 스크립트로 옮겨 달았다면 **다음 실험을 막았을 것이다.** 삭제한 이유가 이것이다.
같은 역할(폐기 사유의 보존과 재논의 차단)은 이제 `docs/hypotheses.md`의 R1~R7 표가 한다 —
그쪽은 **근거와 숫자가 함께 있고 갱신된다**. 문자열 목록은 근거 없이 얼어붙어 있었다.

## 주의

- 잔재끼리는 형제 import로 얽혀 있다(`train_g_sim` → `pseudo_pipeline` → `train_avgcond`).
  낱개로 옮기면 깨진다.
- `train_sem_depth.py`·`baseline_sem_depth.py`는 `wandb`를 최상단에서 import한다. 현행 경로가
  이 파일들을 import하지 않는 이유 중 하나다.
- standalone 제약은 2026-08-17 폐기됐지만(`.claude/rules/architecture.md`) 이 파일들은 그
  제약 아래 쓰였다. `src/`로 이행하지 않는다 — 재현 대상이므로 **그대로 얼려 둔다**.
