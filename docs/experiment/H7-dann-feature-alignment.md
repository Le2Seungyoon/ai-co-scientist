---
id: H7
status: 측정됨
verdict: 미검증
axis: 갭
lane: worker-task_205e65519182
registry: [EXP-023]
---

# H7. DANN 특징 도메인 정렬

## 질문
구조 회귀기의 중간 특징에서 sim/real 도메인 판별을 어렵게 만들면 real 도메인 갭이 줄어드는가? 동일 코드 경로의 `lambda=0` 대조군과 DANN arm을 직접 비교한다.

## 조건

- **X**: 구조 회귀에는 sim train SEM 138,648장만 사용한다. 도메인 판별에는 sim train과 real train SEM 60,664장을 사용하되 real test는 사용하지 않는다.
- **y**: 구조 회귀 target은 sim depth GT에서 계산한 `s=(L-depth)/L`; 도메인 target은 sim=0, real=1이다. real depth·평균 depth·pseudo-label은 사용하지 않는다.
- **모델**: 기존 PlainMLP의 128차원 bottleneck을 공유하고, domain discriminator `Linear(128,256) → ReLU → Linear(256,256) → ReLU → Linear(256,1)`을 붙인다. discriminator에는 BatchNorm을 넣지 않는다.
- **학습**: L1 structure loss + binary domain loss, batch size 128(각 step에서 sim/real 별도 batch), AdamW `lr=1e-3`, cosine schedule, 15 epochs, seed 42. GRL은 `lambda=lambda_max*(2/(1+exp(-10*p))-1)`, `p=global_step/total_steps`를 optimizer step마다 적용한다. `global_step` 범위는 `[0,total_steps)`라 마지막 값도 정확히 1은 아니다.
- **arm A**: 같은 DANN 코드 경로에서 `lambda_max=0`. real batch의 forward와 encoder BatchNorm 통계 갱신이 남으므로 EXP-005 재현이 아니라 DANN 내부 대조군이다.
- **arm B**: `lambda_max=1.0`.
- sampler와 augmentation의 RNG를 모델 초기화 RNG와 분리한다. arm A/B의 차이는 `lambda_max`뿐이어야 한다. AMP는 두 arm에 동일하게 고정하고 manifest parity로 검사하며 기본값은 off다.
- real train은 site 단위 seed 0 split으로 80% domain 학습, 20% 고정 domain probe에 사용한다. sim probe는 EXP-005 구조 holdout의 depth-map pair 중 real probe와 level별 개수가 같도록 seed 0으로 뽑으며 sim 구조/domain 학습 표본과 겹치지 않는다.
- probe는 sim과 real을 서로 다른 batch로 forward한다. encoder BatchNorm만 batch-stat 모드, 나머지 모듈은 eval 모드로 두고 running buffer를 측정 뒤 복원한다. 매 epoch 기록하되 checkpoint 선택에는 쓰지 않고 final epoch AUC만 기전 지표로 보고한다.
- 최종 epoch checkpoint만 추론용으로 발행하고, 중단 재개용 checkpoint와 discriminator checkpoint는 별도 artifact로 둔다. probe AUC는 리더보드 판정을 대체하지 않는다.
- **공통 추론**: EXP-019의 level 경로와 shuffled real AdaBN seed 42, `tau=0`, level smoothing 9를 사용한다.
- **예산**: 학습 2회, 제출 최대 2회.

## 무엇이 답인가

- 1차 지표는 `leaderboard_rmse`, `(X, y) = (real test SEM, hidden real depth GT)`다.
- **채택**: arm B가 arm A보다 public/private 모두 낮고, 두 split 중 작은 개선폭이 0.02 이상이다.
- **조건부**: 두 split 모두 개선했지만 0.02 미만이거나 방향이 갈린다. domain probe AUC가 낮아졌더라도 자동 채택하지 않는다.
- **기각**: 두 split 모두 악화하거나 한 split에서 0.02 이상 악화한다.
- domain probe AUC가 0.80보다 높으면 정렬이 충분히 일어나지 않았다는 기전상 실패로 기록한다. 단, 최종 판정은 사전등록한 리더보드 비교를 따른다.
- **실행 전 중단**: arm 사이의 manifest·seed·학습 step·optimizer 상태·AMP 설정이 달라졌거나 test 입력이 domain 학습에 들어가면 실행을 폐기하고 report를 갱신한다.
- `0.02`는 EXP-017의 약 0.003 무효 변동보다 크고 EXP-018의 약 0.028 명확한 악화보다 작은 운영 임계값이다.

## 결과

EXP-023을 실행 commit `444209ea0723ad2c7bac6f89a120010743d58ec5`에서 seed 42, AMP off로 직렬 실행했다. 두 arm의 manifest parity는 통과했다. 아래 수치는 최종 epoch의 기전 지표이며 leaderboard 성능은 아직 측정하지 않았다.

| report_id | arm | 지표 | 비고 |
|---|---|---|---|
| EXP-023 | A (`lambda_max=0`) | train L1 0.00616294 · domain loss 0.00996802 · probe AUC 0.99982544 | 232.816초 |
| EXP-023 | B (`lambda_max=1`) | train L1 0.00851187 · domain loss 0.69313661 · probe AUC 0.50318308 | 206.696초 |

두 제출 후보 ZIP은 각각 25,988장, 최대값 `{140,150,160,170}`, 허용 범위 밖 0장으로 검증됐다. 실제 제출은 하지 않았다.

## 관찰

- arm B의 probe AUC 0.5032와 domain loss 0.6931은 사전등록한 기전 기준(AUC 0.80 미만)을 충족한다. 같은 특징에서 sim/real 판별 정보가 arm A보다 크게 줄었다.
- 동시에 arm B의 sim train L1은 arm A보다 높다. 도메인 정보 제거가 구조 회귀 적합도와 맞바뀌었을 가능성이 있으나, 이 수치로 real depth 성능을 추론하지 않는다.
- A/B manifest parity와 제출본 검증은 통과했다. 실제 leaderboard 비교 전에는 채택·기각하지 않는다.

## 판정 · 미검증

**판정**: 미검증.

**미검증**: DANN이 실제 leaderboard gap을 줄이는지는 아직 측정하지 않았다. 사전등록한 public/private 차이와 0.02 임계값 판정에는 A/B 제출 점수가 모두 필요하다.

## 이관 범위

채택 시 128차원 bottleneck·discriminator·GRL schedule과 exact manifest에만 결론을 적용한다. 다른 backbone이나 lambda에는 재검증 없이 일반화하지 않는다.
