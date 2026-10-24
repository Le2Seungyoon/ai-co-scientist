---
id: H8
status: 판정
verdict: 채택
axis: sim 구조
lane: worker-task_205e65519182
registry: [EXP-024]
---

# H8. 마스크·양의 깊이 2-head 구조 회귀

## 질문
zero-inflated `s`를 하나의 L1 회귀값으로 맞추는 대신 마스크와 양의 깊이를 분리하면 동일한 PlainMLP backbone에서 real leaderboard 성능이 좋아지는가? sim holdout 개선만으로 채택하지 않고 동일 commit의 single-head control과 직접 비교한다.

## 조건

- **X**: sim train SEM 138,648장. real test는 최종 추론과 AdaBN 외에는 사용하지 않는다.
- **y**: sim depth GT에서 `s=(L-depth)/L`를 계산한다. `m = 1[s>0]`, `s_pos=s` for `m=1`로 분해한다.
- **공통 backbone**: 기존 PlainMLP, batch size 128, AdamW `lr=1e-3`, cosine schedule, 15 epochs, seed 42.
- **arm A**: 기존 single-head `s` 회귀. 출력 `sigmoid(logit)`, 전체 픽셀 L1 loss.
- **arm B**: 같은 backbone(encoder + decoder 마지막 Linear 이전까지)에서 두 head를 낸다. depth head는 arm A 출력층과 같은 초기화값을 재사용하고 **arm A와 같은 sigmoid**를 거친다(`s_pos_hat = sigmoid(depth_logit)`). mask head는 새 `Linear(1024, H·W)`다.
  - 손실: `BCEWithLogits(mask_logit, m) + L1(sigmoid(depth_logit), s | m=1), 1:1`
  - 추론: `sigmoid(mask_logit) * sigmoid(depth_logit)` — soft product, hard threshold 없음.
- **마스킹이 걸리는 정확한 지점**: `m = 1[s>0]`는 GT에서 학습 시에만 만든다. BCE는 배치 전체 픽셀 평균, L1은 **배치 안 m=1 픽셀 전체의 평균**(이미지별 평균 아님)이며 배치에 m=1이 없으면 0이다. m=0 픽셀은 depth head에 gradient를 주지 않는다. 두 head 사이 stop-gradient는 없다 — 두 손실 모두 공유 trunk로 역전파된다.
- **arm 간 차이 (단일 변수가 아니다)**: 동일 초기화에서 arm B의 depth 경로는 arm A와 함수적으로 같다(`tests/test_two_head.py`가 bit 단위로 고정). 그 위에 다음 세 가지가 **한 묶음**으로 달라진다.
  1. mask head 추가 — 출력층 2배(+100%), 전체 파라미터 8,468,736 → 12,011,136 (+41.8%).
  2. 손실 분해 — 전체 픽셀 L1 → BCE + m=1 픽셀 L1. depth head는 m=0 픽셀에서 학습 신호를 잃는다.
  3. 출력 조립 — 단일 sigmoid → 두 sigmoid의 곱.
  이 셋은 이 설계에서 분리할 수 없다. 따라서 이 실험이 답하는 것은 **"2-head 묶음 vs single"**이며, 효과를 zero-inflation 분해 자체에 귀속하지 않는다(용량 증가·손실 형태와 구별 불가). backbone 폭·epoch·optimizer·배치 순서(전용 generator)는 변경하지 않는다.
- sim holdout은 site split seed 0으로 고정하고 structure RMSE, mask AUROC, positive-pixel RMSE를 기록한다. 이 값은 위생/기전 지표이며 채택 지표가 아니다. mask AUROC의 점수원은 arm마다 다르다(arm A `s_hat`, arm B `sigmoid(mask_logit)`) — 두 arm끼리 비교하지 않는다.
- **공통 추론**: EXP-019의 level 경로와 shuffled real AdaBN seed 42, `tau=0`, level smoothing 9를 사용한다.
- **예산**: 학습 2회, 제출 최대 2회.

## 무엇이 답인가

- 1차 지표는 `leaderboard_rmse`, `(X, y) = (real test SEM, hidden real depth GT)`다.
- **채택**: arm B가 arm A보다 public/private 모두 낮고, 두 split 중 작은 개선폭이 0.02 이상이다.
- **조건부**: sim holdout만 개선, 양쪽 leaderboard 개선이 0.02 미만, 또는 split 방향이 갈린다. 이 경우 R6의 sim↔real 순위 반전 위험 때문에 결론 없음으로 닫는다.
- **기각**: 두 split 모두 악화하거나 한 split에서 0.02 이상 악화한다.
- **제출 전 중단**: NaN/Inf, 출력 범위 위반, arm A/B의 backbone·manifest·seed 불일치, 또는 mask 붕괴가 있으면 zip을 만들지 않는다.
  - **mask 붕괴 게이트가 걸리는 정확한 지점** (arm B만): 학습 **마지막 epoch의 sim holdout**(split seed 0, val_frac 0.2)에서 `sigmoid(mask_logit) > 0.5`인 픽셀 비율과 같은 holdout의 GT `s>0` 비율을 비교해, 차이가 10 percentage points 이상이면 실패다. 판정은 학습 시점에 manifest(`sim_mask_gate`)에 고정된다 — ckpt는 진단용으로 저장되지만 `infer_two_head.py`가 zip 생성을 거부한다.
  - real test의 mask 양성률은 **로깅 전용**이다. real depth GT가 없어 sim GT 비율과 비교할 근거가 없으므로 게이트로 쓰지 않는다.
- 과거 branch의 실행되지 않은 사후 종결안은 증거로 사용하지 않는다. 현재 문서의 사전 기준으로 새로 판단한다.
- `0.02`는 H7과 같은 운영 임계값이며 통계적 신뢰구간이 아니다.

## 결과

EXP-024를 실행 commit `9532525128b59d8bdf38a0eafc3197d3297f76dc`에서 seed 42, AMP off로 직렬 실행했다. 두 arm의 manifest parity와 추론 게이트는 통과했다. 아래 수치는 마지막 epoch의 sim holdout 위생 지표이며 채택 지표가 아니다.

| report_id | arm | sim holdout 지표 | leaderboard | 비고 |
|---|---|---|---|---|
| EXP-024 | A (`single`) | depth RMSE 2.06512102 · positive RMSE 0.01879547 · mask AUROC 0.99773738 | public 3.0578249986 · private 3.0049607176 | 제출 1583722 · 254.674초 |
| EXP-024 | B (`two_head`) | depth RMSE 2.00870747 · positive RMSE 0.01832591 · mask AUROC 0.99989729 | public 2.9460719789 · private 2.8952703674 | 제출 1583723 · 274.122초 |

arm B의 마지막 sim holdout mask 양성률은 예측 0.48798323, GT 0.48820918로 차이가 약 0.000226이어서 0.10 붕괴 게이트를 통과했다. 두 제출 후보 ZIP은 각각 25,988장, 최대값 `{140,150,160,170}`, 허용 범위 밖 0장으로 검증됐다. 2026-09-24에 A(`single`)를 먼저 제출해 `submitted=true`를 확인한 뒤 B(`two_head`)를 제출했으며, 두 요청 모두 `verified=true`, `detail=Success`로 끝났다. B는 A보다 public `0.1117530197`, private `0.1096903502` 개선했다.

## 관찰

- arm B의 sim holdout depth RMSE는 arm A보다 0.05641355 낮았고 positive-pixel RMSE도 낮았다. mask 붕괴 없이 사전등록한 2-head 묶음이 sim 지표를 개선했다.
- A/B 추론의 output positive rate는 각각 0.49027485와 0.49001481로 비슷했고, B의 real-test mask rate는 0.48814989였다. real test mask 수치는 로깅 전용이며 성공 판정에 쓰지 않는다.
- 이번에는 sim holdout의 방향이 real leaderboard 양쪽으로 전이됐다. 다만 R6의 반례는 그대로이므로 이것을 sim holdout 일반의 유효성으로 확대하지 않는다. 같은 backbone과 학습 조건에서 사전등록한 2-head 묶음의 paired 비교에만 해당한다.

## 판정 · 채택

**판정**: 채택. arm B가 arm A보다 public/private 모두 낮고, 더 작은 개선폭도 `0.1096903502`로 사전등록 임계값 `0.02`를 넘었다.

**미검증**: 효과가 zero-inflated target 분해·mask head 용량 증가·손실 형태·soft product 추론 중 무엇에서 오는지는 이 설계로 분리되지 않는다 — 분리하려면 별도 가설(예: 파라미터를 맞춘 single-head control)이 필요하다.

## 이관 범위

2-head target 정의·sigmoid depth head·1:1 loss·soft product 추론까지 한 단위로 이관한다. loss weight, hard threshold, backbone을 바꾸면 별도 가설로 다시 등록한다.
