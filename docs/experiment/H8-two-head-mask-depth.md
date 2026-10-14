---
id: H8
status: 계획
verdict: 미검증
axis: sim 구조
lane: null
registry: []
---

# H8. 마스크·양의 깊이 2-head 구조 회귀

## 질문
zero-inflated `s`를 하나의 L1 회귀값으로 맞추는 대신 마스크와 양의 깊이를 분리하면 동일한 PlainMLP backbone에서 real leaderboard 성능이 좋아지는가? sim holdout 개선만으로 채택하지 않고 동일 commit의 single-head control과 직접 비교한다.

## 조건

- **X**: sim train SEM 138,648장. real test는 최종 추론과 AdaBN 외에는 사용하지 않는다.
- **y**: sim depth GT에서 `s=(L-depth)/L`를 계산한다. `m = 1[s>0]`, `s_pos=s` for `m=1`로 분해한다.
- **공통 backbone**: 기존 PlainMLP, batch size 128, AdamW `lr=1e-3`, cosine schedule, 15 epochs, seed 42.
- **arm A**: 기존 single-head `s` 회귀, 전체 픽셀 L1 loss.
- **arm B**: 같은 backbone에서 mask logit과 positive-depth 두 출력을 만든다. `BCEWithLogits(mask_logit,m) + L1(s_pos_hat,s_pos | m=1)`을 1:1로 합산하고, 추론값은 `sigmoid(mask_logit) * clamp(s_pos_hat,0,1)`로 구성한다.
- arm B의 마지막 출력층 증가(+약 41.8% output-layer parameters)는 구조적 confound로 명시한다. backbone 폭·epoch·optimizer는 변경하지 않는다.
- sim holdout은 site split seed 0으로 고정하고 structure RMSE, mask AUROC, positive-pixel RMSE를 기록한다. 이 값은 위생/기전 지표이며 채택 지표가 아니다.
- **공통 추론**: EXP-019의 level 경로와 shuffled real AdaBN seed 42, `tau=0`, level smoothing 9를 사용한다.
- **예산**: 학습 2회, 제출 최대 2회.

## 무엇이 답인가

- 1차 지표는 `leaderboard_rmse`, `(X, y) = (real test SEM, hidden real depth GT)`다.
- **채택**: arm B가 arm A보다 public/private 모두 낮고, 두 split 중 작은 개선폭이 0.02 이상이다.
- **조건부**: sim holdout만 개선, 양쪽 leaderboard 개선이 0.02 미만, 또는 split 방향이 갈린다. 이 경우 R6의 sim↔real 순위 반전 위험 때문에 결론 없음으로 닫는다.
- **기각**: 두 split 모두 악화하거나 한 split에서 0.02 이상 악화한다.
- **제출 전 중단**: NaN/Inf, mask 양성률이 sim GT 양성률에서 10 percentage points 이상 벗어남, 출력 범위 위반, arm A/B의 backbone·manifest·seed 불일치가 있으면 zip을 만들지 않는다.
- 과거 branch의 실행되지 않은 사후 종결안은 증거로 사용하지 않는다. 현재 문서의 사전 기준으로 새로 판단한다.
- `0.02`는 H7과 같은 운영 임계값이며 통계적 신뢰구간이 아니다.

## 결과

아직 실행하지 않았다. report_id는 전용 실행 lane dispatch 직전에 main checkout에서 발급한다.

| report_id | arm | 지표 | 비고 |
|---|---|---|---|

## 관찰

사전등록 단계다. sim 구조 오차가 가장 큰 예산 성분이지만, EXP-020/R6 때문에 sim 개선이 real 개선으로 이어진다는 가정은 금지한다.

## 판정 · 미검증

**판정**: 미검증.

**미검증**: zero-inflated target 분해가 real 성능을 높이는지와 출력층 증가가 효과의 원인인지 아직 분리되지 않았다.

## 이관 범위

채택 시 2-head target 정의·1:1 loss·soft product 추론까지 한 단위로 이관한다. loss weight, hard threshold, backbone을 바꾸면 별도 가설로 다시 등록한다.
