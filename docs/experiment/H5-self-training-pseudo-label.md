---
id: H5
status: 계획
verdict: 미검증
axis: 갭
lane: null
registry: []
---

# H5. 실데이터 의사 라벨을 이용한 2단계 자기학습

## 질문
sim GT로 학습한 구조 예측기가 만든 real-train 의사 라벨을 다시 학습 데이터에 넣으면 sim→real 도메인 갭이 줄어드는가? 단순한 재학습 seed 차이보다 큰 개선이 public/private 리더보드 양쪽에서 재현되는지를 묻는다.

## 조건

- **공통 X**: sim train SEM 138,648장. 자기학습 arm만 real train SEM 60,664장을 추가한다. real test SEM은 AdaBN과 최종 추론 외에는 학습·의사 라벨 생성·선택에 사용하지 않는다.
- **y**: sim은 픽셀별 `s = (L-depth)/L` GT, real train은 EXP-005 계열 PlainMLP teacher가 만든 연속값 soft pseudo-label `s`다. 실제 real depth GT는 사용하지 않는다.
- **teacher**: EXP-005 구조 checkpoint를 고정하고, real train 전체에 `--adabn real --adabn-shuffle 42` 조건으로 의사 라벨을 한 번 생성한다. 생성물의 파일 수·shape·유한값·해시를 manifest에 기록한다.
- **student**: PlainMLP, L1 loss, batch size 128, AdamW, learning rate `1e-3`, cosine schedule, 15 epochs. teacher checkpoint에서 이어 학습하지 않고 scratch에서 시작한다.
- **arm 0**: sim-only, seed 42.
- **arm 0b**: sim-only, seed 43. arm 0과의 차이를 seed 변동폭으로 사용한다.
- **arm 1**: sim + real pseudo-label, seed 42. sim/real 샘플 수 차이 때문에 real이 과대표집되지 않도록 epoch당 real 샘플 수를 sim 샘플 수 이하로 제한하고 sampler 구성을 manifest에 기록한다.
- **공통 추론**: EXP-019의 level 경로와 후처리(`--level-source cnn`, EXP-013 level checkpoint, shuffled real AdaBN seed 42, `tau=0`, level smoothing 9)를 그대로 사용한다.
- **예산**: 구조 학습 3회, 추론 zip 최대 3개, 리더보드 제출 최대 3회. 모든 arm은 같은 구현 commit에서 실행한다.

## 무엇이 답인가

- 1차 지표는 `leaderboard_rmse`이며 `(X, y) = (real test SEM, hidden real depth GT)`다.
- `seed_band = max(|arm0b_public-arm0_public|, |arm0b_private-arm0_private|)`로 정의한다.
- **채택**: arm 1이 arm 0보다 public/private 모두 낮고, 두 split 중 작은 개선폭도 `seed_band + 0.02` 이상이다.
- **조건부**: 두 split 모두 개선했지만 위 여유폭을 넘지 못하거나, 한 split만 개선했다. 이 경우 결론 없음으로 닫고 추가 제출을 자동 승인하지 않는다.
- **기각**: 두 split 모두 악화하거나, 의사 라벨을 넣은 arm이 seed 대조군 범위 밖의 개선을 만들지 못한다.
- **실행 전 중단**: pseudo-label 생성에 test 입력이 섞임, 파일 수/shape 불일치, NaN/Inf, 동일 명령의 해시 불일치, arm 사이에 seed와 데이터 구성 이외의 설정 차이가 발견되면 학습하지 않는다.
- `0.02`는 관측된 무효 수준(EXP-017의 약 0.003)과 명확한 악화(EXP-018의 약 0.028)를 구분하기 위한 사전 운영 임계값이지 통계적 신뢰구간이 아니다.

## 결과

아직 실행하지 않았다. report_id는 전용 실행 lane을 dispatch하기 직전에 main checkout에서 발급한다.

| report_id | arm | 지표 | 비고 |
|---|---|---|---|

## 관찰

사전등록 단계다. 결과 해석은 seed 대조군과 양쪽 리더보드 split을 함께 본 뒤 기록한다.

## 판정 · 미검증

**판정**: 미검증.

**미검증**: real pseudo-label이 실제 도메인 갭 정보를 전달하는지, 아니면 teacher 오류를 증폭하는지 아직 측정하지 않았다.

## 이관 범위

채택되면 pseudo-label manifest, student checkpoint, 재현 명령과 추론 zip만 다음 실험의 입력으로 이관한다. teacher·데이터·추론 조건이 하나라도 달라지면 H5의 결론을 재사용하지 않는다.
