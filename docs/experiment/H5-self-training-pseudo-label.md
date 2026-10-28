---
id: H5
status: 판정
verdict: 기각
axis: 갭
lane: worker-task_205e65519182
registry: [EXP-025]
---

# H5. 실데이터 의사 라벨을 이용한 2단계 자기학습

## 질문
sim GT로 학습한 구조 예측기가 만든 real-train 의사 라벨을 다시 학습 데이터에 넣으면 sim→real 도메인 갭이 줄어드는가? 단순한 재학습 seed 차이보다 큰 개선이 public/private 리더보드 양쪽에서 재현되는지를 묻는다.

## 조건

- **공통 X**: sim train SEM 138,648장. 자기학습 arm만 real train SEM 60,664장을 추가한다. real test SEM은 AdaBN과 최종 추론 외에는 학습·의사 라벨 생성·선택에 사용하지 않는다.
- **y**: sim은 픽셀별 `s = (L-depth)/L` GT, real train은 EXP-005 계열 PlainMLP teacher가 만든 연속값 soft pseudo-label `s`다. 실제 real depth GT는 사용하지 않는다.
- **teacher**: EXP-005 구조 checkpoint를 고정하고, real train 전체에 `--adabn real --adabn-shuffle 42` 조건으로 의사 라벨을 한 번 생성한다. 생성물의 파일 수·shape·유한값·해시를 manifest에 기록한다.
- **student**: PlainMLP, L1 loss, batch size 128, AdamW, learning rate `1e-3`, 15 exposure rounds. teacher checkpoint에서 이어 학습하지 않고 scratch에서 시작한다. cosine schedule은 epoch가 아니라 총 optimizer step을 따른다.
- **arm 0**: sim-only, seed 42. 각 round에 sim 138,648장을 한 번씩 보고, 별도 RNG로 sim 60,664장을 비복원 추가 표집한다.
- **arm 0b**: sim-only, seed 43. arm 0과 동일한 추가 sim 표집 규칙을 쓰며 arm 0과의 차이를 seed 변동폭으로 사용한다.
- **arm 1**: sim 138,648장 + real pseudo-label 60,664장, seed 42. real은 각 round에 전량을 한 번씩만 사용한다.
- **학습량 parity**: 세 arm 모두 round당 199,312 presentations, `drop_last=False`에서 1,558 optimizer steps, 15 rounds에서 총 23,370 steps다. `total_optimizer_steps`, `sim_presentations`, `real_presentations`, `extra_sampler_seed`와 sampler 구성을 manifest에 기록하고 arm parity에서 재계산한다.
- sim 입력 3종(`sim_sem`, `sim_depth`, `sim_case`)과 최종 student checkpoint의 SHA-256을 manifest에 기록한다. `--resume`은 전용 resume state가 없으면 새 학습으로 넘어가지 않고 실패하며, 완료 checkpoint·manifest·중단 state를 덮어쓰지 않는다.
- **공통 추론**: EXP-019의 level 경로와 후처리(`--level-source cnn`, EXP-013 level checkpoint, shuffled real AdaBN seed 42, `tau=0`, level smoothing 9)를 그대로 사용한다.
- **예산**: 구조 학습 3회, 추론 zip 최대 3개, 리더보드 제출 최대 3회. 모든 arm은 같은 구현 commit에서 실행한다.

## 무엇이 답인가

- 1차 지표는 `leaderboard_rmse`이며 `(X, y) = (real test SEM, hidden real depth GT)`다.
- `seed_band = max(|arm0b_public-arm0_public|, |arm0b_private-arm0_private|)`로 정의한다.
- **채택**: arm 1이 arm 0보다 public/private 모두 낮고, 두 split 중 작은 개선폭도 `seed_band + 0.02` 이상이다.
- **조건부**: 두 split 모두 개선했지만 위 여유폭을 넘지 못하거나, 한 split만 개선했다. 이 경우 결론 없음으로 닫고 추가 제출을 자동 승인하지 않는다.
- **기각**: 두 split 모두 악화하거나, 의사 라벨을 넣은 arm이 seed 대조군 범위 밖의 개선을 만들지 못한다.
- **실행 전 중단**: pseudo-label 생성에 test 입력이 섞임, 파일 수/shape 불일치, NaN/Inf, 동일 명령의 artifact 재해시·핵심 계약 비교 불일치, 계획한 optimizer step·sample presentation 불일치, arm 사이에 seed와 데이터 구성 이외의 설정 차이가 발견되면 학습하지 않는다.
- `0.02`는 관측된 무효 수준(EXP-017의 약 0.003)과 명확한 악화(EXP-018의 약 0.028)를 구분하기 위한 사전 운영 임계값이지 통계적 신뢰구간이 아니다.

## 결과

EXP-025를 commit `3dcc750b23f1e43c91be04694f2ad513c6065fcd`에서 직렬 실행했다. 고정 teacher로 만든 real-train 의사 라벨은 독립 2회 생성에서 label SHA-256 `777a49a897d9e4b7cd0cbfd8c3cb23ea6bd7b2d3a0dfdf9ac839ddfc9a7cade7`로 일치했고, 60,664장 모두 shape `[72, 48]`, finite float32였다. 세 arm의 사전·사후 parity gate도 모두 통과했다.

| report_id | arm | 학습 지표 | leaderboard | 비고 |
|---|---|---|---|---|
| EXP-025 | arm 0, sim-only seed 42 | final train L1 0.00573 | public 2.9931130178 · private 2.9437985746 | 제출 1583741 · 194.802초 |
| EXP-025 | arm 0b, sim-only seed 43 | final train L1 0.00574 | public 2.9793695292 · private 2.9341431261 | 제출 1583742 · 190.188초 |
| EXP-025 | arm 1, sim + real pseudo-label seed 42 | final train L1 0.00517 | public 3.2516599469 · private 3.2023513897 | 제출 1583743 · 202.035초 |

세 zip은 공통 EXP-019 추론 조건으로 만들었고 각각 25,988개 파일, maxima `{140, 150, 160, 170}`, 범위 밖 픽셀 0으로 `verify-only`를 통과했다. arm 0(`[EXP-025] H5 arm0 sim-only seed42`) → arm 0b(`[EXP-025] H5 arm0b sim-only seed43`) → arm 1(`[EXP-025] H5 arm1 sim-plus-real-pseudo seed42`) 순서로 재시도 없이 제출했고, 세 요청 모두 `submitted=true`, `verified=true`, `detail=Success`로 끝났다. DACON 화면에서 확인한 제출 ID는 각각 1583741, 1583742, 1583743이며, 가장 낮은 점수는 arm 0b의 public `2.9793695292`, private `2.9341431261`이다.

## 관찰

- arm 0과 arm 0b의 seed 차이는 public `0.0137434886`, private `0.0096554485`이고 사전 정의한 `seed_band`는 `0.0137434886`이다.
- arm 1은 같은 seed의 arm 0보다 public `+0.2585469291`, private `+0.2585528151` 악화했다. 두 split 모두 악화했으므로 seed band나 `0.02` 여유폭을 따질 필요 없이 사전등록 기각 기준을 충족한다.
- arm 1의 final train L1 `0.00517`은 arm 0의 `0.00573`과 arm 0b의 `0.00574`보다 낮았지만 real leaderboard로 전이되지 않았다. 서로 다른 target 혼합에서 계산한 낮은 학습 손실은 도메인 갭 개선의 증거가 아니다.

## 판정 · 기각

**판정**: 기각. 의사 라벨 arm 1이 sim-only arm 0보다 public/private 모두 약 `0.259` 악화해 사전등록 기준을 충족했다.

**미검증**: 다른 teacher, student backbone, pseudo-label 생성법·혼합비 또는 추론 조건에서도 같은 방향인지는 측정하지 않았다. 이 기각은 고정 EXP-005 teacher, scratch PlainMLP student, 등록된 pseudo-label 혼합과 EXP-019 추론 설계에 한정하며 자기학습 일반을 기각하지 않는다.

## 이관 범위

현재 고정 teacher·student·혼합·추론 설계는 다음 실험으로 이관하지 않는다. 다른 teacher, student backbone, pseudo-label 생성법·혼합비 또는 추론 조건은 H5의 기각을 재사용하지 말고 별도 가설로 등록한다.
