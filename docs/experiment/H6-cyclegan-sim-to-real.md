---
id: H6
status: 계획
verdict: 미검증
axis: 갭
lane: worker-task_205e65519182
registry: [EXP-026]
---

# H6. CycleGAN 기반 sim→real 외관 변환

## 질문
sim의 픽셀 대응 depth GT를 보존한 채 외관만 real 쪽으로 옮기면 구조 예측기의 real 도메인 갭이 줄어드는가? 변환기가 기하를 움직이지 않는다는 위생 검사를 먼저 통과한 경우에만 downstream 학습과 리더보드 평가로 진행한다.

## 조건

- **CycleGAN X와 고정 분할**: 전체 sim cache 173,304장을 `sem.map_level_split(sim_case, val_frac=0.2, seed=42)`로 나눈다. CycleGAN은 그중 train 138,648장과 real train SEM 60,664장을 unpaired로 학습하고, sim validation 34,656장은 학습에서 제외한다. real test SEM과 모든 depth GT도 변환기 학습에 넣지 않는다.
- **CycleGAN y**: 명시적 depth target 없음. LSGAN adversarial loss와 cycle/identity loss만 사용한다.
- **변환기**: 1-channel ResNet generator 2개(6 residual blocks), 70×70 PatchGAN discriminator 2개, InstanceNorm, Adam `lr=2e-4`, `betas=(0.5, 0.999)`, batch size 8, seed 42. 50 epochs 고정 learning rate 뒤 50 epochs linear decay를 사용한다. `lambda_cycle=10`, `lambda_identity=5`다.
- **기하 위생 gate**: 위 34,656장 validation pool에서 seed 42로 뽑은 고정 전역 인덱스 2,048장에 대해 원본과 sim→real 변환본의 phase-correlation 이동량 중앙값이 0.5 pixel 이하이고 p95가 1.0 pixel 이하이어야 한다. round-trip `sim→real→sim`의 `[0,1]` 정규화 MAE는 0.10 이하여야 한다. 경계값은 통과이고 NaN은 실패다. 정지 규칙은 정확히 이 세 기준의 AND이며, 하나라도 실패하면 downstream 구조 학습 없이 종료한다(`evaluate_gate`의 `passed`·`failures`). 이 gate는 성능 증거가 아니라 픽셀 GT 보존을 위한 안전 조건이다.
- **국소 진단 probe**: 같은 표본에서 24×24 타일 NCC 블록 매칭 변위(`local_probe`)를 기록하지만 진단 전용이며 판정에 관여하지 않는다 — 통과를 실패로도, 실패를 통과로도 바꾸지 않는다. 미보정이다: 합성 입력에서 기하를 바꾸지 않는 블러+감마에도 플래그를 올리고, 한 타일 안에 중심이 있는 대칭 팽창은 보지 못한다. 전역 phase correlation도 대칭 팽창과 국소 왜곡을 보지 못하므로, 이 gate가 형태 보존을 보증하지 않는다는 것이 사전등록된 잔여 위험이다. probe를 정지 규칙으로 올리려면 실제 sim 표본에서 외관 전용 변환으로 보정한 뒤 이 문서를 개정해야 한다.
- **downstream y**: 변환 전 sim 원본의 픽셀별 `s = (L-depth)/L` GT를 변환본과 동일 좌표로 사용한다.
- **구조 모델**: H5/H7과 같은 PlainMLP, L1 loss, batch size 128, AdamW, `lr=1e-3`, cosine schedule, 15 epochs, seed 42.
- **arm 0**: 원본 sim-only 구조 학습.
- **arm 1**: CycleGAN으로 sim 173,304장 전체를 한 번 변환해 manifest와 함께 고정한다. 변환기 학습에는 train 138,648장만 쓰지만, downstream `train_structure.py`가 원본과 똑같은 138,648/34,656 paired split을 다시 만들 수 있도록 depth/case와 행 수를 그대로 보존한다. 구조 학습은 `--cache-manifest <cache-dir>/manifest.json`을 반드시 함께 받아 배열을 열기 전에 manifest를 재검증하고, 그 파일의 절대경로·SHA-256·report/hypothesis/domain provenance를 checkpoint에 결속한다. 이 SHA-256은 자유롭게 편집 가능한 JSON의 인증 서명이 아니라 검증 뒤 변경을 탐지하기 위한 결속이다.
- **공통 추론**: EXP-019의 level 경로, shuffled real AdaBN seed 42, `tau=0`, level smoothing 9를 유지한다.
- **예산**: CycleGAN 1회, 구조 학습 최대 2회, 제출 최대 2회. 다른 가설의 control을 재사용하려면 코드 commit·입력 manifest·seed·모든 hyperparameter가 동일해야 하며, 하나라도 다르면 재학습한다.

## 무엇이 답인가

- 1차 지표는 `leaderboard_rmse`, `(X, y) = (real test SEM, hidden real depth GT)`다.
- **채택**: arm 1이 arm 0보다 public/private 모두 낮고, 두 split 중 작은 개선폭이 0.02 이상이다.
- **조건부**: 양쪽 모두 개선했지만 0.02 미만이거나 split 방향이 갈린다. 이 경우 변환 예시와 기하 gate를 보존하되 후속 채택은 하지 않는다.
- **기각**: 기하 위생 gate 실패, 두 split 모두 악화, 또는 한 split에서 0.02 이상 악화한다.
- CycleGAN loss와 §7 입력 통계 유사도는 기전 관찰값일 뿐 채택 지표가 아니다. 입력 통계가 real에 가까워져도 리더보드가 나빠질 수 있다는 EXP-018의 반증을 유지한다.
- `0.02`는 H5와 같은 운영 임계값이며 통계적 신뢰구간이 아니다.

## 결과

EXP-026은 이미 발급되었다. 첫 실행은 GPU 학습 전에 전체 cache 173,304장을 train 수 138,648장으로 잘못 검증한 계약 오류로 거부되었고, checkpoint나 gate 결과는 생성되지 않았다. 위 분할 설명은 이 pre-GPU 실패 뒤 기존 EXP-005 paired split과 사전등록의 train/validation 역할을 명시한 것이며, 학습 결과를 본 뒤 조건을 바꾼 것이 아니다.

| report_id | arm | 지표 | 비고 |
|---|---|---|---|

## 관찰

사전등록 단계다. 현재 근거는 단순 입력 통계 정합이 실패했다는 점뿐이며, 학습형 변환이 유효하다는 증거는 아니다.

2026-09-24에 실제 `sim_sem.npy`의 **과거 혼합 pool** 표본 2,048장(seed 42)으로 CPU-only 합성
진단을 수행했다. 당시 표본은 수정 전 `range(138648)` sampler에서 뽑혀, 현재 분할 기준 train
1,640장과 validation 408장이 섞였고 현재 validation-only gate 표본과 겹치는 행은 22장뿐이다.
따라서 아래 표는 수정된 gate 표본을 calibration한 결과가 아니라 legacy/historical diagnostics다.
정확한 생성 코드도 커밋돼 있지 않으므로 수치를 재생성 가능하다고 해석하지 않는다. 모델·GPU·
depth GT·registry는 사용하지 않았고 어떤 runtime 산출물도 쓰지 않았다. 아래 값은 각각 표본별
이동량의 median / p95이며, 이 계약 수정으로 임계값이나 local probe의 진단 전용 역할은 바꾸지
않는다.

| 변환 | 전역 phase correlation | local probe | 전역 판정 |
|---|---:|---:|---|
| blur `(0.8, 1.2)` + gamma `1.6` | 0.008 / 0.016 | 0.399 / 0.451 | 통과 |
| blur `(1.5, 0.8)` + gamma `1.8` | 0.098 / 0.252 | 0.713 / 1.250 | 통과 |
| 비순환 x 이동 `0.5 px` | 0.456 / 0.489 | 0.520 / 0.529 | 통과 |
| 비순환 x 이동 `1.0 px` | 0.999 / 1.004 | 1.019 / 1.025 | 실패 |
| 중심 확대 `1.02×` | 0.016 / 0.035 | 0.633 / 0.664 | 통과 |
| sinusoidal x warp 진폭 `0.75 px` | 0.018 / 0.043 | 0.614 / 0.658 | 통과 |

이 과거 진단은 local probe를 정지 규칙으로 올리지 않는 참고 근거다. 강한 외관 전용 변환이 확대·국소
warp보다 더 큰 local 값을 내므로 현재 값에는 둘을 가르는 임계값이 없다. 동시에 전역 기준이 1px
이동은 막지만 확대·국소 warp는 놓친다는 잔여 위험을 실제 sim 표본에서도 재현했다. 따라서 현행
3기준 gate는 강체 이동과 cycle 붕괴를 거르는 제한된 위생 검사로만 해석하고, 형태 보존 보증으로
승격하지 않는다. 이 calibration은 CycleGAN 출력 자체의 분포를 측정한 것이 아니므로 성능 또는
실제 generator의 안전성 증거도 아니며, 수정된 validation-only gate 표본의 calibration도 아니다.

## 판정 · 미검증

**판정**: 미검증.

**미검증**: 외관 변환이 실제 gap을 줄이는지와 기하 보존 gate가 downstream 오류를 충분히 방지하는지 모두 미측정이다.

## 이관 범위

채택되더라도 변환기 checkpoint 단독이 아니라 입력 manifest, 변환 이미지 해시, 기하 gate 결과, downstream checkpoint를 한 묶음으로 이관한다. 다른 해상도·generator·loss 조합에는 결론을 일반화하지 않는다.
