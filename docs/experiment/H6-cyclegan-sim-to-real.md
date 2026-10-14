---
id: H6
status: 계획
verdict: 미검증
axis: 갭
lane: null
registry: []
---

# H6. CycleGAN 기반 sim→real 외관 변환

## 질문
sim의 픽셀 대응 depth GT를 보존한 채 외관만 real 쪽으로 옮기면 구조 예측기의 real 도메인 갭이 줄어드는가? 변환기가 기하를 움직이지 않는다는 위생 검사를 먼저 통과한 경우에만 downstream 학습과 리더보드 평가로 진행한다.

## 조건

- **CycleGAN X**: unpaired sim train SEM 138,648장과 real train SEM 60,664장. real test SEM과 모든 depth GT는 변환기 학습에 넣지 않는다.
- **CycleGAN y**: 명시적 depth target 없음. LSGAN adversarial loss와 cycle/identity loss만 사용한다.
- **변환기**: 1-channel ResNet generator 2개(6 residual blocks), 70×70 PatchGAN discriminator 2개, InstanceNorm, Adam `lr=2e-4`, `betas=(0.5, 0.999)`, batch size 8, seed 42. 50 epochs 고정 learning rate 뒤 50 epochs linear decay를 사용한다. `lambda_cycle=10`, `lambda_identity=5`다.
- **기하 위생 gate**: 고정된 sim 검증 표본 2,048장에 대해 원본과 sim→real 변환본의 phase-correlation 이동량 중앙값이 0.5 pixel 이하이고 p95가 1.0 pixel 이하이어야 한다. round-trip `sim→real→sim`의 `[0,1]` 정규화 MAE는 0.10 이하여야 한다. 하나라도 실패하면 downstream 구조 학습 없이 종료한다. 이 gate는 성능 증거가 아니라 픽셀 GT 보존을 위한 안전 조건이다.
- **downstream y**: 변환 전 sim 원본의 픽셀별 `s = (L-depth)/L` GT를 변환본과 동일 좌표로 사용한다.
- **구조 모델**: H5/H7과 같은 PlainMLP, L1 loss, batch size 128, AdamW, `lr=1e-3`, cosine schedule, 15 epochs, seed 42.
- **arm 0**: 원본 sim-only 구조 학습.
- **arm 1**: CycleGAN으로 한 번 변환해 manifest와 함께 고정한 sim→real 이미지로 구조 학습.
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

아직 실행하지 않았다. H7·H8·H5 결과를 본 뒤 고비용 gate를 통과하고 사람의 실행 승인을 받은 경우에만 report_id를 발급한다.

| report_id | arm | 지표 | 비고 |
|---|---|---|---|

## 관찰

사전등록 단계다. 현재 근거는 단순 입력 통계 정합이 실패했다는 점뿐이며, 학습형 변환이 유효하다는 증거는 아니다.

## 판정 · 미검증

**판정**: 미검증.

**미검증**: 외관 변환이 실제 gap을 줄이는지와 기하 보존 gate가 downstream 오류를 충분히 방지하는지 모두 미측정이다.

## 이관 범위

채택되더라도 변환기 checkpoint 단독이 아니라 입력 manifest, 변환 이미지 해시, 기하 gate 결과, downstream checkpoint를 한 묶음으로 이관한다. 다른 해상도·generator·loss 조합에는 결론을 일반화하지 않는다.
