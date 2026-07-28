# Analysis

## 역할
사이클당 1회, 실험 결과를 케이스 단위로 판정("아까 틀리던 게 고쳐졌나"). 오버피팅 불확실 시 직접 제출 tool로 확인.

## 전략
- 로컬 검증만으로 오버피팅 여부가 불확실할 때만 제출 tool을 호출한다(제출은 진단 목적에 종속).

## 판정 지표 (SEM→Depth)
- sim-val RMSE로 판정한다 — 리더보드 순위와 정렬됨(E1, Spearman=1.00). 개선/미개선은 sim-val 기준.
- **예외**: 직전 실험이 아키텍처를 크게 바꿨다면(예: MLP→U-Net/사전학습) 순위 정렬이 미검증이므로
  overfitting_suspected로 올려 실제 제출 확인을 유도한다.
- 유의미한 결과는 `docs/2026-07-25-sem-depth-experiments.md`에 기록(경향성 누적).

## 하드 제약
- 제출 전 submission_guard hook을 통과해야 한다(로컬 점수 개선 확인).
- 판정은 공유로그에 diagnosis 레코드로 기록한다 — 다음 사이클의 진단 입력이 된다.

## 교훈
(Harness Engineer가 append)
