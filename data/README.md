# data/

DACON "2025 Samsung AI Challenge - AI Co-Scientist" 대회 데이터셋을 담는 곳.

이 폴더 안의 실제 데이터 파일(csv, zip, 이미지 등)은 용량 문제로 git에 커밋하지 않는다
(`.gitignore` 참고 — 이 README만 추적됨).

```
data/
├── train/                                  실측(real)
│   ├── SEM/Depth_{110,120,130,140}/site_*/*.png   60,664장 / 2,059 사이트 (사이트당 ~31장)
│   └── average_depth.csv                          사이트별 스칼라 2,059행
├── test/
│   └── SEM/*.png                                  25,988장 (익명화·평면, 사이트 정보 없음)
├── simulation_data/                        시뮬레이션(sim)
│   ├── SEM/Case_{1,2,3,4}/*/*.png                 173,304장 (depth map 1장당 itr0/itr1 2장)
│   └── Depth/Case_{1,2,3,4}/*/*.png               86,652장
└── sample_submission.zip
```

**SEM은 원본 영상이 아니라 Hole 단위로 자른 조각이다.** `average_depth.csv`는 그 조각들이
아니라 **원본 전체 영상**의 평균 깊이이므로 예측 대상(조각의 depth map)의 평균이 아니다.
데이터 구조의 확정 사실은 `docs/data-facts.md`를 볼 것 — 실험 설계 전 필독.
