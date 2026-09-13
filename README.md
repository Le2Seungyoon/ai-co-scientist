# AI Co-Scientist

2025 Samsung AI Challenge - AI Co-Scientist(DACON) 참가 프로젝트.

## 아키텍처 요약

사람(프로젝트 리더) → 메인 Claude(PM) → `.claude/agents/`의 sub-agent(researcher·reviewer·
engineer·harness-manager·analyst·executor) → `scripts/` CLI → 데이터/GPU/제출. 별도 서버·
프로토콜 없이 메인 Claude 세션 안에서 sub-agent를 호출하는 구조이고, 에이전트 간 상태 공유는 실험 기록소
(`runtime/registry.jsonl` → `docs/experiment-registry.md`) 하나로 한다.

## 빠른 시작

```bash
uv sync                                   # 의존성 설치
cp .env.example .env                      # 실 백엔드(DACON/wandb) 연동 시 채울 것
uv run pytest -q                          # 전체 테스트 (오프라인, API 키 불필요)
uv run ruff check src tests scripts # lint
```

## 실험 실행 절차

1. 선보고 등록 → `report_id` 발급
   ```bash
   uv run python scripts/exp.py new --title "..." --x-domain sim --x-desc "..." \
     --y-source sim_depth_gt --y-desc "..." --model "..." --method "..." --purpose "..." \
     --metric-name sim_val_rmse --metric-x sim --metric-y sim_depth_gt
   ```
2. 학습 (로컬 GPU) → `scripts/train_level.py` / `scripts/train_structure.py`
3. 결과 기입 → `scripts/exp.py result <report_id> --val '<json>'`
4. 제출 → `scripts/infer_decomposed.py --submit ...` → `scripts/dacon_submit.py --report-id <report_id>`
5. 리더보드 점수 확인 후 → `scripts/exp.py lb <report_id> --public ... --private ...`
6. 문서 갱신 → `scripts/exp.py render`

## 베이스라인 재현

### 1. 데이터 다운로드

[2022 Samsung AI Challenge (3D Metrology)](https://dacon.io/competitions/official/235954/data) 페이지에서
(DACON 로그인 + 대회 참가 신청 필요) Data 탭 다운로드:

- `train.zip` — 실측 SEM 60,664장 + `average_depth.csv`(사이트별 평균 depth 스칼라, 실측엔 픽셀 단위 정답이 없음)
- `simulation_data.zip` — 시뮬레이터 생성 SEM+Depth 페어 259,956장(SEM은 depth 1장당 itr0/itr1 2장)
- `test.zip` — 실측 SEM 25,988장 (정답 없음, 채점용)
- `sample_submission.zip` — 제출 포맷 참고용

압축 해제해서 `data/` 밑에 다음 구조로 배치 (`data/`는 `.gitignore`로 git 미추적, `data/README.md`만 추적):

```
data/
├── simulation_data/{SEM,Depth}/Case_*/.../*.png
├── train/{SEM/.../*.png, average_depth.csv}
├── test/SEM/*.png
└── sample_submission.zip
```

### 2. 베이스라인 학습 + 제출 파일 생성

```bash
uv sync --group baseline                              # torch(cuda12.4)/opencv/pandas/scikit-learn/tqdm 추가
uv run --group baseline python scripts/legacy/baseline_sem_depth.py --output-dir runtime/baseline_output
# → runtime/baseline_output/submission.zip (10 epoch, val RMSE 최저 시점 모델로 추론)
```

대회 공식 baseline 노트북을 `scripts/legacy/baseline_sem_depth.py`로 포팅한 것 — Windows에서 깨지는 두 지점(`glob` 결과의 경로 구분자 때문에
`path.split('/')[-1]`이 파일명을 잘못 추출하는 문제, `DataLoader(num_workers>0)`에 필요한 `__main__` 가드)만
고쳤고 학습 로직 자체는 동일하다. 시뮬레이션 데이터로만 학습해 실측 test에 그대로 추론하는 구조라 도메인 갭이
점수에 그대로 반영된다 — baseline의 알려진 한계.

### 2-1. 현행 파이프라인 (baseline이 아니라 이쪽이 현역)

depth = L(배경 레벨 4택1) × (1 − s(정규화 구조)) 분해 구조다 (`docs/data-facts.md` §2).

```bash
uv run python scripts/train_level.py                        # 레벨 분류기 (real→real)
uv run python scripts/train_structure.py --arch mlp         # 구조 회귀기 (백본 교체 가능)
```

**최고 기록 LB 3.0493 / private 2.9961 (EXP-019) 재현** — 학습 불필요, 두 ckpt는 `runtime/ckpt/`에 있다:

```bash
uv run python scripts/infer_decomposed.py \
    --ckpt runtime/ckpt/EXP-005-structure.pt \
    --level-source cnn --level-ckpt runtime/ckpt/EXP-013-level-cnn.pt \
    --adabn real --adabn-shuffle 42 --tau 0.0 --level-smooth 9 \
    --submit runtime/submissions/<report_id>.zip
```

네 플래그가 전부 실험으로 얻은 것이라 하나라도 빼면 점수가 떨어진다. 제출 전 zip 검증 필수:
25,988장 · 모든 이미지의 max가 {140,150,160,170} · 이탈 0.00%.

분해 이전 스크립트(`train_sem_depth.py`, `infer_submit.py` 등)는 `scripts/legacy/`에 있다 —
재현 전용으로 동결이며 현행 경로가 아니다 (`scripts/legacy/README.md`).

### 3. DACON 제출 API 활성화

DACON이 forum(https://dacon.io/forum/403557)에 공식 배포하는 `dacon_submit_api` 패키지는 내부적으로
`POST https://openapi.dacon.io/submission`에 파일을 multipart form으로 올리는 게 전부라, 그 패키지 자체를
설치할 필요 없이 프로젝트에 이미 있는 `httpx`로 직접 호출하도록 `src/ai_co_scientist/backends/dacon.py`에
포팅해뒀다(`submit`). 별도 wheel 설치·vendoring 불필요.

1. DACON 마이페이지 > 계정관리에서 개인 토큰 발급 → `.env`의 `DACON_API_TOKEN`에 채움
2. `.env`의 `DACON_CPT_ID`(대회 URL의 숫자), `DACON_TEAM_NAME`(대회 페이지 "Teams" 탭 표기 그대로) 채움
3. 제출:
   ```bash
   uv run python scripts/dacon_submit.py runtime/submissions/EXP-001.zip --report-id EXP-001
   # --cpt-id/--team-name 생략 시 .env 값 사용. 성공 시 {'isSubmitted': True, 'detail': 'Success', ...}
   # --report-id를 주면 memo에 report_id가 붙어 리더보드 행 ↔ 기록소 항목이 이어진다.
   ```

## 프로젝트 구조

```
├── README.md              # 이 파일
├── CLAUDE.md              # 하네스 라우터 (에이전트가 먼저 읽는 파일)
├── docs/data-facts.md     # 데이터 구조 확정 사실 — 실험 설계 전 필독
├── docs/hypotheses.md     # 가설 백로그 + 오차 예산
├── docs/experiment-registry.md # 실험 기록소 렌더 문서 (scripts/exp.py render 산출물)
├── config.yaml            # 경로·타깃 도메인·학습 기본값
├── .claude/agents/        # researcher/reviewer/engineer/harness-manager/analyst/executor — 역할 프롬프트
├── .claude/rules/         # 이 저장소에서 일하는 규칙 (agents/와 다른 층)
├── .claude/skills/        # refactor-agent-rules — 지시 파일 재구조화 판단 방법
├── data/                  # 대회 데이터셋 (git 미추적, "베이스라인 재현" §1 참고해 직접 받아서 채울 것)
├── scripts/               # exp.py · train_level.py · train_structure.py · infer_decomposed.py ·
│   │                      # probe_level.py · dacon_submit.py
│   └── legacy/            # 분해 이전 스크립트 — 재현 전용 동결
├── src/ai_co_scientist/
│   ├── config.py          # config.yaml 단일 로더 + .env 로더 + UTF-8 콘솔 가드
│   ├── registry.py        # 실험 기록소 (runtime/registry.jsonl 읽기/쓰기/렌더)
│   ├── sem.py             # 분할·재매개화·평활·QDA 등 순수 로직
│   └── backends/          # dacon.py (제출 API)
└── tests/                 # config·registry·sem·backends·train manifest 결정적 테스트
```
