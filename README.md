# AI Co-Scientist

2025 Samsung AI Challenge - AI Co-Scientist(DACON) 참가 프로젝트.

## 아키텍처 요약

연구팀(리서치·분석) + 엔지니어링팀(Coder·Executor) + 총괄(PM) + 품질/거버넌스(Critic·Harness Engineer)
7개 A2A 에이전트가 로컬 여러 포트에서 각자 프로세스로 돌고, 공유 로그(MCP)로 실험 이력을 주고받는다.

## 빠른 시작

```bash
uv sync                                  # 의존성 설치
cp .env.example .env                     # 실 API 키 연동 시 채울 것 (현재는 전부 mock이라 불필요)
uv run pytest -q                         # 전체 테스트 (mock 기반, API 키 불필요)
uv run ruff check src tests hooks        # lint

uv run python -m ai_co_scientist.runner --skeleton   # M1: PM↔Research 왕복 1회
uv run python -m ai_co_scientist.runner --cycle 3    # 사이클 3바퀴 자동 실행
```

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
uv run --group baseline python scripts/baseline_sem_depth.py --output-dir runtime/baseline_output
# → runtime/baseline_output/submission.zip (10 epoch, val RMSE 최저 시점 모델로 추론)
```

원본은 `docs/[Baseline]_Simulation SEM 영상으로부터 Depth Map 생성 학습.ipynb`(대회 공식 baseline 노트북)를
`scripts/baseline_sem_depth.py`로 포팅한 것 — Windows에서 깨지는 두 지점(`glob` 결과의 경로 구분자 때문에
`path.split('/')[-1]`이 파일명을 잘못 추출하는 문제, `DataLoader(num_workers>0)`에 필요한 `__main__` 가드)만
고쳤고 학습 로직 자체는 동일하다. 시뮬레이션 데이터로만 학습해 실측 test에 그대로 추론하는 구조라 도메인 갭이
점수에 그대로 반영된다 — baseline의 알려진 한계.

### 3. DACON 제출 API 활성화

DACON이 forum(https://dacon.io/forum/403557)에 공식 배포하는 `dacon_submit_api` 패키지는 내부적으로
`POST https://openapi.dacon.io/submission`에 파일을 multipart form으로 올리는 게 전부라, 그 패키지 자체를
설치할 필요 없이 프로젝트에 이미 있는 `httpx`로 직접 호출하도록 `mcp_servers/dacon/server.py`에 포팅해뒀다
(`_real_submit`). 별도 wheel 설치·vendoring 불필요.

1. DACON 마이페이지 > 계정관리에서 개인 토큰 발급 → `.env`의 `DACON_API_TOKEN`에 채움
2. `.env`의 `DACON_CPT_ID`(대회 URL의 숫자), `DACON_TEAM_NAME`(대회 페이지 "Teams" 탭 표기 그대로) 채움
3. `config.yaml`의 `mock.dacon`을 `false`로 바꾸면 Analysis Agent가 사이클 안에서 실제로 제출한다.
   수동으로 한 번만 보내려면:
   ```bash
   uv run python scripts/dacon_submit.py runtime/baseline_output/submission.zip
   # --cpt-id/--team-name 생략 시 .env 값 사용. 성공 시 {'isSubmitted': True, 'detail': 'Success', ...}
   ```

### 4. Lightning AI GPU 잡 제출 활성화

`mcp_servers/lightning/server.py`의 `_real_submit_job`/`_real_poll_job`/`_real_get_credits`가
`lightning-sdk`로 원격 GPU(기본 T4)에 잡을 제출한다. mock과 달리 **비동기**다 — `submit_job`은
잡을 던지고 즉시 `"running"`으로 반환하고, `poll_job`을 호출할 때마다 실제 상태를 조회한다.
로컬 entrypoint 스크립트 하나만 이미지 안으로 base64 인라인해 실행하는 구조라, 별도 데이터
파일에 의존하는 스크립트는 아직 지원 밖이다.

1. https://lightning.ai → Settings → API Key 발급, `.env`의 `LIGHTNING_API_KEY`/
   `LIGHTNING_USER_ID`/`LIGHTNING_TEAMSPACE` 채움
2. `uv sync --group lightning`으로 `lightning-sdk` 설치
3. `config.yaml`의 `mock.lightning`을 `false`로 바꾸면 Executor가 사이클 안에서 실제 T4에 제출한다.
   수동으로 한 번만 검증하려면:
   ```bash
   uv run --group lightning python scripts/lightning_submit.py path/to/entrypoint.py
   ```
4. GPU 종류는 `COSCIENTIST_LIGHTNING_MACHINE`(기본 `T4`), 이미지는
   `COSCIENTIST_LIGHTNING_IMAGE`(기본 `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`) 환경변수로 override 가능.
   크레딧은 teamspace 잔액을 그대로 조회한다 — 리필 주기는 미확인(스펙 §7-②)이라 소진 시
   "시간으로 해결" 대기 로직은 아직 Executor/PM 쪽에 배선돼 있지 않다.

## 프로젝트 구조

```
├── README.md              # 이 파일
├── SPEC.md, docs/SPEC.md  # 설계 스펙 (심사기준 해석·에이전트 아키텍처·실험 플로우)
├── docs/PLAN.md           # 스캐폴딩 계획·마일스톤(M0~M6)
├── config.yaml            # 포트·경로·mock 스위치·임계값
├── rules/                 # <agent>.md — 역할·전략·교훈 (Harness Engineer가 갱신)
├── hooks/                 # 결정적 가드레일 (LLM 호출 없음) — rules/와 동일 레벨, Harness Engineer 소관
├── data/                  # 대회 데이터셋 (git 미추적, "베이스라인 재현" §1 참고해 직접 받아서 채울 것)
├── scripts/               # 일회성 수동 스크립트 — baseline_sem_depth.py, dacon_submit.py
├── src/ai_co_scientist/
│   ├── core/              # config/schema/failure — 메시지 계약, 단일 config 로더
│   ├── llm/                # 역할→모델 라우터 (mock | gemini)
│   ├── a2a/                # A2A 서버 공통 골격 + PM 전용 클라이언트
│   ├── agents/              # pm/research/analysis/coder/executor/critic/harness_engineer
│   ├── mcp_servers/         # shared_log/lightning/dacon/wandb_tools/websearch (mock↔real 스위치)
│   └── toy_task/            # 오프라인 E2E용 합성 데이터·학습 시나리오
└── tests/                  # 각 계층의 mock 기반 결정적 테스트
```
