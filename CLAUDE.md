# ai-co-scientist

2025 Samsung AI Challenge — AI Co-Scientist 후속. **여러 자율 에이전트가 협업해 딥러닝 과제를
푸는 것**이 목적이다(대회 취지). 과제는 SEM 이미지 → depth map 회귀. 사람은 프로젝트 리더,
메인 Claude가 PM/오케스트레이터, 실무는 `.claude/agents/`의 sub-agent가 맡는다.

A2A 다중서버·MCP 서버 구조는 2026-07-30에 제거했다 — 이 규모에서는 과설계였고, 대회가 요구하는
것은 특정 프로토콜이 아니라 자율 협업이다. 지금은 sub-agent가 `scripts/` CLI를 직접 호출한다.

## 핵심 불변식

- **선보고 없이 실험 없음** — X / y / 모델·하이퍼 / 방법론 / 목적을 먼저 기록소에 등록해야
  `report_id`가 나온다. `.claude/rules/workflow.md` → Experiment Pre-Report.
- **기록소가 단일 진실** — `docs/experiment-registry.md`(렌더) ← `runtime/registry.jsonl`.
  2026-07-29 리셋 이전 실험·결론은 전부 폐기(void)이며 인용하지 않는다.
- **판정은 리더보드** — 지표의 (X, y)가 타깃(real SEM → real depth)과 다르면 그 지표로 real
  성능을 주장하지 않는다.
- **데이터 사실이 우선** — depth map 구조(배경 레벨 L 4택1 + 정규화 구조)와 `average_depth`의
  의미는 `docs/data-facts.md`가 확정본이다. 이 값과 어긋나는 코드·문서는 버그로 취급한다.
- **테스트는 오프라인** — API 키 없이 `uv run pytest -q` 전체 통과.
- **로직은 `src/`, `scripts/`는 얇게** — standalone 제약은 2026-08-17 폐기됐다(Lightning 잠정
  폐기). 근거가 한 번도 실현되지 않은 채 중복만 낳았다. 이행이 끝나지 않았으니 **새 코드는
  `src/`에 쓰고 스크립트에 로직을 늘리지 말 것**. `.claude/rules/architecture.md` 참조.
- **병렬은 `experimenter`만 배타** — sub-agent를 동시에 돌릴 때 GPU·제출을 쓰는 `experimenter`는
  1개, `research`/`critic`/`analyst`는 병렬 가능. `.claude/rules/architecture.md` → 병렬 실행 계약.

## Commands

```bash
uv sync                                   # 환경 재현
uv run pytest -q                          # 테스트 (오프라인)
uv run ruff check src tests scripts # 린트 (line-length 100)

uv run python scripts/exp.py new --title ... # 선보고 → report_id
uv run python scripts/exp.py list | render   # 기록소 조회 / 문서 갱신

# 현행 파이프라인 — depth = L(레벨 4택1) × (1 − s(정규화 구조)).  docs/data-facts.md §2
uv run python scripts/train_level.py                              # 레벨 분류기 (real→real)
uv run python scripts/train_structure.py --arch mlp               # 구조 회귀기 (백본 교체 가능)
uv run python scripts/dacon_submit.py runtime/submissions/EXP-0NN-arm.zip --report-id EXP-0NN
```

### 현재 최고 기록 재현 — LB **3.0493** / private 2.9961 (EXP-019)

```bash
uv run python scripts/infer_decomposed.py \
    --ckpt runtime/ckpt/EXP-005-structure.pt \
    --level-source cnn --level-ckpt runtime/ckpt/EXP-013-level-cnn.pt \
    --adabn real --adabn-shuffle 42 --tau 0.0 --level-smooth 9 \
    --submit runtime/submissions/<report_id>.zip
```

학습 불필요 — 두 ckpt는 `runtime/ckpt/`에 있다. **네 플래그가 전부 실험으로 얻은 것**이라
하나라도 빼면 점수가 떨어진다: `--adabn real`(EXP-010, −0.52) · `--adabn-shuffle 42`
(EXP-016, **−0.51**) · `--level-source cnn`(EXP-014, −0.47) · `--level-smooth 9`(EXP-019, −0.07).
`--tau 0.0`은 클램프 없음이 최적(EXP-005·007 전 구간).

**제출본은 반드시 검증한다**: 25,988장 · 모든 이미지의 max가 {140,150,160,170} · 이탈 0.00%.

분해 이전 스크립트는 `scripts/legacy/`로 옮겼다 — 재현 전용이며 현행 경로가 아니다
(`scripts/legacy/README.md`).
`--arch smp:*`는 `uv run --group baseline`이 필요하다. 8GB 카드에서 학습이 죽으면 **배치를
줄이기 전에 원인부터 가릴 것** — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`는 Windows에서
쓰면 안 되고(0x10E 유발), PC가 통째로 재부팅되는 건 OOM이 아니라 드라이버 버그체크다.
`.claude/rules/coding-patterns.md` Gotchas 참조.

## Configuration

| What | Where |
|------|-------|
| Secrets (DACON_API_TOKEN, WANDB_API_KEY) | `.env` (`.env.example` 복사) |
| Tunables (paths, target, train defaults) | `config.yaml` via `ai_co_scientist.config.load_config()` |
| Dependencies | `pyproject.toml` + `uv.lock` (`uv`로만 관리, `uv pip install` 금지) |

## Rules

| File | When to read |
|------|--------------|
| `.claude/rules/workflow.md` | **모든 실험 전** (Experiment Pre-Report) · 계획 · 학습 기록 |
| `.claude/rules/git-workflow.md` | 파일을 바꾸는 모든 작업 전 |
| `.claude/rules/architecture.md` | 하네스 구조 · 레이어 경계 · **sub-agent 병렬 계약** |
| `.claude/rules/coding-patterns.md` | `src/**`, `scripts/**` 수정 시 |
| `.claude/rules/testing.md` | 테스트 작성/수정 시 |

## 인계 메모 (2026-08-17 세션 종료 시점)

- **최고 기록 LB 3.0493 / 2.9961 (EXP-019).** 대조군 EXP-003의 7.3491 대비 −58.5%.
  큰 개선 넷 중 **셋이 학습 0회**로 나왔다 — 값싼 축을 먼저 훑는 것이 이 과제에서 유효했다.
- **다음 후보는 `docs/hypotheses.md` 활성 표에서 고른다.** 세 예산 성분이 4.3 / 2.5 / 2.4로
  평평해져 압도적 레버가 없다. 재검토 1순위는 **최대 성분인 `sim 구조 한계`(46.7%)** 인데
  R6로 막혀 있고, 그 R6는 갭 6 이상 구간에서 관측돼 현재 갭 1.59에서는 **미검증**이다 —
  백본 하나만 다시 재면 갈린다.
- **커밋되지 않은 변경이 크게 쌓여 있다** (하네스·docs 추적 전환, Lightning 제거,
  `scripts/legacy/` 이동, 기록소 락, `src/ai_co_scientist/sem.py` 신설). 사용자 지시로
  커밋을 보류했다 — 새 세션에서 먼저 `git status`를 확인하고 커밋 여부를 물을 것.
- **미완**: 코드 정리 3단계(`train_sem_depth.py`의 smp 모델 동물원을 흡수)는 안 했다.
  `scripts/legacy/`는 재현 전용으로 **동결**이며 손대지 않는다.

## References

- `docs/data-facts.md` — 데이터 구조 확정 사실 (측정 + 운영자 공식 답변). **실험 설계 전 필독**
- `docs/experiment-registry.md` — 실험 기록소 (유일한 실험 근거)
- `docs/hypotheses.md` — 가설 백로그 + **오차 예산**. 다음 실험을 고를 때 여기서 기대 이득을
  계산한다. 기각 사유도 남아 있으니 재논의 전에 확인할 것
- README — 데이터 준비 / 베이스라인 재현 / 실 백엔드 활성화
