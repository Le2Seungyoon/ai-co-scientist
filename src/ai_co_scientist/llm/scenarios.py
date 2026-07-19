"""MockLLM 시나리오 테이블 — (호출 인덱스, 입력) -> 구조화 응답의 결정적 매핑.

시나리오가 반환하는 코드는 실제로 파일에 기록·lint·실행되므로
"마지막 줄에 metrics JSON 출력" 계약을 지켜야 한다 (Executor가 파싱).
"""

GOOD_CODE = '''\
"""MockLLM 생성 실험 스크립트 (결정적) — 마지막 줄에 metrics JSON 출력."""
import json

train_mse = {train}
val_mse = {val}
print(json.dumps({{"train_mse": train_mse, "val_mse": val_mse}}))
'''

BUGGY_CODE = '''\
"""MockLLM 생성 실험 스크립트 — 의도적 버그(F821 undefined name)."""
import json

print(json.dumps({"train_mse": train_mse, "val_mse": val_mse}))
'''

SLOW_CODE = '''\
"""MockLLM 생성 실험 스크립트 — 1차 실행 타임아웃 유도(복구 룰 검증용)."""
import json
import time

time.sleep({sleep})
print(json.dumps({{"train_mse": 0.2, "val_mse": 0.25}}))
'''

TOY_CODE = '''\
"""MockLLM 생성 실험 코드 — toy 데이터 실학습 (degree={degree}, alpha={alpha}, train_rows={train_rows})."""
import csv
import json
import os
from pathlib import Path

import numpy as np


def load(name):
    path = Path(os.environ["COSCIENTIST_TOY_DATA"]) / name
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    return x, y


def features(x, degree):
    # x∈[-2,2] → [-1,1] 스케일 후 거듭제곱 — 고차에서 정규방정식 조건수 안정화 (BLAS 플랫폼 편차 방지)
    return np.vstack([(x / 2.0) ** d for d in range(degree + 1)]).T


x_train, y_train = load("train.csv")
x_val, y_val = load("val.csv")
x_hold, _ = load("holdout.csv")

# 과적합 시나리오용 표본 축소 — 0이면 전체 사용
train_rows = {train_rows}
if train_rows > 0:
    x_train, y_train = x_train[:train_rows], y_train[:train_rows]

degree, alpha = {degree}, {alpha}
phi = features(x_train, degree)
w = np.linalg.solve(phi.T @ phi + alpha * np.eye(phi.shape[1]), phi.T @ y_train)

train_mse = float(np.mean((phi @ w - y_train) ** 2))
val_mse = float(np.mean((features(x_val, degree) @ w - y_val) ** 2))
preds = features(x_hold, degree) @ w

out_path = Path(__file__).parent / "predictions.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "pred"])
    for i, p in enumerate(preds):
        writer.writerow([i, f"{{float(p):.6f}}"])

print(json.dumps({{"train_mse": round(train_mse, 6), "val_mse": round(val_mse, 6),
                  "predictions_path": str(out_path)}}))
'''

# 주의: research 서사(_TOY_VARIABLES)와 coder 실제 스텝(_TOY_STEPS)은 느슨하게 대응 — mock 결정성이 목적
_TOY_STEPS = [(1, 0.0), (2, 0.0), (2, 0.1), (3, 0.1)]
_TOY_VARIABLES = ["poly_degree=2", "ridge_alpha=0.1", "poly_degree=3"]


def _research_toy(idx: int, task_input: dict) -> dict:
    var = _TOY_VARIABLES[idx % len(_TOY_VARIABLES)]
    base = _research_default(idx, task_input)
    base["hypothesis"]["statement"] = f"{var} 적용으로 val_mse 개선"
    base["hypothesis"]["single_variable"] = var
    base["design"]["change"] = var
    base["design"]["keep_fixed"] = [v for v in _TOY_VARIABLES if v != var]
    return base


def _coder_toy(idx: int, task_input: dict) -> dict:
    degree, alpha = _TOY_STEPS[min(idx, len(_TOY_STEPS) - 1)]
    return {"code": TOY_CODE.format(degree=degree, alpha=alpha, train_rows=0)}


# toy_overfit이 사용하는 (표본 축소, degree) — 테스트(조건수 회귀 검증)에서도 재사용해
# 코드/테스트 드리프트 방지. 근거는 아래 _coder_toy_overfit 주석 및
# .superpowers/sdd/task-7-report.md의 Fix Report 2 참고.
TOY_OVERFIT_TRAIN_ROWS = 8
TOY_OVERFIT_DEGREE = 2


def _coder_toy_overfit(idx: int, task_input: dict) -> dict:
    # 재설계(M6 Fix Report 2, cfec230 이후): 이전 시도(deg28~30, 기저 정규화만으로
    # 고차 다항 과적합 재현)는 cond(phi.T@phi)≈2.18e18로 여전히 솔버 노이즈 영역이었다.
    # 원인 재진단: 60개 학습 샘플·저노이즈(σ=0.1)의 sin(2x)+0.5x²는 매끈해서 degree~4면
    # 이미 잘 근사되므로, "안정 조건수(cond<1e12) ∧ 진짜 과적합(gap>0.3)"은 차수를
    # 얼마나 올려도 이 60행 데이터로는 양립 불가(안전 영역은 gap≈0, 유의미한 gap은
    # 오직 near-singular 영역에서만 등장 — 실측표는 task-7-report.md 참고).
    # 그래서 축(x축)을 degree에서 "표본 수"로 바꿨다: 용량(=degree+1개 파라미터) 대비
    # 데이터가 부족해야 과적합이 생긴다는 원 논리를 표본을 줄이는 쪽으로 적용.
    # train_rows∈{5..15} × degree∈{1..9} 전수 실측 스캔 결과(표는 task-7-report.md):
    #   - degree를 키우면(예: degree>=6, train_rows=7~10) 여전히 파라미터 수가 표본 수에
    #     근접해 근방정식이 near-singular해지고, 이전과 동일한 병리(인접 조합 간 gap이
    #     0.1대 → 수천대로 비단조 요동, cond가 1e8~1e12대까지 치솟음)가 재현된다 — 즉
    #     controller가 예로 든 degree=6~8은 실측상 불안정해 기각.
    #   - 반대로 degree=1~2처럼 표본 대비 파라미터가 적은 영역은 cond가 항상 20 안팎으로
    #     극히 낮고, train_rows를 6~9 사이에서 ±1 흔들어도 gap이 0.30~0.44 범위에서
    #     완만하게만 움직인다(요동 없음) — 소수 표본이 만드는 "진짜" 추정 분산이 원인.
    # 선정: train_rows=8, degree=2, alpha=0.0 — cond(phi.T@phi)≈21.6(<<1e12),
    # gap≈0.4216(임계 0.3 충족, 기존 테스트 임계 0.1의 약 4.2배 여유), 인접
    # (train_rows=7/9, 동일 degree)도 각각 gap 0.370/0.439로 안정.
    return {"code": TOY_CODE.format(degree=TOY_OVERFIT_DEGREE, alpha=0.0,
                                    train_rows=TOY_OVERFIT_TRAIN_ROWS)}


_VARIABLES = ["learning_rate", "n_estimators", "max_depth"]


def _research_default(idx: int, task_input: dict) -> dict:
    var = _VARIABLES[idx % len(_VARIABLES)]
    cycle_id = task_input["cycle_id"]
    consensus = task_input.get("consensus_summary", "") or "없음"
    return {
        "hypothesis": {
            "cycle_id": cycle_id,
            "statement": f"{var} 조정으로 val_mse가 개선된다",
            "single_variable": var,
            "rationale": f"컨센서스({consensus})와 직전 진단을 반영한 mock 가설",
        },
        "design": {
            "cycle_id": cycle_id,
            "change": f"{var} 조정",
            "keep_fixed": [v for v in _VARIABLES if v != var],
            "expected_effect": "val_mse 감소",
        },
    }


def _coder_default(idx: int, task_input: dict) -> dict:
    val = max(0.05, round(0.35 - 0.05 * idx, 4))
    return {"code": GOOD_CODE.format(train=round(val - 0.02, 4), val=val)}


def _coder_selfcorrect(idx: int, task_input: dict) -> dict:
    if idx == 0:
        return {"code": BUGGY_CODE}
    return _coder_default(idx, task_input)


def _coder_slow(idx: int, task_input: dict) -> dict:
    return {"code": SLOW_CODE.format(sleep=1.5)}


def _critic_default(idx: int, task_input: dict) -> dict:
    return {"target": task_input.get("draft_type", "unknown"), "attacks": [], "verdict": "pass"}


# 주의: idx는 프로세스 수명 기준 'critic 역할 최초 호출'(사이클/대상 무관) — 사이클당 3회 호출되므로 첫 호출=cycle1 research 검토
def _critic_revise_once(idx: int, task_input: dict) -> dict:
    if idx == 0:
        return {
            "target": task_input.get("draft_type", "unknown"),
            "attacks": ["근거 부족: mock 공격 — 데이터 근거를 제시할 것"],
            "verdict": "revise",
        }
    return _critic_default(idx, task_input)


_CATEGORY_OWNER = {"impl_bug": "coder"}


def _harness_default(idx: int, task_input: dict) -> dict:
    category = task_input.get("failure_category", "") or "unknown"
    agent = _CATEGORY_OWNER.get(category, "executor") if task_input.get("kind") == "A" else "pm"
    return {"agent": agent,
            "lesson": f"{category} 재발 방지: 트리거 {task_input.get('kind')} — 원인 패턴을 점검할 것"}


SCENARIOS: dict[str, dict] = {
    "default": {"research": _research_default, "coder": _coder_default, "critic": _critic_default,
                "harness_engineer": _harness_default},
    "coder_selfcorrect": {"research": _research_default, "coder": _coder_selfcorrect, "critic": _critic_default,
                          "harness_engineer": _harness_default},
    "coder_slow": {"research": _research_default, "coder": _coder_slow, "critic": _critic_default,
                   "harness_engineer": _harness_default},
    "critic_revise": {
        "research": _research_default,
        "coder": _coder_default,
        "critic": _critic_revise_once,
        "harness_engineer": _harness_default,
    },
    "toy_task": {"research": _research_toy, "coder": _coder_toy, "critic": _critic_default,
                 "harness_engineer": _harness_default},
    "toy_overfit": {"research": _research_toy, "coder": _coder_toy_overfit, "critic": _critic_default,
                    "harness_engineer": _harness_default},
}
