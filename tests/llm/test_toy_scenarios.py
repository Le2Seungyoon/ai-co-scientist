import csv
import json
import os
import subprocess
import sys

import numpy as np

from hooks.lint_check import lint_check
from ai_co_scientist.llm.router import LLMRouter
from ai_co_scientist.llm.scenarios import TOY_CODE, TOY_OVERFIT_DEGREE, TOY_OVERFIT_TRAIN_ROWS


def _run_generated(code: str, tmp_path, data_dir):
    path = tmp_path / "exp.py"
    path.write_text(code, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "COSCIENTIST_TOY_DATA": str(data_dir)},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


def _gen_data(tmp_path):
    data_dir = tmp_path / "toy"
    subprocess.run([sys.executable, "-m", "ai_co_scientist.toy_task.generate", str(data_dir)],
                   check=True)
    return data_dir


def test_degree2_actually_improves_over_degree1(tmp_path):
    data_dir = _gen_data(tmp_path)
    router = LLMRouter(scenario="toy_task")
    first = _run_generated(router.invoke("coder", {"design": {}, "error": ""})["code"],
                           tmp_path, data_dir)   # idx0: degree1
    second = _run_generated(router.invoke("coder", {"design": {}, "error": ""})["code"],
                            tmp_path, data_dir)  # idx1: degree2
    assert second["val_mse"] < first["val_mse"]  # 실제 반응성 — 가설이 점수를 바꾼다
    assert "predictions_path" in second


def test_overfit_scenario_has_train_val_gap(tmp_path):
    data_dir = _gen_data(tmp_path)
    router = LLMRouter(scenario="toy_overfit")
    result = _run_generated(router.invoke("coder", {"design": {}, "error": ""})["code"],
                            tmp_path, data_dir)
    assert (result["val_mse"] - result["train_mse"]) > 0.1  # analysis overfit gap 발동 조건


def _load_x(data_dir, name):
    with open(data_dir / name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["x"]) for r in rows])


def test_overfit_features_matrix_conditioning_is_bounded(tmp_path):
    """조건수 안정성 회귀 테스트 (M6 Fix Report 2 — 표본 축소 재설계 대응).

    toy_overfit이 실제로 생성하는 TOY_CODE.features()와 동일한 수식
    ((x/2.0)**d, TOY_CODE 51~53행과 일치)을 재현하고, TOY_CODE와 동일하게
    train_rows로 표본을 축소한 뒤 조건수를 직접 계산한다.

    이전 설계(차수만 올려 과적합을 유도, degree<=~30)는 기저를 재정규화해도
    cond(phi.T@phi)≈2.18e18로 여전히 솔버 노이즈 영역이었다(60 train 샘플·저노이즈
    데이터는 sin(2x)+0.5x²를 degree~4면 이미 잘 근사하므로, cond<1e12인 안전
    영역에서는 진짜 과적합 갭이 전혀 나타나지 않았음 — 상세는
    .superpowers/sdd/task-7-report.md의 Fix Report 참고). 재설계 후에는 차수 대신
    "표본 수"를 줄여 용량 대비 데이터 부족을 재현한다(train_rows=8, degree=2) —
    이 조합은 cond가 항상 20 안팎으로 극히 낮아 절대 기준 cond<1e12를 안정적으로
    만족한다(Fix Report 2의 train_rows×degree 전수 스캔 표 참고).
    """
    data_dir = _gen_data(tmp_path)
    x_train_full = _load_x(data_dir, "train.csv")
    degree = TOY_OVERFIT_DEGREE
    x_train = x_train_full[:TOY_OVERFIT_TRAIN_ROWS]

    def features_scaled(x, d):
        return np.vstack([(x / 2.0) ** dd for dd in range(d + 1)]).T

    phi_scaled = features_scaled(x_train, degree)
    cond_scaled = np.linalg.cond(phi_scaled.T @ phi_scaled)

    assert cond_scaled < 1e12  # 표본 축소 재설계로 이제 달성 가능한 절대 안전 기준


def test_toy_code_passes_lint_check(tmp_path):
    path = tmp_path / "exp.py"
    path.write_text(TOY_CODE.format(degree=2, alpha=0.1, train_rows=0), encoding="utf-8")
    ok, reason = lint_check(str(path))
    assert ok, reason
