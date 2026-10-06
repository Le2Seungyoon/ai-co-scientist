"""레벨 평활 스윕 — torch 없이 돌고, 런 구조가 실제로 평활에 영향을 주는지 본다.

이 스크립트는 워크트리(dev 그룹만 sync — torch 없음)에서 레인이 돌리는 것이 존재 이유다.
그래서 **import 계약**이 수치 못지않게 중요하다.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_level_smoothing.py"


def _fixture(tmp_path, n_sites=40, per_site=8):
    """사이트 단위 합성 데이터. 사이트 하나는 라벨 하나를 공유한다(실제 구조와 같다)."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), n_sites // 4)
    rng.shuffle(y)
    site = np.repeat(np.arange(n_sites), per_site)
    labels = np.repeat(y, per_site)

    proba = np.full((len(labels), 4), 0.05, dtype=np.float32)
    proba[np.arange(len(labels)), labels] = 0.85
    flip = rng.choice(len(labels), size=len(labels) // 12, replace=False)  # 고립 오분류
    for i in flip:
        wrong = (labels[i] + 1) % 4
        proba[i] = 0.05
        proba[i, wrong] = 0.85
    proba /= proba.sum(1, keepdims=True)

    np.save(tmp_path / "proba.npy", proba.astype(np.float32))
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "site.npy", site)
    return proba, labels, site


def _run(tmp_path, *extra):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--proba", str(tmp_path / "proba.npy"),
         "--labels", str(tmp_path / "labels.npy"),
         "--site", str(tmp_path / "site.npy"), *extra],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_source_imports_neither_torch_nor_cv2():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import cv2" not in source


def test_smoothing_improves_accuracy_on_isolated_errors(tmp_path):
    """런 한가운데 고립된 오분류는 최빈값 필터가 흡수해야 한다."""
    _fixture(tmp_path)
    out = _run(tmp_path, "--arm", "smooth", "--values", "1,5")
    got = {a["value"]: a["holdout_accuracy"] for a in out["arms"]}
    assert got[5] > got[1]


def test_reports_the_run_count_so_comparability_is_visible(tmp_path):
    """런 길이 분포가 test와 다르면 최적 k가 옮겨가지 않는다 — 읽는 사람이 판단해야 한다."""
    _fixture(tmp_path)
    out = _run(tmp_path, "--arm", "smooth", "--values", "5")
    assert out["run_count"] > 0
    assert out["n"] == 320


def test_seed_is_reproducible(tmp_path):
    """같은 시드는 같은 순열 — 재현 가능한 실험의 최소 조건."""
    _fixture(tmp_path)
    a = _run(tmp_path, "--arm", "smooth", "--values", "5", "--seed", "7")
    b = _run(tmp_path, "--arm", "smooth", "--values", "5", "--seed", "7")
    assert a["arms"] == b["arms"]


def test_hmm_arm_runs_and_reports(tmp_path):
    _fixture(tmp_path)
    out = _run(tmp_path, "--arm", "hmm", "--values", "0.9,0.974")
    assert [a["value"] for a in out["arms"]] == [0.9, 0.974]
    assert all("holdout_accuracy" in a for a in out["arms"])


def test_declares_its_domain(tmp_path):
    """real SEM → real group label. 리더보드 타깃이 아니다 — 사전 선별 지표다."""
    _fixture(tmp_path)
    out = _run(tmp_path, "--arm", "smooth", "--values", "5")
    assert out["x_domain"] == "real"
    assert out["y_source"] == "real_group_label"
    assert out["is_leaderboard_proxy"] is False
