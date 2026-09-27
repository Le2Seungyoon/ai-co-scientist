"""CPU 재조립 진입점의 계약 — torch/cv2 없이 돌고, 레벨 후처리 인자가 실제로 결과를 바꾼다.

이 진입점은 워크트리(dev 그룹만 sync — torch도 cv2도 없다)에서 병렬로 도는 것이 존재 이유다.
그래서 **import 계약**이 성능 못지않게 중요하다.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from ai_co_scientist.submission import decode_png_gray8

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "assemble_submission.py"


def _fixture(tmp_path, n=6, h=4, w=3):
    """ŝ 덤프 · 레벨 사후확률 덤프 · 파일명 목록을 합성한다."""
    rng = np.random.default_rng(0)
    structure = rng.random((n, h, w), dtype=np.float32)
    proba = np.full((n, 4), 0.1, dtype=np.float32)
    proba[:, 0] = 0.7
    proba[3, :] = [0.1, 0.7, 0.1, 0.1]  # 런 한가운데의 고립된 한 장 — 평활 대상
    names = [f"{i:06d}.png" for i in range(n)]
    np.save(tmp_path / "s.npy", structure)
    np.save(tmp_path / "p.npy", proba)
    (tmp_path / "names.json").write_text(json.dumps(names), encoding="utf-8")
    return structure, proba, names


def _run(tmp_path, *extra):
    out = tmp_path / "sub.zip"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--structure", str(tmp_path / "s.npy"),
         "--level-proba", str(tmp_path / "p.npy"),
         "--names", str(tmp_path / "names.json"),
         "--submit", str(out), *extra],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return out, json.loads(proc.stdout.strip().splitlines()[-1])


def test_produces_readable_zip_without_torch_or_cv2(tmp_path):
    _fixture(tmp_path)
    out, diag = _run(tmp_path)
    assert diag["n"] == 6
    with zipfile.ZipFile(out) as zf:
        assert len(zf.namelist()) == 6
        assert decode_png_gray8(zf.read("000000.png")).shape == (4, 3)


def test_source_imports_neither_torch_nor_cv2():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import cv2" not in source


def test_level_smooth_changes_the_isolated_image(tmp_path):
    """k=3 최빈값 필터는 런 한가운데 고립된 한 장을 흡수해야 한다."""
    _fixture(tmp_path)
    _, plain = _run(tmp_path)
    _, smoothed = _run(tmp_path, "--level-smooth", "3")
    assert smoothed["level_diag"]["changed"] == 1
    assert plain["level_smooth"] == 0
    assert smoothed["level_smooth"] == 3


def test_smooth_and_hmm_are_mutually_exclusive(tmp_path):
    """평활기를 두 번 겹치면 어느 쪽이 점수를 움직였는지 분리되지 않는다."""
    _fixture(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--structure", str(tmp_path / "s.npy"),
         "--level-proba", str(tmp_path / "p.npy"),
         "--names", str(tmp_path / "names.json"),
         "--submit", str(tmp_path / "x.zip"),
         "--level-smooth", "3", "--level-hmm"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode != 0
    assert "함께 쓸 수 없다" in proc.stderr


def test_declares_leaderboard_target():
    """리더보드의 y는 숨은 real depth GT다 — average_depth로 잘못 라벨링하면 안 된다."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"x_domain": "real"' in source
    assert '"y_source": "real_depth_gt"' in source
    assert "real_average_depth" not in source
