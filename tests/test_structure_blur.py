"""train_structure.StructureDataset의 blur_sigma 계약 — H4. GPU/캐시 불필요, 합성 데이터.

blur_sigma=0(기본)은 no-op이어야 하고(EXP-005 등 기존 실험 재현성), blur_sigma>0은 실제로
입력을 바꿔야 한다 — 안 바꾸면 H4 실험 자체가 무의미해진다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from train_structure import CASE_LEVEL, StructureDataset  # noqa: E402


def _synthetic(n=2):
    """지그재그 무늬 SEM(블러가 눈에 띄게 바뀌도록) + depth GT + case."""
    rng = np.random.default_rng(0)
    sem = np.zeros((n, 72, 48), dtype=np.uint8)
    sem[:, ::2, :] = 255  # 가로줄 무늬 — 블러가 반드시 값을 바꾼다
    depth = rng.integers(0, 141, size=(n, 72, 48)).astype(np.uint8)
    case = np.ones(n, dtype=np.int64)  # Case_1 → L=140
    return sem, depth, case


def test_zero_sigma_is_a_no_op():
    sem, depth, case = _synthetic()
    ds = StructureDataset(sem, depth, case, blur_sigma=0.0)
    x, s = ds[0]
    expected_x = sem[0].astype(np.float32) / 255.0
    assert np.array_equal(x.numpy()[0], expected_x), "blur_sigma=0인데 입력이 원본과 다르다"
    lv = CASE_LEVEL[int(case[0])]
    expected_s = (lv - depth[0].astype(np.float32)) / lv
    assert np.allclose(s.numpy()[0], expected_s), "s 재매개화가 blur_sigma 도입으로 깨졌다"


def test_positive_sigma_changes_the_input():
    sem, depth, case = _synthetic()
    raw = StructureDataset(sem, depth, case, blur_sigma=0.0)
    blurred = StructureDataset(sem, depth, case, blur_sigma=0.60)

    x_raw, s_raw = raw[0]
    x_blur, s_blur = blurred[0]

    assert not np.array_equal(x_raw.numpy(), x_blur.numpy()), (
        "blur_sigma=0.60인데 입력이 원본과 동일하다 — GaussianBlur가 걸리지 않았다")
    # 타깃 s는 depth GT에서 계산되므로 입력 블러와 무관하게 동일해야 한다
    assert np.array_equal(s_raw.numpy(), s_blur.numpy()), (
        "블러가 타깃 s까지 건드렸다 — 입력에만 걸려야 한다")
    # 지그재그 무늬는 블러로 완화되므로 표준편차가 줄어야 한다
    assert x_blur.numpy().std() < x_raw.numpy().std()


def test_blur_sigma_defaults_to_zero():
    sem, depth, case = _synthetic()
    ds = StructureDataset(sem, depth, case)
    assert ds.blur_sigma == pytest.approx(0.0)
