"""`ai_co_scientist.cyclegan` — H6 CycleGAN sim→real 순수 로직을 **행동**으로 검사한다.

합성 numpy 픽스처만 쓴다. torch/cv2는 이 워크트리에 없다 — torch가 필요한 케이스는
`pytest.importorskip("torch")`로 건너뛴다(실제로는 skip된다. 그것이 이 파일이 서 있는 이유다).
"""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_co_scientist import cyclegan


# ── 모듈 ──


# ── 설정: to_dict/from_dict 왕복, 타입 엄격 비교 ────────────────

def test_config_to_dict_json_roundtrips():
    d = cyclegan.CycleGANConfig().to_dict()
    assert json.loads(json.dumps(d)) == d


def test_config_from_dict_reconstructs_default():
    cfg = cyclegan.CycleGANConfig()
    assert cyclegan.CycleGANConfig.from_dict(cfg.to_dict()) == cfg


def test_config_from_dict_rejects_missing_key():
    d = cyclegan.CycleGANConfig().to_dict()
    del d["lr"]
    with pytest.raises(ValueError, match="lr"):
        cyclegan.CycleGANConfig.from_dict(d)


def test_config_from_dict_rejects_unknown_key():
    d = cyclegan.CycleGANConfig().to_dict()
    d["extra_field"] = 1
    with pytest.raises(ValueError, match="extra_field"):
        cyclegan.CycleGANConfig.from_dict(d)


def test_config_total_epochs_is_sum():
    cfg = cyclegan.CycleGANConfig()
    assert cfg.total_epochs == 100


def test_validate_config_accepts_preregistered_default():
    cyclegan.validate_config(cyclegan.CycleGANConfig())  # 예외 없음


def test_validate_config_lists_every_deviating_field():
    # int vs float 타입 차이(8 vs 8.0)까지 잡아야 한다 — 값만 비교하면 놓친다
    cfg = cyclegan.CycleGANConfig(batch_size=16, lr=1e-3)
    with pytest.raises(ValueError) as exc:
        cyclegan.validate_config(cfg)
    msg = str(exc.value)
    assert "batch_size" in msg and "lr" in msg
    assert "ngf" not in msg  # 안 바뀐 필드는 나열되지 않아야 한다


def test_validate_config_type_strict_int_vs_float():
    # from_dict가 아니라 직접 dataclass를 만들어 타입을 강제로 섞는다
    cfg = cyclegan.CycleGANConfig(lambda_cycle=10)  # 사전등록은 10.0(float), 이건 10(int)
    with pytest.raises(ValueError, match="lambda_cycle"):
        cyclegan.validate_config(cfg)


# ── lr_multiplier: 정확한 경계값 ────────────────────────────────

def test_lr_multiplier_is_one_during_fixed_phase():
    cfg = cyclegan.CycleGANConfig()
    assert cyclegan.lr_multiplier(0, cfg) == 1.0
    assert cyclegan.lr_multiplier(49, cfg) == 1.0


def test_lr_multiplier_exact_decay_values():
    cfg = cyclegan.CycleGANConfig()
    assert cyclegan.lr_multiplier(50, cfg) == pytest.approx(50 / 51)
    assert cyclegan.lr_multiplier(99, cfg) == pytest.approx(1 / 51)


def test_lr_multiplier_rejects_out_of_range_epoch():
    cfg = cyclegan.CycleGANConfig()
    with pytest.raises(ValueError):
        cyclegan.lr_multiplier(-1, cfg)
    with pytest.raises(ValueError):
        cyclegan.lr_multiplier(100, cfg)


# ── 경로 누수 방지 ──────────────────────────────────────────────

def test_reject_test_paths_rejects_real_test_dir_component():
    with pytest.raises(ValueError):
        cyclegan.reject_test_paths(["data/test/SEM/x.png"])


def test_reject_test_paths_rejects_forbidden_filenames():
    with pytest.raises(ValueError):
        cyclegan.reject_test_paths(["runtime/cache/test_sem.npy"])
    with pytest.raises(ValueError):
        cyclegan.reject_test_paths(["runtime/cache/test_names.json"])


def test_reject_test_paths_accepts_pytest_tmp_dir_named_test_foo0(tmp_path):
    # "test_foo0"는 정확히 "test"가 아니다 — pytest tmp_path 이름 패턴이 걸리면 안 된다
    d = tmp_path / "test_foo0" / "cache"
    d.mkdir(parents=True)
    p = d / "sim_sem.npy"
    p.write_bytes(b"")
    cyclegan.reject_test_paths([p])  # 예외 없음


def test_reject_test_paths_rejects_windows_backslash_test_dir():
    with pytest.raises(ValueError):
        cyclegan.reject_test_paths(["runtime\\test\\cache\\sim_sem.npy"])


def test_require_source_name_accepts_matching_name(tmp_path):
    cyclegan.require_source_name(tmp_path / "sim_sem.npy", "sim_sem.npy")  # 예외 없음


def test_require_source_name_rejects_mismatch(tmp_path):
    with pytest.raises(ValueError):
        cyclegan.require_source_name(tmp_path / "real_sem.npy", "sim_sem.npy")


# ── 형태 계약 ─────────────────────────────────────────────────

def test_check_sem_array_accepts_valid_array():
    arr = np.zeros((5, cyclegan.IMG_H, cyclegan.IMG_W), dtype=np.uint8)
    cyclegan.check_sem_array(arr, "sim_sem", expected_n=5)  # 예외 없음


def test_check_sem_array_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="sim_sem"):
        cyclegan.check_sem_array(np.zeros((5, 72), dtype=np.uint8), "sim_sem")


def test_check_sem_array_rejects_wrong_hw():
    with pytest.raises(ValueError):
        cyclegan.check_sem_array(np.zeros((5, 71, 48), dtype=np.uint8), "sim_sem")


def test_check_sem_array_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        cyclegan.check_sem_array(np.zeros((5, 72, 48), dtype=np.float32), "sim_sem")


def test_check_sem_array_rejects_wrong_length():
    with pytest.raises(ValueError):
        cyclegan.check_sem_array(
            np.zeros((5, 72, 48), dtype=np.uint8), "sim_sem", expected_n=6)


def test_check_batch_shape_accepts_valid_shape():
    cyclegan.check_batch_shape((4, 1, 72, 48))  # 예외 없음


def test_check_batch_shape_rejects_zero_batch():
    with pytest.raises(ValueError):
        cyclegan.check_batch_shape((0, 1, 72, 48))


def test_check_batch_shape_rejects_wrong_hw():
    with pytest.raises(ValueError):
        cyclegan.check_batch_shape((4, 1, 48, 72))


# ── 픽셀 스케일 왕복 ─────────────────────────────────────────────

def test_signed_to_u8_roundtrips_all_256_values_exactly():
    u8 = np.arange(256, dtype=np.uint8)
    assert np.array_equal(cyclegan.signed_to_u8(cyclegan.to_signed(u8)), u8)


def test_to_unit_maps_0_and_255_to_bounds():
    u8 = np.array([0, 255], dtype=np.uint8)
    assert np.allclose(cyclegan.to_unit(u8), [0.0, 1.0])


def test_to_signed_maps_0_and_255_to_bounds():
    u8 = np.array([0, 255], dtype=np.uint8)
    assert np.allclose(cyclegan.to_signed(u8), [-1.0, 1.0])


def test_signed_to_u8_clips_out_of_range():
    x = np.array([-2.0, 2.0], dtype=np.float32)
    assert cyclegan.signed_to_u8(x).tolist() == [0, 255]


# ── gate 표본 인덱스 ─────────────────────────────────────────────

def test_gate_sample_indices_is_sorted_unique_and_deterministic():
    idx1 = cyclegan.gate_sample_indices(10_000, 100, seed=1)
    idx2 = cyclegan.gate_sample_indices(10_000, 100, seed=1)
    assert np.array_equal(idx1, idx2)
    assert idx1.dtype == np.int64
    assert (np.diff(idx1) > 0).all()  # 정렬 + 고유


def test_gate_sample_indices_rejects_oversized_sample():
    with pytest.raises(ValueError):
        cyclegan.gate_sample_indices(10, 20)


def _small_paired_case_fixture():
    """Production split shape in miniature: eight depth maps, two SEM iterations each."""
    return np.repeat(np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int8), 2)


def _small_split_contract():
    return {
        "expected_total": 16,
        "expected_train": 12,
        "expected_val": 4,
        "val_frac": 0.25,
        "seed": 7,
    }


def test_h6_split_contract_distinguishes_full_cache_from_training_rows():
    # Regression: H6 treated the 138,648 training count as the raw cache length, so the real
    # 173,304-row cache was rejected before GPU work instead of selecting EXP-005's train split.
    assert cyclegan.SIM_TOTAL_N == 173_304
    assert cyclegan.SIM_TRAIN_N == 138_648
    assert cyclegan.SIM_VAL_N == 34_656
    assert cyclegan.SIM_TRAIN_N + cyclegan.SIM_VAL_N == cyclegan.SIM_TOTAL_N


def test_sim_split_indices_match_literal_paired_expectation_and_are_disjoint():
    # Literal indices are independently pinned so a helper that merely partitions at 12 rows
    # cannot masquerade as the established case-stratified, depth-map-paired split.
    train_idx, val_idx = cyclegan.sim_split_indices(
        _small_paired_case_fixture(), **_small_split_contract())
    assert train_idx.tolist() == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15]
    assert val_idx.tolist() == [6, 7, 12, 13]
    assert np.intersect1d(train_idx, val_idx).size == 0
    assert np.sort(np.concatenate([train_idx, val_idx])).tolist() == list(range(16))


def test_validation_gate_indices_are_global_members_of_validation_pool():
    # Regression: sampling 0..138647 fingerprints local/train positions, not the global validation
    # rows required by H6.  Seed 11 selects validation-pool positions 0 and 3 => globals 6 and 13.
    gate_idx = cyclegan.validation_gate_indices(
        _small_paired_case_fixture(), n_sample=2, gate_seed=11, **_small_split_contract())
    assert gate_idx.tolist() == [6, 13]
    assert set(gate_idx).issubset({6, 7, 12, 13})


def test_train_input_contract_accepts_full_cache_and_selects_only_train_globals(tmp_path):
    # Regression: the train path must validate a full cache, then expose only its train globals;
    # validating the image array against expected_train recreates the 173,304-vs-138,648 failure.
    sem = np.zeros((16, cyclegan.IMG_H, cyclegan.IMG_W), dtype=np.uint8)
    sem[:, 0, 0] = np.arange(16, dtype=np.uint8)
    sem_path = tmp_path / "sim_sem.npy"
    case_path = tmp_path / "sim_case.npy"
    np.save(sem_path, sem)
    np.save(case_path, _small_paired_case_fixture())

    loaded, case, train_idx, val_idx = cyclegan.load_sim_cache_split(
        sem_path, case_path, **_small_split_contract())
    assert len(loaded) == 16 and len(case) == 16
    assert loaded[train_idx, 0, 0].tolist() == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15]
    assert loaded[val_idx, 0, 0].tolist() == [6, 7, 12, 13]


def test_split_provenance_hashes_literal_global_indices():
    train_idx = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15], dtype=np.int64)
    val_idx = np.array([6, 7, 12, 13], dtype=np.int64)
    provenance = cyclegan.sim_split_provenance(
        train_idx, val_idx, total_n=16, val_frac=0.25, seed=7)
    assert provenance == {
        "method": "map_level_split",
        "total_n": 16,
        "train_n": 12,
        "val_n": 4,
        "val_frac": 0.25,
        "seed": 7,
        "train_indices_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(val_idx.tobytes()).hexdigest(),
    }


# ── phase correlation: 정수 roll 복원 ────────────────────────────

@pytest.mark.parametrize("dy,dx", [(2, 1), (-3, 2), (4, -3), (-4, 3), (1, -1), (-1, -1)])
def test_phase_correlation_recovers_exact_integer_roll(dy, dx):
    # 홀수 크기(9x7)를 써서 "정확히 절반 주기"의 부호 모호성을 피한다 — 그 경우만 +half/-half가
    # 같은 roll을 내는 진짜 비유일성이 있다(모듈 docstring 참조).
    rng = np.random.default_rng(3)
    a = rng.random((9, 7))
    b = np.roll(a, (dy, dx), axis=(0, 1))
    got_dy, got_dx = cyclegan.phase_correlation_shift(a, b)
    assert got_dy == pytest.approx(dy, abs=1e-9)
    assert got_dx == pytest.approx(dx, abs=1e-9)


def test_phase_correlation_identical_inputs_is_exact_zero():
    rng = np.random.default_rng(4)
    a = rng.random((9, 7))
    assert cyclegan.phase_correlation_shift(a, a) == (0.0, 0.0)


def test_phase_correlation_constant_image_is_zero_not_nan():
    a = np.full((9, 7), 5.0)
    dy, dx = cyclegan.phase_correlation_shift(a, a)
    assert dy == 0.0 and dx == 0.0
    assert not np.isnan(dy) and not np.isnan(dx)


def test_phase_correlation_subpixel_recovers_fourier_shifted_image():
    # 대역제한 신호를 위상램프로 소수점 이동시키면 이론적으로 정확한 subpixel 이동이 나온다.
    # 다만 정규화(백색화)된 phase correlation의 주peak은 sinc형으로 매우 뾰족해서(1픽셀 근방),
    # 포물선 3점 근사는 정밀 추정이 아니라 "대략 맞는 방향과 크기"를 보장하는 근사다.
    # 그래서 gate 임계값(중앙값 0.5px, p95 1.0px)과 같은 자리수인 0.2px를 허용오차로 쓴다 —
    # 이 함수가 gate 판정에 실제로 쓰일 정밀도 수준을 반영한 값이다.
    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.sin(2 * np.pi * xx / 16) + np.cos(2 * np.pi * yy / 20)

    def _fourier_shift(image, dy, dx):
        f = np.fft.fft2(image)
        u = np.fft.fftfreq(h).reshape(-1, 1)
        v = np.fft.fftfreq(w).reshape(1, -1)
        phase = np.exp(-2j * np.pi * (u * dy + v * dx))
        return np.fft.ifft2(f * phase).real

    for true_dy, true_dx in [(0.3, -0.4), (1.25, 2.6), (-2.1, 0.7)]:
        shifted = _fourier_shift(img, true_dy, true_dx)
        got_dy, got_dx = cyclegan.phase_correlation_shift(img, shifted)
        assert got_dy == pytest.approx(true_dy, abs=0.2)
        assert got_dx == pytest.approx(true_dx, abs=0.2)


def test_phase_correlation_is_blind_to_symmetric_dilation():
    # 알려진 사각지대: 전역 phase correlation은 강체 이동만 잰다. 배경 텍스처는 그대로 두고
    # 중앙 원판 구멍만 1px 대칭 팽창시키면(중심은 그대로) 기하가 실제로 바뀌었는데도, 안 바뀐
    # 배경이 상관을 지배해 이동량은 ~0으로 나와 gate를 통과한다. 순수 이진 원판(텍스처 없음)은
    # 회전 대칭 때문에 여러 위치의 상관값이 동률이 되어 peak 자체가 불안정해지므로(측정함:
    # 최댓값 0.078 vs 근접 후보들 0.077 수준), 배경에 고정 텍스처를 둬 진짜 지배적 peak가
    # (0,0)에 서게 만든다 — 이 함수가 "이동" 검사이지 "형태 보존" 검사가 아니라는 것을 고정한다.
    h, w = 40, 40
    rng = np.random.default_rng(0)
    texture = rng.random((h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = h / 2, w / 2
    r = np.hypot(yy - cy, xx - cx)
    a = np.where(r < 10, 0.0, texture)
    b = np.where(r < 11, 0.0, texture)  # 대칭 팽창 1px, 중심 이동 없음
    dy, dx = cyclegan.phase_correlation_shift(a, b)
    assert abs(dy) < 0.5 and abs(dx) < 0.5  # gate 임계값 이내 — 위양성 통과가 재현된다


def test_phase_correlation_shifts_matches_hypot_of_phase_correlation_shift():
    rng = np.random.default_rng(5)
    orig = rng.random((4, 9, 7))
    moved = np.stack([np.roll(orig[i], (1, -1), axis=(0, 1)) for i in range(4)])
    signed = cyclegan.phase_correlation_shifts(orig, moved)
    assert signed.shape == (4, 2)
    assert signed.dtype == np.float64
    mags = cyclegan.shift_magnitudes(orig, moved)
    assert np.allclose(mags, np.hypot(signed[:, 0], signed[:, 1]))


def test_shift_magnitudes_shape_and_dtype():
    rng = np.random.default_rng(6)
    orig = rng.random((3, 9, 7))
    moved = orig.copy()
    mags = cyclegan.shift_magnitudes(orig, moved)
    assert mags.shape == (3,)
    assert mags.dtype == np.float64
    assert (mags == 0.0).all()


def test_roundtrip_mae_is_mean_abs_diff_over_255():
    a = np.array([[0, 255]], dtype=np.uint8)
    b = np.array([[0, 0]], dtype=np.uint8)
    assert cyclegan.roundtrip_mae(a, b) == pytest.approx(255 / 2 / 255)


# ── evaluate_gate ────────────────────────────────────────────

def _shifts(n, value):
    return np.full(n, value, dtype=np.float64)


_ZL = np.zeros(cyclegan.GATE_SAMPLE_N)  # 국소 probe가 조용한 기본값 — 판정 기준만 떼어 볼 때


def test_evaluate_gate_passes_at_exact_boundary():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, cyclegan.SHIFT_MEDIAN_MAX)  # 중앙값=p95=경계
    r = cyclegan.evaluate_gate(shifts, cyclegan.ROUNDTRIP_MAE_MAX, local=_ZL)
    assert r["passed"] is True
    assert r["failures"] == []


def test_evaluate_gate_fails_on_shift_median_alone():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, cyclegan.SHIFT_MEDIAN_MAX + 0.01)
    r = cyclegan.evaluate_gate(shifts, 0.0, local=_ZL)
    assert r["passed"] is False
    assert any("shift_median" in f for f in r["failures"])
    assert not any("shift_p95" in f for f in r["failures"])
    assert not any("roundtrip_mae" in f for f in r["failures"])


def test_evaluate_gate_fails_on_shift_p95_alone():
    n = cyclegan.GATE_SAMPLE_N
    shifts = np.zeros(n, dtype=np.float64)
    # 상위 10%를 크게 튀운다 — np.percentile(linear)이 95번째 백분위를 이 구간에 걸치게
    # 하려면 5%보다 여유를 둬야 한다(정확히 5%는 보간 위치가 여전히 0쪽에 걸린다).
    shifts[: n // 10] = cyclegan.SHIFT_P95_MAX + 5.0
    r = cyclegan.evaluate_gate(shifts, 0.0, local=_ZL)
    assert r["passed"] is False
    assert any("shift_p95" in f for f in r["failures"])
    assert not any("shift_median" in f for f in r["failures"])


def test_evaluate_gate_fails_on_roundtrip_mae_alone():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, 0.0)
    r = cyclegan.evaluate_gate(shifts, cyclegan.ROUNDTRIP_MAE_MAX + 0.01, local=_ZL)
    assert r["passed"] is False
    assert any("roundtrip_mae" in f for f in r["failures"])
    assert not any("shift_median" in f or "shift_p95" in f for f in r["failures"])


def test_evaluate_gate_nan_shifts_fails_not_silently_passes():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, 0.0)
    shifts[0] = np.nan
    r = cyclegan.evaluate_gate(shifts, 0.0, local=_ZL)
    assert r["passed"] is False
    assert np.isnan(r["shift_median"]) and np.isnan(r["shift_p95"])


def test_evaluate_gate_nan_mae_fails():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, 0.0)
    r = cyclegan.evaluate_gate(shifts, float("nan"), local=_ZL)
    assert r["passed"] is False
    assert any("roundtrip_mae" in f for f in r["failures"])


def test_evaluate_gate_wrong_n_raises():
    with pytest.raises(ValueError):
        cyclegan.evaluate_gate(np.zeros(10, dtype=np.float64), 0.0, local=np.zeros(10))


def test_evaluate_gate_diagnostics_never_flip_passed():
    # diagnostics(p99, max, mean_dy/dx)는 참고용이다 — 5개 기준 통계만 passed를 결정한다.
    # 상위 2%(45/2048)를 극단으로 밀면: p95 인덱스(0.95*2047≈1944.65)는 여전히 0 구간에 있어
    # median=p95=0을 유지해 통과하지만, p99 인덱스(0.99*2047≈2026.53)는 그 45개 구간에 걸려
    # p99만 임계값을 넘는다 — diagnostics가 gating과 분리돼 있다는 것을 수치로 고정한다.
    n = cyclegan.GATE_SAMPLE_N
    k = 45
    shifts = np.zeros(n, dtype=np.float64)
    shifts[n - k:] = 999.0
    signed = np.zeros((n, 2), dtype=np.float64)
    signed[n - k:] = [999.0, -999.0]
    r = cyclegan.evaluate_gate(shifts, 0.0, local=_ZL, signed=signed)
    assert r["passed"] is True  # diagnostics가 극단이어도 gating 기준(median/p95/mae) 통과면 통과
    assert r["diagnostics"]["shift_max"] == 999.0
    assert r["diagnostics"]["shift_p99"] > cyclegan.SHIFT_P95_MAX  # 진단값 자체는 크다


def test_evaluate_gate_diagnostics_contain_p99_and_max():
    shifts = np.linspace(0.0, 1.0, cyclegan.GATE_SAMPLE_N)
    r = cyclegan.evaluate_gate(shifts, 0.0, local=_ZL)
    assert r["diagnostics"]["shift_max"] == pytest.approx(1.0)
    assert r["diagnostics"]["shift_p99"] == pytest.approx(np.percentile(shifts, 99))


def test_evaluate_gate_diagnostics_mean_dy_dx_from_signed():
    n = cyclegan.GATE_SAMPLE_N
    shifts = np.zeros(n, dtype=np.float64)
    signed = np.zeros((n, 2), dtype=np.float64)
    signed[:, 0] = 2.0
    signed[:, 1] = -3.0
    r = cyclegan.evaluate_gate(shifts, 0.0, local=_ZL, signed=signed)
    assert r["diagnostics"]["mean_dy"] == pytest.approx(2.0)
    assert r["diagnostics"]["mean_dx"] == pytest.approx(-3.0)


def test_evaluate_gate_diagnostics_omit_mean_dy_dx_without_signed():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, 0.0)
    r = cyclegan.evaluate_gate(shifts, 0.0, local=_ZL)
    assert "mean_dy" not in r["diagnostics"]
    assert "mean_dx" not in r["diagnostics"]


# ── 국소 기하: 블록 매칭 (전역 phase correlation의 사각지대) ─────────

def _hole_scene(warp=None, appearance=False, seed=0):
    """72x48 합성 hole crop: 부드러운 원판 구멍 + 저주파 텍스처. `warp(yy, xx) -> (sy, sx)`는
    출력 좌표가 원본의 어디를 읽는지(역사상)다. `appearance=True`는 기하를 바꾸지 않는 외관
    변화(블러 + 아핀 밝기 + 잡음)를 얹는다 — 변환기가 해도 되는 일의 대역이다."""
    h, w = cyclegan.IMG_H, cyclegan.IMG_W
    rng = np.random.default_rng(seed)
    big = rng.normal(size=(h * 3, w * 3))
    ky = np.fft.fftfreq(big.shape[0])[:, None]
    kx = np.fft.fftfreq(big.shape[1])[None]
    tex = np.fft.ifft2(np.fft.fft2(big) * np.exp(-8 * np.pi ** 2 * (ky ** 2 + kx ** 2))).real * 60
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sy, sx = (yy, xx) if warp is None else warp(yy, xx)
    r = np.hypot(sy - h / 2, sx - w / 2)
    y, x = sy + h, sx + w
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - y0, x - x0
    t = (tex[y0, x0] * (1 - fy) * (1 - fx) + tex[y0 + 1, x0] * fy * (1 - fx)
         + tex[y0, x0 + 1] * (1 - fy) * fx + tex[y0 + 1, x0 + 1] * fy * fx)
    img = 170 - 120 / (1 + np.exp((r - 14) / 1.2)) + t
    if appearance:
        k = np.exp(-2 * (np.pi * 0.8) ** 2 * (np.fft.fftfreq(h)[:, None] ** 2
                                             + np.fft.fftfreq(w)[None] ** 2))
        img = np.fft.ifft2(np.fft.fft2(img) * k).real * 1.1 + 10 + rng.normal(0, 3, img.shape)
    return img


def _left_half_shift(d):
    return lambda y, x: (y, np.where(x < cyclegan.IMG_W / 2, x - d, x))


def _smooth_warp(a):
    return lambda y, x: (y + a * np.sin(2 * np.pi * y / cyclegan.IMG_H),
                         x + a * np.sin(2 * np.pi * x / cyclegan.IMG_W + 1))


def _dilate(k):
    cy, cx = cyclegan.IMG_H / 2, cyclegan.IMG_W / 2
    return lambda y, x: (cy + (y - cy) / (1 + k), cx + (x - cx) / (1 + k))


def _gate_on(orig, moved):
    """한 쌍을 GATE_SAMPLE_N장으로 복제해 전역+국소 gate를 모두 돌린다 (mae=0)."""
    g = np.full(cyclegan.GATE_SAMPLE_N, np.hypot(*cyclegan.phase_correlation_shift(orig, moved)))
    loc = np.full(cyclegan.GATE_SAMPLE_N, cyclegan.local_shift_max(orig[None], moved[None])[0])
    return cyclegan.evaluate_gate(g, 0.0, local=loc)


def test_block_match_identical_is_exactly_zero():
    a = _hole_scene()
    np.testing.assert_array_equal(cyclegan.block_match_shifts(a, a), np.zeros((6, 2)))


def test_block_match_recovers_integer_shift_in_every_tile():
    # 비순환 이동(큰 장면에서 잘라낸 두 창)이라 phase correlation의 경계 편향이 없는 설정.
    # 포물선 subpixel 보정이 비대칭 NCC 면에서 ~0.1 px 치우치므로 허용오차 0.15
    big = np.random.default_rng(3).normal(size=(90, 66)).cumsum(0).cumsum(1)
    a = big[8:80, 8:56]
    for dy, dx in [(1, 0), (0, -2), (2, 1), (-1, -1)]:
        b = big[8 - dy:80 - dy, 8 - dx:56 - dx]  # b[p] = a[p - d] — 내용이 +d로 이동
        est = cyclegan.block_match_shifts(a, b)
        np.testing.assert_allclose(est, np.tile([dy, dx], (6, 1)), atol=0.15)


def test_block_match_skips_flat_tiles_as_nan():
    # 평탄한 타일은 변위가 정의되지 않는다 — 0으로 적으면 "움직이지 않았다"는 거짓 증거가 된다
    a = _hole_scene()
    a[:24, :24] = 128.0
    est = cyclegan.block_match_shifts(a, a + 1.0)
    assert np.isnan(est[0]).all()
    assert not np.isnan(est[1:]).any()


def test_local_shift_max_of_all_flat_image_is_nan_and_is_flagged_by_the_probe():
    # 전부 평탄하면 국소 기하를 잴 수 없다 — 진단 probe는 "측정 불가"로 적고 0으로 숨기지 않는다.
    # 판정은 사전등록 기준만 한다(진단 probe는 passed를 바꾸지 않는다)
    a = np.full((1, cyclegan.IMG_H, cyclegan.IMG_W), 128.0)
    loc = cyclegan.local_shift_max(a, a + 3.0)
    assert np.isnan(loc[0])
    res = cyclegan.evaluate_gate(np.zeros(cyclegan.GATE_SAMPLE_N), 0.0,
                                 local=np.full(cyclegan.GATE_SAMPLE_N, loc[0]))
    assert res["passed"] is True
    probe = res["local_probe"]
    assert probe["n_unmeasurable"] == cyclegan.GATE_SAMPLE_N
    assert probe["median"] is None and probe["p95"] is None
    assert probe["flags"] == [f"local_shift 측정 불가 {cyclegan.GATE_SAMPLE_N}장"]


@pytest.mark.parametrize("name,warp", [
    ("left half shifted 1 px", _left_half_shift(1.0)),
    ("smooth 1 px warp", _smooth_warp(1.0)),
    ("4 % hole dilation", _dilate(0.04)),
])
def test_local_warp_is_a_known_residual_of_the_authoritative_gate_and_only_flagged(name, warp):
    # 잔여 위험 고정: 픽셀 GT 대응을 깨는 국소 왜곡을 사전등록 전역 phase correlation은 통과시킨다
    # (사각지대). 국소 probe는 미보정 진단이라 이를 **기록만** 하고 정지시키지 않는다 — 실제 sim
    # 보정 없이 정지 규칙에 넣으면 외관 전용 변환도 멈춘다(`test_appearance_only_blur_gamma_...`).
    # 전역 값은 장면에 따라 흔들리므로(seed 0: 0.30~0.38 px, 20 seed 최대 ~0.75) 장면을 seed 0에
    # 고정하고 크기를 사각지대 안으로 잡았다. 국소 값은 seed 0에서 0.80~1.08 px다
    orig = _hole_scene()
    moved = _hole_scene(warp=warp, appearance=True)
    res = _gate_on(orig, moved)
    assert res["shift_median"] <= cyclegan.SHIFT_MEDIAN_MAX, name  # 전역 기준은 못 본다
    assert res["passed"] is True, name
    assert res["failures"] == [], name
    assert any(f.startswith("local_shift_median") for f in res["local_probe"]["flags"]), name


def test_appearance_only_change_raises_no_probe_flag():
    # 음성 대조: 기하는 그대로 두고 블러·아핀 밝기·잡음만 바꾸면 진단 probe도 조용해야 한다
    res = _gate_on(_hole_scene(), _hole_scene(appearance=True))
    assert res["passed"] is True, res["failures"]
    assert res["local_probe"]["flags"] == []
    assert res["local_probe"]["median"] < 0.3


def test_global_phase_correlation_underestimates_noncircular_subpixel_shift():
    # 발견 고정: 실제 0.5 px 비순환 전역 이동을 전역 phase correlation은 ~0.05로 읽는다
    # (crop 경계의 불연속이 백색화 스펙트럼을 지배). 사전등록 전역 기준이 느슨하다는 증거(잔여
    # 위험) — 진단 probe의 블록 매칭은 같은 입력을 ~0.5로 읽는다
    orig = _hole_scene()
    moved = _hole_scene(warp=lambda y, x: (y - 0.5, x))
    assert np.hypot(*cyclegan.phase_correlation_shift(orig, moved)) < 0.1
    assert cyclegan.local_shift_max(orig[None], moved[None])[0] > 0.45


def test_local_probe_flags_are_inclusive_and_named():
    zeros = _ZL
    at = np.full(cyclegan.GATE_SAMPLE_N, cyclegan.SHIFT_MEDIAN_MAX)
    res = cyclegan.evaluate_gate(zeros, 0.0, local=at)
    assert res["local_probe"]["flags"] == []
    over = cyclegan.evaluate_gate(zeros, 0.0, local=at + 1e-9)
    assert over["passed"] is True and over["failures"] == []
    assert over["local_probe"]["flags"] == [f"local_shift_median {cyclegan.SHIFT_MEDIAN_MAX + 1e-9}"
                                            f" > {cyclegan.SHIFT_MEDIAN_MAX}"]


def test_require_gate_passed_accepts_gate_without_local_shifts(tmp_path):
    # 판정 권한은 사전등록 기준에만 있다 — 진단 probe 원본이 없어도 재판정은 완결된다
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-920")
    del gate["local_shifts"], gate["local_probe"]
    _rewrite(gate_path, gate)
    assert cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-920")["passed"] is True


def test_require_gate_passed_rejects_probe_summary_without_its_raw_values(tmp_path):
    # 원본 없이 남은 probe 요약은 재현할 수 없다 — 사람이 읽을 값이 검증되지 않으면 거부
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-922")
    del gate["local_shifts"]
    _rewrite(gate_path, gate)
    with pytest.raises(cyclegan.GateFailedError, match="요약값"):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-922")


def test_require_gate_passed_does_not_stop_on_diagnostic_local_values(tmp_path):
    # 큰 국소 값은 probe 플래그로 기록될 뿐 하드 스톱이 아니다(미보정 — docs/experiment/H6 §조건)
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-921")
    local = np.full(cyclegan.GATE_SAMPLE_N, 2.0)
    gate["local_shifts"] = local.tolist()
    gate["local_probe"] = cyclegan.evaluate_gate(_ZL, 0.0, local=local)["local_probe"]
    _rewrite(gate_path, gate)
    rechecked = cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-921")
    assert rechecked["passed"] is True
    assert rechecked["local_probe"]["flags"]


# ── manifest — 덮어쓰기 거부 + 해시 검증 ──────────────────────────

def test_write_once_json_refuses_second_write_and_keeps_first_content(tmp_path):
    p = tmp_path / "gate.json"
    cyclegan.write_once_json(p, {"a": 1})
    with pytest.raises(FileExistsError):
        cyclegan.write_once_json(p, {"a": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _passing_gate(report_id="EXP-900", ckpt=None, sim_case=None) -> dict:
    # manifest에 들어가는 gate는 재검증 가능한 원본 값과 결속 정보(ckpt·report_id·config)를 든다
    shifts = np.zeros(cyclegan.GATE_SAMPLE_N, dtype=np.float64)
    gate = cyclegan.evaluate_gate(shifts, 0.0, local=_ZL)
    gate.update({"shifts": shifts.tolist(), "local_shifts": _ZL.tolist(), "report_id": report_id,
                 "config": dict(cyclegan.PREREGISTERED),
                 "ckpt_sha256": cyclegan.sha256_file(ckpt) if ckpt is not None else None})
    if sim_case is not None:
        case = np.load(sim_case)
        train_idx, val_idx = cyclegan.sim_split_indices(case)
        gate.update({
            "sim_case_sha256": cyclegan.sha256_file(sim_case),
            "split": cyclegan.sim_split_provenance(train_idx, val_idx),
            "indices_sha256": cyclegan.indices_sha256(cyclegan.validation_gate_indices(case)),
        })
    return gate


def _manifest_parts(tmp_path, report_id, out_name="out"):
    """결속을 모두 만족하는 manifest 재료: ckpt, source 4종, output 3종(depth/case는 source 사본)."""
    ckpt = tmp_path / f"{report_id}-cyclegan.pt"
    _write_bytes(ckpt, b"ckpt-bytes")
    cache = tmp_path / "cache"
    sources = {}
    for name, data in (("sim_sem", b"sim"), ("sim_depth", b"depth"), ("real_sem", b"real")):
        sources[name] = cache / f"{name}.npy"
        _write_bytes(sources[name], data)
    sources["sim_case"] = cache / "sim_case.npy"
    sources["sim_case"].parent.mkdir(parents=True, exist_ok=True)
    np.save(sources["sim_case"], _production_case_fixture())
    out = tmp_path / out_name
    outputs = {"sim_sem": out / "sim_sem.npy", "sim_depth": out / "sim_depth.npy",
               "sim_case": out / "sim_case.npy"}
    _write_bytes(outputs["sim_sem"], b"translated")
    _write_bytes(outputs["sim_depth"], b"depth")
    _write_bytes(outputs["sim_case"], sources["sim_case"].read_bytes())
    return dict(report_id=report_id, config=cyclegan.PREREGISTERED,
                gate=_passing_gate(report_id, ckpt, sources["sim_case"]), ckpt_path=ckpt,
                source_files=sources,
                output_files=outputs, git_commit="d" * 40)


def test_build_manifest_refuses_failed_gate(tmp_path):
    parts = _manifest_parts(tmp_path, "EXP-900")
    parts["gate"] = cyclegan.evaluate_gate(
        _shifts(cyclegan.GATE_SAMPLE_N, cyclegan.SHIFT_MEDIAN_MAX + 1.0), 0.0, local=_ZL)
    with pytest.raises(ValueError):
        cyclegan.build_manifest(**parts)


def test_build_manifest_and_verify_manifest_roundtrip(tmp_path):
    parts = _manifest_parts(tmp_path, "EXP-901")
    manifest_path = tmp_path / "out" / "manifest.json"
    cyclegan.write_once_json(manifest_path, cyclegan.build_manifest(**parts))

    verified = cyclegan.verify_manifest(manifest_path)
    assert verified["hypothesis"] == "H6"
    assert verified["x_domain"] == "sim_translated_to_real_appearance"
    assert verified["y_source"] == "sim_depth_gt"
    assert Path(verified["ckpt"]["path"]).is_absolute()  # 다른 cwd에서도 검증 가능


def test_verify_manifest_detects_a_single_flipped_byte(tmp_path):
    parts = _manifest_parts(tmp_path, "EXP-902")
    manifest_path = tmp_path / "out" / "manifest.json"
    cyclegan.write_once_json(manifest_path, cyclegan.build_manifest(**parts))

    out = parts["output_files"]["sim_sem"]
    data = bytearray(out.read_bytes())
    data[0] ^= 0x01  # 바이트 1개만 뒤집는다
    out.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="sim_sem"):
        cyclegan.verify_manifest(manifest_path)


def test_manifest_survives_rename_of_its_directory(tmp_path):
    # 회귀: translate_sim.py는 `<out>.partial/`에 쓰고 manifest를 만든 뒤 `<out>`으로 rename한다.
    # 출력 경로를 절대경로로 적으면 rename 직후 모든 output이 "파일 없음"이 되어, 정상 산출물이
    # 영구히 검증 불가가 됐다. 출력은 manifest 디렉터리 기준 상대 이름으로 적어야 한다.
    parts = _manifest_parts(tmp_path, "EXP-903", out_name="EXP-903-translated.partial")
    partial = tmp_path / "EXP-903-translated.partial"
    cyclegan.write_once_json(partial / "manifest.json", cyclegan.build_manifest(**parts))
    final = tmp_path / "EXP-903-translated"
    partial.rename(final)

    verified = cyclegan.verify_manifest(final / "manifest.json")
    assert verified["output_files"]["sim_sem"]["path"] == "sim_sem.npy"
    assert verified["output_files"]["sim_depth"]["path"] == "sim_depth.npy"


def test_build_manifest_rejects_outputs_outside_one_directory(tmp_path):
    # 출력이 상대 이름으로 적히므로, 두 디렉터리에 흩어진 출력은 manifest 하나로 기술할 수 없다
    parts = _manifest_parts(tmp_path, "EXP-904")
    moved = tmp_path / "b" / "sim_depth.npy"
    _write_bytes(moved, b"depth")
    parts["output_files"]["sim_depth"] = moved
    with pytest.raises(ValueError, match="한 디렉터리"):
        cyclegan.build_manifest(**parts)


@pytest.mark.parametrize("mutate,expect", [
    (lambda p: p["output_files"].pop("sim_case"), "output 목록"),
    (lambda p: p["output_files"]["sim_depth"].write_bytes(b"depth-CHANGED"), "sim_depth"),
    (lambda p: p["gate"].update(ckpt_sha256="0" * 64), "ckpt_sha256"),
    (lambda p: p["gate"].update(report_id="EXP-OTHER"), "report_id"),
    (lambda p: p["gate"].update(sim_case_sha256="0" * 64), "sim_case_sha256"),
    (lambda p: p["gate"]["split"].update(seed=43), "split"),
    (lambda p: p["gate"].update(indices_sha256="0" * 64), "indices_sha256"),
    (lambda p: p.update(config={**cyclegan.PREREGISTERED, "seed": 43}), "config"),
])
def test_build_manifest_refuses_unbound_parts(tmp_path, mutate, expect):
    # 해시가 전부 맞아도 부품끼리 안 맞으면(다른 실행의 gate, 바뀐 y, 빠진 산출물) 만들 수 없다
    parts = _manifest_parts(tmp_path, "EXP-905")
    mutate(parts)
    with pytest.raises(ValueError, match=expect):
        cyclegan.build_manifest(**parts)


@pytest.mark.parametrize("mutate,expect", [
    (lambda p: p["source_files"].pop("real_sem"), "source 목록"),
    (lambda p: p["source_files"].update(extra=p["source_files"]["real_sem"]), "source 목록"),
    (lambda p: p.update(git_commit=""), "git_commit"),
    (lambda p: p.update(git_commit="deadbeef"), "git_commit"),
])
def test_build_manifest_rejects_incomplete_source_or_commit_binding(tmp_path, mutate, expect):
    """소스 집합이나 commit 결속이 느슨해지면 다른 변환 실행을 같은 manifest로 오인한다."""
    parts = _manifest_parts(tmp_path, "EXP-906")
    mutate(parts)
    with pytest.raises(ValueError, match=expect):
        cyclegan.build_manifest(**parts)


# ── require_gate_passed: 강화된 하드 스톱 ──────────────────────

def _production_case_fixture():
    """Four balanced cases whose paired 80/20 split is exactly 138,648 / 34,656."""
    return np.repeat(np.repeat(np.arange(1, 5, dtype=np.int8), 21_663), 2)


def _valid_gate_and_ckpt(tmp_path, *, report_id="EXP-910"):
    """`require_gate_passed`를 통과해야 하는 최소 gate JSON + ckpt 쌍을 만든다."""
    ckpt = tmp_path / cyclegan.expected_ckpt_name(report_id)
    _write_bytes(ckpt, b"ckpt-bytes")
    case = _production_case_fixture()
    case_path = tmp_path / "sim_case.npy"
    np.save(case_path, case)
    train_idx, val_idx = cyclegan.sim_split_indices(case)
    idx = cyclegan.validation_gate_indices(case)
    shifts = [0.0] * cyclegan.GATE_SAMPLE_N
    evaluated = cyclegan.evaluate_gate(np.asarray(shifts, dtype=np.float64), 0.0,
                                       local=np.zeros(cyclegan.GATE_SAMPLE_N))
    gate = dict(evaluated)
    gate.update({
        "local_shifts": [0.0] * cyclegan.GATE_SAMPLE_N,
        "report_id": report_id,
        "ckpt_sha256": cyclegan.sha256_file(ckpt),
        "ckpt_epoch": cyclegan.PREREGISTERED["epochs_fixed"] + cyclegan.PREREGISTERED["epochs_decay"],
        "config": cyclegan.PREREGISTERED,
        "shifts": shifts,
        "signed_shifts": [[0.0, 0.0]] * cyclegan.GATE_SAMPLE_N,
        "indices_sha256": cyclegan.indices_sha256(idx),
        "sim_case_sha256": cyclegan.sha256_file(case_path),
        "split": cyclegan.sim_split_provenance(train_idx, val_idx),
    })
    gate_path = tmp_path / "gate.json"
    cyclegan.write_once_json(gate_path, gate)
    return gate_path, ckpt, gate


def test_require_gate_passed_accepts_correct_gate(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path)
    result = cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-910")
    assert result["ckpt_sha256"] == gate["ckpt_sha256"]


def test_require_gate_passed_raises_for_missing_file(tmp_path):
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(
            tmp_path / "nope.json", tmp_path / "nope.pt", report_id="EXP-911")


def test_require_gate_passed_raises_when_stored_passed_is_false(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-912")
    data = json.loads(gate_path.read_text(encoding="utf-8"))
    data["passed"] = False
    gate_path.unlink()
    cyclegan.write_once_json(gate_path, data)
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-912")


def test_require_gate_passed_never_trusts_stored_passed_true(tmp_path):
    # passed=True로 조작했지만 원본 shifts를 재평가하면 실패해야 한다 — 저장된 bool을 믿지 않는다.
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-913")
    data = json.loads(gate_path.read_text(encoding="utf-8"))
    data["passed"] = True
    data["shifts"] = [cyclegan.SHIFT_MEDIAN_MAX + 10.0] * cyclegan.GATE_SAMPLE_N
    gate_path.unlink()
    cyclegan.write_once_json(gate_path, data)
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-913")


def test_require_gate_passed_raises_on_altered_thresholds(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-914")
    data = json.loads(gate_path.read_text(encoding="utf-8"))
    data["thresholds"]["shift_median_max"] = 999.0
    gate_path.unlink()
    cyclegan.write_once_json(gate_path, data)
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-914")


def test_require_gate_passed_raises_on_wrong_n(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-915")
    data = json.loads(gate_path.read_text(encoding="utf-8"))
    data["n"] = 10
    gate_path.unlink()
    cyclegan.write_once_json(gate_path, data)
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-915")


def test_require_gate_passed_raises_on_ckpt_sha_mismatch(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-916")
    ckpt.write_bytes(b"different-bytes")  # ckpt가 gate 이후 바뀌었다
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-916")


def test_require_gate_passed_raises_on_report_id_mismatch(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-917")
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-999")


def test_require_gate_passed_raises_on_wrong_ckpt_epoch(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-918")
    data = json.loads(gate_path.read_text(encoding="utf-8"))
    data["ckpt_epoch"] = 50  # 총 epoch(100)이 아니라 중간 epoch
    gate_path.unlink()
    cyclegan.write_once_json(gate_path, data)
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-918")


def test_require_gate_passed_raises_on_config_deviation(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-919")
    data = json.loads(gate_path.read_text(encoding="utf-8"))
    data["config"]["batch_size"] = 32
    gate_path.unlink()
    cyclegan.write_once_json(gate_path, data)
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-919")


def test_require_gate_passed_raises_on_indices_sha_mismatch(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-920")
    data = json.loads(gate_path.read_text(encoding="utf-8"))
    data["indices_sha256"] = "0" * 64
    gate_path.unlink()
    cyclegan.write_once_json(gate_path, data)
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-920")


def test_require_gate_passed_rejects_wrong_sim_case_before_runtime(tmp_path):
    gate_path, ckpt, _gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-923")
    case_path = tmp_path / "sim_case.npy"
    tampered = np.load(case_path).copy()
    tampered[:2] = 4
    np.save(case_path, tampered)
    with pytest.raises(cyclegan.GateFailedError, match="sim_case"):
        cyclegan.require_gate_passed(
            gate_path, ckpt, report_id="EXP-923", sim_case_path=case_path)


def test_require_gate_passed_rejects_resume_ckpt_filename(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-921")
    resume_ckpt = tmp_path / "EXP-921-cyclegan-resume.pt"
    resume_ckpt.write_bytes(ckpt.read_bytes())
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(gate_path, resume_ckpt, report_id="EXP-921")


def test_require_gate_passed_checks_sim_sem_sha_when_given(tmp_path):
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-922")
    sim_sem = tmp_path / "sim_sem.npy"
    sim_sem.write_bytes(b"sim-sem-bytes")
    data = json.loads(gate_path.read_text(encoding="utf-8"))
    data["sim_sem_sha256"] = cyclegan.sha256_file(sim_sem)
    gate_path.unlink()
    cyclegan.write_once_json(gate_path, data)
    result = cyclegan.require_gate_passed(
        gate_path, ckpt, report_id="EXP-922", sim_sem_path=sim_sem)
    assert result["sim_sem_sha256"] == cyclegan.sha256_file(sim_sem)

    sim_sem.write_bytes(b"tampered-bytes")
    with pytest.raises(cyclegan.GateFailedError):
        cyclegan.require_gate_passed(
            gate_path, ckpt, report_id="EXP-922", sim_sem_path=sim_sem)


def test_expected_ckpt_name_format():
    assert cyclegan.expected_ckpt_name("EXP-042") == "EXP-042-cyclegan.pt"


def test_indices_sha256_matches_manual_sha256_of_bytes():
    idx = cyclegan.gate_sample_indices(1000, 10, seed=1)
    expected = hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()
    assert cyclegan.indices_sha256(idx) == expected


# ── import 경계: torch/cv2가 없어야 한다 ─────────────────────────

def test_importing_cyclegan_does_not_load_torch_or_cv2():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import ai_co_scientist.cyclegan; import sys; "
         "assert 'torch' not in sys.modules; assert 'cv2' not in sys.modules; "
         "print('ok')"],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_module_has_no_top_level_torch_or_cv2_import():
    source = Path(cyclegan.__file__).read_text(encoding="utf-8")
    import ast
    tree = ast.parse(source)
    top_level_modules = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_modules.add(node.module.split(".")[0])
    assert "torch" not in top_level_modules
    assert "cv2" not in top_level_modules


# ── PatchGAN 산술 (torch 없이 검증 가능) ─────────────────────────

def test_patchgan_output_shape_matches_known_arithmetic():
    assert cyclegan.patchgan_output_shape(72, 48) == (7, 4)


def test_build_generator_raises_importerror_without_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch가 설치되어 있어 이 워크트리 전용 경로를 테스트할 수 없다")
    with pytest.raises(ImportError, match="uv run --group baseline"):
        cyclegan.build_generator(cyclegan.CycleGANConfig())


def test_build_discriminator_raises_importerror_without_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch가 설치되어 있어 이 워크트리 전용 경로를 테스트할 수 없다")
    with pytest.raises(ImportError, match="uv run --group baseline"):
        cyclegan.build_discriminator(cyclegan.CycleGANConfig())


def test_build_generator_shape_roundtrips_with_torch():
    torch = pytest.importorskip("torch")
    cfg = cyclegan.CycleGANConfig()
    gen = cyclegan.build_generator(cfg)
    x = torch.zeros(2, 1, cyclegan.IMG_H, cyclegan.IMG_W)
    y = gen(x)
    assert tuple(y.shape) == (2, 1, cyclegan.IMG_H, cyclegan.IMG_W)


def test_build_discriminator_shape_matches_patchgan_output_shape_with_torch():
    torch = pytest.importorskip("torch")
    cfg = cyclegan.CycleGANConfig()
    disc = cyclegan.build_discriminator(cfg)
    x = torch.zeros(2, 1, cyclegan.IMG_H, cyclegan.IMG_W)
    y = disc(x)
    expected_h, expected_w = cyclegan.patchgan_output_shape(cyclegan.IMG_H, cyclegan.IMG_W)
    assert tuple(y.shape) == (2, 1, expected_h, expected_w)


def test_load_generators_refuses_final_checkpoint_without_training_data(tmp_path):
    torch = pytest.importorskip("torch")
    ckpt = tmp_path / "EXP-026-cyclegan.pt"
    torch.save({
        "config": dict(cyclegan.PREREGISTERED),
        "epoch": 100,
        "state_dict": {"G_sim2real": {}, "G_real2sim": {}},
    }, ckpt)
    with pytest.raises(cyclegan.GateFailedError, match="training_data"):
        cyclegan.load_generators(ckpt, torch.device("cpu"))


# ── CLI ──
# scripts/train_cyclegan.py · scripts/translate_sim.py를 subprocess로 돌린다
# (tests/test_assemble_submission.py와 같은 패턴). 이 워크트리엔 torch가 없으므로 모든 거부 경로가
# torch import **전에** 일어났음을 함께 증명한다.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TRAIN_SCRIPT = SCRIPTS / "train_cyclegan.py"
TRANSLATE_SCRIPT = SCRIPTS / "translate_sim.py"


def _run(script, *args):
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, encoding="utf-8")


def _module_level_import_roots(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _no_torch_leak(proc) -> bool:
    """torch가 없는 이 워크트리에서, 거부 경로가 torch를 먼저 건드렸다면 반드시 이 문자열이
    출력에 남는다 -- 없다는 것 자체가 "torch를 아직 안 불렀다"는 증거다."""
    return "No module named 'torch'" not in (proc.stdout + proc.stderr)


def _touch_sem_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "sim_sem.npy").touch()
    (cache_dir / "sim_case.npy").touch()
    (cache_dir / "real_sem.npy").touch()


def _expected_ckpt_name(report_id: str) -> str:
    from ai_co_scientist.cyclegan import expected_ckpt_name
    return expected_ckpt_name(report_id)


def _make_gate_json(path: Path, ckpt_path: Path, *, report_id="H6-TEST",
                    shifts_ok=True, sha_override=None, report_id_field=None,
                    ckpt_epoch_override=None, config_override=None, sim_sem_path=None,
                    sim_case_path=None):
    """`require_gate_passed`가 요구하는 모든 필드를 채운 gate JSON을 합성한다.

    기본값은 전부 "통과"하도록 만들어 두고, 파라미터로 정확히 하나씩만 어긋나게 할 수 있다
    -- require_gate_passed의 검사 순서(이름→report_id→epoch→passed→config→임계값→n→
    shifts 재평가→인덱스 지문→ckpt sha256→sim_sem sha256)를 그대로 따라가며 테스트한다.
    """
    from ai_co_scientist.cyclegan import (
        GATE_SAMPLE_N,
        GATE_THRESHOLDS,
        PREREGISTERED,
        SIM_TRAIN_N,
        gate_sample_indices,
        indices_sha256,
        sha256_file,
    )
    if sim_case_path is None:
        idx = gate_sample_indices(SIM_TRAIN_N, GATE_SAMPLE_N, seed=42)
        train_idx = val_idx = None
    else:
        case = np.load(sim_case_path)
        train_idx, val_idx = cyclegan.sim_split_indices(case)
        idx = cyclegan.validation_gate_indices(case)
    shift_val = 0.0 if shifts_ok else 10.0  # 10.0 > SHIFT_P95_MAX(1.0) -- 재평가하면 반드시 실패
    total_epochs = PREREGISTERED["epochs_fixed"] + PREREGISTERED["epochs_decay"]
    # 요약값은 evaluate_gate가 만든 그대로(recheck_gate가 정확 일치를 본다). passed만 True로
    # 덮는다 -- require_gate_passed는 이 값을 신뢰하지 않고 원본 값으로 재평가한다
    summary = cyclegan.evaluate_gate(np.full(GATE_SAMPLE_N, shift_val), 0.0,
                                     local=np.zeros(GATE_SAMPLE_N))
    gate = {
        **summary,
        "passed": True,
        "thresholds": GATE_THRESHOLDS,
        "shifts": [shift_val] * GATE_SAMPLE_N,
        "local_shifts": [0.0] * GATE_SAMPLE_N,
        "indices_sha256": indices_sha256(idx),
        "ckpt_sha256": sha_override if sha_override is not None else sha256_file(ckpt_path),
        "ckpt_epoch": total_epochs if ckpt_epoch_override is None else ckpt_epoch_override,
        "config": dict(PREREGISTERED) if config_override is None else config_override,
        "report_id": report_id if report_id_field is None else report_id_field,
    }
    if sim_sem_path is not None:
        gate["sim_sem_sha256"] = sha256_file(sim_sem_path)
    if sim_case_path is not None:
        gate["sim_case_sha256"] = sha256_file(sim_case_path)
        gate["split"] = cyclegan.sim_split_provenance(train_idx, val_idx)
    path.write_text(json.dumps(gate), encoding="utf-8")
    return path


def _full_environment(tmp_path, *, report_id="H6-TEST"):
    """require_gate_passed를 완전히 통과하는 ckpt/gate.json/cache를 갖춘 시나리오 하나를 만든다
    -- out-dir 안전성처럼 gate **이후** 단계를 테스트하려면 이 환경이 필요하다."""
    ckpt = tmp_path / _expected_ckpt_name(report_id)
    ckpt.write_bytes(b"dummy-ckpt-bytes")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "sim_sem.npy").write_bytes(b"sim-bytes")
    (cache / "sim_depth.npy").write_bytes(b"depth-bytes")
    np.save(cache / "sim_case.npy", _production_case_fixture())
    (cache / "real_sem.npy").write_bytes(b"real-bytes")
    gate_json = _make_gate_json(tmp_path / "gate.json", ckpt, report_id=report_id,
                                sim_sem_path=cache / "sim_sem.npy",
                                sim_case_path=cache / "sim_case.npy")
    return ckpt, cache, gate_json


# ── import 계약 ──────────────────────────────────────────────

def test_no_top_level_torch_or_cv2_imports():
    for script in (TRAIN_SCRIPT, TRANSLATE_SCRIPT):
        roots = _module_level_import_roots(script)
        assert "torch" not in roots, f"{script.name}가 최상단에서 torch를 임포트한다"
        assert "cv2" not in roots, f"{script.name}가 최상단에서 cv2를 임포트한다"


def test_help_works_without_torch():
    for script in (TRAIN_SCRIPT, TRANSLATE_SCRIPT):
        proc = _run(script, "--help")
        assert proc.returncode == 0, proc.stderr
        assert _no_torch_leak(proc)


def test_train_cyclegan_subcommand_help_without_torch():
    for sub in ("plan", "train", "gate"):
        proc = _run(TRAIN_SCRIPT, sub, "--help")
        assert proc.returncode == 0, proc.stderr
        assert _no_torch_leak(proc)


# ── plan ─────────────────────────────────────────────────────

def test_plan_emits_preregistered_config(tmp_path):
    _touch_sem_cache(tmp_path)
    proc = _run(TRAIN_SCRIPT, "plan", "--report-id", "H6-TEST", "--cache-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])

    from ai_co_scientist.cyclegan import PREREGISTERED
    assert last["config"] == PREREGISTERED
    assert last["expected_ckpt_name"] == _expected_ckpt_name("H6-TEST")


def test_plan_declares_full_cache_train_validation_contract(tmp_path):
    _touch_sem_cache(tmp_path)
    proc = _run(TRAIN_SCRIPT, "plan", "--report-id", "H6-TEST", "--cache-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout.strip().splitlines()[-1])
    assert plan["inputs"]["sim_case"].endswith("sim_case.npy")
    assert plan["split"] == {
        "method": "map_level_split",
        "total_n": 173_304,
        "train_n": 138_648,
        "val_n": 34_656,
        "val_frac": 0.2,
        "seed": 42,
        "gate_pool": "validation",
        "gate_sample_n": 2_048,
    }


def test_plan_rejects_config_deviation(tmp_path):
    _touch_sem_cache(tmp_path)
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"seed": 43}), encoding="utf-8")
    proc = _run(TRAIN_SCRIPT, "plan", "--report-id", "H6-TEST", "--cache-dir", str(tmp_path),
               "--config-json", str(override))
    assert proc.returncode != 0
    assert "seed" in proc.stderr
    assert _no_torch_leak(proc)


# ── train: 거부 경로 (전부 torch 이전) ──────────────────────────

def test_train_rejects_test_directory_before_torch(tmp_path):
    bad_cache = tmp_path / "test" / "cache"
    _touch_sem_cache(bad_cache)
    proc = _run(TRAIN_SCRIPT, "train", "--report-id", "H6-TEST",
               "--cache-dir", str(bad_cache), "--out-dir", str(tmp_path / "ckpt"))
    assert proc.returncode != 0
    assert _no_torch_leak(proc)
    assert not (tmp_path / "ckpt").exists()


def test_train_refuses_when_ckpt_already_exists(tmp_path):
    cache = tmp_path / "cache"
    _touch_sem_cache(cache)
    out_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    (out_dir / _expected_ckpt_name("H6-TEST")).write_bytes(b"existing")
    proc = _run(TRAIN_SCRIPT, "train", "--report-id", "H6-TEST",
               "--cache-dir", str(cache), "--out-dir", str(out_dir))
    assert proc.returncode != 0
    assert _no_torch_leak(proc)


# ── gate: 거부 경로 (전부 torch 이전) ───────────────────────────

def test_gate_rejects_test_directory_before_torch(tmp_path):
    bad_cache = tmp_path / "test" / "cache"
    bad_cache.mkdir(parents=True)
    (bad_cache / "sim_sem.npy").touch()
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"dummy")
    out_json = tmp_path / "gate.json"
    proc = _run(TRAIN_SCRIPT, "gate", "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--cache-dir", str(bad_cache), "--out-json", str(out_json))
    assert proc.returncode != 0
    assert _no_torch_leak(proc)
    assert not out_json.exists()


def test_gate_rejects_wrong_ckpt_name_before_torch(tmp_path):
    """리뷰 지적: 체크포인트 파일명이 규약(expected_ckpt_name)과 다르면 torch 이전에 거부.
    재개점(.resume.pt)이 gate에 조용히 먹히는 것을 막는 검사가 바로 이것이다."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "sim_sem.npy").touch()
    wrong_name_ckpt = tmp_path / f"{'H6-TEST'}-cyclegan.resume.pt"  # 재개점 이름
    wrong_name_ckpt.write_bytes(b"dummy")
    out_json = tmp_path / "gate.json"
    proc = _run(TRAIN_SCRIPT, "gate", "--report-id", "H6-TEST", "--ckpt", str(wrong_name_ckpt),
               "--cache-dir", str(cache), "--out-json", str(out_json))
    assert proc.returncode == 3
    assert _no_torch_leak(proc)
    assert not out_json.exists()
    assert "resume ckpt" in proc.stderr


def test_gate_refuses_when_out_json_already_exists(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "sim_sem.npy").touch()
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"dummy")
    out_json = tmp_path / "gate.json"
    out_json.write_text("{}", encoding="utf-8")
    proc = _run(TRAIN_SCRIPT, "gate", "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--cache-dir", str(cache), "--out-json", str(out_json))
    assert proc.returncode != 0
    assert _no_torch_leak(proc)


# ── translate_sim: 거부 경로 (전부 torch 이전, 산출물 없음) ─────

def test_translate_sim_refuses_missing_gate_json(tmp_path):
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"dummy-ckpt-bytes")
    missing_gate = tmp_path / "missing_gate.json"
    out_dir = tmp_path / "out"
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--gate-json", str(missing_gate), "--cache-dir", str(tmp_path / "cache"),
               "--out-cache-dir", str(out_dir))
    assert proc.returncode == 3, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last["status"] == "refused"
    assert _no_torch_leak(proc)
    assert not out_dir.exists()
    assert not Path(str(out_dir) + ".partial").exists()


def test_translate_sim_refuses_wrong_ckpt_name(tmp_path):
    """resume ckpt 이름을 그대로 넘기면 -- gate.json이 존재하고 그 외엔 다 맞아도 -- 이름
    검사 하나로 거부된다(`require_gate_passed`가 내부에서 하는 검사, 여전히 torch 이전)."""
    wrong_ckpt = tmp_path / "H6-TEST-cyclegan.resume.pt"
    wrong_ckpt.write_bytes(b"dummy-ckpt-bytes")
    gate_json = _make_gate_json(tmp_path / "gate.json", wrong_ckpt, report_id="H6-TEST")
    out_dir = tmp_path / "out"
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(wrong_ckpt),
               "--gate-json", str(gate_json), "--cache-dir", str(tmp_path / "cache"),
               "--out-cache-dir", str(out_dir))
    assert proc.returncode == 3, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last["status"] == "refused"
    assert "파일명" in last["reason"]
    assert _no_torch_leak(proc)
    assert not out_dir.exists()
    assert not Path(str(out_dir) + ".partial").exists()


def test_translate_sim_refuses_ckpt_sha_mismatch(tmp_path):
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"dummy-ckpt-bytes")
    gate_json = _make_gate_json(tmp_path / "gate.json", ckpt, sha_override="0" * 64)
    out_dir = tmp_path / "out"
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--gate-json", str(gate_json), "--cache-dir", str(tmp_path / "cache"),
               "--out-cache-dir", str(out_dir))
    assert proc.returncode == 3, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last["status"] == "refused"
    assert _no_torch_leak(proc)
    assert not out_dir.exists()
    assert not Path(str(out_dir) + ".partial").exists()


def test_translate_sim_refuses_failed_gate(tmp_path):
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"dummy-ckpt-bytes")
    gate_json = _make_gate_json(tmp_path / "gate.json", ckpt, shifts_ok=False)
    out_dir = tmp_path / "out"
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--gate-json", str(gate_json), "--cache-dir", str(tmp_path / "cache"),
               "--out-cache-dir", str(out_dir))
    assert proc.returncode == 3, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last["status"] == "refused"
    assert _no_torch_leak(proc)
    assert not out_dir.exists()
    assert not Path(str(out_dir) + ".partial").exists()


def test_translate_sim_refuses_report_id_mismatch(tmp_path):
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"dummy-ckpt-bytes")
    gate_json = _make_gate_json(tmp_path / "gate.json", ckpt, report_id_field="H6-OTHER")
    out_dir = tmp_path / "out"
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--gate-json", str(gate_json), "--cache-dir", str(tmp_path / "cache"),
               "--out-cache-dir", str(out_dir))
    assert proc.returncode == 3, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last["status"] == "refused"
    assert _no_torch_leak(proc)
    assert not out_dir.exists()


def test_translate_sim_refuses_config_deviation(tmp_path):
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"dummy-ckpt-bytes")
    from ai_co_scientist.cyclegan import PREREGISTERED
    bad_config = {**PREREGISTERED, "seed": 43}
    gate_json = _make_gate_json(tmp_path / "gate.json", ckpt, config_override=bad_config)
    out_dir = tmp_path / "out"
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--gate-json", str(gate_json), "--cache-dir", str(tmp_path / "cache"),
               "--out-cache-dir", str(out_dir))
    assert proc.returncode == 3, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last["status"] == "refused"
    assert _no_torch_leak(proc)
    assert not out_dir.exists()


# ── translate_sim: gate를 통과한 뒤(torch 이전)의 안전장치들 ─────

def test_translate_sim_leakage_guard_reachable_before_torch(tmp_path):
    """cache/test_sem.npy와 real_sem.npy가 바이트 단위로 같으면(이름만 바꾼 사본) 거부한다 --
    gate/ckpt-이름 검사를 모두 통과한 뒤에도 순수 해시 비교라 torch 이전이다."""
    ckpt, cache, gate_json = _full_environment(tmp_path)
    payload = b"same-bytes-for-both"
    (cache / "real_sem.npy").write_bytes(payload)
    (cache / "test_sem.npy").write_bytes(payload)  # real_sem.npy와 바이트가 같은 사본

    out_dir = tmp_path / "out"
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--gate-json", str(gate_json), "--cache-dir", str(cache),
               "--out-cache-dir", str(out_dir))
    assert proc.returncode == 3, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last["status"] == "refused"
    assert "test_sem" in last["reason"] or "sha256" in last["reason"]
    assert _no_torch_leak(proc)
    assert not out_dir.exists()
    assert not Path(str(out_dir) + ".partial").exists()


def test_translate_sim_refuses_existing_out_dir(tmp_path):
    ckpt, cache, gate_json = _full_environment(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--gate-json", str(gate_json), "--cache-dir", str(cache),
               "--out-cache-dir", str(out_dir))
    assert proc.returncode == 3, proc.stderr
    assert _no_torch_leak(proc)


def test_translate_sim_refuses_existing_partial_dir(tmp_path):
    ckpt, cache, gate_json = _full_environment(tmp_path)
    out_dir = tmp_path / "out"
    partial_dir = tmp_path / "out.partial"
    partial_dir.mkdir()
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--gate-json", str(gate_json), "--cache-dir", str(cache),
               "--out-cache-dir", str(out_dir))
    assert proc.returncode == 3, proc.stderr
    assert _no_torch_leak(proc)
    assert not out_dir.exists()


def test_translate_sim_refuses_out_dir_equal_to_cache_dir(tmp_path):
    ckpt, cache, _ = _full_environment(tmp_path)
    gate_json = _make_gate_json(tmp_path / "gate2.json", ckpt, sim_sem_path=cache / "sim_sem.npy")
    proc = _run(TRANSLATE_SCRIPT, "--report-id", "H6-TEST", "--ckpt", str(ckpt),
               "--gate-json", str(gate_json), "--cache-dir", str(cache),
               "--out-cache-dir", str(cache))
    assert proc.returncode == 3, proc.stderr
    assert _no_torch_leak(proc)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="ast 최신 기능 불필요 -- 자리표시")
def test_scripts_share_torch_helpers_through_src_not_each_other():
    # 두 스크립트가 공유하는 ckpt 로드·배치 번역은 src/에 산다(architecture.md: scripts는 얇게).
    # 스크립트끼리 import하면 scripts/가 sys.path에 있어야만 동작하고 로직이 scripts에 자란다
    assert "from train_cyclegan import" not in TRANSLATE_SCRIPT.read_text(encoding="utf-8")
    for script in (TRAIN_SCRIPT, TRANSLATE_SCRIPT):
        source = script.read_text(encoding="utf-8")
        assert "def translate_batches(" not in source, script.name
        assert "def load_generators(" not in source, script.name
        assert "def load_and_validate_ckpt(" not in source, script.name


# ── 경로 위생 강화: 대소문자 무시 + resolve ─────────────────────

@pytest.mark.parametrize("path", [
    "runtime/cache/TEST_SEM.NPY",
    "runtime/cache/Test_Names.json",
    "runtime/cache/Test_anything.NPY",
    "data/TeSt/SEM/x.png",
])
def test_reject_test_paths_is_case_insensitive(path):
    # Windows 파일시스템은 대소문자를 구분하지 않는다 — TEST_SEM.NPY는 test_sem.npy와 같은 파일이다
    with pytest.raises(ValueError):
        cyclegan.reject_test_paths([path])


def test_reject_test_paths_checks_resolved_symlink_target(tmp_path):
    # 이름은 무해한 sim_sem.npy지만 실제로는 test_sem.npy를 가리키는 링크 — resolve()로 잡는다
    target = tmp_path / "test_sem.npy"
    target.write_bytes(b"x")
    link = tmp_path / "cache" / "sim_sem.npy"
    link.parent.mkdir()
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("이 환경은 symlink를 만들 권한이 없다")
    with pytest.raises(ValueError):
        cyclegan.reject_test_paths([link])


# ── gate 호출 계약: 사전등록 3기준만 판정 · 국소 probe는 진단 전용 · 요약값 정확 일치 ──

def test_gate_thresholds_are_exactly_the_three_preregistered_criteria():
    # 정지 규칙 = 사전등록 문서의 세 문턱. 미보정 국소 probe의 설정은 여기 없다
    assert cyclegan.GATE_THRESHOLDS == {"shift_median_max": 0.5, "shift_p95_max": 1.0,
                                        "roundtrip_mae_max": 0.10}


def test_evaluate_gate_without_local_is_a_complete_verdict():
    r = cyclegan.evaluate_gate(np.zeros(cyclegan.GATE_SAMPLE_N), 0.0)
    assert r["passed"] is True
    assert "local_probe" not in r


def test_gate_result_carries_no_local_verdict_and_labels_the_probe_diagnostic():
    r = cyclegan.evaluate_gate(_ZL, 0.0, local=_ZL)
    assert "local_passed" not in r and "preregistered_passed" not in r
    assert r["local_probe"]["role"] == cyclegan.LOCAL_PROBE_ROLE == "diagnostic_only_uncalibrated"
    assert r["local_probe"]["method"] == cyclegan.LOCAL_PROBE_METHOD


@pytest.mark.parametrize("local", [
    np.full(cyclegan.GATE_SAMPLE_N, 50.0),
    np.full(cyclegan.GATE_SAMPLE_N, np.nan),
])
def test_local_probe_never_flips_passed_either_way(local):
    # 통과 쪽: 진단 값이 아무리 나빠도 사전등록 통과를 뒤집지 않는다
    assert cyclegan.evaluate_gate(_ZL, 0.0, local=local)["passed"] is True
    # 실패 쪽: 진단 값이 완벽해도 사전등록 실패를 구하지 못한다(fail-closed)
    bad = cyclegan.evaluate_gate(_shifts(cyclegan.GATE_SAMPLE_N, 0.6), 0.0, local=_ZL)
    assert bad["passed"] is False


def test_gate_passed_is_exactly_the_and_of_three_preregistered_criteria():
    ok = cyclegan.evaluate_gate(_ZL, 0.0, local=_ZL)
    assert ok["passed"] is True
    p95_bad = np.zeros(cyclegan.GATE_SAMPLE_N)
    p95_bad[: cyclegan.GATE_SAMPLE_N // 10] = cyclegan.SHIFT_P95_MAX + 1.0
    bad = {
        "shift_median": dict(shifts=_shifts(cyclegan.GATE_SAMPLE_N, 0.6)),
        "shift_p95": dict(shifts=p95_bad),
        "roundtrip_mae": dict(mae=cyclegan.ROUNDTRIP_MAE_MAX + 1e-9),
    }
    for key, kw in bad.items():
        r = cyclegan.evaluate_gate(kw.get("shifts", _ZL), kw.get("mae", 0.0), local=_ZL)
        assert r["passed"] is False, key
        assert [f.split(" ")[0] for f in r["failures"]] == [key], r["failures"]


@pytest.mark.parametrize("field,value", [
    ("shift_median", -1.0),
    ("local_probe", {"role": "authoritative"}),
    ("failures", ["whatever"]),
])
def test_require_gate_passed_rejects_summary_that_disagrees_with_raw_values(tmp_path, field, value):
    # 원본 per-image 값은 통과인데 요약만 손댄 gate — 사람이 읽는 값과 재평가 값이 따로 논다
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-930")
    gate[field] = value
    gate_path.unlink()
    cyclegan.write_once_json(gate_path, gate)
    with pytest.raises(cyclegan.GateFailedError, match="요약값"):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-930")


def test_measure_geometry_identical_is_zero_and_warp_is_caught():
    orig = np.clip(_hole_scene(), 0, 255).astype(np.uint8)[None]
    same = cyclegan.measure_geometry(orig, orig.copy())
    assert same["shifts"].tolist() == [0.0]
    assert same["local"].tolist() == [0.0]
    assert same["signed"].shape == (1, 2)
    warped = np.clip(_hole_scene(warp=_smooth_warp(1.0)), 0, 255).astype(np.uint8)[None]
    assert cyclegan.measure_geometry(orig, warped)["local"][0] > cyclegan.SHIFT_MEDIAN_MAX


def test_measure_geometry_enforces_uint8_shape_contract():
    with pytest.raises(ValueError):
        cyclegan.measure_geometry(np.zeros((1, 72, 48)), np.zeros((1, 72, 48)))


# ── manifest 불변성: 해시 + 내용 계약 ──────────────────────────────

def _built_manifest(tmp_path, report_id="EXP-940"):
    parts = _manifest_parts(tmp_path, report_id)
    path = tmp_path / "out" / "manifest.json"
    manifest = cyclegan.build_manifest(**parts)
    cyclegan.write_once_json(path, manifest)
    return path, manifest


def _rewrite(path, manifest):
    path.unlink()
    cyclegan.write_once_json(path, manifest)


def test_build_manifest_refuses_gate_without_raw_values(tmp_path):
    parts = _manifest_parts(tmp_path, "EXP-941")
    for k in ("shifts", "local_shifts"):  # passed=True 요약만 남기고 원본 값을 뺀다
        del parts["gate"][k]
    with pytest.raises(ValueError, match="재검증"):
        cyclegan.build_manifest(**parts)


def test_build_manifest_refuses_test_source_path(tmp_path):
    parts = _manifest_parts(tmp_path, "EXP-942")
    src = tmp_path / "cache" / "test_sem.npy"
    _write_bytes(src, b"real")
    parts["source_files"]["real_sem"] = src
    with pytest.raises(ValueError):
        cyclegan.build_manifest(**parts)


@pytest.mark.parametrize("mutate,expect", [
    (lambda m: m["output_files"]["sim_sem"].update(path="../sim_sem.npy"), "맨 파일명"),
    (lambda m: m["gate"].update(local_shifts=[5.0] * cyclegan.GATE_SAMPLE_N), "gate 재검증"),
    (lambda m: m["gate"].update(shifts=[5.0] * cyclegan.GATE_SAMPLE_N), "gate 재검증"),
    (lambda m: m["gate"].update(shift_p95=0.25), "gate 재검증"),
    (lambda m: m.update(hypothesis="H7"), "hypothesis"),
    (lambda m: m["gate"].update(report_id="EXP-OTHER"), "report_id"),
    (lambda m: m["output_files"]["sim_depth"].update(sha256="0" * 64), "sim_depth"),
    (lambda m: m.pop("ckpt"), "ckpt"),
    (lambda m: m["source_files"]["sim_sem"].update(path="runtime/cache/TEST_SEM.npy"), "경로 위생"),
    (lambda m: m["source_files"].pop("real_sem"), "source 목록"),
    (lambda m: m["source_files"].update(extra=m["source_files"]["real_sem"]), "source 목록"),
    (lambda m: m.update(x_domain="sim"), "x_domain"),
    (lambda m: m.update(y_source="none"), "y_source"),
    (lambda m: m.update(git_commit="not-a-commit"), "git_commit"),
])
def test_verify_manifest_rejects_content_tampering_not_only_hash_tampering(tmp_path, mutate,
                                                                          expect):
    # 해시가 안 바뀌는 조작(manifest 본문만 고침)도 재검증에서 잡혀야 manifest를 믿을 수 있다
    path, manifest = _built_manifest(tmp_path)
    cyclegan.verify_manifest(path)  # 원본은 통과
    mutate(manifest)
    _rewrite(path, manifest)
    with pytest.raises(ValueError, match=expect):
        cyclegan.verify_manifest(path)


def test_cyclegan_training_data_provenance_binds_all_inputs_and_split(tmp_path, monkeypatch):
    """resume에서 입력 하나라도 바뀌면 model/optimizer state를 적용하기 전에 거부해야 한다."""
    monkeypatch.setattr(cyclegan, "SIM_TOTAL_N", 16)
    monkeypatch.setattr(cyclegan, "SIM_TRAIN_N", 12)
    monkeypatch.setattr(cyclegan, "SIM_VAL_N", 4)
    monkeypatch.setattr(cyclegan, "SIM_SPLIT_VAL_FRAC", 0.25)
    monkeypatch.setattr(cyclegan, "SIM_SPLIT_SEED", 7)
    cache = tmp_path / "cache"
    _write_small_split_cache(cache)
    (cache / "real_sem.npy").write_bytes(b"real-domain")
    case = np.load(cache / "sim_case.npy")
    train_idx, val_idx = cyclegan.sim_split_indices(case)

    got = cyclegan.cyclegan_training_data_provenance(
        cache / "sim_sem.npy", cache / "sim_case.npy", cache / "real_sem.npy",
        train_idx, val_idx)

    assert set(got["inputs"]) == {"sim_sem", "sim_case", "real_sem"}
    assert got["split"]["train_n"] == 12
    assert got["split"]["val_n"] == 4
    assert cyclegan.validate_cyclegan_training_data(got) == got


def test_cyclegan_resume_provenance_rejects_changed_input_before_state_load(tmp_path,
                                                                            monkeypatch):
    monkeypatch.setattr(cyclegan, "SIM_TOTAL_N", 16)
    monkeypatch.setattr(cyclegan, "SIM_TRAIN_N", 12)
    monkeypatch.setattr(cyclegan, "SIM_VAL_N", 4)
    monkeypatch.setattr(cyclegan, "SIM_SPLIT_VAL_FRAC", 0.25)
    monkeypatch.setattr(cyclegan, "SIM_SPLIT_SEED", 7)
    cache = tmp_path / "cache"
    _write_small_split_cache(cache)
    (cache / "real_sem.npy").write_bytes(b"real-domain")
    case = np.load(cache / "sim_case.npy")
    train_idx, val_idx = cyclegan.sim_split_indices(case)
    expected = cyclegan.cyclegan_training_data_provenance(
        cache / "sim_sem.npy", cache / "sim_case.npy", cache / "real_sem.npy",
        train_idx, val_idx)
    stored = json.loads(json.dumps(expected))
    stored["inputs"]["real_sem"]["sha256"] = "0" * 64

    with pytest.raises(cyclegan.GateFailedError, match="training_data"):
        cyclegan.require_matching_training_data(stored, expected)


# ── GPU 락 · 덮어쓰기 정책 (스크립트 in-process, 가짜 락 — 기계 전역 gpu-0은 안 건드린다) ──

def _load_script(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_small_split_contract(monkeypatch, *script_modules):
    values = {
        "SIM_TOTAL_N": 16,
        "SIM_TRAIN_N": 12,
        "SIM_VAL_N": 4,
        "SIM_SPLIT_VAL_FRAC": 0.25,
        "SIM_SPLIT_SEED": 7,
        "GATE_SAMPLE_N": 2,
    }
    for name, value in values.items():
        monkeypatch.setattr(cyclegan, name, value)
        for mod in script_modules:
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, value)


def _write_small_split_cache(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    sem = np.zeros((16, cyclegan.IMG_H, cyclegan.IMG_W), dtype=np.uint8)
    sem[:, 0, 0] = np.arange(16, dtype=np.uint8)
    np.save(cache / "sim_sem.npy", sem)
    np.save(cache / "sim_case.npy", _small_paired_case_fixture())
    np.save(cache / "sim_depth.npy", np.arange(16, dtype=np.uint8))
    return sem


class _FakeLock:
    """`resource_lock` 대역. 진입·해제를 `events`에 적고, `busy`면 진입 시 `ResourceBusy`."""

    def __init__(self, events, busy=False):
        self.events = events
        self.busy = busy
        self.names = []

    def __call__(self, name, **kwargs):
        from contextlib import contextmanager

        from ai_co_scientist.locks import ResourceBusy
        self.names.append(name)

        @contextmanager
        def cm():
            if self.busy:
                self.events.append("busy")
                raise ResourceBusy(f"fake: {name}")
            self.events.append("lock")
            try:
                yield
            finally:
                self.events.append("unlock")
        return cm()


def _trap_data_reads(monkeypatch, mod, events):
    """스크립트의 `np.load`를 기록 후 중단시킨다 — 락 안에서 데이터 읽기가 처음 일어나는지 본다."""
    def _load(*a, **k):
        events.append("np.load")
        raise RuntimeError("stop: 데이터 읽기 지점 도달")
    monkeypatch.setattr(mod.np, "load", _load)


def _train_args(tmp_path, **kw):
    import argparse
    cache = tmp_path / "cache"
    _touch_sem_cache(cache)
    base = dict(report_id="H6-TEST", cache_dir=str(cache), out_dir=str(tmp_path / "ckpt"),
                num_workers=0, amp=False, resume=False, config_json="")
    base.update(kw)
    return argparse.Namespace(**base)


def _gate_args(tmp_path):
    import argparse
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "sim_sem.npy").touch()
    (cache / "sim_case.npy").touch()
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"dummy")
    return argparse.Namespace(report_id="H6-TEST", ckpt=str(ckpt), cache_dir=str(cache),
                              out_json=str(tmp_path / "gate.json"), batch_size=8)


def test_gpu_lock_name_is_gpu0():
    assert cyclegan.GPU_LOCK == "gpu-0"


@pytest.mark.parametrize("sub", ["train", "gate"])
def test_train_cli_busy_gpu_refuses_before_reading_data(tmp_path, monkeypatch, sub):
    from ai_co_scientist.locks import ResourceBusy
    mod = _load_script(TRAIN_SCRIPT, f"_h6_train_busy_{sub}")
    events = []
    fake = _FakeLock(events, busy=True)
    monkeypatch.setattr(mod, "resource_lock", fake)
    _trap_data_reads(monkeypatch, mod, events)
    args = _train_args(tmp_path) if sub == "train" else _gate_args(tmp_path)
    with pytest.raises(ResourceBusy):
        getattr(mod, sub)(args)
    assert fake.names == ["gpu-0"]
    assert events == ["busy"]  # 데이터를 한 바이트도 안 읽었다
    assert not (tmp_path / "ckpt").exists() and not (tmp_path / "gate.json").exists()


@pytest.mark.parametrize("sub", ["train", "gate"])
def test_train_cli_reads_data_only_inside_gpu_lock_and_releases_it(tmp_path, monkeypatch, sub):
    mod = _load_script(TRAIN_SCRIPT, f"_h6_train_lock_{sub}")
    events = []
    monkeypatch.setattr(mod, "resource_lock", _FakeLock(events))
    _trap_data_reads(monkeypatch, mod, events)
    args = _train_args(tmp_path) if sub == "train" else _gate_args(tmp_path)
    with pytest.raises(RuntimeError, match="stop"):
        getattr(mod, sub)(args)
    assert events == ["lock", "np.load", "unlock"]  # 읽기는 락 안에서, 실패해도 락은 풀린다


def test_train_cli_main_maps_busy_gpu_to_exit_4(tmp_path, monkeypatch, capsys):
    mod = _load_script(TRAIN_SCRIPT, "_h6_train_main_busy")
    monkeypatch.setattr(mod, "resource_lock", _FakeLock([], busy=True))
    monkeypatch.setattr(mod, "ensure_utf8_console", lambda: None)
    args = _gate_args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_cyclegan.py", "gate", "--report-id", "H6-TEST",
                                      "--ckpt", args.ckpt, "--cache-dir", args.cache_dir,
                                      "--out-json", args.out_json])
    assert mod.main() == 4
    assert "gpu-0" in capsys.readouterr().err


def test_plan_takes_no_gpu_lock(tmp_path, monkeypatch):
    mod = _load_script(TRAIN_SCRIPT, "_h6_train_plan")
    fake = _FakeLock([], busy=True)
    monkeypatch.setattr(mod, "resource_lock", fake)
    args = _train_args(tmp_path)
    assert mod.plan(args) == 0
    assert fake.names == []


def test_gate_payload_fingerprints_global_validation_indices_and_case(tmp_path, monkeypatch):
    # The old gate sampled 0..138647 and could not prove which case-derived validation rows it saw.
    mod = _load_script(TRAIN_SCRIPT, "_h6_gate_split_contract")
    _patch_small_split_contract(monkeypatch, mod)
    cache = tmp_path / "cache"
    sem = _write_small_split_cache(cache)
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"checkpoint")
    out_json = tmp_path / "gate.json"
    args = argparse.Namespace(report_id="H6-TEST", batch_size=2)

    monkeypatch.setattr(mod, "load_generators", lambda *_a, **_k: (
        cyclegan.CycleGANConfig(), 100, object(), object()))
    monkeypatch.setattr(mod, "translate_u8", lambda _model, arr, _bs, _device: arr.copy())
    monkeypatch.setattr(mod, "measure_geometry", lambda orig, moved: {
        "shifts": np.zeros(len(orig)), "local": np.zeros(len(orig)),
        "signed": np.zeros((len(orig), 2)),
    })
    monkeypatch.setattr(mod, "roundtrip_mae", lambda _orig, _roundtrip: 0.0)

    assert mod._gate_locked(
        args, cache / "sim_sem.npy", cache / "sim_case.npy", ckpt, out_json) == 0
    gate = json.loads(out_json.read_text(encoding="utf-8"))
    expected_global = np.array([6, 13], dtype=np.int64)
    assert gate["indices_sha256"] == hashlib.sha256(expected_global.tobytes()).hexdigest()
    assert gate["sim_case_sha256"] == cyclegan.sha256_file(cache / "sim_case.npy")
    assert gate["split"]["train_n"] == 12 and gate["split"]["val_n"] == 4
    assert sem[expected_global, 0, 0].tolist() == [6, 13]


def test_train_dataset_exposes_only_train_global_rows(monkeypatch):
    # This pins the CLI consumer, not only the split helper: the Dataset length and row mapping
    # must be the 12 train globals while retaining the full 16-row cache behind it.
    torch = pytest.importorskip("torch")
    mod = _load_script(TRAIN_SCRIPT, "_h6_train_dataset_split_contract")
    _patch_small_split_contract(monkeypatch, mod)
    sem = np.zeros((16, cyclegan.IMG_H, cyclegan.IMG_W), dtype=np.uint8)
    sem[:, 0, 0] = np.arange(16, dtype=np.uint8)
    train_idx, _val_idx = cyclegan.sim_split_indices(_small_paired_case_fixture())
    dataset = mod._make_sim_dataset(torch, sem, train_idx)
    assert len(dataset) == 12
    recovered = [int(round(float(dataset[i][0, 0, 0].item() + 1.0) * 127.5))
                 for i in range(len(dataset))]
    assert recovered == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15]


def _translate_main(monkeypatch, mod, ckpt, cache, gate_json, out_dir):
    monkeypatch.setattr(mod, "ensure_utf8_console", lambda: None)
    monkeypatch.setattr(sys, "argv", [
        "translate_sim.py", "--report-id", "H6-TEST", "--ckpt", str(ckpt),
        "--gate-json", str(gate_json), "--cache-dir", str(cache), "--out-cache-dir", str(out_dir)])
    return mod.main()


def test_translate_busy_gpu_refuses_with_exit_4_and_no_outputs(tmp_path, monkeypatch, capsys):
    ckpt, cache, gate_json = _full_environment(tmp_path)
    mod = _load_script(TRANSLATE_SCRIPT, "_h6_translate_busy")
    events = []
    fake = _FakeLock(events, busy=True)
    monkeypatch.setattr(mod, "resource_lock", fake)
    _trap_data_reads(monkeypatch, mod, events)
    out_dir = tmp_path / "out"
    assert _translate_main(monkeypatch, mod, ckpt, cache, gate_json, out_dir) == 4
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["status"] == "busy"
    assert fake.names == ["gpu-0"]
    assert events == ["busy"]
    assert not out_dir.exists() and not Path(str(out_dir) + ".partial").exists()


def test_translate_reads_data_only_inside_gpu_lock(tmp_path, monkeypatch):
    ckpt, cache, gate_json = _full_environment(tmp_path)
    mod = _load_script(TRANSLATE_SCRIPT, "_h6_translate_lock")
    events = []
    monkeypatch.setattr(mod, "resource_lock", _FakeLock(events))
    _trap_data_reads(monkeypatch, mod, events)
    with pytest.raises(RuntimeError, match="stop"):
        _translate_main(monkeypatch, mod, ckpt, cache, gate_json, tmp_path / "out")
    assert events == ["lock", "np.load", "unlock"]


def test_translation_keeps_full_cache_and_post_write_gate_uses_validation_globals(
        tmp_path, monkeypatch):
    # H6 must translate all rows so downstream recreates the paired split; only the post-write
    # hygiene sample is restricted to the original validation globals.
    mod = _load_script(TRANSLATE_SCRIPT, "_h6_translate_split_contract")
    _patch_small_split_contract(monkeypatch, mod)
    cache = tmp_path / "cache"
    source = _write_small_split_cache(cache)
    real_path = cache / "real_sem.npy"
    real_path.write_bytes(b"real")
    ckpt = tmp_path / _expected_ckpt_name("H6-TEST")
    ckpt.write_bytes(b"checkpoint")
    out_dir = tmp_path / "translated"
    partial_dir = tmp_path / "translated.partial"
    args = argparse.Namespace(report_id="H6-TEST", batch_size=4, git_commit="abc123")
    seen = {}

    monkeypatch.setattr(mod, "load_generators", lambda *_a, **_k: (
        cyclegan.CycleGANConfig(), 100, object(), object()))
    def _translate(_model, arr, _batch_size, _device):
        seen["translated_globals"] = arr[:, 0, 0].tolist()
        return arr.copy()
    monkeypatch.setattr(mod, "translate_u8", _translate)
    def _measure(orig, moved):
        seen["gate_globals"] = orig[:, 0, 0].tolist()
        return {"shifts": np.zeros(len(orig)), "local": np.zeros(len(orig)),
                "signed": np.zeros((len(orig), 2))}
    monkeypatch.setattr(mod, "measure_geometry", _measure)
    monkeypatch.setattr(mod, "build_manifest", lambda **_kwargs: {})
    monkeypatch.setattr(mod, "verify_manifest", lambda _path: {})

    gate = {"roundtrip_mae": 0.0}
    assert mod._translate_locked(
        args, gate, cache / "sim_sem.npy", cache / "sim_depth.npy",
        cache / "sim_case.npy", real_path, ckpt, out_dir, partial_dir) == 0
    written = np.load(out_dir / "sim_sem.npy")
    assert len(written) == 16
    assert seen["translated_globals"] == list(range(16))
    assert seen["gate_globals"] == [6, 13]
    assert np.array_equal(written, source)


def test_translate_refuses_test_path_before_hashing_anything(tmp_path, monkeypatch, capsys):
    # 경로 위생은 gate 하드 스톱(파일 해시)보다 먼저다 — test 경로는 열어 보지도 않는다
    ckpt, cache, gate_json = _full_environment(tmp_path)
    mod = _load_script(TRANSLATE_SCRIPT, "_h6_translate_order")
    hashed = []
    monkeypatch.setattr(mod, "require_gate_passed",
                        lambda *a, **k: hashed.append(a) or pytest.fail("gate를 먼저 읽었다"))
    assert _translate_main(monkeypatch, mod, ckpt, cache, gate_json,
                           tmp_path / "test" / "out") == 3
    assert hashed == []
    assert "test" in json.loads(capsys.readouterr().out.strip().splitlines()[-1])["reason"]


def test_train_refuses_existing_resume_without_resume_flag(tmp_path):
    cache = tmp_path / "cache"
    _touch_sem_cache(cache)
    out_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    resume = out_dir / "H6-TEST-cyclegan.resume.pt"
    resume.write_bytes(b"resume-state")
    proc = _run(TRAIN_SCRIPT, "train", "--report-id", "H6-TEST",
               "--cache-dir", str(cache), "--out-dir", str(out_dir))
    assert proc.returncode != 0
    assert "--resume" in proc.stderr
    assert resume.read_bytes() == b"resume-state"  # 조용히 처음부터 돌며 덮지 않았다
    assert _no_torch_leak(proc)


def test_train_refuses_resume_flag_without_resume_file(tmp_path):
    cache = tmp_path / "cache"
    _touch_sem_cache(cache)
    proc = _run(TRAIN_SCRIPT, "train", "--report-id", "H6-TEST", "--resume",
               "--cache-dir", str(cache), "--out-dir", str(tmp_path / "ckpt"))
    assert proc.returncode != 0
    assert "재개점이 없다" in proc.stderr
    assert not (tmp_path / "ckpt").exists()
    assert _no_torch_leak(proc)


# ── 정지 의미론: 사전등록 기준만 판정, 미보정 국소 probe는 기록만 ──────────

def test_local_shift_p95_alone_is_only_a_probe_flag():
    n = cyclegan.GATE_SAMPLE_N
    local = np.zeros(n)
    local[: n // 10] = cyclegan.SHIFT_P95_MAX + 1.0  # 상위 10%만 크게 — median은 0 그대로
    r = cyclegan.evaluate_gate(_ZL, 0.0, local=local)
    assert r["passed"] is True and r["failures"] == []
    probe = r["local_probe"]
    assert probe["flags"] == [f"local_shift_p95 {probe['p95']} > {cyclegan.SHIFT_P95_MAX}"]


def test_recheck_gate_survives_json_roundtrip_of_nontrivial_floats(tmp_path):
    # 요약값 정확 일치 검사가 0이 아닌 부동소수에서도 JSON 왕복 뒤 성립해야 한다(거짓 거부 방지)
    rng = np.random.default_rng(7)
    shifts = rng.uniform(0, 0.45, cyclegan.GATE_SAMPLE_N)
    local = rng.uniform(0, 0.45, cyclegan.GATE_SAMPLE_N)
    gate = cyclegan.evaluate_gate(shifts, 0.0123456789, local=local)
    gate.update({"shifts": shifts.tolist(), "local_shifts": local.tolist()})
    path = tmp_path / "gate.json"
    cyclegan.write_once_json(path, gate)
    assert cyclegan.recheck_gate(json.loads(path.read_text(encoding="utf-8")))["passed"] is True


def test_require_gate_passed_rejects_gate_written_with_other_local_method(tmp_path):
    # 타일·탐색·평탄 기준이 바뀐 뒤에는 옛 gate JSON의 probe 기록이 재현되지 않는다
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-950")
    gate["local_probe"]["method"] = {**gate["local_probe"]["method"], "tile": 12}
    _rewrite(gate_path, gate)
    with pytest.raises(cyclegan.GateFailedError, match="요약값"):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-950")


def test_require_gate_passed_rejects_legacy_gate_that_stopped_on_local_criteria(tmp_path):
    # 국소 기준을 정지 규칙에 넣던 옛 형식(임계값에 local_* 포함)은 현재 정지 규칙과 다르다
    gate_path, ckpt, gate = _valid_gate_and_ckpt(tmp_path, report_id="EXP-951")
    gate["thresholds"] = {**gate["thresholds"], "local_shift_median_max": 0.5,
                          "local_shift_p95_max": 1.0, "local_tile": 24}
    gate["local_passed"] = True
    _rewrite(gate_path, gate)
    with pytest.raises(cyclegan.GateFailedError, match="임계값"):
        cyclegan.require_gate_passed(gate_path, ckpt, report_id="EXP-951")


def _ring(seed, cy=36.0, cx=24.0, k=0.0):
    """경사진 가장자리의 구멍(시그모이드) + 픽셀 잡음. k>0이면 (cy,cx) 중심 대칭 팽창."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:cyclegan.IMG_H, 0:cyclegan.IMG_W].astype(np.float64)
    r = np.hypot(yy - cy, xx - cx) / (1 + k)
    return 40 + 160 / (1 + np.exp(-(r - 10) / 1.2)) + rng.normal(0, 4, r.shape)


def _gblur(img, sy, sx):
    h, w = img.shape
    k = np.exp(-2 * np.pi ** 2 * ((sy * np.fft.fftfreq(h))[:, None] ** 2
                                  + (sx * np.fft.fftfreq(w))[None] ** 2))
    return np.fft.ifft2(np.fft.fft2(img) * k).real


def _gate_on_stack(orig, moved):
    geo = cyclegan.measure_geometry(orig, moved)
    n = cyclegan.GATE_SAMPLE_N
    return geo, cyclegan.evaluate_gate(np.resize(geo["shifts"], n), 0.0,
                                       local=np.resize(geo["local"], n))


def test_appearance_only_blur_gamma_passes_the_gate():
    # 회귀(H6 blocker): 기하는 그대로인데 이방성 블러+감마가 경사 가장자리의 등밝기 윤곽을 ~1 px
    # 옮긴다. 강도 기반 국소 매칭은 이를 기하 이동과 구별하지 못하므로(원리적 한계) 미보정 probe가
    # 플래그를 올릴 수는 있어도 **gate를 멈추면 안 된다** — 외관 변경은 변환기의 정상 동작이다
    orig = np.stack([np.clip(_ring(s), 0, 255) for s in range(8)]).astype(np.uint8)
    app = np.stack([255 * (np.clip(_gblur(_ring(s), 1.5, 0.8), 0, 255) / 255) ** 1.8
                    for s in range(8)]).clip(0, 255).astype(np.uint8)
    _, r = _gate_on_stack(orig, app)
    assert r["passed"] is True, r["failures"]
    assert r["failures"] == []
    assert r["local_probe"]["flags"], "probe의 알려진 거짓 플래그가 사라졌다 — 보정 기록을 갱신할 것"


def test_known_residual_dilation_centred_inside_a_tile_passes_the_gate():
    # 알려진 잔여 위험(사전등록 §조건에 명시): 한 타일(12,12 중심) 안의 대칭 팽창은 전역 phase
    # correlation도 국소 probe도 보지 못한다. gate는 "형태 보존"을 보증하지 않는다 — 이 테스트가
    # 깨지면(=잡게 되면) 사전등록의 잔여 위험 문단을 갱신할 것
    orig = np.clip(_ring(0, cy=12, cx=12), 0, 255).astype(np.uint8)[None]
    grown = np.clip(_ring(0, cy=12, cx=12, k=0.10), 0, 255).astype(np.uint8)[None]
    assert np.abs(orig.astype(int) - grown.astype(int)).max() > 20  # 실제로 모양이 바뀌었다
    geo, r = _gate_on_stack(orig, grown)
    assert geo["local"][0] < cyclegan.SHIFT_MEDIAN_MAX
    assert r["passed"] is True
    assert r["local_probe"]["flags"] == []


def test_preregistration_states_the_executable_stop_rule():
    # 사전등록 문서가 코드의 정지 규칙과 정확히 같아야 한다: 세 문턱, 표본 2,048, 국소 probe는
    # 진단 전용, 팽창 사각지대를 잔여 위험으로 명시
    doc = (Path(__file__).resolve().parents[1] / "docs" / "experiment"
           / "H6-cyclegan-sim-to-real.md").read_text(encoding="utf-8")
    gate_line = next(line for line in doc.splitlines() if line.startswith("- **기하 위생 gate**"))
    for needle in ("2,048", "중앙값이 0.5 pixel 이하", "p95가 1.0 pixel 이하", "MAE는 0.10 이하",
                   "세 기준", "NaN"):
        assert needle in gate_line, needle
    probe_line = next(line for line in doc.splitlines() if line.startswith("- **국소 진단 probe**"))
    for needle in ("진단 전용", "판정에 관여하지 않는다", "미보정", "팽창", "잔여 위험"):
        assert needle in probe_line, needle
