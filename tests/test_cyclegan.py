"""`ai_co_scientist.cyclegan` — H6 CycleGAN sim→real 순수 로직을 **행동**으로 검사한다.

합성 numpy 픽스처만 쓴다. torch/cv2는 이 워크트리에 없다 — torch가 필요한 케이스는
`pytest.importorskip("torch")`로 건너뛴다(실제로는 skip된다. 그것이 이 파일이 서 있는 이유다).
"""
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


def test_evaluate_gate_passes_at_exact_boundary():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, cyclegan.SHIFT_MEDIAN_MAX)  # 중앙값=p95=경계
    r = cyclegan.evaluate_gate(shifts, cyclegan.ROUNDTRIP_MAE_MAX)
    assert r["passed"] is True
    assert r["failures"] == []


def test_evaluate_gate_fails_on_shift_median_alone():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, cyclegan.SHIFT_MEDIAN_MAX + 0.01)
    r = cyclegan.evaluate_gate(shifts, 0.0)
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
    r = cyclegan.evaluate_gate(shifts, 0.0)
    assert r["passed"] is False
    assert any("shift_p95" in f for f in r["failures"])
    assert not any("shift_median" in f for f in r["failures"])


def test_evaluate_gate_fails_on_roundtrip_mae_alone():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, 0.0)
    r = cyclegan.evaluate_gate(shifts, cyclegan.ROUNDTRIP_MAE_MAX + 0.01)
    assert r["passed"] is False
    assert any("roundtrip_mae" in f for f in r["failures"])
    assert not any("shift_median" in f or "shift_p95" in f for f in r["failures"])


def test_evaluate_gate_nan_shifts_fails_not_silently_passes():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, 0.0)
    shifts[0] = np.nan
    r = cyclegan.evaluate_gate(shifts, 0.0)
    assert r["passed"] is False
    assert np.isnan(r["shift_median"]) and np.isnan(r["shift_p95"])


def test_evaluate_gate_nan_mae_fails():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, 0.0)
    r = cyclegan.evaluate_gate(shifts, float("nan"))
    assert r["passed"] is False
    assert any("roundtrip_mae" in f for f in r["failures"])


def test_evaluate_gate_wrong_n_raises():
    with pytest.raises(ValueError):
        cyclegan.evaluate_gate(np.zeros(10, dtype=np.float64), 0.0)


def test_evaluate_gate_diagnostics_never_flip_passed():
    # diagnostics(p99, max, mean_dy/dx)는 참고용이다 — 사전등록된 3개 기준만 passed를 결정한다.
    # 상위 2%(45/2048)를 극단으로 밀면: p95 인덱스(0.95*2047≈1944.65)는 여전히 0 구간에 있어
    # median=p95=0을 유지해 통과하지만, p99 인덱스(0.99*2047≈2026.53)는 그 45개 구간에 걸려
    # p99만 임계값을 넘는다 — diagnostics가 gating과 분리돼 있다는 것을 수치로 고정한다.
    n = cyclegan.GATE_SAMPLE_N
    k = 45
    shifts = np.zeros(n, dtype=np.float64)
    shifts[n - k:] = 999.0
    signed = np.zeros((n, 2), dtype=np.float64)
    signed[n - k:] = [999.0, -999.0]
    r = cyclegan.evaluate_gate(shifts, 0.0, signed=signed)
    assert r["passed"] is True  # diagnostics가 극단이어도 gating 기준(median/p95/mae) 통과면 통과
    assert r["diagnostics"]["shift_max"] == 999.0
    assert r["diagnostics"]["shift_p99"] > cyclegan.SHIFT_P95_MAX  # 진단값 자체는 크다


def test_evaluate_gate_diagnostics_contain_p99_and_max():
    shifts = np.linspace(0.0, 1.0, cyclegan.GATE_SAMPLE_N)
    r = cyclegan.evaluate_gate(shifts, 0.0)
    assert r["diagnostics"]["shift_max"] == pytest.approx(1.0)
    assert r["diagnostics"]["shift_p99"] == pytest.approx(np.percentile(shifts, 99))


def test_evaluate_gate_diagnostics_mean_dy_dx_from_signed():
    n = cyclegan.GATE_SAMPLE_N
    shifts = np.zeros(n, dtype=np.float64)
    signed = np.zeros((n, 2), dtype=np.float64)
    signed[:, 0] = 2.0
    signed[:, 1] = -3.0
    r = cyclegan.evaluate_gate(shifts, 0.0, signed=signed)
    assert r["diagnostics"]["mean_dy"] == pytest.approx(2.0)
    assert r["diagnostics"]["mean_dx"] == pytest.approx(-3.0)


def test_evaluate_gate_diagnostics_omit_mean_dy_dx_without_signed():
    shifts = _shifts(cyclegan.GATE_SAMPLE_N, 0.0)
    r = cyclegan.evaluate_gate(shifts, 0.0)
    assert "mean_dy" not in r["diagnostics"]
    assert "mean_dx" not in r["diagnostics"]


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


def _passing_gate() -> dict:
    shifts = np.zeros(cyclegan.GATE_SAMPLE_N, dtype=np.float64)
    return cyclegan.evaluate_gate(shifts, 0.0)


def test_build_manifest_refuses_failed_gate(tmp_path):
    ckpt = tmp_path / "EXP-900-cyclegan.pt"
    _write_bytes(ckpt, b"ckpt-bytes")
    failing_gate = cyclegan.evaluate_gate(
        _shifts(cyclegan.GATE_SAMPLE_N, cyclegan.SHIFT_MEDIAN_MAX + 1.0), 0.0)
    with pytest.raises(ValueError):
        cyclegan.build_manifest(
            report_id="EXP-900", config=cyclegan.PREREGISTERED, gate=failing_gate,
            ckpt_path=ckpt, source_files={}, output_files={}, git_commit="deadbeef")


def test_build_manifest_and_verify_manifest_roundtrip(tmp_path):
    ckpt = tmp_path / "EXP-901-cyclegan.pt"
    src = tmp_path / "sim_sem.npy"
    out = tmp_path / "out" / "sim_sem.npy"
    _write_bytes(ckpt, b"ckpt-bytes")
    _write_bytes(src, b"source-bytes")
    _write_bytes(out, b"output-bytes")

    manifest = cyclegan.build_manifest(
        report_id="EXP-901", config=cyclegan.PREREGISTERED, gate=_passing_gate(),
        ckpt_path=ckpt, source_files={"sim_sem": src}, output_files={"sim_sem": out},
        git_commit="deadbeef")
    manifest_path = tmp_path / "out" / "manifest.json"
    cyclegan.write_once_json(manifest_path, manifest)

    verified = cyclegan.verify_manifest(manifest_path)
    assert verified["hypothesis"] == "H6"
    assert verified["x_domain"] == "sim_translated_to_real_appearance"
    assert verified["y_source"] == "sim_depth_gt"


def test_verify_manifest_detects_a_single_flipped_byte(tmp_path):
    ckpt = tmp_path / "EXP-902-cyclegan.pt"
    out = tmp_path / "sim_sem.npy"
    _write_bytes(ckpt, b"ckpt-bytes")
    _write_bytes(out, b"output-bytes")

    manifest = cyclegan.build_manifest(
        report_id="EXP-902", config=cyclegan.PREREGISTERED, gate=_passing_gate(),
        ckpt_path=ckpt, source_files={}, output_files={"sim_sem": out}, git_commit="deadbeef")
    manifest_path = tmp_path / "manifest.json"
    cyclegan.write_once_json(manifest_path, manifest)

    data = bytearray(out.read_bytes())
    data[0] ^= 0x01  # 바이트 1개만 뒤집는다
    out.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="sim_sem"):
        cyclegan.verify_manifest(manifest_path)



def test_manifest_survives_rename_of_its_directory(tmp_path):
    # 회귀: translate_sim.py는 `<out>.partial/`에 쓰고 manifest를 만든 뒤 `<out>`으로 rename한다.
    # 출력 경로를 절대경로로 적으면 rename 직후 모든 output이 "파일 없음"이 되어, 정상 산출물이
    # 영구히 검증 불가가 됐다. 출력은 manifest 디렉터리 기준 상대 이름으로 적어야 한다.
    ckpt = tmp_path / "EXP-903-cyclegan.pt"
    src = tmp_path / "cache" / "sim_sem.npy"
    partial = tmp_path / "EXP-903-translated.partial"
    _write_bytes(ckpt, b"ckpt-bytes")
    _write_bytes(src, b"source-bytes")
    _write_bytes(partial / "sim_sem.npy", b"output-bytes")
    _write_bytes(partial / "sim_depth.npy", b"depth-bytes")

    manifest = cyclegan.build_manifest(
        report_id="EXP-903", config=cyclegan.PREREGISTERED, gate=_passing_gate(),
        ckpt_path=ckpt, source_files={"sim_sem": src},
        output_files={"sim_sem": partial / "sim_sem.npy", "sim_depth": partial / "sim_depth.npy"},
        git_commit="deadbeef")
    cyclegan.write_once_json(partial / "manifest.json", manifest)
    final = tmp_path / "EXP-903-translated"
    partial.rename(final)

    verified = cyclegan.verify_manifest(final / "manifest.json")
    assert verified["output_files"]["sim_sem"]["path"] == "sim_sem.npy"
    assert verified["output_files"]["sim_depth"]["path"] == "sim_depth.npy"


def test_build_manifest_rejects_outputs_outside_one_directory(tmp_path):
    # 출력이 상대 이름으로 적히므로, 두 디렉터리에 흩어진 출력은 manifest 하나로 기술할 수 없다
    ckpt = tmp_path / "EXP-904-cyclegan.pt"
    _write_bytes(ckpt, b"ckpt-bytes")
    _write_bytes(tmp_path / "a" / "sim_sem.npy", b"x")
    _write_bytes(tmp_path / "b" / "sim_depth.npy", b"y")
    with pytest.raises(ValueError, match="한 디렉터리"):
        cyclegan.build_manifest(
            report_id="EXP-904", config=cyclegan.PREREGISTERED, gate=_passing_gate(),
            ckpt_path=ckpt, source_files={},
            output_files={"sim_sem": tmp_path / "a" / "sim_sem.npy",
                          "sim_depth": tmp_path / "b" / "sim_depth.npy"},
            git_commit="deadbeef")


# ── require_gate_passed: 강화된 하드 스톱 ──────────────────────

def _valid_gate_and_ckpt(tmp_path, *, report_id="EXP-910"):
    """`require_gate_passed`를 통과해야 하는 최소 gate JSON + ckpt 쌍을 만든다."""
    ckpt = tmp_path / cyclegan.expected_ckpt_name(report_id)
    _write_bytes(ckpt, b"ckpt-bytes")
    idx = cyclegan.gate_sample_indices(cyclegan.SIM_TRAIN_N, cyclegan.GATE_SAMPLE_N, 42)
    shifts = [0.0] * cyclegan.GATE_SAMPLE_N
    evaluated = cyclegan.evaluate_gate(np.asarray(shifts, dtype=np.float64), 0.0)
    gate = dict(evaluated)
    gate.update({
        "report_id": report_id,
        "ckpt_sha256": cyclegan.sha256_file(ckpt),
        "ckpt_epoch": cyclegan.PREREGISTERED["epochs_fixed"] + cyclegan.PREREGISTERED["epochs_decay"],
        "config": cyclegan.PREREGISTERED,
        "shifts": shifts,
        "signed_shifts": [[0.0, 0.0]] * cyclegan.GATE_SAMPLE_N,
        "indices_sha256": cyclegan.indices_sha256(idx),
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
    (cache_dir / "real_sem.npy").touch()


def _expected_ckpt_name(report_id: str) -> str:
    from ai_co_scientist.cyclegan import expected_ckpt_name
    return expected_ckpt_name(report_id)


def _make_gate_json(path: Path, ckpt_path: Path, *, report_id="H6-TEST",
                    shifts_ok=True, sha_override=None, report_id_field=None,
                    ckpt_epoch_override=None, config_override=None, sim_sem_path=None):
    """`require_gate_passed`가 요구하는 모든 필드를 채운 gate JSON을 합성한다.

    기본값은 전부 "통과"하도록 만들어 두고, 파라미터로 정확히 하나씩만 어긋나게 할 수 있다
    -- require_gate_passed의 검사 순서(이름→report_id→epoch→passed→config→임계값→n→
    shifts 재평가→인덱스 지문→ckpt sha256→sim_sem sha256)를 그대로 따라가며 테스트한다.
    """
    from ai_co_scientist.cyclegan import (
        GATE_SAMPLE_N,
        PREREGISTERED,
        ROUNDTRIP_MAE_MAX,
        SHIFT_MEDIAN_MAX,
        SHIFT_P95_MAX,
        SIM_TRAIN_N,
        gate_sample_indices,
        indices_sha256,
        sha256_file,
    )
    idx = gate_sample_indices(SIM_TRAIN_N, GATE_SAMPLE_N, seed=42)
    shift_val = 0.0 if shifts_ok else 10.0  # 10.0 > SHIFT_P95_MAX(1.0) -- 재평가하면 반드시 실패
    total_epochs = PREREGISTERED["epochs_fixed"] + PREREGISTERED["epochs_decay"]
    gate = {
        "passed": True,  # require_gate_passed는 이 값을 신뢰하지 않고 shifts/mae로 재평가한다
        "n": GATE_SAMPLE_N,
        "shift_median": shift_val, "shift_p95": shift_val, "roundtrip_mae": 0.0,
        "thresholds": {"shift_median_max": SHIFT_MEDIAN_MAX, "shift_p95_max": SHIFT_P95_MAX,
                      "roundtrip_mae_max": ROUNDTRIP_MAE_MAX},
        "failures": [],
        "shifts": [shift_val] * GATE_SAMPLE_N,
        "indices_sha256": indices_sha256(idx),
        "ckpt_sha256": sha_override if sha_override is not None else sha256_file(ckpt_path),
        "ckpt_epoch": total_epochs if ckpt_epoch_override is None else ckpt_epoch_override,
        "config": dict(PREREGISTERED) if config_override is None else config_override,
        "report_id": report_id if report_id_field is None else report_id_field,
    }
    if sim_sem_path is not None:
        gate["sim_sem_sha256"] = sha256_file(sim_sem_path)
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
    (cache / "sim_case.npy").write_bytes(b"case-bytes")
    (cache / "real_sem.npy").write_bytes(b"real-bytes")
    gate_json = _make_gate_json(tmp_path / "gate.json", ckpt, report_id=report_id,
                                sim_sem_path=cache / "sim_sem.npy")
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


# ── 인덱스 표본 결정성 (torch 불필요) ────────────────────────────

def test_gate_sample_indices_used_by_scripts_is_deterministic():
    """gate()와 translate_sim()이 같은 (n_total, n_sample, seed)로 gate_sample_indices를 부르므로
    두 쪽에서 표본이 항상 같다 -- 이 계약이 깨지면 쓰기 후 재검증이 다른 표본을 잰다."""
    from ai_co_scientist.cyclegan import GATE_SAMPLE_N, SIM_TRAIN_N, gate_sample_indices

    a = gate_sample_indices(SIM_TRAIN_N, GATE_SAMPLE_N, seed=42)
    b = gate_sample_indices(SIM_TRAIN_N, GATE_SAMPLE_N, seed=42)
    assert np.array_equal(a, b)
    assert len(a) == GATE_SAMPLE_N


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
