"""`ai_co_scientist.two_head` — H8 mask·양의 깊이 2-head 순수 로직을 **행동**으로 검사한다.

sem.py와 같은 원칙: grep 계약이 아니라 실제 배열/딕셔너리 값으로 확인한다. torch 없는
dev 환경에서 전부 통과해야 하므로, torch가 필요한 두 함수(two_head_loss,
compose_output_torch)의 테스트만 `pytest.importorskip("torch")`로 개별 스킵한다 —
모듈 자체나 이 파일은 torch 없이도 수집(collect)되어야 한다.
"""
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from ai_co_scientist import two_head

# scripts/는 패키지가 아니라 sys.path에 경로를 얹어 import한다 — tests/test_infer_decomposed.py와
# 같은 패턴. 두 스크립트 모두 모듈 최상단에서 torch/cv2를 import하지 않으므로(지연 import),
# 이 파일 전체가 torch 없는 dev 환경에서도 수집·실행된다.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import infer_two_head  # noqa: E402
import train_two_head  # noqa: E402

# ── 공용 fixture 헬퍼 ────────────────────────────────────────

PREREG_DEFAULTS = dict(arch="mlp", batch_size=128, lr=1e-3, optimizer="AdamW", schedule="cosine",
                       epochs=15, seed=42, split_seed=0, val_frac=0.2)


def _fingerprint(**overrides) -> dict:
    fp = {"n_sim": 100, "sem_shape": (72, 48), "depth_shape": (72, 48),
          "case_counts": {1: 25, 2: 25, 3: 25, 4: 25}, "case_sha256": "a" * 64,
          "val_idx_sha256": "b" * 64, "depth_sample_sha256": "c" * 64}
    fp.update(overrides)
    return fp


def _sim_gate(arm="single", *, pred_pos_rate=None, gt_pos_rate=0.42, epoch=15) -> dict:
    """arm=two_head 기본값은 pred==gt(오차 0)라 항상 게이트를 통과하는 baseline이다."""
    if arm == "two_head" and pred_pos_rate is None:
        pred_pos_rate = gt_pos_rate
    return two_head.sim_mask_gate(
        {"pred_pos_rate": pred_pos_rate, "gt_pos_rate": gt_pos_rate}, arm, epoch)


def _build(arm="single", *, n_params=None, git_commit="deadbeef", git_dirty=False,
          sim_gt_pos_rate=0.42, resumed_from_epoch=None, fingerprint=None, sim_mask_gate=None,
          **hparam_overrides):
    hparams = dict(PREREG_DEFAULTS)
    hparams.update(hparam_overrides)
    if n_params is None:
        counts = two_head.mlp_param_counts()
        n_params = counts["single"] if arm == "single" else counts["two_head"]
    if sim_mask_gate is None:
        sim_mask_gate = _sim_gate(arm, gt_pos_rate=sim_gt_pos_rate)
    return two_head.build_manifest(
        arm=arm, n_train=100, n_val=20, n_params=n_params,
        data_fingerprint=fingerprint if fingerprint is not None else _fingerprint(),
        git_commit=git_commit, git_dirty=git_dirty, sim_gt_pos_rate=sim_gt_pos_rate,
        sim_mask_gate=sim_mask_gate, resumed_from_epoch=resumed_from_epoch, **hparams)


def _pair():
    fp = _fingerprint()
    a = _build("single", git_commit="c0ffee", fingerprint=fp)
    b = _build("two_head", git_commit="c0ffee", fingerprint=fp)
    return a, b


def _cfg(**overrides) -> dict:
    cfg = dict(two_head.INFERENCE_PREREGISTERED)
    cfg.update(overrides)
    return cfg


# ── split_target ─────────────────────────────────────────────

def test_split_target_basic_values():
    s = np.array([0.0, 0.2, 1.0, 0.0], dtype=np.float32)
    m, s_pos = two_head.split_target(s)
    assert m.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert s_pos.tolist() == pytest.approx([0.0, 0.2, 1.0, 0.0])
    assert m.dtype == np.float32 and s_pos.dtype == np.float32
    assert m.shape == s.shape and s_pos.shape == s.shape


def test_split_target_rejects_nan():
    with pytest.raises(ValueError):
        two_head.split_target(np.array([0.1, np.nan], dtype=np.float32))


def test_split_target_rejects_negative():
    with pytest.raises(ValueError):
        two_head.split_target(np.array([-0.01, 0.2], dtype=np.float32))


def test_split_target_rejects_above_one():
    with pytest.raises(ValueError):
        two_head.split_target(np.array([0.1, 1.01], dtype=np.float32))


# ── compose_output ───────────────────────────────────────────

def test_compose_output_bounds_and_exact_value():
    mask_logit = np.array([0.0, 10.0, -10.0])
    s_pos_hat = np.array([0.5, 1.5, -0.5])  # out of [0,1] on purpose — must be clipped
    out = two_head.compose_output(mask_logit, s_pos_hat)
    assert out.dtype == np.float32
    assert (out >= 0.0).all() and (out <= 1.0).all()
    # sigmoid(0)=0.5, clip(0.5,0,1)=0.5 -> 0.25
    assert out[0] == pytest.approx(0.25, abs=1e-6)


def test_compose_output_extreme_logits_stay_bounded_with_no_warnings():
    mask_logit = np.array([1e4, -1e4])
    s_pos_hat = np.array([0.5, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # RuntimeWarning(overflow) 발생 시 여기서 실패한다
        out = two_head.compose_output(mask_logit, s_pos_hat)
    assert out[0] == pytest.approx(0.5, abs=1e-6)
    assert out[1] == pytest.approx(0.0, abs=1e-6)


# ── two_head_loss_reference (numpy) ──────────────────────────

def test_two_head_loss_reference_matches_explicit_formula():
    logit = np.array([0.0, 2.0, -2.0, 0.0])
    s_pos_hat = np.array([0.3, 0.9, 0.1, 0.7])
    s = np.array([0.0, 0.8, 0.6, 0.0])  # m = [0,1,1,0]
    m = (s > 0).astype(np.float64)
    bce_expected = float(np.mean(np.maximum(logit, 0) - logit * m
                                 + np.log1p(np.exp(-np.abs(logit)))))
    l1_expected = float(np.mean(np.abs(s_pos_hat[m > 0] - s[m > 0])))
    ref = two_head.two_head_loss_reference(logit, s_pos_hat, s)
    assert ref["bce"] == pytest.approx(bce_expected)
    assert ref["l1_pos"] == pytest.approx(l1_expected)
    assert ref["total"] == pytest.approx(bce_expected + l1_expected)
    assert ref["n_pos"] == 2


def test_two_head_loss_reference_l1_pos_ignores_m0_pixels():
    logit = np.array([0.0, 2.0, -2.0, 0.0])
    s_pos_hat = np.array([0.3, 0.9, 0.1, 0.7])
    s = np.array([0.0, 0.8, 0.6, 0.0])
    ref = two_head.two_head_loss_reference(logit, s_pos_hat, s)
    # m=0인 idx 0을 아무 값으로 바꿔도 l1_pos/total이 그대로여야 한다
    perturbed = s_pos_hat.copy()
    perturbed[0] = 999.0
    ref2 = two_head.two_head_loss_reference(logit, perturbed, s)
    assert ref2["l1_pos"] == pytest.approx(ref["l1_pos"])
    assert ref2["total"] == pytest.approx(ref["total"])


def test_two_head_loss_reference_l1_pos_uses_raw_unclamped_s_pos_hat():
    logit = np.array([0.0, 2.0])
    s_pos_hat = np.array([0.3, 1.5])  # idx1 (m=1) 밖의 값 — clip되면 안 된다
    s = np.array([0.0, 0.8])
    ref = two_head.two_head_loss_reference(logit, s_pos_hat, s)
    raw_l1 = abs(1.5 - 0.8)
    clamped_l1 = abs(1.0 - 0.8)  # 잘못 clip했다면 나올 값
    assert ref["l1_pos"] == pytest.approx(raw_l1)
    assert ref["l1_pos"] != pytest.approx(clamped_l1)


def test_two_head_loss_reference_zero_positive_case():
    logit = np.array([-5.0, -5.0])
    s_pos_hat = np.array([0.9, 0.9])
    s = np.array([0.0, 0.0])  # 전부 m=0
    ref = two_head.two_head_loss_reference(logit, s_pos_hat, s)
    assert ref["n_pos"] == 0
    assert ref["l1_pos"] == 0.0
    bce_expected = float(np.mean(np.maximum(logit, 0) + np.log1p(np.exp(-np.abs(logit)))))
    assert ref["bce"] == pytest.approx(bce_expected)
    assert ref["total"] == pytest.approx(bce_expected)


# ── torch 의존 (스킵 가능) ────────────────────────────────────

def test_two_head_loss_matches_reference_torch():
    torch = pytest.importorskip("torch")
    logit = torch.tensor([0.0, 2.0, -2.0, 0.0], dtype=torch.float64, requires_grad=True)
    s_pos_hat = torch.tensor([0.3, 0.9, 0.1, 0.7], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([0.0, 0.8, 0.6, 0.0], dtype=torch.float64)
    total, bce, l1_pos = two_head.two_head_loss(logit, s_pos_hat, s)
    ref = two_head.two_head_loss_reference(logit.detach().numpy(), s_pos_hat.detach().numpy(),
                                           s.numpy())
    assert total.item() == pytest.approx(ref["total"])
    assert bce.item() == pytest.approx(ref["bce"])
    assert l1_pos.item() == pytest.approx(ref["l1_pos"])


def test_two_head_loss_zero_positive_keeps_graph():
    torch = pytest.importorskip("torch")
    logit = torch.tensor([-5.0, -5.0], requires_grad=True)
    s_pos_hat = torch.tensor([0.9, 0.9], requires_grad=True)
    s = torch.tensor([0.0, 0.0])
    total, bce, l1_pos = two_head.two_head_loss(logit, s_pos_hat, s)
    assert l1_pos.item() == 0.0
    assert l1_pos.requires_grad  # 값은 0이지만 그래프는 살아 있어야 한다
    total.backward()
    assert s_pos_hat.grad is not None


def test_compose_output_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    logit = torch.tensor([0.0, 10.0, -10.0, 1e4, -1e4], dtype=torch.float64)
    pos = torch.tensor([0.5, 1.5, -0.5, 0.3, 0.3], dtype=torch.float64)
    out_t = two_head.compose_output_torch(logit, pos)
    out_np = two_head.compose_output(logit.numpy(), pos.numpy())
    assert np.allclose(out_t.detach().numpy().astype(np.float64), out_np.astype(np.float64),
                       atol=1e-6)


# ── BinnedAUROC ──────────────────────────────────────────────

def test_binned_auroc_perfect_separation():
    auc = two_head.BinnedAUROC()
    auc.update(np.array([0.9, 0.8, 0.7]), np.array([1, 1, 1]))
    auc.update(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))
    assert auc.value() == pytest.approx(1.0)


def test_binned_auroc_inverted():
    auc = two_head.BinnedAUROC()
    auc.update(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))
    auc.update(np.array([0.9, 0.8, 0.7]), np.array([0, 0, 0]))
    assert auc.value() == pytest.approx(0.0)


def test_binned_auroc_all_ties():
    auc = two_head.BinnedAUROC()
    auc.update(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 1, 0, 0]))
    assert auc.value() == pytest.approx(0.5)


def test_binned_auroc_empty_class_raises():
    auc = two_head.BinnedAUROC()
    auc.update(np.array([0.5, 0.6]), np.array([1, 1]))
    with pytest.raises(ValueError):
        auc.value()


# ── StructureMetrics ─────────────────────────────────────────

def test_structure_metrics_without_mask_prob():
    sm = two_head.StructureMetrics()
    s = np.array([[[0.0, 0.2], [0.5, 0.0]]], dtype=np.float64)
    s_hat = np.array([[[0.1, 0.3], [0.4, 0.0]]], dtype=np.float64)
    levels = np.array([140.0])
    sm.update(s_hat, s, levels)
    v = sm.value()
    assert v["s_rmse"] == pytest.approx((0.03 / 4) ** 0.5)
    assert v["depth_rmse"] == pytest.approx((588.0 / 4) ** 0.5)
    assert v["pos_rmse"] == pytest.approx(0.1)
    assert v["mask_auroc"] == pytest.approx(1.0)
    assert v["mask_auroc_score"] == "s_hat"
    assert v["pred_pos_rate"] is None
    assert v["gt_pos_rate"] == pytest.approx(0.5)
    assert v["n_pixels"] == 4


def test_structure_metrics_uses_mask_prob_for_auroc_when_given():
    sm = two_head.StructureMetrics()
    s = np.array([[[0.0, 0.2], [0.5, 0.0]]], dtype=np.float64)
    # s_hat 자체는 나쁜 분리자(자기 AUROC=0.0)지만 mask_prob은 완벽한 분리자(1.0) —
    # 실제로 mask_prob이 쓰였는지 구분하는 fixture
    s_hat = np.array([[[0.5, 0.1], [0.2, 0.5]]], dtype=np.float64)
    mask_prob = np.array([[[0.1, 0.9], [0.8, 0.2]]], dtype=np.float64)
    levels = np.array([140.0])
    sm.update(s_hat, s, levels, mask_prob=mask_prob)
    v = sm.value()
    assert v["mask_auroc"] == pytest.approx(1.0)
    assert v["mask_auroc_score"] == "mask_prob"
    assert v["pred_pos_rate"] == pytest.approx(0.5)


def test_structure_metrics_accumulates_across_updates():
    sm = two_head.StructureMetrics()
    s = np.array([[[0.0, 0.2], [0.5, 0.0]]], dtype=np.float64)
    s_hat = np.array([[[0.1, 0.3], [0.4, 0.0]]], dtype=np.float64)
    levels = np.array([140.0])
    sm.update(s_hat, s, levels)
    sm.update(s_hat, s, levels)  # 동일 배치를 두 번 — streaming임을 증명
    v = sm.value()
    assert v["n_pixels"] == 8
    assert v["s_rmse"] == pytest.approx((0.03 / 4) ** 0.5)


# ── RateCounter ──────────────────────────────────────────────

def test_rate_counter_streams_across_updates():
    rc = two_head.RateCounter()
    rc.update(np.array([1, 0, 1, 0]))
    rc.update(np.array([1, 1, 0]))
    assert rc.value() == pytest.approx(4 / 7)


def test_rate_counter_empty_is_zero():
    rc = two_head.RateCounter()
    assert rc.value() == 0.0


# ── sim_gt_pos_rate_from_depth ───────────────────────────────

def test_sim_gt_pos_rate_from_depth_exact():
    depth = np.array([
        [[100, 140], [139, 0]],    # case1 (L=140): 3/4 positive
        [[140, 140], [140, 140]],  # case1: 0/4 positive
        [[0, 0], [0, 0]],          # case2 (L=150): 4/4 positive
        [[150, 149], [151, 0]],    # case2: 2/4 positive
    ], dtype=np.float32)
    case = np.array([1, 1, 2, 2], dtype=np.int64)
    idx_all = np.array([0, 1, 2, 3])
    rate = two_head.sim_gt_pos_rate_from_depth(depth, case, idx_all, chunk=2)
    assert rate == pytest.approx(9 / 16)
    rate_default_chunk = two_head.sim_gt_pos_rate_from_depth(depth, case, idx_all)
    assert rate_default_chunk == pytest.approx(9 / 16)
    idx_subset = np.array([0, 2])
    rate_subset = two_head.sim_gt_pos_rate_from_depth(depth, case, idx_subset, chunk=1)
    assert rate_subset == pytest.approx(7 / 8)


# ── sha256_array ─────────────────────────────────────────────

def test_sha256_array_differs_by_shape_for_same_bytes():
    a = np.arange(6, dtype=np.int32)
    b1 = a.reshape(2, 3)
    b2 = a.reshape(3, 2)
    assert b1.tobytes() == b2.tobytes()  # 바이트는 동일함을 확인
    assert two_head.sha256_array(b1) != two_head.sha256_array(b2)


def test_sha256_array_differs_by_dtype_for_same_bytes():
    z_i32 = np.zeros(4, dtype=np.int32)
    z_f32 = np.zeros(4, dtype=np.float32)
    assert z_i32.tobytes() == z_f32.tobytes()
    assert two_head.sha256_array(z_i32) != two_head.sha256_array(z_f32)


def test_sha256_array_is_deterministic():
    a = np.arange(10, dtype=np.float64)
    assert two_head.sha256_array(a) == two_head.sha256_array(a.copy())


def test_sha256_array_returns_hex_string():
    h = two_head.sha256_array(np.array([1, 2, 3]))
    assert isinstance(h, str) and len(h) == 64
    int(h, 16)  # ValueError가 나지 않아야 유효한 hex


# ── mask_diagnostics ─────────────────────────────────────────

def test_mask_diagnostics_exact():
    mp = np.array([0.05, 0.5, 0.95, 0.5])
    d = two_head.mask_diagnostics(mp)
    assert d["mean_prob"] == pytest.approx(0.5)
    assert d["pred_pos_rate"] == pytest.approx(0.25)
    assert d["confident_frac"] == pytest.approx(0.5)


# ── validate_hparams ─────────────────────────────────────────

def test_validate_hparams_valid_passes():
    assert two_head.validate_hparams(dict(PREREG_DEFAULTS)) == []


@pytest.mark.parametrize("key", list(PREREG_DEFAULTS))
def test_validate_hparams_flags_each_missing_key(key):
    hp = dict(PREREG_DEFAULTS)
    del hp[key]
    errs = two_head.validate_hparams(hp)
    assert any(key in e for e in errs), errs


@pytest.mark.parametrize("key,bad", [
    ("arch", "unet"), ("batch_size", 64), ("lr", 2e-3), ("optimizer", "SGD"),
    ("schedule", "step"), ("epochs", 10), ("seed", 1), ("split_seed", 1), ("val_frac", 0.3),
])
def test_validate_hparams_flags_each_mismatched_key(key, bad):
    hp = dict(PREREG_DEFAULTS)
    hp[key] = bad
    errs = two_head.validate_hparams(hp)
    assert any(key in e for e in errs), errs


# ── sim_mask_gate ────────────────────────────────────────────

def test_sim_mask_gate_single_does_not_apply_and_has_no_errors():
    g = two_head.sim_mask_gate({"pred_pos_rate": None, "gt_pos_rate": 0.5}, "single", epoch=15)
    assert g["applies"] is False
    assert g["errors"] == []
    assert g["threshold"] == two_head.MASK_THRESHOLD
    assert g["tolerance"] == two_head.MASK_RATE_TOLERANCE
    assert g["epoch"] == 15
    assert g["pred_pos_rate"] is None
    assert g["gt_pos_rate"] == 0.5


def test_sim_mask_gate_two_head_within_tolerance_passes():
    g = two_head.sim_mask_gate({"pred_pos_rate": 0.45, "gt_pos_rate": 0.5}, "two_head", epoch=15)
    assert g["applies"] is True
    assert g["errors"] == []


def test_sim_mask_gate_two_head_exceeds_tolerance_fails():
    g = two_head.sim_mask_gate({"pred_pos_rate": 0.7, "gt_pos_rate": 0.5}, "two_head", epoch=15)
    assert g["errors"] != []


def test_sim_mask_gate_two_head_missing_pred_pos_rate_is_itself_an_error():
    g = two_head.sim_mask_gate({"pred_pos_rate": None, "gt_pos_rate": 0.5}, "two_head", epoch=15)
    assert g["errors"] != []
    assert g["pred_pos_rate"] is None


# ── GPU_LOCK ─────────────────────────────────────────────────

def test_gpu_lock_constant():
    assert two_head.GPU_LOCK == "gpu-0"


# ── build_manifest / validate_manifest ───────────────────────

def test_build_manifest_valid_round_trips_through_validate():
    m = _build()
    assert two_head.validate_manifest(m) == []
    assert m["format"] == two_head.CKPT_FORMAT
    assert m["arm"] == "single"
    assert m["x_domain"] == "sim"
    assert m["y_source"] == "sim_depth_gt"
    assert m["target"] == "s = (L - d) / L"
    assert m["resumed_from_epoch"] is None


def test_build_manifest_rejects_unknown_arm():
    with pytest.raises(ValueError):
        two_head.build_manifest(arm="bogus", n_train=1, n_val=1, n_params=1,
                                data_fingerprint=_fingerprint(), git_commit="x", git_dirty=False,
                                sim_gt_pos_rate=0.1, sim_mask_gate={}, **PREREG_DEFAULTS)


def test_build_manifest_rejects_missing_hparam():
    hp = dict(PREREG_DEFAULTS)
    del hp["seed"]
    with pytest.raises(ValueError, match="seed"):
        two_head.build_manifest(arm="single", n_train=1, n_val=1, n_params=1,
                                data_fingerprint=_fingerprint(), git_commit="x", git_dirty=False,
                                sim_gt_pos_rate=0.1, sim_mask_gate={}, **hp)


def test_build_manifest_requires_sim_mask_gate_keyword():
    hp = dict(PREREG_DEFAULTS)
    with pytest.raises(TypeError):
        two_head.build_manifest(arm="single", n_train=1, n_val=1, n_params=1,
                                data_fingerprint=_fingerprint(), git_commit="x", git_dirty=False,
                                sim_gt_pos_rate=0.1, **hp)  # sim_mask_gate 누락 — 기본값이 없다


def test_build_manifest_resumed_from_epoch_value():
    m = _build(resumed_from_epoch=7)
    assert m["resumed_from_epoch"] == 7
    assert two_head.validate_manifest(m) == []


@pytest.mark.parametrize("key,bad", [
    ("arch", "unet"), ("batch_size", 64), ("lr", 2e-3), ("optimizer", "SGD"),
    ("schedule", "step"), ("epochs", 10), ("seed", 1), ("split_seed", 1), ("val_frac", 0.3),
])
def test_validate_manifest_flags_each_preregistered_key(key, bad):
    m = _build()
    assert two_head.validate_manifest(m) == []
    m2 = dict(m)
    m2[key] = bad
    errs = two_head.validate_manifest(m2)
    assert any(key in e for e in errs), errs


def test_validate_manifest_flags_missing_required_keys():
    m = _build()
    required = (list(two_head.PARITY_KEYS)
                + ["arm", "n_params", "loss", "output", "ckpt_selection", "git_dirty",
                   "resumed_from_epoch", "sim_mask_gate"])
    for key in required:
        m2 = dict(m)
        del m2[key]
        errs = two_head.validate_manifest(m2)
        assert any(key in e for e in errs), f"missing-key silent for {key!r}: {errs}"


def test_validate_manifest_flags_incomplete_sim_mask_gate():
    m = _build("two_head")
    m2 = dict(m)
    m2["sim_mask_gate"] = {"applies": True}  # 나머지 서브키가 없다
    errs = two_head.validate_manifest(m2)
    for k in ("threshold", "tolerance", "split", "epoch", "pred_pos_rate", "gt_pos_rate",
             "errors"):
        assert any(k in e for e in errs), f"sim_mask_gate subkey silent for {k!r}: {errs}"


def test_validate_manifest_flags_sim_mask_gate_applies_mismatch():
    m = _build("single")  # single이므로 applies는 False여야 한다
    m2 = dict(m)
    m2["sim_mask_gate"] = dict(m2["sim_mask_gate"])
    m2["sim_mask_gate"]["applies"] = True  # arm=single과 불일치
    errs = two_head.validate_manifest(m2)
    assert any("applies" in e for e in errs), errs


def test_validate_manifest_flags_incomplete_data_fingerprint():
    m = _build()
    m2 = dict(m)
    m2["data_fingerprint"] = {"n_sim": 1}  # 나머지 서브키가 없다
    errs = two_head.validate_manifest(m2)
    for k in ("sem_shape", "depth_shape", "case_counts", "case_sha256", "val_idx_sha256",
             "depth_sample_sha256"):
        assert any(k in e for e in errs), f"data_fingerprint subkey silent for {k!r}: {errs}"


@pytest.mark.parametrize("bad_rate", [-0.1, 1.1, float("nan")])
def test_validate_manifest_flags_bad_sim_gt_pos_rate(bad_rate):
    m = _build()
    m2 = dict(m)
    m2["sim_gt_pos_rate"] = bad_rate
    errs = two_head.validate_manifest(m2)
    assert any("sim_gt_pos_rate" in e for e in errs)


def test_validate_manifest_flags_wrong_domains():
    m = _build()
    m2 = dict(m)
    m2["x_domain"] = "real"
    assert any("x_domain" in e for e in two_head.validate_manifest(m2))
    m3 = dict(m)
    m3["y_source"] = "real_depth_gt"
    assert any("y_source" in e for e in two_head.validate_manifest(m3))


# ── validate_arm_pair ────────────────────────────────────────

def test_validate_arm_pair_clean_pass():
    a, b = _pair()
    assert two_head.validate_arm_pair(a, b) == []


def test_validate_arm_pair_rejects_same_arm_twice():
    a, b = _pair()
    b2 = dict(b)
    b2["arm"] = "single"
    errs = two_head.validate_arm_pair(a, b2)
    assert errs


def test_validate_arm_pair_rejects_dirty_tree():
    a, b = _pair()
    a2 = dict(a)
    a2["git_dirty"] = True
    errs = two_head.validate_arm_pair(a2, b)
    assert any("git_dirty" in e or "dirty" in e for e in errs)


def test_validate_arm_pair_rejects_empty_commit():
    a, b = _pair()
    b2 = dict(b)
    b2["git_commit"] = ""
    errs = two_head.validate_arm_pair(a, b2)
    assert any("git_commit" in e for e in errs)


def test_validate_arm_pair_resumed_from_epoch_may_differ():
    a, b = _pair()
    a2 = dict(a)
    a2["resumed_from_epoch"] = 3
    b2 = dict(b)
    b2["resumed_from_epoch"] = None
    assert two_head.validate_arm_pair(a2, b2) == []
    assert "resumed_from_epoch" not in two_head.PARITY_KEYS


def test_validate_arm_pair_sim_mask_gate_may_differ():
    a, b = _pair()
    a2 = dict(a)
    a2["sim_mask_gate"] = _sim_gate("single", gt_pos_rate=0.1, epoch=3)
    b2 = dict(b)
    b2["sim_mask_gate"] = _sim_gate("two_head", gt_pos_rate=0.42, epoch=9)
    assert two_head.validate_arm_pair(a2, b2) == []
    assert "sim_mask_gate" not in two_head.PARITY_KEYS


_PARITY_PERTURB = {
    "format": "bogus-format/v9",
    "arch": "unet",
    "batch_size": 64,
    "lr": 2e-3,
    "optimizer": "SGD",
    "schedule": "step",
    "epochs": 10,
    "seed": 1,
    "split_seed": 1,
    "val_frac": 0.3,
    "n_train": 999,
    "n_val": 999,
    "data_fingerprint": _fingerprint(n_sim=999),
    "git_commit": "other-commit",
    "sim_gt_pos_rate": 0.9,
    "x_domain": "real",
    "y_source": "other",
    "target": "other-target",
}


@pytest.mark.parametrize("key", two_head.PARITY_KEYS)
def test_validate_arm_pair_flags_each_parity_key(key):
    a, b = _pair()
    b2 = dict(b)
    b2[key] = _PARITY_PERTURB[key]
    errs = two_head.validate_arm_pair(a, b2)
    assert any(key in e for e in errs), errs


def test_parity_keys_covers_all_perturb_cases():
    # 위 파라미터 테이블이 PARITY_KEYS와 정확히 1:1이어야 새 키가 추가돼도 조용히 안 빠진다
    assert set(two_head.PARITY_KEYS) == set(_PARITY_PERTURB)


# ── ckpt_payload / resolve_ckpt ──────────────────────────────

def test_ckpt_payload_and_resolve_round_trip():
    m = _build("two_head")
    obj = two_head.ckpt_payload("two_head", m, {"w": 1})
    assert obj["format"] == two_head.CKPT_FORMAT
    assert obj["arch"] == m["arch"]
    arm, manifest = two_head.resolve_ckpt(obj)
    assert arm == "two_head"
    assert manifest == m


def test_resolve_ckpt_rejects_bare_state_dict():
    bare = {"encoder.0.weight": np.zeros(3), "encoder.0.bias": np.zeros(1)}
    with pytest.raises(ValueError, match="infer_decomposed"):
        two_head.resolve_ckpt(bare)


def test_resolve_ckpt_rejects_train_structure_style_ckpt():
    ts_ckpt = {"arch": "mlp", "state_dict": {"encoder.0.weight": np.zeros(3)}}
    with pytest.raises(ValueError, match="infer_decomposed"):
        two_head.resolve_ckpt(ts_ckpt)


def test_resolve_ckpt_rejects_wrong_format():
    obj = {"format": "other/v9", "arm": "single", "manifest": _build("single"), "state_dict": {}}
    with pytest.raises(ValueError):
        two_head.resolve_ckpt(obj)


def test_resolve_ckpt_rejects_arm_mismatch():
    m = _build("single")
    obj = two_head.ckpt_payload("single", m, {})
    obj["arm"] = "two_head"  # obj의 arm과 manifest의 arm이 어긋난다
    with pytest.raises(ValueError):
        two_head.resolve_ckpt(obj)


def test_resolve_ckpt_propagates_manifest_errors():
    m = _build("single")
    m["seed"] = 999  # PREREGISTERED 위반
    obj = two_head.ckpt_payload("single", m, {})
    with pytest.raises(ValueError):
        two_head.resolve_ckpt(obj)


# ── check_output ─────────────────────────────────────────────

def test_check_output_accepts_valid_float32():
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    assert two_head.check_output(s) == []


def test_check_output_rejects_nan():
    s = np.array([[[0.1, 0.2], [np.nan, 0.4]]], dtype=np.float32)
    assert two_head.check_output(s)


def test_check_output_rejects_inf():
    s = np.array([[[0.1, np.inf], [0.3, 0.4]]], dtype=np.float32)
    assert two_head.check_output(s)


def test_check_output_rejects_just_below_zero():
    s = np.array([[[-1e-6, 0.2], [0.3, 0.4]]], dtype=np.float32)
    errs = two_head.check_output(s)
    assert any("min" in e for e in errs)


def test_check_output_rejects_just_above_one():
    s = np.array([[[0.1, 0.2], [0.3, 1.0 + 1e-6]]], dtype=np.float32)
    errs = two_head.check_output(s)
    assert any("max" in e for e in errs)


def test_check_output_rejects_wrong_ndim():
    s = np.zeros((2, 2), dtype=np.float32)
    errs = two_head.check_output(s)
    assert errs


def test_check_output_rejects_wrong_dtype():
    s = np.full((1, 2, 2), 0.5, dtype=np.float64)
    errs = two_head.check_output(s)
    assert any("dtype" in e for e in errs)


def test_check_output_expected_shape_mismatch():
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    errs = two_head.check_output(s, expected_shape=(1, 3, 3))
    assert any("shape" in e for e in errs)


def test_check_output_expected_shape_match_passes():
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    assert two_head.check_output(s, expected_shape=(1, 2, 2)) == []


# ── check_mask_rate ──────────────────────────────────────────

def test_check_mask_rate_exactly_at_tolerance_stops():
    errs = two_head.check_mask_rate(two_head.MASK_RATE_TOLERANCE, 0.0)
    assert errs


def test_check_mask_rate_just_under_tolerance_passes():
    errs = two_head.check_mask_rate(0.0999, 0.0)
    assert errs == []


def test_check_mask_rate_nan_stops():
    assert two_head.check_mask_rate(float("nan"), 0.3)


# ── validate_inference_config ────────────────────────────────

def test_validate_inference_config_accepts_baseline():
    assert two_head.validate_inference_config(_cfg()) == []


def test_validate_inference_config_accepts_level_ckpt_backslash_spelling():
    cfg = _cfg(level_ckpt="runtime\\ckpt\\EXP-013-level-cnn.pt")
    assert two_head.validate_inference_config(cfg) == []


@pytest.mark.parametrize("key,bad", [
    ("level_source", "qda"), ("adabn", "off"), ("adabn_shuffle", 0),
    ("adabn_stats", "running"), ("tau", 0.05), ("level_smooth", 1),
    ("level_ckpt", "runtime/ckpt/other.pt"), ("histmatch", True),
    ("adabn_drop_last", True), ("level_hmm", True),
])
def test_validate_inference_config_flags_each_key(key, bad):
    cfg = _cfg(**{key: bad})
    errs = two_head.validate_inference_config(cfg)
    assert any(key in e for e in errs), errs


def test_validate_inference_config_flags_missing_key():
    cfg = _cfg()
    del cfg["tau"]
    errs = two_head.validate_inference_config(cfg)
    assert any("tau" in e for e in errs)


def test_inference_preregistered_keys_covered_by_parametrize():
    tested = {k for k, _ in [
        ("level_source", None), ("adabn", None), ("adabn_shuffle", None), ("adabn_stats", None),
        ("tau", None), ("level_smooth", None), ("level_ckpt", None), ("histmatch", None),
        ("adabn_drop_last", None), ("level_hmm", None),
    ]}
    assert tested == set(two_head.INFERENCE_PREREGISTERED)


# ── submission_gate ──────────────────────────────────────────

def test_submission_gate_single_with_mask_rate_errors():
    a, b = _pair()
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    errs = two_head.submission_gate(s, "single", 0.5, a, b, _cfg())
    assert any("mask_rate" in e for e in errs)


def test_submission_gate_single_clean_pass():
    a, b = _pair()
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    errs = two_head.submission_gate(s, "single", None, a, b, _cfg())
    assert errs == []


def test_submission_gate_two_head_allows_none_mask_rate():
    # real test mask_rate는 이제 게이트가 아니라 로그용 진단이다(reviewer H1) — None도 허용된다
    a, b = _pair()  # b: two_head, sim_mask_gate.errors == [] (pred==gt==0.42)
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    errs = two_head.submission_gate(s, "two_head", None, b, a, _cfg())
    assert errs == []


def test_submission_gate_two_head_allows_float_mask_rate_as_diagnostic_only():
    a, b = _pair()
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    # real GT가 없어 sim GT와 비교할 근거가 없으므로, mask_rate가 뭐든(0.99처럼 극단이어도)
    # 게이트는 이 값 자체로는 실패하지 않아야 한다
    errs = two_head.submission_gate(s, "two_head", 0.99, b, a, _cfg())
    assert errs == []


def test_submission_gate_two_head_surfaces_sim_mask_gate_errors():
    a, b = _pair()
    b2 = dict(b)
    # 학습 시점 sim 홀드아웃에서 마스크 예측 양성률이 GT에서 크게 벗어난 상태를 재현한다
    b2["sim_mask_gate"] = _sim_gate("two_head", pred_pos_rate=0.99, gt_pos_rate=0.42)
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    errs = two_head.submission_gate(s, "two_head", None, b2, a, _cfg())
    assert any("sim_mask_gate" in e for e in errs)


def test_submission_gate_expected_shape_passthrough():
    a, b = _pair()
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    errs = two_head.submission_gate(s, "single", None, a, b, _cfg(), expected_shape=(1, 3, 3))
    assert any("shape" in e for e in errs)


# ── mlp_param_counts ─────────────────────────────────────────

def test_mlp_param_counts_exact_numbers():
    d = two_head.mlp_param_counts()
    assert d["single"] == 8_468_736
    assert d["extra"] == 3_542_400
    assert d["two_head"] == 8_468_736 + 3_542_400
    assert d["extra_frac_total"] == pytest.approx(0.41830, abs=1e-5)
    assert d["extra_frac_output_layer"] == 1.0


def test_two_head_manifest_names_sigmoid_depth_path():
    # 매니페스트는 arm B depth 경로가 arm A와 같은 sigmoid임을 적어야 한다 (raw+clamp 아님)
    b = _build("two_head")
    assert b["output"] == "sigmoid(mask_logit) * sigmoid(depth_logit)"
    assert "L1(sigmoid(depth_logit), s | m=1)" in b["loss"]
    assert _build("single")["output"] == "sigmoid(logit)"


def test_resolve_ckpt_rejects_v1_raw_depth_head_format():
    # v1 ckpt는 raw depth head로 학습됐다 — state_dict 키가 같아 조용히 로드되면 의미가 바뀐다
    a, _ = _pair()
    obj = two_head.ckpt_payload("single", {**a, "format": "h8-two-head/v1"}, {})
    obj["format"] = "h8-two-head/v1"
    with pytest.raises(ValueError):
        two_head.resolve_ckpt(obj)


# ── H8 사전등록 문서 ↔ 코드 일치 ─────────────────────────────

H8_DOC = Path(__file__).resolve().parents[1] / "docs" / "experiment" / "H8-two-head-mask-depth.md"


def test_h8_doc_states_code_formulas_verbatim():
    doc = H8_DOC.read_text(encoding="utf-8")
    b = _build("two_head")
    assert b["output"] in doc
    assert b["loss"] in doc


def test_h8_doc_parameter_increase_matches_mlp_param_counts():
    doc = H8_DOC.read_text(encoding="utf-8")
    d = two_head.mlp_param_counts()
    assert f"{d['single']:,}" in doc and f"{d['two_head']:,}" in doc
    assert f"+{d['extra_frac_total'] * 100:.1f}%" in doc  # 전체 파라미터 기준
    assert "+100%" in doc  # 출력층 기준 (정확히 두 배)
    assert "output-layer parameters" not in doc  # 41.8%를 출력층 증가로 적던 옛 문구


def test_h8_doc_states_where_stop_gate_applies():
    doc = H8_DOC.read_text(encoding="utf-8")
    assert f"sigmoid(mask_logit) > {two_head.MASK_THRESHOLD}" in doc
    assert "sim holdout" in doc and "로깅 전용" in doc


# ══════════════════════════════════════════════════════════════
# CLI 계약 — scripts/train_two_head.py, scripts/infer_two_head.py (Engineer B, 읽기 전용)
# ══════════════════════════════════════════════════════════════

# T1 — 두 스크립트를 import해도 torch/cv2/train_structure/infer_decomposed가 로드되지 않는다.
# 이 테스트 파일 자체가 이미(모듈 최상단에서) 둘 다 import했으므로, 오염 없는 별도 프로세스로
# 확인한다.

def test_t1_importing_both_scripts_does_not_pull_in_torch_or_cv2():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {SCRIPTS.as_posix()!r})\n"
        "import train_two_head\n"
        "import infer_two_head\n"
        "leaked = [m for m in ('torch', 'cv2', 'train_structure', 'infer_decomposed')"
        " if m in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"무거운 모듈이 import 시점에 로드됐다: {result.stdout.strip()!r}\n{result.stderr}")


# T2 — train_two_head 파서 기본값은 two_head.PREREGISTERED와 일치해야 한다.

def test_t2_train_two_head_parser_defaults_match_preregistered():
    args = train_two_head.build_parser().parse_args(["--arm", "single", "--out", "x.pt"])
    assert args.epochs == two_head.PREREGISTERED["epochs"]
    assert args.batch_size == two_head.PREREGISTERED["batch_size"]
    assert args.lr == two_head.PREREGISTERED["lr"]
    assert args.seed == two_head.PREREGISTERED["seed"]
    assert args.split_seed == two_head.PREREGISTERED["split_seed"]
    assert args.val_frac == two_head.PREREGISTERED["val_frac"]
    assert args.num_workers == 0
    assert args.resume is False


# T3 — 필수 인자 누락/잘못된 --arm은 SystemExit. choices는 ARMS와 정확히 같다.

def test_t3_train_two_head_missing_arm_exits():
    with pytest.raises(SystemExit):
        train_two_head.build_parser().parse_args(["--out", "x.pt"])


def test_t3_train_two_head_missing_out_exits():
    with pytest.raises(SystemExit):
        train_two_head.build_parser().parse_args(["--arm", "single"])


def test_t3_train_two_head_bogus_arm_exits():
    with pytest.raises(SystemExit):
        train_two_head.build_parser().parse_args(["--arm", "bogus", "--out", "x.pt"])


def test_t3_train_two_head_arm_choices_equal_arms():
    parser = train_two_head.build_parser()
    arm_action = next(a for a in parser._actions if a.dest == "arm")
    assert tuple(arm_action.choices) == tuple(two_head.ARMS)


# T4 — infer_two_head: --ckpt/--peer-ckpt/--submit는 각각 개별적으로 필수. 기본값은
# INFERENCE_PREREGISTERED와 일치(level_ckpt는 Path.as_posix() 정규화). main()과 같은 방식으로
# 만든 cfg가 validate_inference_config를 무오류로 통과한다.

def test_t4_infer_two_head_missing_ckpt_exits():
    with pytest.raises(SystemExit):
        infer_two_head.build_parser().parse_args(["--peer-ckpt", "p.pt", "--submit", "s.zip"])


def test_t4_infer_two_head_missing_peer_ckpt_exits():
    with pytest.raises(SystemExit):
        infer_two_head.build_parser().parse_args(["--ckpt", "c.pt", "--submit", "s.zip"])


def test_t4_infer_two_head_missing_submit_exits():
    with pytest.raises(SystemExit):
        infer_two_head.build_parser().parse_args(["--ckpt", "c.pt", "--peer-ckpt", "p.pt"])


def test_t4_infer_two_head_defaults_match_inference_preregistered_and_pass_gate():
    args = infer_two_head.build_parser().parse_args(
        ["--ckpt", "c.pt", "--peer-ckpt", "p.pt", "--submit", "s.zip"])
    assert args.level_source == two_head.INFERENCE_PREREGISTERED["level_source"]
    assert args.adabn == two_head.INFERENCE_PREREGISTERED["adabn"]
    assert args.adabn_shuffle == two_head.INFERENCE_PREREGISTERED["adabn_shuffle"]
    assert args.tau == two_head.INFERENCE_PREREGISTERED["tau"]
    assert args.level_smooth == two_head.INFERENCE_PREREGISTERED["level_smooth"]
    assert (Path(args.level_ckpt).as_posix()
           == Path(two_head.INFERENCE_PREREGISTERED["level_ckpt"]).as_posix())

    # main()과 동일하게 조립한 cfg (infer_two_head.py에 별도 헬퍼 함수가 없으므로 args + 하드코딩된
    # 3개 키를 그대로 재현한다 — main()의 해당 리터럴과 1:1 대응, coding-patterns.md 참고)
    cfg = {
        "level_source": args.level_source, "adabn": args.adabn,
        "adabn_shuffle": args.adabn_shuffle, "tau": args.tau, "level_smooth": args.level_smooth,
        "level_ckpt": args.level_ckpt,
        "adabn_stats": "batch", "histmatch": False, "adabn_drop_last": False, "level_hmm": False,
    }
    assert two_head.validate_inference_config(cfg) == []


# T5 — `--help`는 PYTHONIOENCODING 없이도 exit 0이어야 한다 (ensure_utf8_console이 argparse
# 전에 불렸는지 핀). 자식이 stdout/stderr를 utf-8로 재구성하므로 부모도 utf-8로 읽어야 한다 —
# 이 콘솔(cp949)의 기본 디코딩에 맡기면 부모 쪽에서 UnicodeDecodeError가 난다(실측).

def _help_env_without_pythonioencoding() -> dict:
    return {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}


def test_t5_train_two_head_help_exits_zero_without_pythonioencoding():
    result = subprocess.run([sys.executable, str(SCRIPTS / "train_two_head.py"), "--help"],
                            capture_output=True, encoding="utf-8",
                            env=_help_env_without_pythonioencoding(), cwd=str(SCRIPTS.parent))
    assert result.returncode == 0, result.stderr


def test_t5_infer_two_head_help_exits_zero_without_pythonioencoding():
    result = subprocess.run([sys.executable, str(SCRIPTS / "infer_two_head.py"), "--help"],
                            capture_output=True, encoding="utf-8",
                            env=_help_env_without_pythonioencoding(), cwd=str(SCRIPTS.parent))
    assert result.returncode == 0, result.stderr


# T6 — infer_two_head._output_pos_rate: round(L*(1-s)) < L인 픽셀 비율. 경계값(s가 1/(2L) 바로
# 아래라 반올림하면 다시 L이 되는 경우)은 세면 안 된다는 게 핵심 계약이다.

def test_t6_output_pos_rate_exact_with_two_levels():
    # image0(L=140): 0.5→70(count) · 0.0→140(no) · 0.0035<1/280→round to 140(no, 경계) · 0.01→139(count)
    # image1(L=150): 0.5→75(count) · 0.0→150(no) · 0.0032<1/300→round to 150(no, 경계) · 0.02→147(count)
    structure = np.array([
        [[0.5, 0.0], [0.0035, 0.01]],
        [[0.5, 0.0], [0.0032, 0.02]],
    ], dtype=np.float32)
    levels = np.array([140.0, 150.0], dtype=np.float32)
    rate = infer_two_head._output_pos_rate(structure, levels)
    assert rate == pytest.approx(4 / 8)


def test_t6_output_pos_rate_just_below_half_lsb_rounds_back_to_background_and_does_not_count():
    level = 140.0
    s = 1.0 / (2.0 * level) - 1e-4  # 1/(2L)에 살짝 못 미친다 → L*(1-s)는 (L-0.5) 바로 위
    structure = np.array([[[s]]], dtype=np.float32)
    levels = np.array([level], dtype=np.float32)
    assert infer_two_head._output_pos_rate(structure, levels) == 0.0


# T7 — 소스 계약: submit 존재 거부가 main() 안에서 첫 torch 사용보다 먼저 나오고, submission_gate
# 호출이 reconstruct_and_zip보다 먼저 나온다.

def test_t7_submit_exists_refusal_precedes_first_torch_usage_in_main():
    src = (SCRIPTS / "infer_two_head.py").read_text(encoding="utf-8")
    main_src = src[src.index("def main():"):]
    # main() 안에서만 찾는다 — 파일 앞쪽 헬퍼 함수(_bn_buffer_hash 등)에도 "import torch"가
    # 있지만 그건 호출 시점이 한참 뒤라 실행 순서와 무관하다. GPU 작업(≈torch) 전에 "덮어쓰면
    # 안 되는 zip"을 먼저 거부해야 비싼 작업을 시작하기 전에 값싸게 멈춘다 (self-review.md와
    # 같은 원칙: 제출 전 중단은 GPU 시간을 쓰기 전에 걸려야 의미가 있다).
    refusal_idx = main_src.index("submit_path.exists()")
    first_torch_idx = main_src.index("import torch")
    assert refusal_idx < first_torch_idx


def test_t7_submission_gate_precedes_reconstruct_and_zip():
    src = (SCRIPTS / "infer_two_head.py").read_text(encoding="utf-8")
    main_src = src[src.index("def main():"):]
    # 게이트 호출이 실제 zip 조립보다 소스상 먼저 나와야, "게이트를 통과하지 않고 zip이 만들어질
    # 수 있는 경로가 없다"를 소스만 읽고도 확인할 수 있다 (조립 함수가 한 곳뿐이라는 계약과 같은
    # 이유 — sem.py assemble_depth 주석 참고).
    gate_idx = main_src.index("submission_gate(")
    zip_idx = main_src.index("reconstruct_and_zip(")
    assert gate_idx < zip_idx


# ══════════════════════════════════════════════════════════════
# RNG/초기화 parity — arm B의 추가 head가 arm A와의 backbone 초기화 동일성을 깨지 않는지
# ══════════════════════════════════════════════════════════════

def test_train_two_head_source_uses_explicit_generator_seeded_from_args_seed():
    src = (SCRIPTS / "train_two_head.py").read_text(encoding="utf-8")
    # B1: 배치 순서 RNG가 전역이 아니라 명시적 torch.Generator여야, arm B가 추가 head(mask_head)
    # 생성으로 전역 RNG를 arm A보다 더 소비해도 두 arm의 배치 순서가 갈라지지 않는다.
    assert "generator=gen" in src
    assert "gen.manual_seed(args.seed)" in src


def test_make_arm_model_rng_and_init_parity():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    torch.manual_seed(42)
    single = train_two_head.make_arm_model("single")
    torch.manual_seed(42)
    two = train_two_head.make_arm_model("two_head")

    def flat(m):
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    assert torch.equal(flat(single.encoder), flat(two.encoder))
    single_decoder_body = nn.Sequential(*list(single.decoder.children())[:-1])
    assert torch.equal(flat(single_decoder_body), flat(two.backbone))
    single_out = list(single.decoder.children())[-1]  # arm A 출력층 — depth_head가 재사용해야 함
    assert torch.equal(flat(single_out), flat(two.depth_head))

    n_single = sum(p.numel() for p in single.parameters())
    n_two = sum(p.numel() for p in two.parameters())
    assert n_single == 8_468_736
    assert n_two == 12_011_136


def test_two_head_depth_path_is_functionally_arm_a_at_init():
    # arm B의 depth 경로는 arm A와 같은 출력 비선형(sigmoid)을 가져야 한다 — raw Linear +
    # 추론 clamp면 "head 분리" 외에 출력 비선형까지 바뀌어 단일 변수 비교가 깨진다.
    # 같은 시드·같은 초기화에서 s_pos_hat은 arm A 출력과 bit 단위로 같아야 한다.
    torch = pytest.importorskip("torch")

    torch.manual_seed(42)
    single = train_two_head.make_arm_model("single").eval()
    torch.manual_seed(42)
    two = train_two_head.make_arm_model("two_head").eval()

    x = torch.rand(3, 1, 72, 48)
    with torch.no_grad():
        s_a = single(x)
        _mask_logit, s_pos_hat = two(x)
    assert torch.equal(s_a, s_pos_hat)
    assert float(s_pos_hat.min()) >= 0.0 and float(s_pos_hat.max()) <= 1.0


def test_subset_random_sampler_explicit_generator_is_immune_to_global_rng_state():
    torch = pytest.importorskip("torch")
    from torch.utils.data import SubsetRandomSampler
    import torch.nn as nn

    idx = list(range(20))
    gen1 = torch.Generator().manual_seed(42)
    order1 = list(SubsetRandomSampler(idx, generator=gen1))

    torch.manual_seed(0)  # 전역 RNG를 건드린다
    _ = nn.Linear(10, 10)  # 파라미터 초기화가 전역 RNG를 소비한다 — arm B의 추가 head와 같은 효과
    torch.manual_seed(999)

    gen2 = torch.Generator().manual_seed(42)
    order2 = list(SubsetRandomSampler(idx, generator=gen2))

    # 명시적 generator를 쓰면 그 사이 전역 RNG를 얼마나 소비했는지와 무관하게 같은 순서가 나온다
    # — 이게 B1이 SubsetRandomSampler에 전역 RNG 대신 명시적 Generator를 넘기는 이유다.
    assert order1 == order2


# ══════════════════════════════════════════════════════════════
# GPU_LOCK — 두 스크립트 모두 B가 이미 resource_lock/validate_hparams/RuntimeError로 갱신했다
# (아래 테스트들은 "pending"이 아니라 현재 소스에 대해 실제로 수집·실행된다).
# ══════════════════════════════════════════════════════════════

def test_train_two_head_main_exits_2_and_never_touches_torch_when_gpu_lock_busy(monkeypatch,
                                                                                tmp_path):
    from contextlib import contextmanager

    from ai_co_scientist.locks import ResourceBusy

    calls = []

    @contextmanager
    def fake_resource_lock(name, timeout=0):
        calls.append(name)
        raise ResourceBusy("locked for test")
        yield  # pragma: no cover — contextmanager 형태를 유지하기 위한 자리, 도달하지 않는다

    monkeypatch.setattr(train_two_head, "resource_lock", fake_resource_lock)
    # git preflight는 락 전에 돈다 — 작업 트리 상태와 무관하게 락 경로까지 가도록 깨끗한 HEAD로 고정
    monkeypatch.setattr(train_two_head, "_git",
                        lambda *a: "abc123" if a[0] == "rev-parse" else "")
    monkeypatch.setattr(sys, "argv", ["x", "--arm", "single", "--out", str(tmp_path / "x.pt")])

    with pytest.raises(SystemExit) as exc_info:
        train_two_head.main()
    assert exc_info.value.code == 2
    assert calls == [two_head.GPU_LOCK]
    # _run()(그 안에서 `import torch`가 일어난다)은 `with resource_lock(...):`의 __enter__가
    # 끝난 뒤에야 불린다 — ResourceBusy가 __enter__에서 나면 _run은 절대 호출되지 않는다. 이
    # 환경엔 애초에 torch가 없으므로, _run이 잘못 호출됐다면 SystemExit(2)가 아니라
    # ModuleNotFoundError로 죽었을 것이다 — 깔끔한 SystemExit(2) 자체가 이미 증거다.
    assert "torch" not in sys.modules


def test_train_two_head_main_validates_hparams_before_lock_and_torch_import():
    src = (SCRIPTS / "train_two_head.py").read_text(encoding="utf-8")
    main_src = src[src.index("def main():"):]
    # P2-3: GPU를 잡기(resource_lock) 전에 하이퍼파라미터 드리프트부터 걸러야, 락을 쥔 채로
    # 실패하는 낭비가 없다. `import torch`는 _run() 안에만 있고 _run은 락 블록 안에서만 불리므로,
    # main() 자체의 텍스트에 "import torch"가 전혀 없다는 것 자체가 이미 이 순서를 보증한다.
    hparams_idx = main_src.index("validate_hparams(")
    lock_idx = main_src.index("resource_lock(")
    assert hparams_idx < lock_idx
    assert "import torch" not in main_src


def test_infer_two_head_resource_lock_wraps_adapt_bn_after_validate_arm_pair():
    src = (SCRIPTS / "infer_two_head.py").read_text(encoding="utf-8")
    main_src = src[src.index("def main():"):]
    # GPU 작업(adapt_bn이 대표) 전체가 resource_lock의 with 블록 **안**에서만 실행돼야 하고,
    # 그 락을 잡기 전에 CPU에서 먼저 arm parity를 검증해 실패를 값싸게 만든다(P2-5).
    validate_idx = main_src.index("validate_arm_pair(")
    lock_idx = main_src.index("resource_lock(")
    adapt_bn_idx = main_src.index("adapt_bn(")
    assert validate_idx < lock_idx < adapt_bn_idx


def test_infer_two_head_build_parser_never_touches_the_lock():
    src = (SCRIPTS / "infer_two_head.py").read_text(encoding="utf-8")
    build_parser_src = src[src.index("def build_parser("):src.index("def main():")]
    assert "resource_lock" not in build_parser_src


def test_infer_two_head_compute_mask_rate_raises_runtimeerror_not_assert():
    src = (SCRIPTS / "infer_two_head.py").read_text(encoding="utf-8")
    start = src.index("def compute_mask_rate(")
    end = src.index("def _output_pos_rate(")
    body = src[start:end]
    # assert는 `python -O`에서 통째로 사라진다 — H2(AdaBN 이후 BN 통계 불변) 위반을 실제로
    # 막으려면 최적화와 무관하게 항상 실행되는 raise RuntimeError여야 한다.
    assert "assert " not in body
    assert body.count("raise RuntimeError(") >= 2  # train 모드 재진입 + BN 버퍼 변경, 최소 2곳


def test_torch_generator_state_round_trips_and_cuda_load_requires_cpu_before_set_state():
    torch = pytest.importorskip("torch")
    import os
    import tempfile

    gen = torch.Generator()
    gen.manual_seed(123)
    _ = torch.randperm(100, generator=gen)  # 상태를 한 번 소비해 초기 시드 상태와 달라지게 한다
    state = gen.get_state()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gen.pt")
        torch.save({"sampler_state": state}, path)

        loaded_cpu = torch.load(path, map_location="cpu", weights_only=False)
        gen2 = torch.Generator()
        gen2.set_state(loaded_cpu["sampler_state"].cpu())
        assert torch.equal(gen2.get_state(), state)
        # 상태가 실제로 복원됐다는 행동 증거: 이어지는 호출이 원본 generator의 다음 호출과 같다
        assert torch.equal(torch.randperm(100, generator=gen2),
                           torch.randperm(100, generator=gen))

        if torch.cuda.is_available():
            # P2-1: map_location="cuda"는 저장된 dict 안의 ByteTensor(sampler_state)까지 GPU로
            # 옮긴다. Generator.set_state는 CPU ByteTensor만 받으므로 .cpu() 없이는 죽는다 —
            # 이게 train_two_head.py의 `gen.set_state(ck["sampler_state"].cpu())`의 이유다.
            loaded_cuda = torch.load(path, map_location="cuda", weights_only=False)
            assert loaded_cuda["sampler_state"].device.type == "cuda"
            with pytest.raises(Exception):  # 정확한 예외형은 torch 버전에 따라 다를 수 있다
                torch.Generator().set_state(loaded_cuda["sampler_state"])  # .cpu() 없이 — 실패해야
            torch.Generator().set_state(loaded_cuda["sampler_state"].cpu())  # .cpu()를 붙이면 통과


# ── git preflight · 산출물 덮어쓰기 정책 ─────────────────────

def test_git_preflight_clean_passes():
    assert two_head.git_preflight_errors("abc123", "") == []


def test_git_preflight_flags_missing_head_and_dirty_tree():
    errs = two_head.git_preflight_errors("", "?? src/ai_co_scientist/new.py" + chr(10))
    assert len(errs) == 2
    assert "HEAD" in errs[0]
    assert "src/ai_co_scientist/new.py" in errs[1]


def test_train_artifact_paths():
    paths = two_head.train_artifact_paths("runtime/ckpt/EXP-0NN-two_head.pt")
    assert paths["ckpt"].name == "EXP-0NN-two_head.pt"
    assert paths["manifest"].name == "EXP-0NN-two_head.manifest.json"
    assert paths["resume"].name == "EXP-0NN-two_head.resume.pt"


def test_train_output_errors_fresh_out_passes(tmp_path):
    assert two_head.train_output_errors(tmp_path / "a.pt", resume=False) == []
    assert two_head.train_output_errors(tmp_path / "a.pt", resume=True) == []


@pytest.mark.parametrize("existing", ["a.pt", "a.manifest.json"])
@pytest.mark.parametrize("resume", [False, True])
def test_train_output_errors_never_overwrites_finished_artifacts(tmp_path, existing, resume):
    (tmp_path / existing).write_bytes(b"x")
    errs = two_head.train_output_errors(tmp_path / "a.pt", resume=resume)
    assert len(errs) == 1 and existing in errs[0]


def test_train_output_errors_stale_resume_requires_flag(tmp_path):
    (tmp_path / "a.resume.pt").write_bytes(b"x")
    errs = two_head.train_output_errors(tmp_path / "a.pt", resume=False)
    assert len(errs) == 1 and "--resume" in errs[0]
    assert two_head.train_output_errors(tmp_path / "a.pt", resume=True) == []


def _patch_lock_recorder(monkeypatch):
    from contextlib import contextmanager

    calls = []

    @contextmanager
    def fake_resource_lock(name, timeout=0):
        calls.append(name)
        yield

    monkeypatch.setattr(train_two_head, "resource_lock", fake_resource_lock)
    return calls


def test_train_two_head_main_refuses_dirty_tree_before_lock(monkeypatch, tmp_path, capsys):
    calls = _patch_lock_recorder(monkeypatch)
    monkeypatch.setattr(train_two_head, "_git",
                        lambda *a: "abc123" if a[0] == "rev-parse" else "?? scripts/x.py")
    monkeypatch.setattr(sys, "argv", ["x", "--arm", "two_head", "--out", str(tmp_path / "b.pt")])
    with pytest.raises(SystemExit) as exc_info:
        train_two_head.main()
    assert exc_info.value.code == 2
    assert calls == []
    assert "scripts/x.py" in capsys.readouterr().out


def test_train_two_head_main_refuses_existing_ckpt_before_lock(monkeypatch, tmp_path, capsys):
    calls = _patch_lock_recorder(monkeypatch)
    monkeypatch.setattr(train_two_head, "_git",
                        lambda *a: "abc123" if a[0] == "rev-parse" else "")
    out = tmp_path / "b.pt"
    out.write_bytes(b"done")
    monkeypatch.setattr(sys, "argv", ["x", "--arm", "two_head", "--out", str(out), "--resume"])
    with pytest.raises(SystemExit) as exc_info:
        train_two_head.main()
    assert exc_info.value.code == 2
    assert calls == []
    assert out.read_bytes() == b"done"
    assert "덮어쓰지 않는다" in capsys.readouterr().out


def test_train_two_head_git_preflight_includes_untracked_files():
    src = (SCRIPTS / "train_two_head.py").read_text(encoding="utf-8")
    assert "--untracked-files=all" in src
    assert "--untracked-files=no" not in src


def test_train_two_head_writes_are_atomic_and_resume_keeps_epoch_log():
    src = (SCRIPTS / "train_two_head.py").read_text(encoding="utf-8")
    run_src = src[src.index("def _run("):src.index("def main():")]
    assert "torch.save(" not in run_src  # 모든 저장은 _atomic_torch_save 경유
    assert ".write_text(" not in run_src
    assert '"epoch_metrics": epoch_metrics' in run_src
    assert 'ck["epoch_metrics"]' in run_src


def test_infer_two_head_refuses_existing_dump_structure_before_torch(monkeypatch, tmp_path,
                                                                     capsys):
    dump = tmp_path / "s.npy"
    dump.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", "a.pt", "--peer-ckpt", "b.pt",
                                      "--submit", str(tmp_path / "z.zip"),
                                      "--dump-structure", str(dump)])
    with pytest.raises(SystemExit) as exc_info:
        infer_two_head.main()
    assert exc_info.value.code == 2
    assert "dump-structure" in capsys.readouterr().out
    assert "torch" not in sys.modules


def test_git_preflight_flags_failed_git_status():
    errs = two_head.git_preflight_errors("abc123", None)
    assert len(errs) == 1 and "git status" in errs[0]
    assert two_head.git_preflight_errors(None, "") != []


def test_train_two_head_git_returns_none_on_nonzero_exit():
    assert train_two_head._git("rev-parse", "--verify", "no-such-ref-h8-xyz") is None


def _resume_identity(**over):
    base = {"arm": "two_head", "epochs": 15, "lr": 1e-3, "seed": 42, "split_seed": 0,
            "val_frac": 0.2, "batch_size": 128, "git_commit": "abc",
            "data_fingerprint": {"val_idx_sha256": "v", "depth_sample_sha256": "d"}}
    return {**base, **over}


def test_resume_identity_matching_passes():
    assert two_head.resume_identity_errors(_resume_identity(), _resume_identity()) == []


@pytest.mark.parametrize("key,bad", [
    ("git_commit", "def"), ("arm", "single"), ("lr", 2e-3),
    ("data_fingerprint", {"val_idx_sha256": "v2", "depth_sample_sha256": "d"}),
])
def test_resume_identity_rejects_drift(key, bad):
    errs = two_head.resume_identity_errors(_resume_identity(**{key: bad}), _resume_identity())
    assert len(errs) == 1 and key in errs[0]


def test_resume_identity_rejects_legacy_resume_file_without_commit():
    saved = _resume_identity()
    del saved["git_commit"]
    assert two_head.resume_identity_errors(saved, _resume_identity()) != []


def test_train_two_head_resume_file_records_identity():
    src = (SCRIPTS / "train_two_head.py").read_text(encoding="utf-8")
    run_src = src[src.index("def _run("):src.index("def main():")]
    assert '"git_commit": start_commit, "data_fingerprint": fingerprint' in run_src
    assert "resume_identity_errors(" in run_src
    # fingerprint는 재개 검증보다 먼저 계산돼야 한다
    assert run_src.index("fingerprint = {") < run_src.index("resume_identity_errors(")


@pytest.mark.parametrize("n,ok", [(129, False), (257, False), (128, True), (130, True)])
def test_batch_errors_flags_size_one_tail(n, ok):
    assert (two_head.batch_errors(n, 128) == []) is ok


def test_infer_two_head_dump_structure_checks_npy_suffixed_path(monkeypatch, tmp_path, capsys):
    (tmp_path / "s.npy").write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", "a.pt", "--peer-ckpt", "b.pt",
                                      "--submit", str(tmp_path / "z.zip"),
                                      "--dump-structure", str(tmp_path / "s")])
    with pytest.raises(SystemExit) as exc_info:
        infer_two_head.main()
    assert exc_info.value.code == 2
    assert "s.npy" in capsys.readouterr().out


def test_infer_two_head_dump_after_gate_and_hashing_inside_lock():
    src = (SCRIPTS / "infer_two_head.py").read_text(encoding="utf-8")
    main_src = src[src.index("def main():"):]
    assert main_src.index("submission_gate(") < main_src.index("np.save(")
    assert main_src.index("ckpt_sha256 = ") < main_src.index("except ResourceBusy")
