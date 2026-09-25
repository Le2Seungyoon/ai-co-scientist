"""dann — H7 DANN 특징 도메인 정렬의 순수 로직 (config·스케줄·분할·매니페스트).

**이 파일은 torch를 module scope에서 import하지 않는다.** 워크트리는 dev 그룹만 sync하므로
torch가 없다 — 순수 numpy 함수들은 항상 돈다. torch가 필요한 함수(`grad_reverse`,
`make_discriminator`, `dann_step_losses`)의 테스트만 별도 섹션에 모아 각 테스트 안에서
`pytest.importorskip("torch")`를 부른다 — 그래야 나머지가 스킵 없이 통과한다.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_co_scientist import sem
from ai_co_scientist.dann import (
    ARM_FIELDS, ARMS, CODE_PATHS, DOMAIN_SOURCES, PREREG_N_REAL, PREREG_N_SIM_TOTAL, PREREG_N_SIM_TRAIN,
    SIM_SPLIT_SEED, SIM_SPLIT_VAL_FRAC, STRUCTURE_SOURCES, DannConfig, array_digest,
    assert_clean_code, assert_disjoint, assert_domain_sources, build_manifest, check_arm_parity,
    check_manifest_files, check_output_policy, check_counts, code_fingerprint, epoch_rng,
    file_fingerprint, ganin_lambda, lambda_schedule, output_paths, probe_batches, probe_order,
    progress,
    publish_once, read_manifest, real_batches, real_domain_split, replace_atomic, roc_auc,
    run_contract, sim_batches, sim_probe_indices, sim_structure_split, step_plan, validate_arm,
    write_json_exclusive, write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# build_manifest 테스트에서 공유하는 유효한 source 지문 4장(구조 3 + domain 1).
SOURCES = [
    {"name": "sim_sem.npy", "size_bytes": 10, "sha256": "a" * 64},
    {"name": "sim_depth.npy", "size_bytes": 11, "sha256": "b" * 64},
    {"name": "sim_case.npy", "size_bytes": 12, "sha256": "c" * 64},
    {"name": "real_sem.npy", "size_bytes": 13, "sha256": "d" * 64},
]

# build_manifest 테스트가 parity_key 안에 들어가는 두 dict를 공유한다(AMENDMENT 1 §4).
OPT_SIG = {"param_groups": 1, "params": [["encoder.weight", [128, 4]]], "lr": 1e-3,
          "weight_decay": 0.0, "betas": [0.9, 0.999]}
ENVIRONMENT = {"torch_version": "2.4.0", "cuda_version": "12.4", "device": "cpu",
              "git_commit": "deadbeef"}


# ── DannConfig / ARMS ──────────────────────────────────────

def test_arms_constants_are_the_only_lambda_knob():
    assert ARMS == {"A": 0.0, "B": 1.0}


def test_structure_sources_excludes_real():
    assert STRUCTURE_SOURCES == ("sim_sem.npy", "sim_depth.npy", "sim_case.npy")


def test_arm_fields_are_exactly_the_manifest_keys_allowed_to_differ():
    assert ARM_FIELDS == frozenset({"arm", "lambda_max", "lambda_digest", "report_id",
                                   "outputs", "source"})


def test_for_arm_sets_lambda_max_from_arms():
    cfg_a = DannConfig.for_arm("A")
    cfg_b = DannConfig.for_arm("B")
    assert cfg_a.arm == "A" and cfg_a.lambda_max == 0.0
    assert cfg_b.arm == "B" and cfg_b.lambda_max == 1.0


def test_for_arm_rejects_unknown_arm():
    with pytest.raises(ValueError):
        DannConfig.for_arm("C")


def test_for_arm_rejects_lambda_max_override():
    with pytest.raises(ValueError):
        DannConfig.for_arm("A", lambda_max=0.5)


def test_for_arm_a_vs_b_differ_only_in_arm_and_lambda_max():
    """arm A/B의 차이는 lambda_max뿐이어야 한다 — 사전등록 조건(H7 pre-report)."""
    da = DannConfig.for_arm("A").to_dict()
    db = DannConfig.for_arm("B").to_dict()
    da.pop("arm"), da.pop("lambda_max")
    db.pop("arm"), db.pop("lambda_max")
    assert da == db


def test_dann_config_amp_defaults_off_and_is_overridable():
    assert DannConfig.for_arm("A").amp is False
    assert DannConfig.for_arm("A", amp=True).amp is True


# ── ganin_lambda ────────────────────────────────────────────

def test_ganin_lambda_zero_at_progress_zero():
    assert ganin_lambda(0.0, lambda_max=1.0, gamma=10.0) == 0.0


def test_ganin_lambda_zero_when_lambda_max_zero():
    for p in (0.0, 0.3, 0.5, 1.0):
        assert ganin_lambda(p, lambda_max=0.0, gamma=10.0) == 0.0


def test_ganin_lambda_known_value_at_half_progress():
    # 2/(1+exp(-gamma*p)) - 1 == tanh(gamma*p/2) — 항등식으로 독립 계산한 기대값과 비교한다.
    expected = math.tanh(10.0 * 0.5 / 2)
    assert ganin_lambda(0.5, lambda_max=1.0, gamma=10.0) == pytest.approx(expected, abs=1e-12)


def test_ganin_lambda_known_value_at_full_progress_scaled():
    expected = 2.0 * math.tanh(10.0 * 1.0 / 2)
    assert ganin_lambda(1.0, lambda_max=2.0, gamma=10.0) == pytest.approx(expected, abs=1e-12)


def test_ganin_lambda_clips_progress_to_unit_interval():
    assert ganin_lambda(-0.3, lambda_max=5.0, gamma=10.0) == 0.0
    assert ganin_lambda(1.7, lambda_max=1.0, gamma=10.0) == pytest.approx(
        ganin_lambda(1.0, lambda_max=1.0, gamma=10.0), abs=1e-12)


# ── real_domain_split ───────────────────────────────────────

def _synthetic_sites():
    """20 사이트 × 이미지 3장, 4클래스에 고르게 분포 — sem.site_split과 직접 비교하기 위함."""
    site = np.repeat(np.arange(20), 3)
    class_of_site = np.arange(20) % 4
    y = class_of_site[site]
    return site, y


def test_real_domain_split_equals_sem_site_split():
    site, y = _synthetic_sites()
    got = real_domain_split(site, y, probe_frac=0.2, seed=7)
    expected = sem.site_split(site, y, 0.2, 7)
    np.testing.assert_array_equal(got, expected)


def test_real_domain_split_is_site_level():
    site, y = _synthetic_sites()
    mask = real_domain_split(site, y, probe_frac=0.2, seed=7)
    for s in range(20):
        vals = set(mask[site == s].tolist())
        assert len(vals) == 1, f"site {s}가 probe/train 양쪽에 걸쳐 있다: {vals}"


def test_real_domain_split_deterministic_by_seed():
    site, y = _synthetic_sites()
    a = real_domain_split(site, y, probe_frac=0.2, seed=7)
    b = real_domain_split(site, y, probe_frac=0.2, seed=7)
    np.testing.assert_array_equal(a, b)


# ── sim_structure_split ─────────────────────────────────────

def test_sim_structure_split_is_train_complement_of_map_level_split():
    """138,648장 = EXP-005 분할(sem.map_level_split, val_frac=0.2, seed=42)의 train 쪽
    (AMENDMENT 1 §1) — sim_structure_split은 그 분할을 다시 구현하지 않고 위임한다."""
    case = np.repeat(np.arange(20) % 4, 2)  # 20개 depth-map, 4 Case에 고르게 분포, 쌍 유지
    train_mask = sim_structure_split(case)
    expected = ~sem.map_level_split(case, SIM_SPLIT_VAL_FRAC, SIM_SPLIT_SEED)
    np.testing.assert_array_equal(train_mask, expected)


def test_sim_structure_split_keeps_depth_map_pairs_together():
    case = np.repeat(np.arange(20) % 4, 2)
    train_mask = sim_structure_split(case)
    assert np.array_equal(train_mask[::2], train_mask[1::2])


# ── epoch_rng ───────────────────────────────────────────────

def test_epoch_rng_deterministic_for_same_key():
    a = epoch_rng(1, "sim", 0).random()
    b = epoch_rng(1, "sim", 0).random()
    assert a == b


def test_epoch_rng_streams_are_independent():
    sim_draw = epoch_rng(1, "sim", 0).random()
    real_draw = epoch_rng(1, "real", 0).random()
    probe_draw = epoch_rng(1, "probe", 0).random()
    assert len({sim_draw, real_draw, probe_draw}) == 3


def test_epoch_rng_changes_with_epoch():
    e0 = epoch_rng(1, "sim", 0).random()
    e1 = epoch_rng(1, "sim", 1).random()
    assert e0 != e1


def test_epoch_rng_raises_on_unknown_stream():
    with pytest.raises(ValueError):
        epoch_rng(1, "bogus", 0)


# ── sim_batches / real_batches / step_plan ───────────────────

def test_sim_batches_shape_permutation_drop_last():
    sim_idx = np.arange(10)
    out = sim_batches(sim_idx, 3, np.random.default_rng(123))
    expected = np.random.default_rng(123).permutation(sim_idx)[:9].reshape(3, 3)
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out, expected)
    assert len(np.unique(out)) == 9  # 10장 중 1장은 drop_last로 버려진다


def test_real_batches_covers_every_index_before_repeats():
    real_idx = np.arange(5)
    out = real_batches(real_idx, batch_size=3, n_steps=3, rng=np.random.default_rng(11))
    assert out.shape == (3, 3)
    flat = out.reshape(-1)
    first_cycle = flat[:5]
    assert sorted(first_cycle.tolist()) == [0, 1, 2, 3, 4]
    second_cycle_prefix = flat[5:9]
    assert len(set(second_cycle_prefix.tolist())) == 4  # 새 순열의 앞부분 — 그 안에서는 중복 없음
    assert set(second_cycle_prefix.tolist()).issubset(set(real_idx.tolist()))


def test_real_batches_deterministic_for_same_rng_seed():
    real_idx = np.arange(5)
    a = real_batches(real_idx, 3, 3, np.random.default_rng(11))
    b = real_batches(real_idx, 3, 3, np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)


def test_step_plan_identical_for_arm_a_and_b():
    """step_plan은 lambda_max/arm과 무관해야 한다 — H7 사전등록 조건."""
    n_sim_idx = np.arange(20)
    real_idx = np.arange(15)
    cfg_a = DannConfig.for_arm("A", batch_size=4, seed=99)
    cfg_b = DannConfig.for_arm("B", batch_size=4, seed=99)
    sb_a, rb_a = step_plan(n_sim_idx, real_idx, cfg_a, epoch=0)
    sb_b, rb_b = step_plan(n_sim_idx, real_idx, cfg_b, epoch=0)
    np.testing.assert_array_equal(sb_a, sb_b)
    np.testing.assert_array_equal(rb_a, rb_b)


def test_step_plan_changes_with_epoch():
    n_sim_idx = np.arange(20)
    real_idx = np.arange(15)
    cfg = DannConfig.for_arm("A", batch_size=4, seed=99)
    sb0, _ = step_plan(n_sim_idx, real_idx, cfg, epoch=0)
    sb1, _ = step_plan(n_sim_idx, real_idx, cfg, epoch=1)
    assert not np.array_equal(sb0, sb1)


# ── sim_probe_indices ───────────────────────────────────────
# AMENDMENT 2: 그룹 대칭 버전 — real_probe_groups(GROUPS 인덱스 0..3)별로 holdout에서 그만큼
# (짝수로 내림한) 쌍을 뽑는다. 그룹 g ↔ sim Case g+1.

def _group_pool(n_pairs_per_group: dict) -> tuple:
    """4그룹 전부 holdout인 풀 — `{g: 쌍 개수}`에서 `(holdout_mask, case)` 생성."""
    case_pairs = []
    for g in range(4):
        case_pairs.extend([g + 1] * n_pairs_per_group.get(g, 0))
    case = np.repeat(np.array(case_pairs, dtype=np.int64), 2)
    mask = np.ones(len(case), dtype=bool)
    return mask, case


def test_sim_probe_indices_matches_floor_even_counts_per_group():
    mask, case = _group_pool({0: 10, 1: 10, 2: 10, 3: 10})  # 그룹당 10쌍 가용
    # 요청 개수 -> floor(n/2)쌍: g0=7->3, g1=4->2, g2=9->4, g3=1->0
    real_probe_groups = np.array([0] * 7 + [1] * 4 + [2] * 9 + [3] * 1)
    res = sim_probe_indices(mask, case, real_probe_groups, seed=5)

    assert list(res) == sorted(res.tolist())
    for i in res.tolist():
        partner = i + 1 if i % 2 == 0 else i - 1
        assert partner in res.tolist()  # 쌍이 항상 함께 뽑힌다

    expected_pairs = {0: 3, 1: 2, 2: 4, 3: 0}
    for g, n_pairs in expected_pairs.items():
        n_imgs = int((case[res] == g + 1).sum())
        assert n_imgs == 2 * n_pairs, g
    assert len(res) == 2 * sum(expected_pairs.values())


def test_sim_probe_indices_deterministic_by_seed():
    mask, case = _group_pool({0: 10, 1: 10, 2: 10, 3: 10})
    real_probe_groups = np.array([0] * 7 + [1] * 4)
    a = sim_probe_indices(mask, case, real_probe_groups, seed=3)
    b = sim_probe_indices(mask, case, real_probe_groups, seed=3)
    np.testing.assert_array_equal(a, b)


def test_sim_probe_indices_raises_when_a_groups_pool_is_too_small():
    mask, case = _group_pool({0: 2, 1: 10, 2: 10, 3: 10})  # group0은 2쌍만 가용
    real_probe_groups = np.array([0] * 10)  # floor(10/2)=5쌍 요청 > 2쌍 가용
    with pytest.raises(ValueError):
        sim_probe_indices(mask, case, real_probe_groups, seed=0)


def test_sim_probe_indices_raises_on_mask_pair_misalignment():
    mask = np.array([True, False, True, True])  # 쌍(0,1) 내부가 서로 다르다
    case = np.array([1, 1, 1, 1])
    with pytest.raises(ValueError):
        sim_probe_indices(mask, case, np.array([0, 0]), seed=0)


def test_sim_probe_indices_raises_on_case_pair_misalignment():
    mask = np.array([True, True, True, True])
    case = np.array([1, 2, 1, 1])  # 쌍(0,1)이 서로 다른 Case
    with pytest.raises(ValueError):
        sim_probe_indices(mask, case, np.array([0, 0]), seed=0)


def test_sim_probe_disjoint_from_sim_train_via_structure_split():
    """probe sim ∩ sim-train == ∅ — sim_structure_split이 만든 진짜 train/holdout 분할 위에서
    확인한다(합성 데이터지만 실제 분할 함수를 그대로 태운다)."""
    map_case = np.concatenate([np.full(60, c) for c in (1, 2, 3, 4)])  # Case당 60개 depth-map
    case = np.repeat(map_case, 2)  # 이미지 단위, 240*2=480장
    train_mask = sim_structure_split(case)
    holdout_mask = ~train_mask

    real_probe_groups = np.array([0] * 5 + [1] * 8 + [2] * 3)  # g3은 0장(빈 그룹도 허용)
    sim_probe = sim_probe_indices(holdout_mask, case, real_probe_groups, seed=1)

    sim_train_idx = np.where(train_mask)[0]
    assert_disjoint(sim_probe, sim_train_idx, "sim probe/train")  # 안 겹치면 raise하지 않는다
    assert set(sim_probe.tolist()).issubset(set(np.where(holdout_mask)[0].tolist()))


def test_real_probe_disjoint_from_real_domain_via_site_split():
    """real-probe ∩ real-domain == ∅ — real_domain_split의 마스크와 그 보수는 정의상 배타적
    이지만, assert_disjoint를 실제 분할 출력 위에서 e2e로 exercise한다."""
    site, y = _synthetic_sites()
    probe_mask = real_domain_split(site, y, probe_frac=0.2, seed=7)
    real_probe_idx = np.where(probe_mask)[0]
    real_domain_idx = np.where(~probe_mask)[0]
    assert_disjoint(real_probe_idx, real_domain_idx, "real probe/domain")


# ── validate_arm ────────────────────────────────────────────

def test_validate_arm_passes_for_correct_pairs():
    assert validate_arm("A", 0.0) is None
    assert validate_arm("B", 1.0) is None


def test_validate_arm_rejects_unknown_arm():
    with pytest.raises(ValueError):
        validate_arm("C", 0.0)


def test_validate_arm_rejects_mismatched_lambda_max():
    with pytest.raises(ValueError):
        validate_arm("A", 1.0)
    with pytest.raises(ValueError):
        validate_arm("B", 0.0)


# ── output_paths / check_output_policy ───────────────────────

def test_output_paths_derives_all_five_from_ckpt_path():
    paths = output_paths("runtime/ckpt/EXP-099-dann-A.pt")
    assert paths["ckpt"] == Path("runtime/ckpt/EXP-099-dann-A.pt")
    assert paths["disc_ckpt"] == Path("runtime/ckpt/EXP-099-dann-A.disc.pt")
    assert paths["resume_ckpt"] == Path("runtime/ckpt/EXP-099-dann-A.resume.pt")
    assert paths["manifest"] == Path("runtime/ckpt/EXP-099-dann-A.manifest.json")
    assert paths["result_json"] == Path("runtime/ckpt/EXP-099-dann-A.result.json")
    assert all(isinstance(v, Path) for v in paths.values())


def test_output_paths_rejects_non_pt_suffix():
    with pytest.raises(ValueError):
        output_paths("runtime/ckpt/EXP-099-dann-A.pth")


def test_check_output_policy_fresh_run_passes_when_nothing_exists(tmp_path):
    paths = output_paths(tmp_path / "m.pt")
    assert check_output_policy(paths, resume=False) is None


def test_check_output_policy_fresh_run_raises_if_any_output_exists(tmp_path):
    paths = output_paths(tmp_path / "m.pt")
    paths["manifest"].write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        check_output_policy(paths, resume=False)


def test_check_output_policy_resume_requires_manifest(tmp_path):
    paths = output_paths(tmp_path / "m.pt")
    with pytest.raises(FileNotFoundError):
        check_output_policy(paths, resume=True)


def test_check_output_policy_resume_rejects_a_finished_run(tmp_path):
    paths = output_paths(tmp_path / "m.pt")
    paths["manifest"].write_text("{}", encoding="utf-8")
    paths["result_json"].write_bytes(b"x")  # 완료 표식
    with pytest.raises(FileExistsError):
        check_output_policy(paths, resume=True)


def test_check_output_policy_resume_allows_a_run_that_died_while_publishing(tmp_path):
    """ckpt는 게시됐지만 result_json 전에 죽은 실행 — 손으로 파일을 지우지 않아도 --resume이
    남은 게시를 마칠 수 있어야 한다(새 실행은 여전히 거부)."""
    paths = output_paths(tmp_path / "m.pt")
    paths["manifest"].write_text("{}", encoding="utf-8")
    paths["resume_ckpt"].write_bytes(b"x")
    paths["ckpt"].write_bytes(b"x")
    assert check_output_policy(paths, resume=True) is None
    with pytest.raises(FileExistsError):
        check_output_policy(paths, resume=False)


def test_check_output_policy_resume_passes_with_manifest_and_resume_ckpt_only(tmp_path):
    paths = output_paths(tmp_path / "m.pt")
    paths["manifest"].write_text("{}", encoding="utf-8")
    paths["resume_ckpt"].write_bytes(b"x")
    assert check_output_policy(paths, resume=True) is None


# ── write_json_exclusive ──────────────────────────────────────

def test_write_json_exclusive_writes_canonical_json_and_refuses_overwrite(tmp_path):
    path = write_json_exclusive(tmp_path / "o.json", {"b": 2, "a": 1})
    assert path.read_text(encoding="utf-8") == json.dumps(
        {"a": 1, "b": 2}, sort_keys=True, ensure_ascii=False, indent=2)
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"a": 99})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}  # 원본 그대로


# ── progress / lambda_schedule ────────────────────────────────

def test_progress_matches_step_over_total():
    assert progress(0, 4) == 0.0
    assert progress(2, 4) == 0.5
    assert progress(3, 4) == 0.75


def test_progress_raises_outside_valid_range():
    with pytest.raises(ValueError):
        progress(-1, 4)
    with pytest.raises(ValueError):
        progress(4, 4)


def test_lambda_schedule_all_zero_for_arm_a():
    sched = lambda_schedule(5, lambda_max=0.0)
    assert sched.dtype == np.float64
    np.testing.assert_array_equal(sched, np.zeros(5))


def test_lambda_schedule_arm_b_starts_zero_increases_and_stays_below_max():
    sched = lambda_schedule(5, lambda_max=1.0, gamma=10.0)
    assert sched[0] == 0.0
    assert np.all(np.diff(sched) > 0)
    assert sched[-1] < 1.0


def test_lambda_schedule_pins_exact_values_for_tiny_t():
    # tanh 항등식(2/(1+exp(-x))-1 == tanh(x/2))으로 독립 계산한 값과 비교한다.
    sched = lambda_schedule(3, lambda_max=2.0, gamma=10.0)
    expected = np.array([2.0 * math.tanh(10.0 * (i / 3) / 2) for i in range(3)])
    np.testing.assert_allclose(sched, expected, atol=1e-12)


# ── probe_batches ───────────────────────────────────────────

def test_probe_batches_covers_all_indices_no_drop_and_merges_short_tail():
    idx = np.arange(7)
    batches = probe_batches(idx, batch_size=3)
    assert [len(b) for b in batches] == [3, 4]  # 마지막 1장이 앞 배치로 합쳐졌다(크기 1 배치 금지)
    np.testing.assert_array_equal(np.concatenate(batches), idx)


def test_probe_batches_exact_multiple_has_no_merge():
    idx = np.arange(6)
    batches = probe_batches(idx, batch_size=3)
    assert [len(b) for b in batches] == [3, 3]


def test_probe_batches_single_batch_when_shorter_than_batch_size():
    idx = np.arange(2)
    batches = probe_batches(idx, batch_size=5)
    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0], idx)


def test_probe_batches_raises_below_two_indices():
    with pytest.raises(ValueError):
        probe_batches(np.arange(1), batch_size=3)
    with pytest.raises(ValueError):
        probe_batches(np.arange(0), batch_size=3)


# ── assert_disjoint ─────────────────────────────────────────

def test_assert_disjoint_passes_when_no_overlap():
    assert assert_disjoint(np.array([1, 2, 3]), np.array([4, 5]), "x") is None


def test_assert_disjoint_raises_naming_what_on_overlap():
    with pytest.raises(ValueError, match="sim probe/train"):
        assert_disjoint(np.array([1, 2, 3]), np.array([3, 4]), "sim probe/train")


# ── roc_auc ─────────────────────────────────────────────────

def test_roc_auc_known_value_no_ties():
    # 손으로 계산: neg={0.1,0.4}, pos={0.35,0.8} -> rank(pos)=2,4 -> AUC=(6-3)/4=0.75
    scores = [0.1, 0.4, 0.35, 0.8]
    labels = [0, 0, 1, 1]
    assert roc_auc(scores, labels) == pytest.approx(0.75, abs=1e-12)


def test_roc_auc_all_tied_scores_gives_half():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert roc_auc(scores, labels) == pytest.approx(0.5, abs=1e-12)


def test_roc_auc_known_value_with_partial_tie():
    # neg={1,2}, pos={2,3} — 점수 2가 한쪽 neg·한쪽 pos에 걸쳐 있음 (동순위 평균 처리).
    # 손계산: (1 + 0.5 + 1 + 1) / 4 = 0.875
    scores = [1, 2, 2, 3]
    labels = [0, 0, 1, 1]
    assert roc_auc(scores, labels) == pytest.approx(0.875, abs=1e-12)


def test_roc_auc_raises_on_single_class():
    with pytest.raises(ValueError):
        roc_auc([0.1, 0.2, 0.3], [0, 0, 0])


# ── assert_domain_sources / check_counts ─────────────────────

def test_assert_domain_sources_accepts_allowed_names():
    assert assert_domain_sources(list(DOMAIN_SOURCES)) is None


def test_assert_domain_sources_rejects_test_sem():
    with pytest.raises(ValueError):
        assert_domain_sources(["sim_sem.npy", "test_sem.npy"])


def test_assert_domain_sources_rejects_non_domain_structure_source():
    with pytest.raises(ValueError):
        assert_domain_sources(["sim_depth.npy"])  # 구조 전용 소스는 domain 판별에 못 들어간다


def test_check_counts_passes_on_prereg_counts():
    assert check_counts(PREREG_N_SIM_TOTAL, PREREG_N_SIM_TRAIN, PREREG_N_REAL) is None


def test_check_counts_raises_on_mismatch():
    with pytest.raises(ValueError):
        check_counts(PREREG_N_SIM_TOTAL - 1, PREREG_N_SIM_TRAIN, PREREG_N_REAL)
    with pytest.raises(ValueError):
        check_counts(PREREG_N_SIM_TOTAL, PREREG_N_SIM_TRAIN - 1, PREREG_N_REAL)
    with pytest.raises(ValueError):
        check_counts(PREREG_N_SIM_TOTAL, PREREG_N_SIM_TRAIN, PREREG_N_REAL - 1)


# ── file_fingerprint / array_digest ───────────────────────────

def test_file_fingerprint_matches_sha256_across_a_1mib_chunk_boundary(tmp_path):
    import os

    data = os.urandom(1_500_000)  # 1 MiB 스트리밍 청크 경계를 실제로 넘긴다
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    fp = file_fingerprint(p)
    assert fp == {"name": "blob.bin", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_array_digest_differs_by_dtype_even_with_same_values():
    a32 = np.array([1, 2, 3], dtype=np.int32)
    a64 = np.array([1, 2, 3], dtype=np.int64)
    assert array_digest(a32) != array_digest(a64)


def test_array_digest_matches_independent_recomputation():
    a = np.array([1, 2, 3], dtype=np.int32)
    expected = hashlib.sha256()
    expected.update(str(a.dtype).encode("utf-8"))
    expected.update(str(a.shape).encode("utf-8"))
    expected.update(a.tobytes())
    assert array_digest(a) == expected.hexdigest()


def test_array_digest_deterministic_and_order_sensitive():
    a = np.arange(5)
    b = np.arange(5, 10)
    assert array_digest(a, b) == array_digest(a, b)
    assert array_digest(a, b) != array_digest(b, a)


# ── build_manifest / check_arm_parity ─────────────────────────

def _manifest(arm, report_id, **kw):
    cfg = DannConfig.for_arm(arm)
    defaults = dict(
        sources=SOURCES, n_sim_train=10, n_real_domain=8, n_real_probe=2, n_sim_probe=4,
        split_digest="d1", plan_digest="p1", lambda_digest=f"lam-{arm}", steps_per_epoch=3,
        outputs={"ckpt": f"{report_id}.pt"},
        optimizer_signature=OPT_SIG, environment=ENVIRONMENT,
    )
    defaults.update(kw)
    return build_manifest(cfg, report_id=report_id, **defaults)


def test_build_manifest_has_required_keys():
    man = _manifest("A", "EXP-100")
    for key in ("hypothesis", "report_id", "arm", "lambda_max", "config", "x_domain",
                "y_source", "metric", "mechanism_metric", "checkpoint_selection", "counts",
                "sources", "split_digest", "plan_digest", "lambda_digest", "steps_per_epoch",
                "total_steps", "outputs", "optimizer_signature", "environment", "grl_schedule",
                "lr_schedule", "epoch_semantics", "rng", "probe", "overwrite_policy",
                "parity_key"):
        assert key in man, key
    assert man["hypothesis"] == "H7"
    assert man["total_steps"] == man["steps_per_epoch"] * man["config"]["epochs"]


def test_build_manifest_mechanism_metric_marks_not_for_selection():
    # AMENDMENT 1 §2: probe AUC는 checkpoint 선택에 절대 쓰이지 않는다 — 이 사실이 manifest에
    # 기계가 읽을 수 있는 형태로 박혀 있어야 한다.
    man = _manifest("A", "EXP-116")
    assert man["mechanism_metric"]["not_for_selection"] is True
    assert man["mechanism_metric"]["epoch"] == "final (all epochs logged)"


def test_build_manifest_rejects_test_source_name():
    with pytest.raises(ValueError):
        _manifest("A", "EXP-101", sources=[{"name": "test_sem.npy", "size_bytes": 1, "sha256": "x"}])


def test_build_manifest_parity_key_equal_across_arms():
    man_a = _manifest("A", "EXP-102")
    man_b = _manifest("B", "EXP-103")
    assert man_a["parity_key"] == man_b["parity_key"]


def test_build_manifest_parity_key_differs_when_split_digest_differs():
    man_a = _manifest("A", "EXP-104", split_digest="d1")
    man_b = _manifest("A", "EXP-105", split_digest="d2")
    assert man_a["parity_key"] != man_b["parity_key"]


def test_build_manifest_parity_key_differs_when_optimizer_signature_differs():
    # AMENDMENT 1 §4: optimizer_signature/environment는 parity_key 안에 들어간다.
    man_a = _manifest("A", "EXP-117", optimizer_signature=OPT_SIG)
    man_b = _manifest("A", "EXP-118", optimizer_signature={**OPT_SIG, "lr": 2e-3})
    assert man_a["parity_key"] != man_b["parity_key"]


def test_build_manifest_parity_key_differs_when_environment_differs():
    man_a = _manifest("A", "EXP-119", environment=ENVIRONMENT)
    man_b = _manifest("A", "EXP-120", environment={**ENVIRONMENT, "device": "cuda"})
    assert man_a["parity_key"] != man_b["parity_key"]


def test_check_arm_parity_passes_on_legitimate_a_vs_b():
    man_a = _manifest("A", "EXP-106")
    man_b = _manifest("B", "EXP-107")
    assert check_arm_parity(man_a, man_b) is None


def test_check_arm_parity_raises_on_same_arm():
    man_a1 = _manifest("A", "EXP-108")
    man_a2 = _manifest("A", "EXP-109")
    with pytest.raises(ValueError):
        check_arm_parity(man_a1, man_a2)


def test_check_arm_parity_raises_on_non_arm_difference():
    man_a = _manifest("A", "EXP-110", split_digest="d1")
    man_b = _manifest("B", "EXP-111", split_digest="d2")  # arm 외에 split_digest도 다르다
    with pytest.raises(ValueError):
        check_arm_parity(man_a, man_b)


# ── write_manifest / read_manifest ────────────────────────────

def test_write_manifest_refuses_to_overwrite_and_keeps_original_bytes(tmp_path):
    man = _manifest("A", "EXP-112")
    path = write_manifest(tmp_path / "m.json", man)
    original = path.read_bytes()

    other = _manifest("B", "EXP-113")
    with pytest.raises(FileExistsError):
        write_manifest(path, other)

    assert path.read_bytes() == original


def test_read_manifest_round_trips(tmp_path):
    man = _manifest("A", "EXP-114")
    path = write_manifest(tmp_path / "m.json", man)
    got = read_manifest(path)
    assert got["report_id"] == man["report_id"]
    assert got["parity_key"] == man["parity_key"]


def test_read_manifest_detects_tampering(tmp_path):
    man = _manifest("A", "EXP-115")
    path = write_manifest(tmp_path / "m.json", man)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["steps_per_epoch"] = data["steps_per_epoch"] + 1  # parity_key 밖 필드를 몰래 바꾼다
    path.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(path)


# ── module-level import hygiene ───────────────────────────────

def test_importing_dann_does_not_import_torch():
    """torch 없는 워크트리에서도 `import ai_co_scientist.dann`이 돌아야 한다 — 서브프로세스로
    확인해야 이 테스트 파일 자신이 이미 torch를 로드했을 가능성과 무관하게 검증된다."""
    r = subprocess.run(
        [sys.executable, "-c",
         "import ai_co_scientist.dann, sys; "
         "assert 'torch' not in sys.modules, 'torch가 모듈 최상위에서 import됐다'"],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ═══════════════════════════════════════════════════════════
# torch-lazy 함수 — torch가 없는 이 워크트리에서는 전부 스킵된다.
# ═══════════════════════════════════════════════════════════

def test_grad_reverse_forward_identity_backward_negates_and_scales():
    torch = pytest.importorskip("torch")
    from ai_co_scientist.dann import grad_reverse

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = grad_reverse(x, 2.0)
    assert torch.equal(y, x)  # forward는 identity

    y.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, -2.0))  # backward는 -lam 배


def test_make_discriminator_architecture_has_no_batchnorm_and_exact_shapes():
    pytest.importorskip("torch")
    import torch.nn as nn

    from ai_co_scientist.dann import make_discriminator

    disc = make_discriminator(feature_dim=128, hidden=256)
    assert not any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                   for m in disc.modules())

    layers = list(disc)
    assert isinstance(layers[0], nn.Linear) and layers[0].in_features == 128 \
        and layers[0].out_features == 256
    assert isinstance(layers[1], nn.ReLU)
    assert isinstance(layers[2], nn.Linear) and layers[2].in_features == 256 \
        and layers[2].out_features == 256
    assert isinstance(layers[3], nn.ReLU)
    assert isinstance(layers[4], nn.Linear) and layers[4].in_features == 256 \
        and layers[4].out_features == 1


class _FakeModel:
    """PlainMLP의 encoder/decoder/out 인터페이스만 흉내낸 최소 모형 — train_structure는 import하지
    않는다(스코프 밖)."""

    def __init__(self, torch):
        import torch.nn as nn

        self.encoder = nn.Linear(4, 128)
        self.decoder = nn.Linear(128, 8)
        self.out = nn.Linear(8, 4)


def test_dann_step_losses_lam_zero_zeroes_encoder_domain_grad_but_not_disc_grad():
    torch = pytest.importorskip("torch")

    from ai_co_scientist.dann import dann_step_losses, make_discriminator

    model = _FakeModel(torch)
    disc = make_discriminator(feature_dim=128, hidden=16)

    x_sim = torch.rand(3, 4)
    s_sim = torch.rand(3, 2, 2)
    x_real = torch.rand(3, 4)

    losses = dann_step_losses(model, disc, x_sim, s_sim, x_real, lam=0.0)
    assert set(losses) == {"l1", "domain", "total"}

    losses["domain"].backward()

    assert model.encoder.weight.grad is not None
    assert torch.equal(model.encoder.weight.grad, torch.zeros_like(model.encoder.weight.grad))

    disc_layers = [m for m in disc if hasattr(m, "weight")]
    assert disc_layers[0].weight.grad is not None
    assert bool((disc_layers[0].weight.grad != 0).any())


def test_bn_buffers_snapshot_and_restore_roundtrip():
    """probe forward(encoder BN train 모드)가 모델을 실제로 바꾸지 않는다는 조건(AMENDMENT 1 §5)
    의 기반 — 스냅샷 이후의 forward가 running stat을 움직이지만 restore가 정확히 되돌린다."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from ai_co_scientist.dann import bn_buffers_restore, bn_buffers_snapshot

    model = nn.Sequential(nn.BatchNorm1d(4))
    model.train()
    model(torch.randn(8, 4))  # 초기값에서 벗어나게 한 뒤에 스냅샷
    snap = bn_buffers_snapshot(model)
    assert set(snap) == {"0"}

    model(torch.randn(8, 4) * 10 + 5)  # 통계를 더 흔든다
    assert not torch.equal(model[0].running_mean, snap["0"]["running_mean"])

    bn_buffers_restore(model, snap)
    assert torch.equal(model[0].running_mean, snap["0"]["running_mean"])
    assert torch.equal(model[0].running_var, snap["0"]["running_var"])
    assert torch.equal(model[0].num_batches_tracked, snap["0"]["num_batches_tracked"])


# ── scripts/train_dann.py CLI 계약 (Engineer B) ─────────────────
# torch 없는 워크트리에서도 `train_dann` 모듈 import·`--help`·`--dry-plan`이 동작해야 한다 —
# 이 섹션은 실제 학습(run())을 절대 건드리지 않는다.

def _repo_root() -> Path:
    """일반적으로는 tests/의 parents[1]. 스크래치패드에서 직접 돌릴 때를 위한 fallback."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "scripts" / "train_dann.py").exists():
        root = Path.cwd()
    return root


def _load_train_dann():
    scripts_dir = str(_repo_root() / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import train_dann
    return train_dann


def _run_cli(cwd, *extra_args, arm="A", lambda_max=0.0, report_id="EXP-900"):
    """공통 서브프로세스 호출자. arm/lambda_max/report_id는 기본값(정합)을 두고 개별 테스트가
    덮어쓴다."""
    script = _repo_root() / "scripts" / "train_dann.py"
    return subprocess.run(
        [sys.executable, str(script), "--arm", arm, "--lambda-max", str(lambda_max),
         "--report-id", report_id, *extra_args],
        cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
    )


# ── import 위생 ─────────────────────────────────────────────

def test_cli_import_does_not_import_torch():
    """torch 없는 워크트리에서도 `import train_dann`이 돌아야 한다 — 서브프로세스로 확인해야
    이 테스트 파일 자신이 이미 torch를 로드했을 가능성과 무관하게 검증된다."""
    scripts_dir = str(_repo_root() / "scripts")
    code = (
        "import sys; sys.path.insert(0, %r); import train_dann; "
        "assert 'torch' not in sys.modules, 'torch가 모듈 최상위에서 import됐다'"
    ) % scripts_dir
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_repo_root()),
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr


def test_cli_help_exits_zero():
    """--help는 argparse가 처리하고 즉시 exit(0)해야 한다 — em dash 등 유니코드가 콘솔을
    죽이지 않는지도 같이 검증된다(ensure_utf8_console이 ArgumentParser 전에 불렸는지).

    캡처 쪽 encoding="utf-8"을 명시해야 한다 — 이게 없으면 자식 프로세스는
    ensure_utf8_console() 덕에 UTF-8로 잘 쓰지만, 부모(subprocess.run)의 text=True는 로케일
    기본값(cp949)으로 디코드를 시도해 em dash에서 리더 스레드가 UnicodeDecodeError로 죽는다
    (측정: PYTHONIOENCODING을 지우고 encoding= 없이 돌리면 스레드 예외가 난다). 이 테스트의
    본래 목적(PYTHONIOENCODING 없이도 스크립트 자신이 안 죽는지)은 env에서 그것만 지우고
    캡처 인코딩은 별개로 고정하는 것으로 만족된다.
    """
    script = _repo_root() / "scripts" / "train_dann.py"
    env = os.environ.copy()
    env.pop("PYTHONIOENCODING", None)  # cp949 콘솔을 흉내내는 게 목적이라 강제 UTF-8을 지운다
    proc = subprocess.run([sys.executable, str(script), "--help"], cwd=str(_repo_root()),
                          capture_output=True, text=True, encoding="utf-8", env=env)
    assert proc.returncode == 0, proc.stderr


# ── argparse 계약 ───────────────────────────────────────────

def test_cli_requires_arm_lambda_max_and_report_id(capsys):
    train_dann = _load_train_dann()
    with pytest.raises(SystemExit):
        train_dann.build_parser().parse_args(["--lambda-max", "0.0", "--report-id", "EXP-1"])
    with pytest.raises(SystemExit):
        train_dann.build_parser().parse_args(["--arm", "A", "--report-id", "EXP-1"])
    with pytest.raises(SystemExit):
        train_dann.build_parser().parse_args(["--arm", "A", "--lambda-max", "0.0"])
    capsys.readouterr()  # argparse가 stderr에 쓰는 usage 메시지를 비운다


def test_cli_arm_choices_exactly_a_b():
    train_dann = _load_train_dann()
    ap = train_dann.build_parser()
    arm_action = next(a for a in ap._actions if a.dest == "arm")
    assert list(arm_action.choices) == ["A", "B"]


def test_cli_rejects_hyperparameter_flags(capsys):
    """lambda/epochs/lr/batch-size/seed는 DannConfig.for_arm(사전등록)에서만 온다 — CLI로
    바꿀 수 있으면 두 arm의 유일한 차이가 lambda_max라는 사전등록 계약이 깨진다. `--lambda-max`
    자체는 예외다(validate_arm 확인용으로 새로 추가된 필수 플래그) — 여기서는 그 외의
    하이퍼파라미터 플래그가 전부 거부되는지만 본다."""
    train_dann = _load_train_dann()
    base = ["--arm", "A", "--lambda-max", "0.0", "--report-id", "EXP-2"]
    for flag, value in (("--epochs", "5"), ("--lr", "0.01"),
                       ("--batch-size", "64"), ("--seed", "1")):
        with pytest.raises(SystemExit):
            train_dann.build_parser().parse_args([*base, flag, value])
    capsys.readouterr()


def test_cli_default_out_derives_from_report_id_and_arm():
    train_dann = _load_train_dann()
    assert train_dann.default_out("EXP-099", "A") == "runtime/ckpt/EXP-099-dann-A.pt"
    assert train_dann.default_out("EXP-099", "B") == "runtime/ckpt/EXP-099-dann-B.pt"
    ns = train_dann.build_parser().parse_args(
        ["--arm", "A", "--lambda-max", "0.0", "--report-id", "EXP-099"])
    assert ns.out == ""  # 빈 문자열 기본값 — main()/run()이 default_out()으로 실제 경로를 채운다


# ── --lambda-max ↔ --arm 정합 (validate_arm) ───────────────────

def test_cli_lambda_max_mismatch_exits_nonzero(tmp_path):
    """--arm A는 lambda_max 0.0, --arm B는 1.0이어야 한다 — 뒤바뀐 값은 즉시 실패해야
    두 arm이 lambda_max 하나만 다르다는 사전등록 불변식이 CLI 레벨에서도 지켜진다."""
    proc_a = _run_cli(tmp_path, arm="A", lambda_max=1.0)  # A인데 B의 값
    assert proc_a.returncode != 0

    proc_b = _run_cli(tmp_path, arm="B", lambda_max=0.0)  # B인데 A의 값
    assert proc_b.returncode != 0


def test_cli_lambda_max_match_reaches_dry_plan(tmp_path):
    """정합하는 (arm, lambda_max) 쌍은 validate_arm을 통과해 --dry-plan까지 도달해야 한다."""
    proc = _run_cli(tmp_path, "--dry-plan", arm="A", lambda_max=0.0)
    assert proc.returncode == 0, proc.stderr
    proc = _run_cli(tmp_path, "--dry-plan", arm="B", lambda_max=1.0)
    assert proc.returncode == 0, proc.stderr


# ── --dry-plan ──────────────────────────────────────────────

def test_cli_dry_plan_arm_a_and_b_differ_only_in_arm_and_lambda(tmp_path):
    """arm A/B의 --dry-plan 출력은 config가 arm·lambda_max에서만 갈려야 한다 — 사전등록의
    핵심 불변식(두 arm은 lambda_max 하나만 다르다)을 CLI 레벨에서도 지킨다. 데이터도 파일도
    건드리면 안 되므로 cwd=tmp_path(빈 디렉터리, cache-dir/data-dir 없음)에서 돌린다."""
    results = {}
    for arm, lam in (("A", 0.0), ("B", 1.0)):
        proc = _run_cli(tmp_path, "--dry-plan", arm=arm, lambda_max=lam)
        assert proc.returncode == 0, proc.stderr
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        results[arm] = json.loads(lines[-1])

    assert results["A"]["arm"] == "A"
    assert results["B"]["arm"] == "B"
    cfg_a, cfg_b = results["A"]["config"], results["B"]["config"]
    diff_keys = {k for k in cfg_a if cfg_a[k] != cfg_b.get(k)}
    assert diff_keys == {"arm", "lambda_max"}, diff_keys

    created = list(tmp_path.rglob("*"))
    assert created == [], f"--dry-plan이 파일을 만들었다: {created}"


def test_cli_dry_plan_prints_the_five_output_paths(tmp_path):
    """output_paths()의 5개 키(ckpt/disc_ckpt/resume_ckpt/manifest/result_json)가 그대로
    --dry-plan JSON에 문자열로 실려야 한다 — executor가 사전에 산출물 경로를 볼 수 있어야 한다."""
    proc = _run_cli(tmp_path, "--dry-plan", arm="A", lambda_max=0.0)
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    out = json.loads(lines[-1])
    assert set(out["paths"]) == {"ckpt", "disc_ckpt", "resume_ckpt", "manifest", "result_json"}
    for v in out["paths"].values():
        assert isinstance(v, str)


def test_cli_dry_plan_refuses_when_out_already_exists_without_resume(tmp_path):
    """--dry-plan도 check_output_policy를 불러 산출물 5종 중 하나라도 이미 있으면(--resume
    없이는) 학습을 시작하기 전에 실패해야 한다 — GPU를 잡기 전에 사고를 잡는 자리다."""
    out = tmp_path / "runtime" / "ckpt" / "EXP-900-dann-A.pt"
    out.parent.mkdir(parents=True)
    out.touch()  # 산출물 5종 중 ckpt만 미리 존재

    proc = _run_cli(tmp_path, "--out", str(out), "--dry-plan", arm="A", lambda_max=0.0)
    assert proc.returncode != 0


# ── resource_lock("gpu-0") 배치 — 행동으로 ────────────────────

def _recording_run(monkeypatch, *, busy=False):
    """prepare/train/resource_lock을 기록기로 바꿔 `run()`의 순서를 본다 — torch 없이도 돈다."""
    train_dann = _load_train_dann()
    events = []

    class _Busy(Exception):
        pass

    class _Lock:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            if busy:
                raise _Busy(self.name)
            events.append(("lock", self.name))

        def __exit__(self, *exc):
            events.append(("unlock", self.name))
            return False

    monkeypatch.setattr(train_dann, "prepare", lambda args: events.append(("prepare",)) or {})
    monkeypatch.setattr(train_dann, "train",
                        lambda args, prep: events.append(("train",)) or {"ok": True})
    monkeypatch.setattr(train_dann, "resource_lock", _Lock)
    return train_dann, events, _Busy


def test_run_prepares_on_cpu_then_trains_only_inside_gpu_lock(monkeypatch, capsys):
    """데이터 해시·분할(prepare)은 락 밖에서 먼저, 학습(train)만 `gpu-0` 안에서 — 락을 쥔 채
    수십 GB를 해시하면 다른 GPU 작업이 이유 없이 막힌다."""
    train_dann, events, _ = _recording_run(monkeypatch)
    train_dann.run(object())
    assert events == [("prepare",), ("lock", "gpu-0"), ("train",), ("unlock", "gpu-0")]
    capsys.readouterr()


def test_run_does_not_train_when_gpu_is_busy(monkeypatch):
    train_dann, events, busy = _recording_run(monkeypatch, busy=True)
    with pytest.raises(busy):
        train_dann.run(object())
    assert ("train",) not in events


def test_train_builds_the_model_only_inside_train():
    """모델 초기화(make_model)는 `train()` 안에만 있다 — `prepare()`가 torch를 만지면 GPU 락
    밖에서 카드를 잡는 경로가 열린다. 행동 테스트(위 둘)는 순서를, 이것은 소재를 고정한다."""
    train_dann = _load_train_dann()
    source = Path(train_dann.__file__).read_text(encoding="utf-8")
    prepare_body = source.split("\ndef prepare(")[1].split("\ndef ")[0]
    train_body = source.split("\ndef train(")[1].split("\ndef run(")[0]
    assert "torch" not in prepare_body
    assert "make_model(" in train_body and "seed_everything(" in train_body


# ── run_contract / lambda_digest ─────────────────────────────

def test_run_contract_is_arm_independent_and_names_the_exact_schedule():
    a, b = run_contract(DannConfig.for_arm("A")), run_contract(DannConfig.for_arm("B"))
    assert a == b
    assert a["grl_schedule"]["formula"] == "lambda = lambda_max * (2 / (1 + exp(-gamma * p)) - 1)"
    assert a["grl_schedule"]["gamma"] == 10.0
    assert a["lr_schedule"]["t_max"] == 15
    assert a["rng"]["streams"] == {"sim": 0, "real": 1, "probe": 2}


def test_lambda_digest_may_differ_across_arms_but_contract_changes_break_parity():
    man_a, man_b = _manifest("A", "EXP-130"), _manifest("B", "EXP-131")
    assert man_a["lambda_digest"] != man_b["lambda_digest"]
    check_arm_parity(man_a, man_b)  # lambda 배열 차이는 허용된다

    tampered = json.loads(json.dumps(man_b))
    tampered["grl_schedule"]["gamma"] = 5.0
    with pytest.raises(ValueError, match="grl_schedule.gamma"):
        check_arm_parity(man_a, tampered)


# ── check_manifest_files / check-parity CLI ──────────────────

def test_parity_holds_between_file_loaded_and_in_memory_manifest_with_tuples(tmp_path):
    """`--parity-with`는 파일에서 읽은 arm A와 메모리의 arm B를 비교한다 — 스크립트가
    optimizer_signature를 tuple로 만들어도 JSON의 list와 같게 판정돼야 한다(리뷰 결함 재현)."""
    sig = {**OPT_SIG, "params": [("model.w", [128, 4]), ("disc.w", [1, 256])]}
    a = write_manifest(tmp_path / "a.json", _manifest("A", "EXP-141", optimizer_signature=sig))
    man_b = _manifest("B", "EXP-142", optimizer_signature=sig)
    check_arm_parity(read_manifest(a), man_b)


def test_source_commit_is_recorded_but_outside_parity():
    man_a = _manifest("A", "EXP-143", source={"git_commit": "aaa", "dirty": False})
    man_b = _manifest("B", "EXP-144", source={"git_commit": "bbb", "dirty": False})
    assert man_a["source"]["git_commit"] == "aaa"
    check_arm_parity(man_a, man_b)


def test_probe_order_is_a_fixed_shuffle_of_each_probe():
    sim, real = np.arange(0, 40, 2), np.arange(100, 130)
    s1, r1 = probe_order(sim, real, 42)
    s2, r2 = probe_order(sim, real, 42)
    assert np.array_equal(s1, s2) and np.array_equal(r1, r2)
    assert sorted(s1.tolist()) == sim.tolist() and sorted(r1.tolist()) == real.tolist()
    assert not np.array_equal(s1, sim) and not np.array_equal(r1, real)


def test_check_manifest_files_passes_on_a_vs_b_and_rejects_drift(tmp_path):
    a = write_manifest(tmp_path / "a.json", _manifest("A", "EXP-132"))
    b = write_manifest(tmp_path / "b.json", _manifest("B", "EXP-133"))
    check_manifest_files(a, b)
    c = write_manifest(tmp_path / "c.json", _manifest("B", "EXP-134", plan_digest="p2"))
    with pytest.raises(ValueError, match="plan_digest"):
        check_manifest_files(a, c)


def test_cli_check_parity_exit_codes(tmp_path):
    script = _repo_root() / "scripts" / "train_dann.py"
    a = write_manifest(tmp_path / "a.json", _manifest("A", "EXP-135"))
    b = write_manifest(tmp_path / "b.json", _manifest("B", "EXP-136"))
    c = write_manifest(tmp_path / "c.json", _manifest("B", "EXP-137", split_digest="d9"))

    def cli(*paths):
        return subprocess.run([sys.executable, str(script), "check-parity", *map(str, paths)],
                              cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8")

    ok = cli(a, b)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["parity"] == "ok"
    assert cli(a, c).returncode != 0


# ── _open_manifest (resume 계약, torch 없이) ──────────────────

def _args(**kw):
    import argparse
    return argparse.Namespace(**{"resume": False, **kw})


def test_open_manifest_fresh_run_writes_once(tmp_path):
    train_dann = _load_train_dann()
    paths = output_paths(tmp_path / "x.pt")
    man = _manifest("A", "EXP-138")
    assert train_dann._open_manifest(_args(), paths, man) == man
    with pytest.raises(FileExistsError):
        train_dann._open_manifest(_args(), paths, man)


def test_open_manifest_resume_returns_stored_and_rejects_other_run(tmp_path):
    train_dann = _load_train_dann()
    paths = output_paths(tmp_path / "x.pt")
    man = _manifest("A", "EXP-139")
    write_manifest(paths["manifest"], man)
    stored = train_dann._open_manifest(_args(resume=True), paths, man)
    assert stored["parity_key"] == man["parity_key"]
    other = _manifest("A", "EXP-139", split_digest="changed")
    with pytest.raises(ValueError, match="parity_key"):
        train_dann._open_manifest(_args(resume=True), paths, other)
    with pytest.raises(ValueError, match="report_id"):
        train_dann._open_manifest(_args(resume=True), paths, _manifest("A", "EXP-140"))


# ── publish_once / replace_atomic ────────────────────────────

def test_publish_once_writes_and_refuses_second_publish(tmp_path):
    path = tmp_path / "ck.pt"
    publish_once(path, lambda tmp: Path(tmp).write_bytes(b"one"))
    with pytest.raises(FileExistsError):
        publish_once(path, lambda tmp: Path(tmp).write_bytes(b"two"))
    assert path.read_bytes() == b"one"
    assert [q.name for q in tmp_path.iterdir()] == ["ck.pt"]  # 임시 파일이 남지 않는다


def test_publish_once_leaves_nothing_when_writer_crashes(tmp_path):
    path = tmp_path / "ck.pt"

    def crash(tmp):
        Path(tmp).write_bytes(b"half")
        raise RuntimeError("died mid-write")

    with pytest.raises(RuntimeError):
        publish_once(path, crash)
    assert list(tmp_path.iterdir()) == []


def test_replace_atomic_keeps_previous_file_when_writer_crashes(tmp_path):
    path = tmp_path / "resume.pt"
    replace_atomic(path, lambda tmp: Path(tmp).write_bytes(b"epoch-1"))
    replace_atomic(path, lambda tmp: Path(tmp).write_bytes(b"epoch-2"))
    assert path.read_bytes() == b"epoch-2"

    def crash(tmp):
        Path(tmp).write_bytes(b"trunc")
        raise RuntimeError("died mid-save")

    with pytest.raises(RuntimeError):
        replace_atomic(path, crash)
    assert path.read_bytes() == b"epoch-2"
    assert [q.name for q in tmp_path.iterdir()] == ["resume.pt"]


# ── code_fingerprint / assert_clean_code ─────────────────────

def _fake_git(head="abc123\n", status="", fail=False):
    def run(cmd, **kw):
        if fail:
            raise subprocess.CalledProcessError(128, cmd)
        out = head if cmd[1] == "rev-parse" else status
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    return run


def test_code_fingerprint_reports_dirty_paths_and_hashes_files(tmp_path):
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    fp = code_fingerprint(tmp_path, paths=("a.py",), run=_fake_git(status=" M a.py\n"))
    assert fp["git_commit"] == "abc123"
    assert fp["dirty"] is True and fp["dirty_paths"] == ["a.py"]
    assert fp["files_sha256"] == {"a.py": hashlib.sha256(b"x = 1\n").hexdigest()}
    with pytest.raises(RuntimeError, match="a.py"):
        assert_clean_code(fp)


def test_code_fingerprint_hash_changes_with_uncommitted_content(tmp_path):
    f = tmp_path / "a.py"
    f.write_bytes(b"x = 1\n")
    before = code_fingerprint(tmp_path, paths=("a.py",), run=_fake_git())
    f.write_bytes(b"x = 2\n")
    after = code_fingerprint(tmp_path, paths=("a.py",), run=_fake_git())
    assert before["git_commit"] == after["git_commit"]
    assert before["files_sha256"] != after["files_sha256"]


def test_assert_clean_code_passes_on_clean_tree_and_rejects_missing_git(tmp_path):
    (tmp_path / "a.py").write_bytes(b"")
    assert_clean_code(code_fingerprint(tmp_path, paths=("a.py",), run=_fake_git()))
    no_git = code_fingerprint(tmp_path, paths=("a.py",), run=_fake_git(fail=True))
    assert no_git["git_commit"] == "unknown"
    with pytest.raises(RuntimeError):
        assert_clean_code(no_git)


def test_code_fingerprint_default_paths_exist_in_repo():
    fp = code_fingerprint(REPO_ROOT)
    assert set(fp["files_sha256"]) == set(CODE_PATHS)


def test_cli_dry_plan_carries_contract_and_code_fingerprint(tmp_path):
    proc = _run_cli(tmp_path, "--dry-plan", arm="B", lambda_max=1.0)
    assert proc.returncode == 0, proc.stderr
    out = json.loads([ln for ln in proc.stdout.splitlines() if ln.strip()][-1])
    assert out["contract"] == run_contract(DannConfig.for_arm("B"))
    assert "files_sha256" in out["code"] and "dirty" in out["code"]


# ── domain_probe_auc (torch) ─────────────────────────────────

def test_domain_probe_auc_restores_bn_and_modes_and_separates_domains():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from ai_co_scientist.dann import domain_probe_auc, make_discriminator

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(4, 128), nn.BatchNorm1d(128))
            self.decoder = nn.Linear(128, 4)

    torch.manual_seed(0)
    model, disc = Model(), make_discriminator(128, 8)
    model.train()
    disc.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    seen = []
    sim = np.zeros((5, 2, 2), dtype=np.float32)
    real = np.ones((7, 2, 2), dtype=np.float32)

    def to_tensor(arr, idx):
        seen.append((arr is real, len(idx)))
        return torch.from_numpy(np.ascontiguousarray(arr[idx]))[:, None]

    auc = domain_probe_auc(model, disc, [(sim, np.arange(5), 0.0), (real, np.arange(7), 1.0)],
                           batch_size=4, to_tensor=to_tensor)
    assert 0.0 <= auc <= 1.0
    assert seen == [(False, 5), (True, 7)]  # 도메인별 배치, 꼬리는 앞 배치에 합쳐진다
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert model.training and disc.training
