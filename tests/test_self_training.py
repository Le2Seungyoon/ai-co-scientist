"""`ai_co_scientist.self_training` — H5 자기학습 계약의 **행동**을 검사한다.

전부 합성 fixture, `tmp_path`만 쓴다 — `data/`도 GPU도 필요 없다. CLI 절(§4)은
`scripts/build_pseudo_labels.py`, `scripts/train_self_training.py`(Engineer B 담당)를
검사하는데, 이 파일을 쓰는 시점에는 아직 존재하지 않을 수 있다. 그 경우 해당 테스트는
`ModuleNotFoundError`로 실패한다 — TDD 원칙상 의도된 것이며, 건너뛰지 않는다.
"""
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_co_scientist import self_training as st

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _import_script(name: str):
    """scripts/<name>.py를 지연 import한다 — 모듈이 없으면 여기서 실패한다(의도됨)."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module(name)


def _synthetic_pseudo_files(tmp_path, n=6, subdir=""):
    """real_sem.npy(uint8) + labels.npy(float32 [0,1]) + teacher.pt(임의 바이트)를 만든다."""
    base = tmp_path / subdir if subdir else tmp_path
    base.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    real = rng.integers(0, 255, size=(n, *st.IMAGE_SHAPE), dtype=np.uint8)
    np.save(base / st.REAL_TRAIN_NPY, real)
    labels = rng.random((n, *st.IMAGE_SHAPE)).astype(np.float32)
    np.save(base / "labels.npy", labels)
    (base / "teacher.pt").write_bytes(b"synthetic-teacher-weights-v1")
    return base


# ── 해시 / 정규 직렬화 ──────────────────────────────────────

def test_sha256_file_matches_hashlib_reference(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world" * 1000)
    assert st.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_sha256_file_streams_in_chunks_smaller_than_file(tmp_path):
    # chunk를 파일보다 훨씬 작게 줘도 전체 내용을 소비해야 한다 — 스트리밍 루프의 핵심 계약
    p = tmp_path / "b.bin"
    p.write_bytes(bytes(range(256)) * 50)
    assert st.sha256_file(p, chunk=7) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_canonical_json_is_key_order_independent():
    a = st.canonical_json({"b": 1, "a": 2})
    b = st.canonical_json({"a": 2, "b": 1})
    assert a == b == '{"a":2,"b":1}'


def test_fingerprint_changes_when_content_changes():
    f1 = st.fingerprint({"lr": 1e-3})
    f2 = st.fingerprint({"lr": 5e-3})
    assert f1 != f2
    assert f1 == st.fingerprint({"lr": 1e-3})  # 결정적이어야 한다


# ── require_real_train_source: 경로 계약 ────────────────────

def test_require_real_train_source_rejects_wrong_name(tmp_path):
    p = tmp_path / "sem_images.npy"
    with pytest.raises(st.ContractError):
        st.require_real_train_source(p)


def test_require_real_train_source_rejects_test_named_file(tmp_path):
    # 파일명에 "test"가 있으면 메시지에도 "test"가 있어야 한다(운영자가 원인을 바로 읽도록)
    p = tmp_path / "test_sem.npy"
    with pytest.raises(st.ContractError, match="test"):
        st.require_real_train_source(p)


def test_require_real_train_source_accepts_despite_test_named_parent_dir(tmp_path):
    # pytest tmp_path 부모 디렉터리 이름은 흔히 "test..."를 포함한다 — 파일명만 봐야 한다
    parent = tmp_path / "test_a_parent_dir_name"
    parent.mkdir()
    n = 3
    arr = np.zeros((n, *st.IMAGE_SHAPE), dtype=np.uint8)
    np.save(parent / st.REAL_TRAIN_NPY, arr)
    resolved = st.require_real_train_source(parent / st.REAL_TRAIN_NPY, expected_n=n)
    assert resolved == (parent / st.REAL_TRAIN_NPY).resolve()


def test_require_real_train_source_rejects_hash_copy_of_test_npy(tmp_path):
    # 이름은 real_sem.npy로 바꿔치기했지만 바이트가 test_sem.npy와 같으면(우회 복사) 거부한다
    n = 3
    arr = np.zeros((n, *st.IMAGE_SHAPE), dtype=np.uint8)
    np.save(tmp_path / st.TEST_NPY, arr)
    shutil.copy2(tmp_path / st.TEST_NPY, tmp_path / st.REAL_TRAIN_NPY)
    with pytest.raises(st.ContractError, match="test"):
        st.require_real_train_source(tmp_path / st.REAL_TRAIN_NPY, cache_dir=tmp_path,
                                     expected_n=n)


def test_require_real_train_source_rejects_samefile_copy_of_test_npy(tmp_path):
    # 하드링크로 같은 inode를 가리키면 바이트 비교 이전에 samefile로 잡혀야 한다
    n = 3
    arr = np.zeros((n, *st.IMAGE_SHAPE), dtype=np.uint8)
    np.save(tmp_path / st.TEST_NPY, arr)
    try:
        import os
        os.link(tmp_path / st.TEST_NPY, tmp_path / st.REAL_TRAIN_NPY)
    except OSError:
        pytest.skip("이 파일시스템은 하드링크를 지원하지 않는다")
    with pytest.raises(st.ContractError, match="test"):
        st.require_real_train_source(tmp_path / st.REAL_TRAIN_NPY, cache_dir=tmp_path,
                                     expected_n=n)


def test_require_real_train_source_allows_different_content_from_test_npy(tmp_path):
    # test_sem.npy가 캐시에 있어도, 내용이 다르면 통과해야 한다 (오탐 방지)
    n = 3
    np.save(tmp_path / st.TEST_NPY, np.zeros((n, *st.IMAGE_SHAPE), dtype=np.uint8))
    np.save(tmp_path / st.REAL_TRAIN_NPY, np.ones((n, *st.IMAGE_SHAPE), dtype=np.uint8))
    st.require_real_train_source(tmp_path / st.REAL_TRAIN_NPY, cache_dir=tmp_path, expected_n=n)


def test_require_real_train_source_rejects_wrong_count(tmp_path):
    arr = np.zeros((3, *st.IMAGE_SHAPE), dtype=np.uint8)
    np.save(tmp_path / st.REAL_TRAIN_NPY, arr)
    with pytest.raises(st.ContractError):
        st.require_real_train_source(tmp_path / st.REAL_TRAIN_NPY, expected_n=4)


def test_require_real_train_source_rejects_wrong_shape(tmp_path):
    arr = np.zeros((3, 10, 10), dtype=np.uint8)
    np.save(tmp_path / st.REAL_TRAIN_NPY, arr)
    with pytest.raises(st.ContractError):
        st.require_real_train_source(tmp_path / st.REAL_TRAIN_NPY, expected_n=3)


def test_require_real_train_source_rejects_wrong_dtype(tmp_path):
    arr = np.zeros((3, *st.IMAGE_SHAPE), dtype=np.float32)
    np.save(tmp_path / st.REAL_TRAIN_NPY, arr)
    with pytest.raises(st.ContractError):
        st.require_real_train_source(tmp_path / st.REAL_TRAIN_NPY, expected_n=3)


def test_require_real_train_source_returns_resolved_path(tmp_path):
    arr = np.zeros((2, *st.IMAGE_SHAPE), dtype=np.uint8)
    np.save(tmp_path / st.REAL_TRAIN_NPY, arr)
    got = st.require_real_train_source(tmp_path / st.REAL_TRAIN_NPY, expected_n=2)
    assert got == (tmp_path / st.REAL_TRAIN_NPY).resolve()
    assert got.is_absolute()


# ── validate_pseudo_labels ──────────────────────────────────

def test_validate_pseudo_labels_rejects_wrong_shape():
    arr = np.zeros((5, 1, 1), dtype=np.float32)
    with pytest.raises(st.ContractError):
        st.validate_pseudo_labels(arr, 5)


def test_validate_pseudo_labels_rejects_wrong_dtype():
    arr = np.zeros((5, *st.IMAGE_SHAPE), dtype=np.float64)
    with pytest.raises(st.ContractError):
        st.validate_pseudo_labels(arr, 5)


def test_validate_pseudo_labels_rejects_nan():
    arr = np.zeros((5, *st.IMAGE_SHAPE), dtype=np.float32)
    arr[0, 0, 0] = np.nan
    with pytest.raises(st.ContractError):
        st.validate_pseudo_labels(arr, 5)


def test_validate_pseudo_labels_rejects_inf():
    arr = np.zeros((5, *st.IMAGE_SHAPE), dtype=np.float32)
    arr[1, 2, 3] = np.inf
    with pytest.raises(st.ContractError):
        st.validate_pseudo_labels(arr, 5)


def test_validate_pseudo_labels_rejects_negative_value():
    arr = np.zeros((5, *st.IMAGE_SHAPE), dtype=np.float32)
    arr[0, 0, 0] = -0.001
    with pytest.raises(st.ContractError):
        st.validate_pseudo_labels(arr, 5)


def test_validate_pseudo_labels_rejects_value_over_one():
    arr = np.zeros((5, *st.IMAGE_SHAPE), dtype=np.float32)
    arr[0, 0, 0] = 1.001
    with pytest.raises(st.ContractError):
        st.validate_pseudo_labels(arr, 5)


def test_validate_pseudo_labels_returns_exact_stats():
    arr = np.zeros((2, *st.IMAGE_SHAPE), dtype=np.float32)
    arr[0, 0, 0] = 1.0
    stats = st.validate_pseudo_labels(arr, 2)
    assert stats["n"] == 2
    assert stats["shape"] == [2, 72, 48]
    assert stats["dtype"] == "float32"
    assert stats["min"] == 0.0
    assert stats["max"] == 1.0
    assert stats["mean"] == pytest.approx(1.0 / (2 * 72 * 48))
    assert isinstance(stats["mean"], float) and not isinstance(stats["mean"], np.floating)


# ── manifest 빌드 / 쓰기 / 읽기 ──────────────────────────────

def test_build_pseudo_manifest_rejects_non_real_adabn_source(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=4)
    with pytest.raises(st.ContractError):
        st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=4,
                                 adabn_source="test")


def test_build_pseudo_manifest_rejects_wrong_adabn_shuffle(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=4)
    with pytest.raises(st.ContractError):
        st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=4,
                                 adabn_shuffle=7)


def test_build_pseudo_manifest_returns_expected_schema_and_fields(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=4)
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=4,
                                 source_commit="abc123")
    assert m["schema"] == st.PSEUDO_SCHEMA
    assert m["x_domain"] == "real"
    assert m["y_source"] == "pseudo_label"
    assert m["teacher"]["adabn_source"] == "real"
    assert m["teacher"]["adabn_shuffle"] == 42
    assert m["teacher"]["sha256"] == st.sha256_file(base / "teacher.pt")
    assert m["source"]["n"] == 4
    assert m["source"]["sha256"] == st.sha256_file(base / st.REAL_TRAIN_NPY)
    assert m["labels"]["sha256"] == st.sha256_file(base / "labels.npy")
    assert m["labels"]["n"] == 4
    assert m["source_commit"] == "abc123"
    assert "timestamp" not in json.dumps(m).lower()  # manifest에 타임스탬프가 없어야 재현성이 성립한다


def test_write_manifest_refuses_overwrite(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=2)
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=2)
    p = st.write_manifest(base / "m.json", m)
    assert p.exists()
    with pytest.raises(st.ContractError):
        st.write_manifest(base / "m.json", m)


def test_write_manifest_roundtrips_via_load_manifest(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=2)
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=2)
    p = st.write_manifest(base / "m.json", m)
    assert st.load_manifest(p) == m
    assert p.read_text(encoding="utf-8").endswith("\n")


# ── verify_pseudo_manifest ───────────────────────────────────

def test_verify_pseudo_manifest_passes_for_untouched_files(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=3)
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=3)
    p = st.write_manifest(base / "m.json", m)
    out = st.verify_pseudo_manifest(p)
    assert out == m


def test_verify_pseudo_manifest_detects_tampered_labels(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=3)
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=3)
    p = st.write_manifest(base / "m.json", m)

    raw = bytearray((base / "labels.npy").read_bytes())
    raw[-1] ^= 0xFF
    (base / "labels.npy").write_bytes(bytes(raw))

    with pytest.raises(st.ContractError, match="labels"):
        st.verify_pseudo_manifest(p)


def test_verify_pseudo_manifest_detects_tampered_source(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=3)
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=3)
    p = st.write_manifest(base / "m.json", m)

    raw = bytearray((base / st.REAL_TRAIN_NPY).read_bytes())
    raw[-1] ^= 0xFF
    (base / st.REAL_TRAIN_NPY).write_bytes(bytes(raw))

    with pytest.raises(st.ContractError, match="source"):
        st.verify_pseudo_manifest(p)


def test_verify_pseudo_manifest_detects_missing_labels_file(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=3)
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=3)
    p = st.write_manifest(base / "m.json", m)
    (base / "labels.npy").unlink()
    with pytest.raises(st.ContractError, match="labels"):
        st.verify_pseudo_manifest(p)


def test_verify_pseudo_manifest_checks_given_teacher_ckpt_matches(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=3)
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=3)
    p = st.write_manifest(base / "m.json", m)

    other = base / "other_teacher.pt"
    other.write_bytes(b"a-different-teacher")
    with pytest.raises(st.ContractError, match="teacher"):
        st.verify_pseudo_manifest(p, teacher_ckpt=other)

    st.verify_pseudo_manifest(p, teacher_ckpt=base / "teacher.pt")  # 예외 없이 통과해야 한다


# ── manifest 비교 / 재현 ─────────────────────────────────────

def test_pseudo_manifest_build_is_reproducible_across_different_paths(tmp_path):
    # manifest 결정성: 같은 내용을 다른 경로에서 두 번 빌드하면 경로 필드만 다르고 나머지는 같다
    src = _synthetic_pseudo_files(tmp_path, n=5, subdir="src")
    dup = tmp_path / "dup"
    dup.mkdir()
    for name in (st.REAL_TRAIN_NPY, "labels.npy", "teacher.pt"):
        shutil.copy2(src / name, dup / name)

    m1 = st.build_pseudo_manifest(labels_path=src / "labels.npy",
                                  source_path=src / st.REAL_TRAIN_NPY,
                                  teacher_ckpt=src / "teacher.pt", expected_n=5,
                                  source_commit="c1")
    m2 = st.build_pseudo_manifest(labels_path=dup / "labels.npy",
                                  source_path=dup / st.REAL_TRAIN_NPY,
                                  teacher_ckpt=dup / "teacher.pt", expected_n=5,
                                  source_commit="c1")

    assert m1["labels"]["path"] != m2["labels"]["path"]  # 경로 자체는 실제로 다르다
    assert st.compare_pseudo_manifests(m1, m2) == []
    st.require_reproduced(m1, m2)  # 예외가 나면 안 된다


def test_compare_pseudo_manifests_reports_content_diff(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=3)
    m1 = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                  source_path=base / st.REAL_TRAIN_NPY,
                                  teacher_ckpt=base / "teacher.pt", expected_n=3,
                                  batch_size=512)
    m2 = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                  source_path=base / st.REAL_TRAIN_NPY,
                                  teacher_ckpt=base / "teacher.pt", expected_n=3,
                                  batch_size=256)
    diffs = st.compare_pseudo_manifests(m1, m2)
    assert diffs == ["batch_size"]


def test_require_reproduced_raises_on_diff(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=3)
    m1 = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                  source_path=base / st.REAL_TRAIN_NPY,
                                  teacher_ckpt=base / "teacher.pt", expected_n=3,
                                  source_commit="c1")
    m2 = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                  source_path=base / st.REAL_TRAIN_NPY,
                                  teacher_ckpt=base / "teacher.pt", expected_n=3,
                                  source_commit="c2")
    with pytest.raises(st.ContractError, match="source_commit"):
        st.require_reproduced(m1, m2)


# ── 경로 별칭 거부 ───────────────────────────────────────────

def test_reject_aliases_raises_when_two_paths_are_the_same(tmp_path):
    shared = tmp_path / "shared.pt"
    with pytest.raises(st.ContractError, match="out"):
        st.reject_aliases(teacher_ckpt=shared, out=shared)


def test_reject_aliases_detects_alias_through_relative_dotdot(tmp_path):
    # 문자열은 다르지만 resolve() 후 같은 파일을 가리키면 잡혀야 한다
    sub = tmp_path / "sub"
    sub.mkdir()
    a = sub / "x.pt"
    b = sub / ".." / "sub" / "x.pt"
    with pytest.raises(st.ContractError):
        st.reject_aliases(out=a, teacher_ckpt=b)


def test_reject_aliases_ignores_none_values(tmp_path):
    st.reject_aliases(out=tmp_path / "a.pt", teacher_ckpt=tmp_path / "b.pt",
                      pseudo_manifest=None, resume=None)  # 예외 없이 통과해야 한다


def test_reject_aliases_allows_distinct_paths(tmp_path):
    st.reject_aliases(out=tmp_path / "a.pt", teacher_ckpt=tmp_path / "b.pt",
                      manifest=tmp_path / "c.json")  # 예외 없이 통과해야 한다


# ── teacher 바이트 복사 거부 ─────────────────────────────────

def test_reject_teacher_copy_raises_for_byte_identical_file(tmp_path):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"weights-v1")
    out = tmp_path / "student.pt"
    out.write_bytes(b"weights-v1")
    with pytest.raises(st.ContractError):
        st.reject_teacher_copy(out, st.sha256_file(teacher))


def test_reject_teacher_copy_allows_missing_path(tmp_path):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"weights-v1")
    out = tmp_path / "does_not_exist.pt"
    st.reject_teacher_copy(out, st.sha256_file(teacher))  # 예외 없이 통과해야 한다


def test_reject_teacher_copy_allows_different_content(tmp_path):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"weights-v1")
    out = tmp_path / "student.pt"
    out.write_bytes(b"weights-v2-different")
    st.reject_teacher_copy(out, st.sha256_file(teacher))  # 예외 없이 통과해야 한다


# ── resume / manifest 경로 파생 ──────────────────────────────

def test_student_resume_path_mirrors_train_structure_convention():
    assert st.student_resume_path("runtime/ckpt/EXP-0NN-student.pt") == \
        Path("runtime/ckpt/EXP-0NN-student.resume.pt")


def test_student_manifest_path_suffix():
    assert st.student_manifest_path("runtime/ckpt/EXP-0NN-student.pt") == \
        Path("runtime/ckpt/EXP-0NN-student.manifest.json")


# ── RealSamplerSpec.build 검증 ───────────────────────────────

def test_real_sampler_spec_build_defaults_to_min_n_real_n_sim():
    a = st.RealSamplerSpec.build(n_sim=10, n_real=3)
    assert a.real_per_epoch == 3
    b = st.RealSamplerSpec.build(n_sim=3, n_real=10)
    assert b.real_per_epoch == 3


def test_real_sampler_spec_build_rejects_negative_real_per_epoch():
    with pytest.raises(st.ContractError):
        st.RealSamplerSpec.build(n_sim=10, n_real=10, real_per_epoch=-1)


def test_real_sampler_spec_build_rejects_real_per_epoch_over_n_real():
    with pytest.raises(st.ContractError):
        st.RealSamplerSpec.build(n_sim=10, n_real=3, real_per_epoch=4)


def test_real_sampler_spec_build_rejects_real_per_epoch_over_n_sim():
    # H5 조건: epoch당 real 샘플 수는 sim 샘플 수를 넘지 않는다
    with pytest.raises(st.ContractError):
        st.RealSamplerSpec.build(n_sim=3, n_real=10, real_per_epoch=4)


def test_real_sampler_spec_build_rejects_nonpositive_n_sim():
    with pytest.raises(st.ContractError):
        st.RealSamplerSpec.build(n_sim=0, n_real=10)


def test_real_sampler_spec_build_rejects_real_per_epoch_positive_with_zero_n_real():
    with pytest.raises(st.ContractError):
        st.RealSamplerSpec.build(n_sim=10, n_real=0, real_per_epoch=1)


def test_real_sampler_spec_build_allows_zero_real_per_epoch_with_zero_n_real():
    spec = st.RealSamplerSpec.build(n_sim=10, n_real=0, real_per_epoch=0)
    assert spec.real_per_epoch == 0


def test_real_sampler_spec_to_dict_declares_sampling_without_replacement():
    spec = st.RealSamplerSpec.build(n_sim=10, n_real=10, real_per_epoch=5, seed=1)
    d = spec.to_dict()
    assert d["replacement"] is False
    assert d["n_sim"] == 10 and d["n_real"] == 10 and d["real_per_epoch"] == 5 and d["seed"] == 1


# ── RealSamplerSpec.epoch_indices: 결정성 / 무작위성 ─────────

def test_epoch_indices_same_spec_and_epoch_are_identical():
    spec = st.RealSamplerSpec.build(n_sim=20, n_real=50, real_per_epoch=10, seed=42)
    a = spec.epoch_indices(3)
    b = spec.epoch_indices(3)
    assert np.array_equal(a, b)
    assert a.dtype == np.int64


def test_epoch_indices_differ_across_epochs():
    spec = st.RealSamplerSpec.build(n_sim=20, n_real=50, real_per_epoch=10, seed=42)
    a = spec.epoch_indices(0)
    b = spec.epoch_indices(1)
    assert not np.array_equal(a, b)


def test_epoch_indices_include_every_sim_index_exactly_once():
    spec = st.RealSamplerSpec.build(n_sim=15, n_real=50, real_per_epoch=9, seed=7)
    for epoch in range(4):
        idx = spec.epoch_indices(epoch)
        sim_part = sorted(idx[idx < 15].tolist())
        assert sim_part == list(range(15))


def test_epoch_indices_real_count_is_bounded_and_no_replacement():
    spec = st.RealSamplerSpec.build(n_sim=15, n_real=50, real_per_epoch=9, seed=7)
    for epoch in range(4):
        idx = spec.epoch_indices(epoch)
        real_part = idx[idx >= 15]
        assert len(real_part) == 9
        assert len(set(real_part.tolist())) == 9  # 비복원 — 중복이 없어야 한다
        assert real_part.min() >= 15 and real_part.max() < 15 + 50


def test_epoch_indices_length_matches_n_sim_plus_real_per_epoch():
    spec = st.RealSamplerSpec.build(n_sim=15, n_real=50, real_per_epoch=9, seed=7)
    assert len(spec.epoch_indices(0)) == 24


def test_epoch_indices_zero_real_per_epoch_is_just_sim_permutation():
    spec = st.RealSamplerSpec.build(n_sim=8, n_real=0, real_per_epoch=0, seed=1)
    idx = spec.epoch_indices(0)
    assert sorted(idx.tolist()) == list(range(8))


# ── arm_spec / student_config ────────────────────────────────

def test_arm_spec_rejects_unknown_arm():
    with pytest.raises(st.ContractError):
        st.arm_spec("arm2")


def test_arm_spec_returns_expected_data_and_seed():
    assert st.arm_spec("arm0") == {"data": "sim_only", "seed": 42}
    assert st.arm_spec("arm0b") == {"data": "sim_only", "seed": 43}
    assert st.arm_spec("arm1") == {"data": "sim_pseudo", "seed": 42}


def test_student_config_sim_only_forbids_real_sampling():
    sampler = st.RealSamplerSpec.build(n_sim=10, n_real=10, real_per_epoch=1, seed=42)
    with pytest.raises(st.ContractError):
        st.student_config("arm0", sampler, source_commit="c1")


def test_student_config_sim_only_forbids_pseudo_manifest():
    sampler = st.RealSamplerSpec.build(n_sim=10, n_real=0, real_per_epoch=0, seed=42)
    with pytest.raises(st.ContractError):
        st.student_config("arm0", sampler, pseudo_manifest_sha256="a" * 64, source_commit="c1")


def test_student_config_sim_pseudo_requires_real_sampling():
    sampler = st.RealSamplerSpec.build(n_sim=10, n_real=10, real_per_epoch=0, seed=42)
    with pytest.raises(st.ContractError):
        st.student_config("arm1", sampler, pseudo_manifest_sha256="a" * 64, source_commit="c1")


def test_student_config_sim_pseudo_requires_pseudo_manifest_sha():
    sampler = st.RealSamplerSpec.build(n_sim=10, n_real=10, real_per_epoch=5, seed=42)
    with pytest.raises(st.ContractError):
        st.student_config("arm1", sampler, pseudo_manifest_sha256=None, source_commit="c1")


def test_student_config_matches_arms_hparams_and_fingerprint_is_deterministic():
    sampler = st.RealSamplerSpec.build(n_sim=10, n_real=0, real_per_epoch=0, seed=42)
    cfg1 = st.student_config("arm0", sampler, source_commit="c1")
    cfg2 = st.student_config("arm0", sampler, source_commit="c1")
    assert cfg1 == cfg2  # 같은 입력 → 같은 config (config_fingerprint 포함)
    assert cfg1["lr"] == 1e-3 and cfg1["epochs"] == 15 and cfg1["init"] == "scratch"
    assert cfg1["config_fingerprint"] == st.fingerprint(
        {k: v for k, v in cfg1.items() if k != "config_fingerprint"})


def test_student_config_fingerprint_changes_when_seed_changes():
    sampler_a = st.RealSamplerSpec.build(n_sim=10, n_real=0, real_per_epoch=0, seed=42)
    sampler_b = st.RealSamplerSpec.build(n_sim=10, n_real=0, real_per_epoch=0, seed=43)
    cfg_a = st.student_config("arm0", sampler_a, source_commit="c1")
    cfg_b = st.student_config("arm0b", sampler_b, source_commit="c1")
    assert cfg_a["config_fingerprint"] != cfg_b["config_fingerprint"]


# ── check_arm_parity ─────────────────────────────────────────

def _three_arm_configs(n_sim=20, n_real=20, source_commit="c1", lr_override=None):
    s0 = st.RealSamplerSpec.build(n_sim, n_real, real_per_epoch=0, seed=42)
    s0b = st.RealSamplerSpec.build(n_sim, n_real, real_per_epoch=0, seed=43)
    s1 = st.RealSamplerSpec.build(n_sim, n_real, real_per_epoch=5, seed=42)
    cfg0 = st.student_config("arm0", s0, source_commit=source_commit)
    cfg0b = st.student_config("arm0b", s0b, source_commit=source_commit)
    cfg1 = st.student_config("arm1", s1, pseudo_manifest_sha256="a" * 64,
                             source_commit=source_commit)
    if lr_override is not None:
        cfg0b = dict(cfg0b)
        cfg0b["lr"] = lr_override
    return [cfg0, cfg0b, cfg1]


def test_check_arm_parity_passes_for_allowed_diffs_only():
    st.check_arm_parity(_three_arm_configs())  # 예외가 없어야 한다


def test_check_arm_parity_rejects_lr_change():
    configs = _three_arm_configs(lr_override=5e-3)
    with pytest.raises(st.ContractError, match="lr"):
        st.check_arm_parity(configs)


def test_check_arm_parity_rejects_different_source_commit():
    s0 = st.RealSamplerSpec.build(20, 20, real_per_epoch=0, seed=42)
    s0b = st.RealSamplerSpec.build(20, 20, real_per_epoch=0, seed=43)
    cfg0 = st.student_config("arm0", s0, source_commit="commit-a")
    cfg0b = st.student_config("arm0b", s0b, source_commit="commit-b")
    with pytest.raises(st.ContractError, match="source_commit"):
        st.check_arm_parity([cfg0, cfg0b])


def test_check_arm_parity_rejects_empty_source_commit():
    s0 = st.RealSamplerSpec.build(20, 20, real_per_epoch=0, seed=42)
    s0b = st.RealSamplerSpec.build(20, 20, real_per_epoch=0, seed=43)
    cfg0 = st.student_config("arm0", s0, source_commit="")
    cfg0b = st.student_config("arm0b", s0b, source_commit="")
    with pytest.raises(st.ContractError, match="source_commit"):
        st.check_arm_parity([cfg0, cfg0b])


def test_check_arm_parity_requires_at_least_two_configs():
    with pytest.raises(st.ContractError):
        st.check_arm_parity(_three_arm_configs()[:1])


def test_check_arm_parity_rejects_mismatched_n_sim():
    s0 = st.RealSamplerSpec.build(n_sim=20, n_real=20, real_per_epoch=0, seed=42)
    s0b = st.RealSamplerSpec.build(n_sim=21, n_real=20, real_per_epoch=0, seed=43)
    cfg0 = st.student_config("arm0", s0, source_commit="c1")
    cfg0b = st.student_config("arm0b", s0b, source_commit="c1")
    with pytest.raises(st.ContractError, match="n_sim"):
        st.check_arm_parity([cfg0, cfg0b])


# ══════════════════════════════════════════════════════════════
# §4 CLI 계약 — scripts/build_pseudo_labels.py, scripts/train_self_training.py
# (Engineer B 담당 — 이 파일을 쓰는 시점엔 아직 없을 수 있고, 그러면 아래는 실패한다)
# ══════════════════════════════════════════════════════════════

def test_cli_build_rejects_test_named_source(capsys):
    bpl = _import_script("build_pseudo_labels")
    with pytest.raises(SystemExit) as exc:
        bpl.main(["build", "--teacher-ckpt", "teacher.pt", "--cache-dir", ".",
                  "--out", "out.npy", "--manifest", "manifest.json",
                  "--source", "test_sem.npy"])
    assert exc.value.code == 2
    assert "test" in capsys.readouterr().err


def test_cli_build_rejects_out_equal_teacher_ckpt(tmp_path, capsys):
    bpl = _import_script("build_pseudo_labels")
    base = _synthetic_pseudo_files(tmp_path, n=4)
    shared = base / "shared.npy"
    with pytest.raises(SystemExit) as exc:
        bpl.main(["build", "--teacher-ckpt", str(shared), "--cache-dir", str(base),
                  "--out", str(shared), "--manifest", str(base / "m.json"),
                  "--source", str(base / st.REAL_TRAIN_NPY), "--expected-n", "4"])
    assert exc.value.code == 2


def test_cli_build_rejects_existing_out(tmp_path, capsys):
    bpl = _import_script("build_pseudo_labels")
    base = _synthetic_pseudo_files(tmp_path, n=4)
    out = base / "preexisting.npy"
    out.write_bytes(b"already-here")
    with pytest.raises(SystemExit) as exc:
        bpl.main(["build", "--teacher-ckpt", str(base / "teacher.pt"), "--cache-dir", str(base),
                  "--out", str(out), "--manifest", str(base / "m.json"),
                  "--source", str(base / st.REAL_TRAIN_NPY), "--expected-n", "4"])
    assert exc.value.code == 2


def test_cli_verify_passes_and_then_detects_tamper(tmp_path, capsys):
    bpl = _import_script("build_pseudo_labels")
    base = _synthetic_pseudo_files(tmp_path, n=4)
    manifest = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                        source_path=base / st.REAL_TRAIN_NPY,
                                        teacher_ckpt=base / "teacher.pt", expected_n=4)
    mpath = st.write_manifest(base / "m.json", manifest)

    bpl.main(["verify", "--manifest", str(mpath)])
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(last)["ok"] is True

    raw = bytearray((base / "labels.npy").read_bytes())
    raw[-1] ^= 0xFF
    (base / "labels.npy").write_bytes(bytes(raw))

    with pytest.raises(SystemExit) as exc:
        bpl.main(["verify", "--manifest", str(mpath)])
    assert exc.value.code == 2


def test_cli_compare_reports_ok_for_reproduced_build(tmp_path, capsys):
    bpl = _import_script("build_pseudo_labels")
    src = _synthetic_pseudo_files(tmp_path, n=4, subdir="src")
    dup = tmp_path / "dup"
    dup.mkdir()
    for name in (st.REAL_TRAIN_NPY, "labels.npy", "teacher.pt"):
        shutil.copy2(src / name, dup / name)

    m1 = st.write_manifest(src / "m.json", st.build_pseudo_manifest(
        labels_path=src / "labels.npy", source_path=src / st.REAL_TRAIN_NPY,
        teacher_ckpt=src / "teacher.pt", expected_n=4))
    m2 = st.write_manifest(dup / "m.json", st.build_pseudo_manifest(
        labels_path=dup / "labels.npy", source_path=dup / st.REAL_TRAIN_NPY,
        teacher_ckpt=dup / "teacher.pt", expected_n=4))

    bpl.main(["compare", str(m1), str(m2)])
    last = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(last)
    assert payload["ok"] is True


def _train_argv(tmp_path, arm, out, teacher, *extra):
    """train 인자 — `--plan`을 넣어야 argparse 필수 인자 누락이 아닌 **계약**으로 exit 2가 난다."""
    return ["train", "--arm", arm, "--cache-dir", str(tmp_path), "--out", str(out),
            "--teacher-ckpt", str(teacher), "--plan", str(tmp_path / "plan.json"), *extra]


def test_cli_train_arm1_without_pseudo_manifest_exits_2(tmp_path, capsys):
    tst = _import_script("train_self_training")
    with pytest.raises(SystemExit) as exc:
        tst.main(_train_argv(tmp_path, "arm1", tmp_path / "student.pt", tmp_path / "teacher.pt"))
    assert exc.value.code == 2
    assert "--pseudo-manifest가 필요하다" in capsys.readouterr().err


def test_cli_train_arm0_with_pseudo_manifest_exits_2(tmp_path, capsys):
    tst = _import_script("train_self_training")
    manifest = tmp_path / "pseudo.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        tst.main(_train_argv(tmp_path, "arm0", tmp_path / "student.pt", tmp_path / "teacher.pt",
                             "--pseudo-manifest", str(manifest)))
    assert exc.value.code == 2
    assert "sim-only 대조군" in capsys.readouterr().err


def test_cli_train_out_equal_teacher_ckpt_exits_2(tmp_path, capsys):
    tst = _import_script("train_self_training")
    shared = tmp_path / "shared.pt"
    shared.write_bytes(b"teacher-weights")
    with pytest.raises(SystemExit) as exc:
        tst.main(_train_argv(tmp_path, "arm0", shared, shared))
    assert exc.value.code == 2
    assert "같은 경로" in capsys.readouterr().err


def test_cli_train_out_is_byte_copy_of_teacher_exits_2(tmp_path, capsys):
    tst = _import_script("train_self_training")
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher-weights")
    out = tmp_path / "student.pt"
    out.write_bytes(b"teacher-weights")  # teacher의 바이트 복사본 — scratch 시작 위반
    with pytest.raises(SystemExit) as exc:
        tst.main(_train_argv(tmp_path, "arm0", out, teacher))
    assert exc.value.code == 2
    assert "바이트 복사본" in capsys.readouterr().err


def _write_arm_manifest(path, arm, n_sim=5, n_real=5, source_commit="deadbeef", lr=None):
    spec = st.arm_spec(arm)
    real_per_epoch = 3 if spec["data"] == "sim_pseudo" else 0
    sampler = st.RealSamplerSpec.build(n_sim, n_real, real_per_epoch=real_per_epoch,
                                       seed=spec["seed"])
    pseudo_sha = "a" * 64 if spec["data"] == "sim_pseudo" else None
    cfg = st.student_config(arm, sampler, pseudo_manifest_sha256=pseudo_sha,
                            source_commit=source_commit)
    if lr is not None:
        cfg = dict(cfg)
        cfg["lr"] = lr
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_cli_parity_three_synthetic_manifests_ok(tmp_path, capsys):
    tst = _import_script("train_self_training")
    m0 = _write_arm_manifest(tmp_path / "arm0.json", "arm0")
    m0b = _write_arm_manifest(tmp_path / "arm0b.json", "arm0b")
    m1 = _write_arm_manifest(tmp_path / "arm1.json", "arm1")

    tst.main(["parity", str(m0), str(m0b), str(m1)])
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(last)["ok"] is True


def test_cli_parity_lr_change_exits_2(tmp_path):
    tst = _import_script("train_self_training")
    m0 = _write_arm_manifest(tmp_path / "arm0.json", "arm0")
    m0b = _write_arm_manifest(tmp_path / "arm0b.json", "arm0b", lr=5e-3)  # 계약 위반: lr 고정

    with pytest.raises(SystemExit) as exc:
        tst.main(["parity", str(m0), str(m0b)])
    assert exc.value.code == 2


def test_lazy_import_does_not_require_torch_or_cv2():
    """torch/cv2가 sys.modules에서 차단돼도 두 스크립트의 import 자체는 성공해야 한다.

    최상위에서 torch를 import하면 import 시점에 즉시 실패해, 이 워크트리(dev 그룹만 sync)에서
    `pytest --collect-only`조차 죽는다 — 지연 import가 계약인 이유.
    """
    scripts_literal = json.dumps(str(SCRIPTS_DIR))
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['cv2'] = None; "
        f"sys.path.insert(0, {scripts_literal}); "
        "import build_pseudo_labels, train_self_training"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("name", ["build_pseudo_labels.py", "train_self_training.py"])
def test_cli_help_exits_0(name):
    script = SCRIPTS_DIR / name
    proc = subprocess.run([sys.executable, str(script), "--help"],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr


def _train_format_manifest(path, arm):
    """`train`이 실제로 디스크에 쓰는 형태 — cfg + out/x_domain/y_source/metric/note."""
    cfg = json.loads(_write_arm_manifest(path, arm).read_text(encoding="utf-8"))
    is_arm1 = cfg["data"] == "sim_pseudo"
    cfg.update({"out": str(path.with_suffix(".pt")),
                "x_domain": "sim+real" if is_arm1 else "sim",
                "y_source": "sim_depth_gt+pseudo_label" if is_arm1 else "sim_depth_gt",
                "metric": None, "note": "judged by leaderboard only"})
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_cli_parity_accepts_manifests_in_train_output_format(tmp_path, capsys):
    # 회귀: x_domain/y_source가 허용 차이에 없어서 train이 쓴 실제 manifest 3개로는
    # parity가 항상 실패했다 — bare student_config로만 검사하면 드러나지 않는다.
    tst = _import_script("train_self_training")
    paths = [_train_format_manifest(tmp_path / f"{a}.manifest.json", a)
             for a in ("arm0", "arm0b", "arm1")]
    tst.main(["parity", *map(str, paths)])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["arms"] == [
        "arm0", "arm0b", "arm1"]


def test_check_arm_parity_rejects_dirty_commit():
    # 같은 커밋 해시라도 작업트리가 dirty면 arm마다 코드가 달랐을 수 있다 — "같은 구현 commit"
    # 조건은 깨끗한 트리에서만 성립한다.
    cfgs = [st.student_config(a, st.RealSamplerSpec.build(5, 0, seed=st.ARMS[a]["seed"]),
                              source_commit="abc" + st.DIRTY_SUFFIX) for a in ("arm0", "arm0b")]
    with pytest.raises(st.ContractError, match="dirty"):
        st.check_arm_parity(cfgs)


def test_git_head_marks_dirty_tree_and_empty_outside_git(tmp_path, monkeypatch):
    # 실행 트리의 HEAD와 dirty 여부가 manifest에 그대로 남아야 parity가 판단할 수 있다.
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True,  # noqa: E731
                                    capture_output=True)
    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    (repo / "f.txt").write_text("a", encoding="utf-8")
    run("add", "f.txt")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "f")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert st.git_head(cwd=repo) == head
    (repo / "f.txt").write_text("b", encoding="utf-8")
    assert st.git_head(cwd=repo) == head + st.DIRTY_SUFFIX
    outside = tmp_path / "nogit"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # 상위 저장소로 새지 않게
    assert st.git_head(cwd=outside) == ""


# ── 리뷰 반영: sim X 고정 · 깨끗한 커밋 · plan 선행 · 덮어쓰기 거부 · manifest 주장 재검증 ──

def _synthetic_case(n_maps=20):
    """depth map n_maps장 × itr0/itr1 = SEM 2*n_maps장, Case 1..4 균등."""
    return np.repeat(np.tile(np.arange(1, 5), n_maps // 4), 2).astype(np.int64)


def test_sim_train_indices_is_exp005_split_independent_of_arm_seed():
    # student sim X는 EXP-005 teacher의 train 분할(map_level_split 0.2, seed 42) 그대로다.
    # arm0b(seed 43)가 분할을 바꾸면 seed 대조군에 데이터 구성 차이가 섞인다.
    from ai_co_scientist.sem import map_level_split
    case = _synthetic_case()
    idx = st.sim_train_indices(case, expected_n=None)
    want = np.where(~map_level_split(case, 0.2, 42))[0]
    assert np.array_equal(idx, want)
    assert len(idx) == 32  # 5 maps/Case × 20퍼센트 = 1 map/Case 홀드아웃 → 16 maps × 2
    # itr0/itr1 쌍이 갈라지지 않는다 — 누수 방지
    assert set((idx[::2] + 1).tolist()) == set(idx[1::2].tolist())


def test_sim_train_indices_rejects_wrong_count():
    with pytest.raises(st.ContractError, match="138648"):
        st.sim_train_indices(_synthetic_case())


def test_sim_subset_record_hashes_the_indices():
    idx = np.arange(6, dtype=np.int64)
    rec = st.sim_subset_record(idx)
    assert rec["n"] == 6 and rec["val_frac"] == 0.2 and rec["seed"] == 42
    assert rec["indices_sha256"] == hashlib.sha256(idx.tobytes()).hexdigest()
    assert st.sim_subset_record(idx[::-1].copy())["indices_sha256"] != rec["indices_sha256"]


@pytest.mark.parametrize("commit", ["", "abc" + st.DIRTY_SUFFIX])
def test_require_clean_commit_rejects_empty_and_dirty(commit):
    with pytest.raises(st.ContractError, match="깨끗한 커밋"):
        st.require_clean_commit(commit)


def test_require_real_train_source_missing_file_is_contract_error(tmp_path):
    # FileNotFoundError 트레이스백(exit 1)이 아니라 ContractError(exit 2)여야 한다.
    with pytest.raises(st.ContractError, match="파일이 없다"):
        st.require_real_train_source(tmp_path / st.REAL_TRAIN_NPY, expected_n=4)


@pytest.mark.parametrize("field,value", [
    (("teacher", "adabn_shuffle"), 7), (("teacher", "adabn_source"), "realtest"),
    (("y_source",), "real_depth_gt"), (("labels", "n"), 3)])
def test_verify_rejects_manifest_claims_that_break_h5(tmp_path, field, value):
    # 파일 해시는 그대로인데 manifest의 주장만 손으로 바꾼 경우 — 해시 검사로는 안 잡힌다.
    base = _synthetic_pseudo_files(tmp_path, n=4)
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=4)
    node = m
    for k in field[:-1]:
        node = node[k]
    node[field[-1]] = value
    path = st.write_manifest(base / "m.json", m)
    with pytest.raises(st.ContractError, match="계약과 다르다"):
        st.verify_pseudo_manifest(path)


def test_pseudo_manifest_records_adabn_recipe_and_runtime(tmp_path):
    base = _synthetic_pseudo_files(tmp_path, n=4)
    rt = {"device": "cuda", "torch": "2.x", "cuda": "12.4"}
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=4, runtime=rt)
    assert m["teacher"]["adabn_batch"] == 512 and m["teacher"]["adabn_drop_last"] is False
    assert m["runtime"] == rt
    other = dict(m, runtime={**rt, "device": "cpu"})
    assert st.compare_pseudo_manifests(m, other) == ["runtime.device"]


def _cfg(arm, commit="deadbeef"):
    spec = st.arm_spec(arm)
    n_real = 5 if spec["data"] == "sim_pseudo" else 0
    sampler = st.RealSamplerSpec.build(5, n_real, seed=spec["seed"])
    return st.student_config(arm, sampler, source_commit=commit,
                             pseudo_manifest_sha256="a" * 64 if n_real else None)


def test_check_arm_parity_rejects_duplicate_arm():
    with pytest.raises(st.ContractError, match="두 번"):
        st.check_arm_parity([_cfg("arm0"), _cfg("arm0")])


def test_check_arm_parity_rejects_seed_not_matching_arms_table():
    bad = dict(_cfg("arm0b"), seed=42)  # arm0b인데 seed 42 — arm0의 복제
    with pytest.raises(st.ContractError, match="ARMS"):
        st.check_arm_parity([_cfg("arm0"), bad])


def test_check_arm_parity_rejects_stale_fingerprint():
    bad = dict(_cfg("arm0b"), config_fingerprint="0" * 64)
    with pytest.raises(st.ContractError, match="config_fingerprint"):
        st.check_arm_parity([_cfg("arm0"), bad])


def test_check_arm_parity_rejects_runtime_difference():
    # train이 덧붙이는 runtime은 허용 차이가 아니다 — arm0 CPU / arm1 GPU를 잡는다.
    a = dict(_cfg("arm0"), runtime={"device": "cuda"})
    b = dict(_cfg("arm0b"), runtime={"device": "cpu"})
    with pytest.raises(st.ContractError, match="runtime"):
        st.check_arm_parity([a, b])


@pytest.fixture
def plan_env(tmp_path, monkeypatch):
    """plan/train CLI가 torch 직전까지 가도록: 합성 cache + 깨끗한 커밋 + 소형 sim 분할."""
    tst = _import_script("train_self_training")
    monkeypatch.setattr(tst, "git_head", lambda cwd=None: "c0ffee")
    monkeypatch.setattr(tst, "sim_train_indices",
                        lambda case: st.sim_train_indices(case, expected_n=None))
    base = _synthetic_pseudo_files(tmp_path, n=4)
    np.save(base / "sim_case.npy", _synthetic_case())
    m = st.build_pseudo_manifest(labels_path=base / "labels.npy",
                                 source_path=base / st.REAL_TRAIN_NPY,
                                 teacher_ckpt=base / "teacher.pt", expected_n=4)
    st.write_manifest(base / "pseudo.json", m)
    return tst, base


def _arm_args(base, arm):
    extra = ["--pseudo-manifest", str(base / "pseudo.json")] if arm == "arm1" else []
    return ["--arm", arm, "--cache-dir", str(base), "--out", str(base / f"{arm}.pt"),
            "--teacher-ckpt", str(base / "teacher.pt"), *extra]


def test_cli_plan_then_parity_before_any_training(plan_env, capsys):
    # 사전보고 "실행 전 중단": arm 설정 차이는 GPU를 쓰기 **전에** 잡혀야 한다.
    tst, base = plan_env
    plans = []
    for arm in ("arm0", "arm0b", "arm1"):
        plans.append(base / f"{arm}.plan.json")
        tst.main(["plan", *_arm_args(base, arm), "--write", str(plans[-1])])
    capsys.readouterr()
    tst.main(["parity", *map(str, plans)])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ok"] is True
    cfg1 = st.load_manifest(plans[2])
    assert cfg1["sampler"]["n_sim"] == 32 and cfg1["sampler"]["real_per_epoch"] == 4
    assert cfg1["sim_subset"] == st.load_manifest(plans[0])["sim_subset"]


def test_cli_train_rejects_plan_of_another_arm(plan_env, capsys):
    tst, base = plan_env
    tst.main(["plan", *_arm_args(base, "arm0"), "--write", str(base / "arm0.plan.json")])
    with pytest.raises(SystemExit) as exc:
        tst.main(["train", *_arm_args(base, "arm0b"), "--plan", str(base / "arm0.plan.json")])
    assert exc.value.code == 2
    assert "--plan과 지금 설정이 다르다" in capsys.readouterr().err


def test_cli_train_refuses_to_overwrite_existing_out(plan_env, capsys):
    # --out이 기존 ckpt(예: EXP-013 레벨 분류기)를 가리키면 덮어쓰지 않는다 — 재개만 예외.
    tst, base = plan_env
    tst.main(["plan", *_arm_args(base, "arm0"), "--write", str(base / "arm0.plan.json")])
    (base / "arm0.pt").write_bytes(b"precious-checkpoint")
    with pytest.raises(SystemExit) as exc:
        tst.main(["train", *_arm_args(base, "arm0"), "--plan", str(base / "arm0.plan.json")])
    assert exc.value.code == 2
    assert "--out이 이미 존재한다" in capsys.readouterr().err
    assert (base / "arm0.pt").read_bytes() == b"precious-checkpoint"


def test_cli_plan_rejects_dirty_tree(plan_env, monkeypatch, capsys):
    tst, base = plan_env
    monkeypatch.setattr(tst, "git_head", lambda cwd=None: "c0ffee" + st.DIRTY_SUFFIX)
    with pytest.raises(SystemExit) as exc:
        tst.main(["plan", *_arm_args(base, "arm0"), "--write", str(base / "p.json")])
    assert exc.value.code == 2
    assert "깨끗한 커밋" in capsys.readouterr().err
    assert not (base / "p.json").exists()


def test_cli_build_rejects_dirty_tree_before_torch(tmp_path, monkeypatch, capsys):
    bpl = _import_script("build_pseudo_labels")
    monkeypatch.setattr(bpl, "git_head", lambda cwd=None: "")
    base = _synthetic_pseudo_files(tmp_path, n=4)
    with pytest.raises(SystemExit) as exc:
        bpl.main(["build", "--teacher-ckpt", str(base / "teacher.pt"), "--cache-dir", str(base),
                  "--out", str(base / "o.npy"), "--manifest", str(base / "m.json"),
                  "--expected-n", "4"])
    assert exc.value.code == 2
    assert "깨끗한 커밋" in capsys.readouterr().err


# ── GPU 락: 실제 GPU 구간만 `resource_lock(GPU_LOCK)` 안에서 돈다 ──
# 진짜 락(기계 단위 temp 경로)과 GPU 함수를 모두 가짜로 바꿔, 순서만 기록한다. 락 밖에서 GPU
# 함수가 불리거나, 검증 실패인데 락을 잡거나, 바쁜 락을 무시하고 GPU를 쓰면 여기서 빨개진다.

class _LockRecorder:
    def __init__(self, busy=False):
        self.events, self.busy = [], busy

    def __call__(self, name, **kw):
        from contextlib import contextmanager

        from ai_co_scientist.locks import ResourceBusy

        @contextmanager
        def cm():
            if self.busy:
                raise ResourceBusy(f"{name} held by another run")
            self.events.append(f"acquire:{name}")
            try:
                yield
            finally:
                self.events.append(f"release:{name}")
        return cm()


def test_gpu_lock_name_is_the_shared_machine_resource():
    # tests/test_locks.py와 같은 이름 — 다른 이름이면 infer/train 레인과 서로 막지 못한다.
    assert st.GPU_LOCK == "gpu-0"


def test_cli_train_runs_gpu_section_inside_gpu_lock(plan_env, monkeypatch):
    tst, base = plan_env
    tst.main(["plan", *_arm_args(base, "arm0"), "--write", str(base / "arm0.plan.json")])
    rec = _LockRecorder()
    monkeypatch.setattr(tst, "resource_lock", rec)
    monkeypatch.setattr(tst, "_train_on_gpu",
                        lambda *a, **k: rec.events.append("gpu"))
    tst.main(["train", *_arm_args(base, "arm0"), "--plan", str(base / "arm0.plan.json")])
    assert rec.events == ["acquire:gpu-0", "gpu", "release:gpu-0"]


def test_cli_train_releases_gpu_lock_when_gpu_section_fails(plan_env, monkeypatch):
    tst, base = plan_env
    tst.main(["plan", *_arm_args(base, "arm0"), "--write", str(base / "arm0.plan.json")])
    rec = _LockRecorder()
    monkeypatch.setattr(tst, "resource_lock", rec)

    def boom(*a, **k):
        rec.events.append("gpu")
        raise SystemExit(2)  # 예: --resume 지문 불일치로 ap.error
    monkeypatch.setattr(tst, "_train_on_gpu", boom)
    with pytest.raises(SystemExit):
        tst.main(["train", *_arm_args(base, "arm0"), "--plan", str(base / "arm0.plan.json")])
    assert rec.events == ["acquire:gpu-0", "gpu", "release:gpu-0"]


def test_cli_train_busy_gpu_exits_2_without_training(plan_env, monkeypatch, capsys):
    tst, base = plan_env
    tst.main(["plan", *_arm_args(base, "arm0"), "--write", str(base / "arm0.plan.json")])
    rec = _LockRecorder(busy=True)
    monkeypatch.setattr(tst, "resource_lock", rec)
    monkeypatch.setattr(tst, "_train_on_gpu", lambda *a, **k: rec.events.append("gpu"))
    with pytest.raises(SystemExit) as exc:
        tst.main(["train", *_arm_args(base, "arm0"), "--plan", str(base / "arm0.plan.json")])
    assert exc.value.code == 2
    assert "gpu-0" in capsys.readouterr().err
    assert rec.events == []


def test_cli_plan_and_failed_train_validation_never_take_gpu_lock(plan_env, monkeypatch):
    # plan은 CPU 전용이고, 검증에서 떨어진 train은 GPU 락을 건드리지 않아야 한다 — 그래야
    # 다른 레인이 학습 중일 때도 plan/parity/검증을 돌릴 수 있다.
    tst, base = plan_env
    rec = _LockRecorder()
    monkeypatch.setattr(tst, "resource_lock", rec)
    monkeypatch.setattr(tst, "_train_on_gpu", lambda *a, **k: rec.events.append("gpu"))
    tst.main(["plan", *_arm_args(base, "arm0"), "--write", str(base / "arm0.plan.json")])
    with pytest.raises(SystemExit):
        tst.main(["train", *_arm_args(base, "arm0b"), "--plan", str(base / "arm0.plan.json")])
    assert rec.events == []


@pytest.fixture
def build_env(tmp_path, monkeypatch):
    """build CLI가 GPU 구간 직전까지 가도록: 합성 real_sem + 깨끗한 커밋 + 가짜 락."""
    bpl = _import_script("build_pseudo_labels")
    monkeypatch.setattr(bpl, "git_head", lambda cwd=None: "c0ffee")
    base = _synthetic_pseudo_files(tmp_path, n=4)
    rec = _LockRecorder()
    monkeypatch.setattr(bpl, "resource_lock", rec)

    def fake_label(args, ap, cache, source, out_path):
        rec.events.append("gpu")
        np.save(out_path, np.full((4, *st.IMAGE_SHAPE), 0.5, dtype=np.float32))
        return {"device": "fake", "torch": "none", "cuda": None}
    monkeypatch.setattr(bpl, "_label_on_gpu", fake_label)
    argv = ["build", "--teacher-ckpt", str(base / "teacher.pt"), "--cache-dir", str(base),
            "--out", str(base / "pseudo.npy"), "--manifest", str(base / "pseudo.json"),
            "--expected-n", "4"]
    return bpl, base, rec, argv


def test_cli_build_labels_inside_gpu_lock_then_writes_manifest(build_env, capsys):
    bpl, base, rec, argv = build_env
    bpl.main(argv)
    assert rec.events == ["acquire:gpu-0", "gpu", "release:gpu-0"]
    m = st.verify_pseudo_manifest(base / "pseudo.json", teacher_ckpt=base / "teacher.pt")
    assert m["runtime"] == {"device": "fake", "torch": "none", "cuda": None}
    assert m["source_commit"] == "c0ffee"
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["labels"]["n"] == 4


def test_cli_build_busy_gpu_exits_2_without_labels(build_env, capsys):
    bpl, base, rec, argv = build_env
    rec.busy = True
    with pytest.raises(SystemExit) as exc:
        bpl.main(argv)
    assert exc.value.code == 2
    assert "gpu-0" in capsys.readouterr().err
    assert rec.events == []
    assert not (base / "pseudo.npy").exists() and not (base / "pseudo.json").exists()


def test_cli_build_test_source_never_takes_gpu_lock(build_env):
    bpl, base, rec, argv = build_env
    np.save(base / st.TEST_NPY, np.zeros((4, *st.IMAGE_SHAPE), dtype=np.uint8))
    with pytest.raises(SystemExit):
        bpl.main([*argv, "--source", str(base / st.TEST_NPY)])
    assert rec.events == []
