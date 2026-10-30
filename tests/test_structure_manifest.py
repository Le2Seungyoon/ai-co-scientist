"""H6 translated-cache manifest must be verified inside structure training."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ai_co_scientist import cyclegan


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _valid_translated_cache(tmp_path: Path) -> tuple[Path, Path]:
    report_id = "EXP-026"
    ckpt = tmp_path / f"{report_id}-cyclegan.pt"
    _write(ckpt, b"cycle-checkpoint")
    source = tmp_path / "source"
    sim_case = source / "sim_case.npy"
    source.mkdir()
    np.save(sim_case, np.repeat(np.repeat(np.arange(1, 5, dtype=np.int8), 21_663), 2))
    sources = {
        "sim_sem": source / "sim_sem.npy",
        "sim_depth": source / "sim_depth.npy",
        "sim_case": sim_case,
        "real_sem": source / "real_sem.npy",
    }
    for name in ("sim_sem", "sim_depth", "real_sem"):
        _write(sources[name], name.encode())
    cache = tmp_path / "translated"
    outputs = {
        "sim_sem": cache / "sim_sem.npy",
        "sim_depth": cache / "sim_depth.npy",
        "sim_case": cache / "sim_case.npy",
    }
    _write(outputs["sim_sem"], b"translated-sem")
    _write(outputs["sim_depth"], sources["sim_depth"].read_bytes())
    _write(outputs["sim_case"], sim_case.read_bytes())
    case = np.load(sim_case)
    train_idx, val_idx = cyclegan.sim_split_indices(case)
    shifts = np.zeros(cyclegan.GATE_SAMPLE_N)
    gate = cyclegan.evaluate_gate(shifts, 0.0, local=shifts)
    gate.update({
        "shifts": shifts.tolist(), "local_shifts": shifts.tolist(),
        "report_id": report_id, "config": dict(cyclegan.PREREGISTERED),
        "ckpt_sha256": cyclegan.sha256_file(ckpt),
        "sim_case_sha256": cyclegan.sha256_file(sim_case),
        "split": cyclegan.sim_split_provenance(train_idx, val_idx),
        "indices_sha256": cyclegan.indices_sha256(cyclegan.validation_gate_indices(case)),
        "training_data": cyclegan.cyclegan_training_data_provenance(
            sources["sim_sem"], sim_case, sources["real_sem"], train_idx, val_idx),
    })
    manifest = cyclegan.build_manifest(
        report_id=report_id, config=dict(cyclegan.PREREGISTERED), gate=gate,
        ckpt_path=ckpt, source_files=sources, output_files=outputs,
        git_commit="a" * 40)
    manifest_path = cache / "manifest.json"
    cyclegan.write_once_json(manifest_path, manifest)
    return cache, manifest_path


def test_translated_cache_provenance_verifies_manifest_and_preserves_domains(tmp_path):
    cache, manifest_path = _valid_translated_cache(tmp_path)

    got = cyclegan.resolve_structure_cache_provenance(cache, manifest_path)

    assert got["cache_manifest"] == {
        "path": str(manifest_path.resolve()),
        "sha256": cyclegan.sha256_file(manifest_path),
    }
    assert got["report_id"] == "EXP-026"
    assert got["hypothesis"] == "H6"
    assert got["x_domain"] == "sim_translated_to_real_appearance"
    assert got["y_source"] == "sim_depth_gt"
    bound = cyclegan.bind_structure_checkpoint({"state_dict": {"weight": 1}}, got)
    assert bound["data_provenance"] == got
    assert bound["data_provenance"]["x_domain"] != "sim"
    result = cyclegan.structure_result_provenance(got)
    assert result["x_domain"] == "sim_translated_to_real_appearance"
    assert result["y_source"] == "sim_depth_gt"
    assert result["data_provenance"] == got


def test_raw_control_without_manifest_keeps_existing_sim_domain(tmp_path):
    got = cyclegan.resolve_structure_cache_provenance(tmp_path / "raw-cache")
    assert cyclegan.structure_result_provenance(got) == {
        "x_domain": "sim", "y_source": "sim_depth_gt",
    }
    assert cyclegan.bind_structure_checkpoint({"state_dict": {}}, got)["data_provenance"] == got


def test_cache_with_manifest_cannot_be_consumed_as_raw_control(tmp_path):
    cache, _manifest_path = _valid_translated_cache(tmp_path)
    with pytest.raises(ValueError, match="--cache-manifest"):
        cyclegan.resolve_structure_cache_provenance(cache)


def test_structure_manifest_must_be_exactly_inside_selected_cache(tmp_path):
    cache, manifest_path = _valid_translated_cache(tmp_path)
    outside = tmp_path / "copied-manifest.json"
    outside.write_bytes(manifest_path.read_bytes())

    with pytest.raises(ValueError, match="cache-dir.*manifest.json"):
        cyclegan.resolve_structure_cache_provenance(cache, outside)


def test_structure_manifest_missing_refuses_before_cache_data_can_be_returned(tmp_path):
    cache = tmp_path / "translated"
    cache.mkdir()
    with pytest.raises(ValueError, match="manifest"):
        cyclegan.resolve_structure_cache_provenance(cache, cache / "manifest.json")


def test_structure_manifest_tamper_refuses_before_cache_data_can_be_returned(tmp_path):
    cache, manifest_path = _valid_translated_cache(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["x_domain"] = "sim"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="x_domain"):
        cyclegan.resolve_structure_cache_provenance(cache, manifest_path)


def test_structure_resume_refuses_different_verified_manifest_provenance(tmp_path):
    cache, manifest_path = _valid_translated_cache(tmp_path)
    current = cyclegan.resolve_structure_cache_provenance(cache, manifest_path)
    stored = json.loads(json.dumps(current))
    stored["cache_manifest"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="data_provenance"):
        cyclegan.require_matching_structure_provenance(stored, current)


def test_structure_resume_allows_legacy_missing_provenance_only_for_raw_cache(tmp_path):
    raw = cyclegan.resolve_structure_cache_provenance(tmp_path / "raw")
    assert cyclegan.require_matching_structure_provenance(None, raw) is None
    cache, manifest_path = _valid_translated_cache(tmp_path)
    translated = cyclegan.resolve_structure_cache_provenance(cache, manifest_path)
    with pytest.raises(ValueError, match="data_provenance"):
        cyclegan.require_matching_structure_provenance(None, translated)


def test_train_structure_rejects_tampered_manifest_before_opening_cache_arrays(tmp_path):
    cache = tmp_path / "translated"
    cache.mkdir()
    manifest = cache / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_structure.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--cache-dir", str(cache),
         "--cache-manifest", str(manifest), "--epochs", "0"],
        capture_output=True, text=True, encoding="utf-8")

    assert proc.returncode != 0
    assert "cache manifest 검증 실패" in proc.stderr
    assert "sim_sem.npy" not in proc.stderr
