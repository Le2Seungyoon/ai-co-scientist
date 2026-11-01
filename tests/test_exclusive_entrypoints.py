"""원래 CLI의 배타 경계: GPU/HTTP 대신 실패 센티널, 락은 임시 경로의 실제 구현."""
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from ai_co_scientist import locks
from ai_co_scientist.submission import write_submission_zip

ROOT = Path(__file__).resolve().parents[1]
GPU_SCRIPTS = [
    ("train_level", [], "seed_everything"),
    ("train_structure", [], "seed_everything"),
    ("infer_decomposed", ["--submit", "out.zip"], "load_model"),
    ("dump_level_proba", ["--out", "out.npy"], "predict_levels_cnn"),
]


def load_script(monkeypatch, name):
    if name != "dacon_submit":
        torch = pytest.importorskip("torch")
        pytest.importorskip("cv2")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = importlib.util.spec_from_file_location("_exclusive_" + name,
                                                ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ensure_utf8_console", lambda: None)
    return module


@pytest.mark.parametrize("name,args,backend", GPU_SCRIPTS)
def test_busy_gpu_never_enters_backend_or_writes(tmp_path, monkeypatch, name, args, backend):
    # Removing any original main() guard reaches the sentinel (never a real GPU).
    monkeypatch.chdir(tmp_path)
    real_lock = locks.resource_lock
    monkeypatch.setattr(locks, "resource_lock", lambda name, **kw:
                        real_lock(name, root=tmp_path / "locks", **kw))
    module = load_script(monkeypatch, name)
    events = []
    if name == "train_structure":
        monkeypatch.setattr(module, "resolve_structure_cache_provenance",
                            lambda *a: events.append("verified") or {"source": "verified"})
    monkeypatch.setattr(module, backend, lambda *a, **kw: pytest.fail("protected work reached"))
    monkeypatch.setattr(sys, "argv", [name] + args)
    with real_lock("gpu-0", root=tmp_path / "locks"):
        with pytest.raises(SystemExit) as exc:
            module.main()
    assert exc.value.code == 2
    assert events == (["verified"] if name == "train_structure" else [])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["locks"]


@pytest.mark.parametrize("name,args,backend", GPU_SCRIPTS)
@pytest.mark.parametrize("argument,code", [("--help", 0), ("--unknown-option", 2)])
def test_parser_finishes_before_lock(monkeypatch, name, args, backend, argument, code):
    monkeypatch.setattr(locks, "resource_lock", lambda *a, **kw: pytest.fail("lock before parse"))
    module = load_script(monkeypatch, name)
    monkeypatch.setattr(sys, "argv", [name, argument])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == code


def test_inference_probability_validation_precedes_lock_and_model(monkeypatch):
    monkeypatch.setattr(locks, "resource_lock", lambda *a, **kw: pytest.fail("lock before validation"))
    module = load_script(monkeypatch, "infer_decomposed")
    monkeypatch.setattr(module, "load_model", lambda *a: pytest.fail("model before validation"))
    monkeypatch.setattr(sys, "argv", ["infer", "--submit", "out.zip", "--level-source",
                                     "mean_only", "--dump-level-proba", "out.npy"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_structure_verified_provenance_passes_into_locked_work(monkeypatch):
    events = []
    provenance = {"cache_manifest_sha256": "verified-hash"}

    @contextmanager
    def lock(name):
        events.append(("lock", name))
        yield
        events.append("unlock")

    monkeypatch.setattr(locks, "resource_lock", lock)
    module = load_script(monkeypatch, "train_structure")
    monkeypatch.setattr(module, "resolve_structure_cache_provenance",
                        lambda *a: events.append("verify") or provenance)
    monkeypatch.setattr(module, "_run_locked", lambda args, verified:
                        events.append(("run", verified is provenance)), raising=False)
    monkeypatch.setattr(module, "seed_everything", lambda *a: pytest.fail("unwrapped training"))
    monkeypatch.setattr(sys, "argv", ["train_structure"])
    module.main()
    assert events == ["verify", ("lock", "gpu-0"), ("run", True), "unlock"]


def test_structure_invalid_provenance_never_acquires(monkeypatch):
    monkeypatch.setattr(locks, "resource_lock", lambda *a, **kw: pytest.fail("lock before provenance"))
    module = load_script(monkeypatch, "train_structure")

    def invalid(*a):
        raise ValueError("manifest hash mismatch")

    monkeypatch.setattr(module, "resolve_structure_cache_provenance", invalid)
    monkeypatch.setattr(sys, "argv", ["train_structure"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["busy", "verify-only", "invalid", "submit"])
def test_dacon_only_actual_submit_holds_slot(tmp_path, monkeypatch, mode):
    path = tmp_path / "submission.zip"
    write_submission_zip(np.full((1, 2, 2), 140, dtype=np.uint8), ["a.png"], path)
    real_lock = locks.resource_lock
    events = []

    @contextmanager
    def acquire(name):
        events.append(("lock", name))
        with real_lock(name, root=tmp_path / "locks"):
            yield

    monkeypatch.setattr(locks, "resource_lock", acquire)
    module = load_script(monkeypatch, "dacon_submit")
    real_verify = module.verify_submission

    def verify(*a, **kw):
        assert events == []
        events.append("verify")
        return real_verify(*a, **kw)

    def submit(*a):
        assert events == ["verify", ("lock", "dacon-slot")]
        with pytest.raises(locks.ResourceBusy), real_lock("dacon-slot", root=tmp_path / "locks"):
            pytest.fail("backend ran without owning slot")
        events.append("submit")
        return {"isSubmitted": True}

    monkeypatch.setattr(module, "verify_submission", verify)
    monkeypatch.setattr(module.dacon, "submit", submit if mode == "submit" else
                        lambda *a: pytest.fail("unexpected backend call"))
    args = ["dacon", str(path), "--expected-files", "2" if mode == "invalid" else "1"]
    if mode == "verify-only":
        args.append("--verify-only")
    monkeypatch.setattr(sys, "argv", args)
    if mode == "busy":
        with real_lock("dacon-slot", root=tmp_path / "locks"):
            assert module.main() == 1
    else:
        assert module.main() == (2 if mode == "invalid" else 0)
    assert events == (["verify", ("lock", "dacon-slot"), "submit"] if mode == "submit" else
                      ["verify", ("lock", "dacon-slot")] if mode == "busy" else ["verify"])
