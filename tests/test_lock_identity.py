"""PID 재사용, 회수 경쟁, 파일 실패 경로의 오프라인 회귀 검사."""
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from ai_co_scientist import locks


def aged_lock(tmp_path, token):
    path = tmp_path / "resource.lock"
    path.write_text(token, encoding="utf-8")
    old = time.time() - 10000
    os.utime(path, (old, old))
    return path


def test_token_records_current_process_start(tmp_path):
    with locks.file_lock(tmp_path / "lock", timeout=0) as path:
        fields = path.read_text().split(":")
        assert len(fields) == 3
        assert fields[1] == str(os.getpid())
        assert int(fields[2]) > 0


@pytest.mark.parametrize("start,protected", [(123, True), (124, False), (None, True)])
def test_pid_reuse_compares_start_and_unknown_identity_fails_closed(tmp_path, monkeypatch,
                                                                 start, protected):
    path = aged_lock(tmp_path, "owner:42:123")
    monkeypatch.setattr(locks, "_process_identity", lambda pid: (True, start), raising=False)
    if protected:
        with pytest.raises(locks.ResourceBusy), locks.file_lock(path, timeout=0):
            pytest.fail("live or unreadable owner displaced")
    else:
        with locks.file_lock(path, timeout=0):
            assert path.read_text() != "owner:42:123"


@pytest.mark.parametrize("token", ["old:42", "old:42:", "old:42:broken"])
def test_legacy_tokens_preserve_live_owner(tmp_path, monkeypatch, token):
    path = aged_lock(tmp_path, token)
    monkeypatch.setattr(locks, "_process_identity", lambda pid: (True, 999), raising=False)
    with pytest.raises(locks.ResourceBusy), locks.file_lock(path, timeout=0):
        pytest.fail("legacy live owner displaced")


def test_linux_start_field_handles_parentheses(monkeypatch):
    # comm may itself contain spaces and ')'; starttime is field 22.
    raw = "42 (odd ) name) S " + " ".join(["0"] * 18 + ["98765", "99"])
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: raw)
    assert locks._linux_process_identity(42) == (True, 98765)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows process lookup")
def test_windows_exact_pid_lookup_rejects_substring(monkeypatch):
    # Old tasklist substring matching accepts this unrelated PID in a process name.
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: type(
        "Output", (), {"stdout": '"process4294967295.exe","1234"'})())
    assert not locks._pid_alive(4294967295)


def test_stale_reclaim_rechecks_exact_token(tmp_path, monkeypatch):
    path = aged_lock(tmp_path, "old:42:123")
    replacement = "new:43:456"

    def identity(pid):
        if pid == 42:
            path.write_text(replacement)
            return False, None
        return True, 456

    monkeypatch.setattr(locks, "_process_identity", identity, raising=False)
    with pytest.raises(locks.ResourceBusy), locks.file_lock(path, timeout=0):
        pytest.fail("replacement lock removed")
    assert path.read_text() == replacement


@pytest.mark.parametrize("operation", ["stat", "read_text"])
def test_unreadable_lock_is_bounded_resource_busy(tmp_path, monkeypatch, operation):
    path = aged_lock(tmp_path, "dead:42:123")
    original = getattr(Path, operation)

    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("unreadable lock")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, operation, denied)
    with pytest.raises(locks.ResourceBusy), locks.file_lock(path, timeout=0):
        pytest.fail("unreadable lock reclaimed")


@pytest.mark.parametrize("short_write", [False, True])
def test_failed_token_write_closes_and_removes_lock(tmp_path, monkeypatch, short_write):
    path = tmp_path / "resource.lock"
    descriptors = []
    real_write = os.write

    def fail(fd, payload):
        descriptors.append(fd)
        if short_write:
            return real_write(fd, payload[:1])
        raise OSError("disk full")

    monkeypatch.setattr(os, "write", fail)
    with pytest.raises(OSError), locks.file_lock(path, timeout=0):
        pytest.fail("entered body without complete ownership token")
    assert not path.exists()
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_default_resource_contends_across_cwd(tmp_path):
    name = "cross-cwd-" + uuid.uuid4().hex
    code = (
        "from ai_co_scientist.locks import resource_lock, ResourceBusy\n"
        "try:\n"
        f"    with resource_lock({name!r}): pass\n"
        "except ResourceBusy: raise SystemExit(19)\n"
    )
    with locks.resource_lock(name):
        proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                              capture_output=True, text=True, timeout=15)
    assert proc.returncode == 19, proc.stderr


def test_fresh_contender_does_not_open_token_during_owner_release(tmp_path, monkeypatch):
    # Windows readers can deny the owner's unlink. Fresh contention needs only mtime;
    # opening every token left reg.lock behind and timed out seven of eight writers.
    path = tmp_path / "resource.lock"
    with locks.file_lock(path, timeout=0):
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", lambda *a, **kw:
                          pytest.fail("fresh contender opened holder's token"))
            with pytest.raises(locks.ResourceBusy), locks.file_lock(path, timeout=0):
                pytest.fail("contender acquired")
    assert not path.exists()
