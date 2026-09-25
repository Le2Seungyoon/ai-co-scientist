"""자원 락 — 배타성과 스테일 회수를 **행동으로** 검사한다 (오프라인, 합성 경로).

`flock`은 이 호스트에 없다. `O_CREAT|O_EXCL`이 그 자리를 채우므로, 이 파일이 재는 것은
"두 번째 획득이 실제로 거부되는가"와 "죽은 보유자의 락이 회수되는가"다.
"""
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from ai_co_scientist.locks import ResourceBusy, file_lock, resource_lock


def test_second_acquire_fails_immediately_at_zero_timeout(tmp_path):
    """timeout=0은 큐잉하지 않고 즉시 거부한다 — flock -w 0과 같은 의미."""
    lock = tmp_path / "gpu.lock"
    with file_lock(lock, timeout=0.0):
        with pytest.raises(ResourceBusy):
            with file_lock(lock, timeout=0.0):
                pass


def test_lock_is_released_on_exit(tmp_path):
    lock = tmp_path / "gpu.lock"
    with file_lock(lock, timeout=0.0):
        pass
    with file_lock(lock, timeout=0.0):  # 재획득 가능해야 한다
        pass
    assert not lock.exists()


def test_lock_is_released_when_the_body_raises(tmp_path):
    """예외로 빠져나가도 락이 남으면 그 자원은 영구 점유된다."""
    lock = tmp_path / "gpu.lock"
    with pytest.raises(ZeroDivisionError):
        with file_lock(lock, timeout=0.0):
            1 / 0
    assert not lock.exists()


def test_stale_lock_is_reclaimed(tmp_path):
    """죽은 프로세스가 남긴 락은 회수된다 — 아니면 재부팅까지 자원이 잠긴다."""
    lock = tmp_path / "gpu.lock"
    lock.write_text("99999", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    with file_lock(lock, timeout=0.0, stale=120.0):
        pass


def test_fresh_lock_is_not_reclaimed(tmp_path):
    """스테일 판정이 너무 공격적이면 살아있는 보유자를 밀어낸다."""
    lock = tmp_path / "gpu.lock"
    lock.write_text("99999", encoding="utf-8")
    with pytest.raises(ResourceBusy):
        with file_lock(lock, timeout=0.0, stale=120.0):
            pass


def test_resource_lock_default_root_is_outside_the_repo(tmp_path):
    """워크트리마다 다른 파일을 잠그면 락이 아니라 장식이다 —
    `architecture.md`가 이미 이름 붙인 `registry.locked()`의 트리별 경로 결함."""
    import tempfile

    from ai_co_scientist.config import project_root

    # The default lock directory is intentionally shared by every checkout.  A unique probe
    # name keeps independent pytest processes from testing each other instead of this contract.
    with resource_lock(f"probe-default-root-{uuid.uuid4().hex}") as held:
        assert Path(tempfile.gettempdir()).resolve() in Path(held).resolve().parents
        assert project_root() not in Path(held).resolve().parents


def test_resource_lock_excludes_by_name(tmp_path):
    with resource_lock("gpu-0", root=tmp_path):
        with pytest.raises(ResourceBusy):
            with resource_lock("gpu-0", root=tmp_path):
                pass
        with resource_lock("dacon-slot", root=tmp_path):  # 다른 자원은 막히지 않는다
            pass


def test_holder_does_not_delete_replacement_lock(tmp_path):
    """락이 회수되면 새 보유자가 그 파일을 들고 있다. 원래 보유자가 빠져나갈 때
    unlink하면 새 보유자의 락을 깨트린다 — 토큰 검증으로 방지한다."""
    lock = tmp_path / "resource.lock"
    with file_lock(lock, timeout=0.0):
        # 일반적인 상황: 보유 중 다른 진행이 락을 회수하고 획득한다
        new_token = "replaced_token"
        lock.write_text(new_token, encoding="utf-8")
    # 원래 보유자가 빠져나갔지만, unlink하지 않았어야 한다 (토큰 불일치)
    assert lock.exists(), "새 보유자의 락이 원래 보유자에 의해 삭제됐다"
    assert lock.read_text(encoding="utf-8") == new_token


def test_stale_lock_with_live_pid_is_not_reclaimed(tmp_path):
    """나이만으로 회수하면, 정당하게 오래 걸리거나 디버거에 멈춘 보유자의 락을
    GPU를 쥔 채로 빼앗는다 — 토큰의 pid가 살아있으면 나이가 지나도 회수하지 않는다."""
    lock = tmp_path / "gpu.lock"
    lock.write_text(f"sometoken:{os.getpid()}", encoding="utf-8")  # 이 테스트 프로세스 자신 = 확실히 살아있다
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    with pytest.raises(ResourceBusy):
        with file_lock(lock, timeout=0.0, stale=120.0):
            pass


def test_stale_lock_with_dead_pid_is_reclaimed(tmp_path):
    """죽은 프로세스의 pid는 나이가 지나면 여전히 회수돼야 한다."""
    lock = tmp_path / "gpu.lock"
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    dead_pid = proc.pid  # 이미 종료됐다 — 이 pid는 더 이상 살아있지 않다
    lock.write_text(f"sometoken:{dead_pid}", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    with file_lock(lock, timeout=0.0, stale=120.0):
        pass


def test_stale_lock_with_unparseable_pid_still_reclaims_on_age(tmp_path):
    """토큰이 `uuid:pid` 형식이 아니면(구형 락, 손상된 파일 등) pid를 판단할 수 없다 —
    영원히 막지 않고 기존 나이 기반 회수로 되돌아간다."""
    lock = tmp_path / "gpu.lock"
    lock.write_text("not-a-valid-token", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    with file_lock(lock, timeout=0.0, stale=120.0):
        pass


def test_resource_lock_default_stale_exceeds_registry_stale():
    """자원의 스테일은 레지스트리보다 훨씬 커야 한다. 레지스트리는 밀리초 규모 작업이고,
    GPU는 시간 규모 작업이다."""
    from ai_co_scientist.locks import LOCK_STALE, LOCK_STALE_RESOURCE

    assert LOCK_STALE_RESOURCE > LOCK_STALE, (
        f"LOCK_STALE_RESOURCE({LOCK_STALE_RESOURCE}) should exceed "
        f"LOCK_STALE({LOCK_STALE})"
    )
