"""자원 락 — 배타성과 스테일 회수를 **행동으로** 검사한다 (오프라인, 합성 경로).

`flock`은 이 호스트에 없다. `O_CREAT|O_EXCL`이 그 자리를 채우므로, 이 파일이 재는 것은
"두 번째 획득이 실제로 거부되는가"와 "죽은 보유자의 락이 회수되는가"다.
"""
import os
import time
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

    with resource_lock("probe-default-root") as held:
        assert Path(tempfile.gettempdir()).resolve() in Path(held).resolve().parents
        assert project_root() not in Path(held).resolve().parents


def test_resource_lock_excludes_by_name(tmp_path):
    with resource_lock("gpu-0", root=tmp_path):
        with pytest.raises(ResourceBusy):
            with resource_lock("gpu-0", root=tmp_path):
                pass
        with resource_lock("dacon-slot", root=tmp_path):  # 다른 자원은 막히지 않는다
            pass
