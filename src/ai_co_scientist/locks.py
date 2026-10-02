"""크로스플랫폼 배타 락 — 자원 하나에 실행 하나.

`flock`은 이 호스트(Windows)에 없다. `O_CREAT|O_EXCL`은 POSIX·Windows 모두에서 원자적이라
별도 의존성 없이 같은 보장을 준다. 획득 루프는 `registry.locked()`가 쓰던 것을 자원 이름으로
일반화한 것이며, **Windows 전용 실패 경로를 그대로 들고 온다** — 8스레드 x 40회 타격에서
재현됐고, 안 잡으면 그 스레드의 작업이 통째로 소실된다.

**기본 경로가 저장소 바깥인 것이 핵심이다.** `project_root()`는 `__file__` 기준이라 워크트리마다
다른 경로를 준다. 레지스트리에는 그것이 의도지만(복제 금지), GPU와 제출 슬롯은 모든 워크트리가
같은 자원을 두고 다투므로 기계 단위 경로여야 한다.
"""
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_STALE = 120.0  # 이보다 오래된 락은 죽은 프로세스가 남긴 것으로 보고 회수한다
_LOCK_DIRNAME = "ai-co-scientist-locks"


class ResourceBusy(RuntimeError):
    """다른 보유자가 자원을 들고 있다. `timeout=0`에서는 즉시 난다."""


@contextmanager
def file_lock(lock_path, *, timeout: float, stale: float = LOCK_STALE):
    """이 경로를 배타적으로 잡는다. 보유 중 경로를 yield한다.

    timeout=0은 큐잉 없이 즉시 실패한다 — `flock -w 0`과 같은 의미다. 못 잡은 쪽은 기다리며
    그렇게 말해야 하고, 다른 자원으로 옮기면 안 된다(그것이 두 작업을 한 카드에 올리는 경로다).
    """
    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            # PermissionError는 **Windows 전용 경로**다. 막 unlink된 파일이 delete-pending
            # 상태면 O_CREAT|O_EXCL이 EEXIST가 아니라 EACCES를 던진다 — POSIX 가정으로 짜면
            # 놓친다. `exists()`와 `stat()` 사이에 해제되면 FileNotFoundError(TOCTOU).
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue  # 방금 해제됐다 — 즉시 재시도
            if age > stale:
                lock.unlink(missing_ok=True)  # 죽은 프로세스가 남긴 락 회수
                continue
            if time.monotonic() >= deadline:
                raise ResourceBusy(f"자원이 사용 중이다({timeout}초 대기): {lock}")
            time.sleep(0.05)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield lock
    finally:
        lock.unlink(missing_ok=True)


def resource_lock(name: str, *, timeout: float = 0.0, root=None):
    """이름 붙은 기계 단위 자원을 잡는다 — `gpu-0`, `dacon-slot`.

    root를 주지 않으면 저장소 바깥(tempfile.gettempdir())에 놓는다. 그래야 워크트리에서 잡은
    락이 main의 실행을 실제로 막는다.
    """
    base = Path(root) if root is not None else Path(tempfile.gettempdir()) / _LOCK_DIRNAME
    return file_lock(base / f"{name}.lock", timeout=timeout)
