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
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

LOCK_STALE = 120.0  # 레지스트리용. 이보다 오래된 락은 죽은 프로세스가 남긴 것으로 보고 회수한다
LOCK_STALE_RESOURCE = 6 * 3600  # GPU 같은 자원용. 학습은 1.5-2시간 지속되므로 6시간이 합리적이다
_LOCK_DIRNAME = "ai-co-scientist-locks"
GPU_LOCK = "gpu-0"
DACON_LOCK = "dacon-slot"


class ResourceBusy(RuntimeError):
    """다른 보유자가 자원을 들고 있다. `timeout=0`에서는 즉시 난다."""


def _parse_token(token: str) -> "tuple[int, str] | None":
    """신형 uuid:pid:start와 구형 uuid:pid를 모두 읽는다."""
    parts = token.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        pid = int(parts[1])
    except ValueError:
        return None
    if not 0 < pid <= 0xFFFFFFFF:
        return None
    return pid, parts[2] if len(parts) == 3 else ""


def _linux_process_identity(pid: int) -> "tuple[bool, int | None]":
    """comm 안의 괄호를 건너뛰고 /proc stat의 22번째 필드(starttime)를 읽는다."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False, None
    except OSError:
        return True, None  # 조회 불가를 사망으로 오인하지 않는다
    try:
        return True, int(raw[raw.rfind(")") + 2:].split()[19])
    except (ValueError, IndexError):
        return True, None


def _windows_process_identity(pid: int) -> "tuple[bool, int | None]":
    """정확한 PID의 실행 상태와 FILETIME 생성시각을 조회한다 (문자열 부분일치 금지)."""
    import ctypes
    from ctypes import wintypes

    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        time_ptr = ctypes.POINTER(wintypes.FILETIME)
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [time_ptr] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87, None  # ERROR_INVALID_PARAMETER만 사망
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True, None
            if exit_code.value != 259:  # STILL_ACTIVE; 종료 뒤 핸들만 남은 프로세스 제외
                return False, None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return True, None
            return True, (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        finally:
            kernel.CloseHandle(handle)
    except (OSError, AttributeError, ValueError):
        return True, None


def _process_identity(pid: int) -> "tuple[bool, int | None]":
    """같은 PID라도 시작시각이 다르면 재사용된 별개 프로세스다."""
    if sys.platform == "win32":
        return _windows_process_identity(pid)
    if sys.platform.startswith("linux"):
        return _linux_process_identity(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, None
    except OSError:
        return True, None
    return True, None


def _pid_alive(pid: int) -> bool:
    return _process_identity(pid)[0]


def _token_still_alive(token: str) -> bool:
    """읽었던 바로 그 토큰의 소유자만 검사한다. 조회 불가는 보수적으로 점유로 본다."""
    parsed = _parse_token(token)
    if parsed is None:
        return False
    pid, start = parsed
    alive, actual = _process_identity(pid)
    if not alive:
        return False
    try:
        expected = int(start)
    except ValueError:
        return True  # 구형·시작시각 없는 토큰은 PID 생존으로 판단
    return actual is None or actual == expected


@contextmanager
def file_lock(lock_path, *, timeout: float, stale: float = LOCK_STALE):
    """이 경로를 배타적으로 잡는다. 보유 중 경로를 yield한다.

    timeout=0은 큐잉 없이 즉시 실패한다 — `flock -w 0`과 같은 의미다. 못 잡은 쪽은 기다리며
    그렇게 말해야 하고, 다른 자원으로 옮기면 안 된다(그것이 두 작업을 한 카드에 올리는 경로다).

    스테일 기간을 초과한 락은 회수되므로, 스테일 기간은 정당한 보유 시간보다 길어야 한다.
    레지스트리의 LOCK_STALE(120초)는 밀리초 규모 read-modify-write에 적합하다.
    GPU 같은 자원의 LOCK_STALE_RESOURCE(6시간)는 1.5-2시간 학습에 대비한다.
    """
    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    _, start = _process_identity(os.getpid())
    token = f"{uuid.uuid4().hex}:{os.getpid()}:{start if start is not None else ''}"
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            # PermissionError는 **Windows 전용 경로**다. 막 unlink된 파일이 delete-pending
            # 상태면 O_CREAT|O_EXCL이 EEXIST가 아니라 EACCES를 던진다 — POSIX 가정으로 짜면
            # 놓친다. `exists()`와 `stat()` 사이에 해제되면 FileNotFoundError(TOCTOU).
            try:
                age = time.time() - lock.stat().st_mtime
                # 새 락을 읽는 핸들은 Windows에서 소유자의 unlink를 방해할 수 있다.
                judged_token = (lock.read_text(encoding="utf-8", errors="replace").strip()
                                if age > stale else "")
            except FileNotFoundError:
                continue  # 방금 해제됐다 — 즉시 재시도
            except OSError:
                age, judged_token = 0.0, ""  # 못 읽으면 회수하지 않고 데드라인을 검사
            if age > stale and not _token_still_alive(judged_token):
                try:
                    current = lock.read_text(encoding="utf-8", errors="replace").strip()
                    # 비교와 unlink 사이의 작은 경쟁 창은 남는다 (원자적 compare-delete 없음).
                    if current == judged_token:
                        lock.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass  # 재확인·삭제 불가도 대기 한도 안에서 ResourceBusy로 종료
            # age > stale인데 보유자가 살아있으면 회수하지 않는다 — 정당하게 오래 걸리거나
            # 디버거에 멈춘 보유자가 자원을 쥔 채로 새 보유자와 부딪히는 것을 막는다.
            if time.monotonic() >= deadline:
                raise ResourceBusy(f"자원이 사용 중이다({timeout}초 대기): {lock}")
            time.sleep(0.05)
    try:
        payload = token.encode()
        if os.write(fd, payload) != len(payload):
            raise OSError("incomplete lock token write")
    except OSError:
        os.close(fd)
        lock.unlink(missing_ok=True)
        raise
    os.close(fd)
    try:
        yield lock
    finally:
        # 이 획득이 여전히 락 파일을 소유하는지 확인하고, 맞을 때만 unlink한다.
        # 스테일 판정으로 회수되면 다른 진행이 지금 이 파일을 들고 있다.
        try:
            # errors="replace": 이 디렉터리는 기계 전역 공유 temp라 외부 프로세스가 남긴
            # non-UTF-8 바이트가 닿을 수 있다. UnicodeDecodeError(ValueError)는 OSError가
            # 아니라 아래 except를 빠져나가 본문 예외를 가리고 락을 남긴다 — 실측 결함.
            held_token = lock.read_text(encoding="utf-8", errors="replace").strip()
            if held_token == token:
                lock.unlink(missing_ok=True)
            else:
                # 우리가 못 잡은 사이에 다른 진행이 획득했다. unlink하면 그것을 깨트린다.
                print(f"경고: 락이 회수됨 (보유 중): {lock}", file=sys.stderr)
        except OSError:
            # 읽기 실패는 예외로 전파하지 않는다. 이미 해제됐을 수 있다.
            # FileNotFoundError는 OSError의 하위 클래스라 따로 잡지 않는다.
            pass


def resource_lock(name: str, *, timeout: float = 0.0, root=None,
                  stale: float = LOCK_STALE_RESOURCE):
    """이름 붙은 기계 단위 자원을 잡는다 — `gpu-0`, `dacon-slot`.

    root를 주지 않으면 저장소 바깥(tempfile.gettempdir())에 놓는다. 그래야 워크트리에서 잡은
    락이 main의 실행을 실제로 막는다.

    stale 기본값은 LOCK_STALE_RESOURCE(6시간)다. 학습 같은 오래 걸리는 작업이 정상적으로
    진행하는 동안 회수되지 않도록 충분히 크다.
    """
    base = Path(root) if root is not None else Path(tempfile.gettempdir()) / _LOCK_DIRNAME
    return file_lock(base / f"{name}.lock", timeout=timeout, stale=stale)
