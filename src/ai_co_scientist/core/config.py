"""루트 config.yaml 단일 로더. 다른 모듈은 yaml을 직접 열지 않는다."""
import os
import sys
from functools import lru_cache
from pathlib import Path

import yaml


def project_root() -> Path:
    """저장소 루트 (config.yaml이 있는 곳). src/ai_co_scientist/core/ 에서 3단계 위."""
    return Path(__file__).resolve().parents[3]


def ensure_utf8_console() -> None:
    """Windows의 비-UTF8 콘솔 코드페이지(cp949 등)에서 유니코드 문자(—, → 등) print()가
    UnicodeEncodeError로 크래시하는 걸 방지. 프로세스 진입점에서 최초 1회 호출."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(project_root() / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dotenv() -> None:
    """.env → os.environ. 이미 설정된 키는 덮어쓰지 않음 (python-dotenv 없이 최소 구현).

    real 백엔드(M5+)가 토큰류를 읽기 전에 호출한다. mock 경로는 절대 호출하지 않음 —
    .env 파일이 없어도(참가 초기) 전체 사이클이 그대로 돌아야 하므로."""
    env_path = project_root() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()
