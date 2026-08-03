"""루트 config.yaml 단일 로더. 다른 모듈은 yaml을 직접 열지 않는다."""
import os
import sys
from functools import lru_cache
from pathlib import Path

import yaml


def project_root() -> Path:
    """저장소 루트 (config.yaml이 있는 곳). src/ai_co_scientist/ 에서 2단계 위."""
    return Path(__file__).resolve().parents[2]


def ensure_utf8_console() -> None:
    """Windows의 비-UTF8 콘솔(cp949 등)에서 유니코드 print가 UnicodeEncodeError로
    죽는 걸 방지. 프로세스 진입점에서 최초 1회 호출."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(project_root() / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dotenv() -> None:
    """.env → os.environ. 이미 설정된 키는 덮어쓰지 않음.

    실 백엔드(DACON/Lightning)만 호출한다 — 테스트는 .env 없이도 전부 통과해야 한다."""
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
