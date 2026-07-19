"""rules/<agent>.md 로더 — 에이전트 기동 시 시스템 프롬프트 앞부분에 주입.

M4: Harness Engineer가 '## 교훈' 섹션에 append할 수 있도록 append_lesson 제공.
rules 디렉토리는 env COSCIENTIST_RULES_DIR가 있으면 그것을 우선하고
(단위/E2E 테스트가 실제 rules/ 디렉토리를 오염시키지 않도록 격리), 없으면 config 경로.
"""
import os
from pathlib import Path

from ai_co_scientist.core.config import load_config, project_root


def _rules_dir() -> Path:
    override = os.environ.get("COSCIENTIST_RULES_DIR")
    if override:
        return Path(override)
    return project_root() / load_config()["paths"]["rules_dir"]


def load_rules(agent: str) -> str:
    path = _rules_dir() / f"{agent}.md"
    if not path.exists():
        raise FileNotFoundError(f"rules 파일 없음: {path}")
    return path.read_text(encoding="utf-8")


def append_lesson(agent: str, lesson: str) -> bool:
    """rules/<agent>.md 교훈 섹션에 append. 동일 항목 라인(`- <lesson>`)이 있으면 False.

    부분 문자열 판정은 실 LLM의 표현 변형에 무력해 정확 라인 매치로 교체(M4→M5 이월 ③).
    """
    path = _rules_dir() / f"{agent}.md"
    text = path.read_text(encoding="utf-8")
    entry = f"- {lesson}"
    if entry in {line.strip() for line in text.splitlines()}:
        return False
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + entry + "\n", encoding="utf-8")
    return True
