"""Critic 에이전트 (M4: MockLLM 기반 공격) — 메시지 모드. 공격만 하고 산출물은 만들지 않는다."""
from ai_co_scientist.a2a.base import serve
from ai_co_scientist.core.config import load_config
from ai_co_scientist.core.rules import load_rules
from ai_co_scientist.core.schema import CritiqueReport, to_payload
from ai_co_scientist.llm.router import LLMRouter

_router: LLMRouter | None = None
_rules: str | None = None


def _get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


def _get_rules() -> str:
    global _rules
    if _rules is None:
        _rules = load_rules("critic")
    return _rules


async def handle(payload: dict) -> dict:
    out = _get_router().invoke("critic", {
        "draft_type": payload.get("type", "unknown"),
        "draft": payload.get("data", {}),
        "rules": _get_rules(),
    })
    return to_payload(CritiqueReport(
        target=out["target"], attacks=out["attacks"], verdict=out["verdict"]))


def main() -> None:
    load_rules("critic")
    serve("critic", "산출물 공격 담당 (M4 MockLLM)",
          load_config()["agents"]["critic"]["port"], handle)


if __name__ == "__main__":
    main()
