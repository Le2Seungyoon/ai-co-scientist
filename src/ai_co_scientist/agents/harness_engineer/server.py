"""Harness Engineer 에이전트 (M4: MockLLM 기반 결정적 트리거 대응) — 메시지 모드.

HarnessTrigger를 받아 대상 에이전트와 교훈 한 줄을 담은 HarnessProposal을 반환한다.
실제 rules 파일 append는 호출측(pm.cycle.run_cycles)의 몫 — 여기는 제안만 한다.
"""
from ai_co_scientist.a2a.base import serve
from ai_co_scientist.core.config import load_config
from ai_co_scientist.core.rules import load_rules
from ai_co_scientist.core.schema import HarnessProposal, HarnessTrigger, to_payload
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
        _rules = load_rules("harness_engineer")
    return _rules


async def handle(payload: dict) -> dict:
    trigger = HarnessTrigger.model_validate(payload.get("data", {}))
    out = _get_router().invoke("harness_engineer", {
        "kind": trigger.kind,
        "failure_category": str(trigger.failure_category or ""),
        "escalation_count": trigger.escalation_count,
        "rules": _get_rules(),
    })
    return to_payload(HarnessProposal(agent=out["agent"], lesson=out["lesson"]))


def main() -> None:
    load_rules("harness_engineer")
    serve("harness_engineer", "rules/hook 제안 담당 (M4 MockLLM)",
          load_config()["agents"]["harness_engineer"]["port"], handle)


if __name__ == "__main__":
    main()
