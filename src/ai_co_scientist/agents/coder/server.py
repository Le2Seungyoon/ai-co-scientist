"""Coder 에이전트 (M3: self-correct graph) — 메시지 모드."""
import os

from ai_co_scientist.a2a.base import serve
from ai_co_scientist.agents.coder.graph import CoderGraph
from ai_co_scientist.core.config import load_config
from ai_co_scientist.core.failure import AgentTaskFailure, FailureCategory
from ai_co_scientist.core.rules import load_rules
from ai_co_scientist.core.schema import ExperimentDesign, FailureEvent, parse_payload, to_payload

_graph: CoderGraph | None = None


def _get_graph() -> CoderGraph:
    global _graph
    if _graph is None:
        _graph = CoderGraph()
    return _graph


async def handle(payload: dict) -> dict:
    design = parse_payload(payload)
    if not isinstance(design, ExperimentDesign):
        raise ValueError(f"ExperimentDesign 아님: {payload.get('type')}")
    # M2 E2E 계약: 주입된 구현 실패는 graph 진입 전 즉시 신고
    if os.environ.get("COSCIENTIST_INJECT_FAILURE") == "impl_bug":
        raise AgentTaskFailure(to_payload(FailureEvent(
            cycle_id=design.cycle_id, category=FailureCategory.IMPL_BUG,
            detail="주입된 실패: impl_bug")))
    return await _get_graph().run(payload["data"], cycle_id=design.cycle_id)


def main() -> None:
    load_rules("coder")
    serve("coder", "실험설계 구현 담당 (M3 self-correct)",
          load_config()["agents"]["coder"]["port"], handle)


if __name__ == "__main__":
    main()
