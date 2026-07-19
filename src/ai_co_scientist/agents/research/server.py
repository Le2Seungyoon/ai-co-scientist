"""Research 에이전트 (M3: 컨센서스 기반 가설 graph) — 메시지 모드."""
from ai_co_scientist.a2a.base import serve
from ai_co_scientist.agents.research.graph import ResearchGraph
from ai_co_scientist.core.config import load_config
from ai_co_scientist.core.rules import load_rules
from ai_co_scientist.core.schema import CycleContext, parse_payload

_graph: ResearchGraph | None = None


def _get_graph() -> ResearchGraph:
    global _graph
    if _graph is None:
        _graph = ResearchGraph()
    return _graph


async def handle(payload: dict) -> dict:
    ctx = parse_payload(payload)
    if not isinstance(ctx, CycleContext):
        # python -O에서도 살아있는 명시적 검증 (assert는 -O에서 제거됨)
        raise ValueError(f"CycleContext 아님: {payload.get('type')}")
    return await _get_graph().run(ctx)


def main() -> None:
    load_rules("research")
    port = load_config()["agents"]["research"]["port"]
    serve("research", "가설·실험설계 담당 (M3 graph)", port, handle)


if __name__ == "__main__":
    main()
