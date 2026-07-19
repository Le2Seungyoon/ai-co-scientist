"""M1 walking skeleton — PM이 Research에 CycleContext를 보내고 응답을 받는다."""
from ai_co_scientist.a2a.client import PMClient
from ai_co_scientist.core.config import load_config
from ai_co_scientist.core.schema import CycleContext, parse_payload, to_payload


async def run_skeleton() -> dict:
    port = load_config()["agents"]["research"]["port"]
    client = PMClient(f"http://127.0.0.1:{port}")

    try:
        ctx = CycleContext(cycle_id=1, consensus_summary="(빈 컨센서스 — 첫 사이클)")
        result = await client.send(to_payload(ctx))

        out = parse_payload(result)
        print(f"[pm] {result['type']} 수신: 가설='{out.hypothesis.statement}'")
        return result
    finally:
        await client.aclose()
