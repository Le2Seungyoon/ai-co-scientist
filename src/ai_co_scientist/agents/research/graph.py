"""Research graph — 공유로그 컨텍스트 수집 → MockLLM 가설 생성 → 단일 변인 검증 → 기록."""
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ai_co_scientist.core.failure import AgentTaskFailure, FailureCategory
from ai_co_scientist.core.mcp_client import mcp_session, tool_result_data
from ai_co_scientist.core.rules import load_rules
from ai_co_scientist.core.schema import (
    CycleContext,
    ExperimentDesign,
    FailureEvent,
    Hypothesis,
    ResearchOutput,
    to_payload,
)

SHARED_LOG_SERVER = "ai_co_scientist.mcp_servers.shared_log.server"


class ResearchState(TypedDict, total=False):
    ctx: dict            # CycleContext dump
    consensus_summary: str
    recent_diagnoses: str
    hypothesis: dict
    design: dict
    validation_error: str
    regenerated: bool
    output: dict | None
    failure: dict | None


class ResearchGraph:
    def __init__(self, router=None):
        from ai_co_scientist.llm.router import LLMRouter

        self._router = router or LLMRouter()
        self._rules = load_rules("research")
        self._graph = self._build()

    async def run(self, ctx: CycleContext) -> dict:
        final = await self._graph.ainvoke({"ctx": ctx.model_dump(mode="json")})
        if final.get("failure") is not None:
            raise AgentTaskFailure(final["failure"])
        return final["output"]

    async def _gather(self, state: ResearchState) -> dict:
        """공유로그에서 컨센서스·최근 진단 조회 — 조회 실패는 격리(빈 컨텍스트로 진행)."""
        consensus, diagnoses = "", ""
        try:
            async with mcp_session(SHARED_LOG_SERVER) as session:
                # tool 수준 실패는 예외가 아니라 isError 응답 — 명시 확인해야 격리됨
                res = await session.call_tool("get_consensus", {})
                if not res.isError:
                    data = tool_result_data(res)
                    if isinstance(data, dict):
                        data = data.get("result", data)
                    if data:
                        consensus = str(data)
                res = await session.call_tool(
                    "query_ledger", {"record_type": "diagnosis", "limit": 3})
                if not res.isError:
                    rows = tool_result_data(res)
                    if isinstance(rows, dict):
                        rows = rows.get("result", rows)
                    if isinstance(rows, list) and rows:
                        diagnoses = " / ".join(
                            str(r["content"].get("diagnosis", "")) for r in rows)
        except Exception as e:  # noqa: BLE001 — 조회 실패가 가설 생성을 막지 않게 격리
            print(f"[research] 공유로그 조회 실패(빈 컨텍스트로 진행): {e}", file=sys.stderr)
        return {"consensus_summary": consensus, "recent_diagnoses": diagnoses}

    async def _generate(self, state: ResearchState) -> dict:
        ctx = state["ctx"]
        out = self._router.invoke("research", {
            "cycle_id": ctx["cycle_id"],
            "consensus_summary": state.get("consensus_summary", "") or ctx.get("consensus_summary", ""),
            "recent_diagnoses": state.get("recent_diagnoses", ""),
            "resource_constraints": ctx.get("resource_constraints", ""),
            "rules": self._rules,
        })
        return {"hypothesis": out["hypothesis"], "design": out["design"]}

    async def _validate(self, state: ResearchState) -> dict:
        """단일 변인 하드 제약 (rules/research.md) — 결정적 검사."""
        var = state["hypothesis"].get("single_variable", "")
        keep_fixed = state["design"].get("keep_fixed", [])
        if not var:
            return {"validation_error": "single_variable 비어 있음"}
        if var in keep_fixed:
            return {"validation_error": f"단일 변인 '{var}'이 keep_fixed에 포함됨"}
        return {"validation_error": ""}

    async def _record(self, state: ResearchState) -> dict:
        hypothesis = Hypothesis.model_validate(state["hypothesis"])
        design = ExperimentDesign.model_validate(state["design"])
        try:
            async with mcp_session(SHARED_LOG_SERVER) as session:
                res = await session.call_tool("log_append", {
                    "cycle_id": hypothesis.cycle_id, "record_type": "hypothesis",
                    "owner": "research",
                    "content": hypothesis.model_dump(mode="json"),
                })
                # tool 수준 실패도 isError 응답 — 격리 유지, 관측만 (스펙 §9 M3 교훈)
                if res.isError:
                    print(f"[research] log_append tool 실패(무시): "
                          f"{res.content[0].text if res.content else ''}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — 기록 실패 격리 (M2 이월 ①)
            print(f"[research] 공유로그 기록 실패(무시): {e}", file=sys.stderr)
        return {"output": to_payload(ResearchOutput(hypothesis=hypothesis, design=design))}

    async def _fail(self, state: ResearchState) -> dict:
        failure = FailureEvent(
            cycle_id=state["ctx"]["cycle_id"],
            category=FailureCategory.LOGIC_INCONSISTENT,
            detail=f"단일 변인 검증 실패(재생성 후에도): {state['validation_error']}",
        )
        return {"failure": to_payload(failure)}

    def _after_validate(self, state: ResearchState) -> str:
        if not state.get("validation_error"):
            return "record"
        if not state.get("regenerated"):
            return "mark_regenerated"
        return "fail"

    async def _mark_regenerated(self, state: ResearchState) -> dict:
        return {"regenerated": True}

    def _build(self):
        builder = StateGraph(ResearchState)
        builder.add_node("gather", self._gather)
        builder.add_node("generate", self._generate)
        builder.add_node("validate", self._validate)
        builder.add_node("mark_regenerated", self._mark_regenerated)
        builder.add_node("record", self._record)
        builder.add_node("fail", self._fail)
        builder.add_edge(START, "gather")
        builder.add_edge("gather", "generate")
        builder.add_edge("generate", "validate")
        builder.add_conditional_edges("validate", self._after_validate)
        builder.add_edge("mark_regenerated", "generate")
        builder.add_edge("record", END)
        builder.add_edge("fail", END)
        return builder.compile()
