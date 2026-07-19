"""PM 사이클 — 정상 경로는 StateGraph 고정 엣지, 예외만 결정적 라우팅 (스펙 §3).

M2에서 LLM 판단 지점 2곳(모호한 재라우팅·에스컬레이션 판단)은 결정적
룰 테이블로 대체돼 있다 — MockLLM 주입은 M3에서.
"""
import sys
from functools import partial
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ai_co_scientist.a2a.client import PMClient
from ai_co_scientist.core.config import load_config
from ai_co_scientist.core.mcp_client import mcp_session, tool_result_data
from ai_co_scientist.core.rules import append_lesson
from ai_co_scientist.core.schema import CycleContext, HarnessProposal, HarnessTrigger, to_payload
from hooks.cycle_log_guard import cycle_log_guard

SHARED_LOG_SERVER = "ai_co_scientist.mcp_servers.shared_log.server"


def detect_harness_triggers(infra_rows: list[dict], escalations_in_window: int,
                            repeat_threshold: int, escalation_threshold: int) -> list[HarnessTrigger]:
    """결정적 트리거 감지 (스펙 §5) — (A) 동일 실패 재발, (B) 에스컬레이션 빈도.

    (A)의 failure_category는 원장에 쌓인 문자열을 그대로 받으므로, FailureCategory에
    없는 미등록 값이 있으면 pydantic ValidationError가 난다 — 그 카테고리만 skip하고
    stderr에 남긴다(스펙 코너케이스 — 원장 오염이 트리거 판단 전체를 막지 않게).
    """
    triggers: list[HarnessTrigger] = []
    counts: dict[str, int] = {}
    for row in infra_rows:
        cat = row.get("failure_category") or ""
        if cat:
            counts[cat] = counts.get(cat, 0) + 1
    for cat, n in counts.items():
        if n >= repeat_threshold:
            try:
                triggers.append(HarnessTrigger(kind="A", failure_category=cat,
                                               escalation_count=0, window=0))
            except ValidationError as e:
                print(f"[pm] 미등록 failure_category '{cat}' — harness 트리거 skip: {e}",
                      file=sys.stderr)
    if escalations_in_window >= escalation_threshold:
        triggers.append(HarnessTrigger(kind="B", failure_category=None,
                                       escalation_count=escalations_in_window, window=0))
    return triggers


class CycleState(TypedDict, total=False):
    cycle_id: int
    infra_feedback: str
    prev_verdict_summary: str
    critique: str
    last_critique: dict
    research_output: dict
    code_artifact: dict
    run_result: dict
    verdict: dict
    failure: dict | None
    coder_retries: int
    replans: int
    research_critic_rounds: int
    coder_critic_rounds: int
    analysis_critic_rounds: int
    outcome: str            # "ok" | "escalated"
    escalation_reason: str
    human_resumes: int
    resume_target: str


async def console_gate(summary: str) -> str:
    """기본 사람 게이트 — 입력 불가(EOF)면 자동 abort (E2E 행 방지).

    sys.stdin.isatty() 대신 EOFError를 직접 잡는다 — Windows CRT는 stdin=DEVNULL도
    character device라 isatty()가 True를 반환해 비대화형 감지가 안 되는 문제가 있었음.
    """
    print(f"[pm][에스컬레이션] {summary}")
    try:
        return input("지시 입력(abort=중단): ")
    except EOFError:
        print("[pm] 입력 불가(EOF) — 자동 'abort'")
        return "abort"


class PMCycle:
    def __init__(self, clients: dict, human_gate=None):
        self._clients = clients
        self._human_gate = human_gate or console_gate
        cfg = load_config()["pm"]
        self._coder_retry_limit = cfg["coder_retry_limit"]
        self._replan_limit = cfg["replan_limit"]
        self._poll_interval = cfg["poll_interval_s"]
        self._poll_timeout = cfg["poll_timeout_s"]
        self._critic_round_limit = cfg["critic_round_limit"]
        self._human_resume_limit = cfg["human_resume_limit"]
        self.graph = self._build()

    # ── 노드 ──────────────────────────────────────────────

    async def _resource_check(self, state: CycleState) -> dict:
        # M2: 자원 확인 자리(항상 통과). M5에서 lightning 크레딧 조회로 대체.
        return {"failure": None}

    async def _dispatch_research(self, state: CycleState) -> dict:
        ctx = CycleContext(
            cycle_id=state["cycle_id"],
            resource_constraints=state.get("infra_feedback", ""),
            prev_verdict_summary=state.get("prev_verdict_summary", ""),
            critique=state.get("critique", ""),
        )
        out = await self._clients["research"].send(to_payload(ctx))
        if out.get("type") == "failure_event":
            return {"failure": out, "research_output": None}
        return {"research_output": out, "failure": None}

    async def _critic_round(self, state: CycleState, draft_key: str) -> dict:
        report = await self._clients["critic"].send(state[draft_key])
        return {"last_critique": report}

    def _after_critic(self, state: CycleState, rounds_key: str,
                       pass_target: str, revise_target: str) -> str:
        verdict = state.get("last_critique", {}).get("data", {}).get("verdict", "pass")
        if verdict != "revise":
            return pass_target
        if state.get(rounds_key, 0) < self._critic_round_limit:
            return revise_target
        return f"residual_{rounds_key}"   # 라운드 소진 → 잔여 공격 기록 후 확정

    async def _revise_prep(self, state: CycleState, rounds_key: str) -> dict:
        attacks = state.get("last_critique", {}).get("data", {}).get("attacks", [])
        updates: dict = {rounds_key: state.get(rounds_key, 0) + 1}
        if rounds_key == "research_critic_rounds":
            updates["critique"] = " / ".join(attacks)   # research는 critique를 입력으로 받음
        return updates

    async def _record_residual(self, state: CycleState, rounds_key: str) -> dict:
        """라운드 소진 시 잔여 공격을 공유로그에 남기고 확정 (스펙 §5 — 흔적 보존)."""
        report = state.get("last_critique", {}).get("data", {})
        try:
            async with mcp_session(SHARED_LOG_SERVER) as session:
                res = await session.call_tool("log_append", {
                    "cycle_id": state["cycle_id"], "record_type": "diagnosis",
                    "owner": "critic",
                    "content": {"unresolved_attacks": report.get("attacks", []),
                                "target": report.get("target", "")},
                })
                if res.isError:  # tool 수준 실패도 isError 응답 — 관측만 (스펙 §9 M3 교훈)
                    print(f"[pm] 잔여 공격 기록 실패(무시): {res.content[0].text if res.content else ''}",
                          file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — 기록 실패 격리
            print(f"[pm] 잔여 공격 기록 실패(무시): {e}", file=sys.stderr)
        return {}

    async def _dispatch_coder(self, state: CycleState) -> dict:
        design_payload = {
            "type": "experiment_design",
            "data": state["research_output"]["data"]["design"],
        }
        result = await self._clients["coder"].send(design_payload)
        if result["type"] == "failure_event":
            return {"failure": result,
                    "coder_retries": state.get("coder_retries", 0) + 1}
        return {"code_artifact": result, "failure": None}

    async def _dispatch_executor(self, state: CycleState) -> dict:
        client = self._clients["executor"]
        task_id = await client.submit(state["code_artifact"])
        outcome = await client.poll(
            task_id, interval=self._poll_interval, timeout=self._poll_timeout)
        if outcome.state != "completed":
            failure = outcome.payload if outcome.payload is not None else {
                "type": "failure_event",
                "data": {"cycle_id": state["cycle_id"], "category": "unknown",
                         "detail": f"executor {outcome.state}"},
            }
            category = failure.get("data", {}).get("category", "")
            updates: dict = {"failure": failure}
            if category != "impl_bug":
                updates["replans"] = state.get("replans", 0) + 1
            return updates
        return {"run_result": outcome.payload, "failure": None}

    async def _infra_feedback(self, state: CycleState) -> dict:
        detail = state["failure"]["data"].get("detail", "인프라 실패")
        return {"infra_feedback":
                f"직전 실행 인프라 실패: {detail}. 자원 제약을 반영해 재설계할 것.",
                "coder_retries": 0, "critique": ""}  # 새 설계는 깨끗하게 — 통과된 draft에 대한 이전 critic 공격을 재생하지 않는다 (게이트 재개와 동일 원칙)

    async def _dispatch_analysis(self, state: CycleState) -> dict:
        result = await self._clients["analysis"].send(state["run_result"])
        if result.get("type") == "failure_event":
            return {"failure": result, "verdict": None}
        return {"verdict": result, "failure": None}

    async def _impl_bug_feedback(self, state: CycleState) -> dict:
        """실행 실패를 Coder에 재지시 — MockLLM 호출 인덱스가 진행되며 새 코드 생성(M3)."""
        return {"coder_retries": state.get("coder_retries", 0) + 1, "failure": None}

    async def _check_cycle_log(self, state: CycleState) -> dict:
        """decide 진입 전 필수 레코드 확인 — 누락은 관측 기록(차단 아님, M4 단순화)."""
        try:
            async with mcp_session(SHARED_LOG_SERVER) as session:
                res = await session.call_tool("query_ledger", {"cycle_id": state["cycle_id"]})
                if res.isError:  # isError 응답 확인 (스펙 §9 M3 교훈)
                    print("[pm] cycle_log_guard 조회 실패 — 판정 보류", file=sys.stderr)
                    return {}
                rows = tool_result_data(res)
                if isinstance(rows, dict):
                    rows = rows.get("result", rows)
                ok, missing = cycle_log_guard(rows if isinstance(rows, list) else [],
                                              state["cycle_id"])
                if not ok:
                    print(f"[pm] 필수 레코드 누락: {missing}", file=sys.stderr)
                    res2 = await session.call_tool("log_append", {
                        "cycle_id": state["cycle_id"], "record_type": "infra_event",
                        "owner": "pm", "content": {"missing_records": missing},
                        "failure_category": "logic_inconsistent",
                    })
                    if res2.isError:
                        print("[pm] 누락 기록 실패(무시)", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — guard 실패가 사이클을 막지 않게 격리
            print(f"[pm] cycle_log_guard 실패(무시): {e}", file=sys.stderr)
        return {}

    async def _decide(self, state: CycleState) -> dict:
        diagnosis = state.get("verdict", {}).get("data", {}).get("diagnosis", "")
        return {"outcome": "ok", "prev_verdict_summary": diagnosis}

    async def _escalate(self, state: CycleState) -> dict:
        """반복 실패의 사람 게이트 — 응답 기반 재개(M4). abort/한도 소진 시 종료."""
        reason = f"cycle {state['cycle_id']}: 반복 실패 — {state.get('failure')}"
        answer = (await self._human_gate(reason)).strip().lower()
        resumes = state.get("human_resumes", 0)
        if answer == "abort" or resumes >= self._human_resume_limit:
            return {"outcome": "escalated", "escalation_reason": reason, "resume_target": ""}
        category = (state.get("failure") or {}).get("data", {}).get("category", "")
        target = "dispatch_coder" if category == "impl_bug" else "dispatch_research"
        return {"human_resumes": resumes + 1, "coder_retries": 0, "replans": 0,
                "failure": None, "resume_target": target,
                "infra_feedback": "", "critique": ""}  # 재개는 깨끗한 재설계 — 이전 라운드의 잔존 컨텍스트를 재생하지 않는다

    # ── 조건부 엣지 ────────────────────────────────────────

    def _after_coder(self, state: CycleState) -> str:
        if state.get("failure") is None:
            return "critic_coder"
        if state["coder_retries"] <= self._coder_retry_limit:
            return "dispatch_coder"          # 구현 실패 → Coder 재지시
        return "escalate"

    def _after_research(self, state: CycleState) -> str:
        return "critic_research" if state.get("failure") is None else "escalate"

    def _after_analysis(self, state: CycleState) -> str:
        return "critic_analysis" if state.get("failure") is None else "escalate"

    def _after_executor(self, state: CycleState) -> str:
        if state.get("failure") is None:
            return "dispatch_analysis"
        category = state["failure"].get("data", {}).get("category", "")
        if category == "impl_bug":
            # 실행 중 드러난 구현 실패 → Coder 재지시 (스펙 §5). 한도는 coder_retries 공유
            # 증가 전 검사이므로 한도식이 직접 실패 경로(증가 후 <= limit)와 같아지려면 < limit — 양 경로 모두 총 3회 dispatch 후 에스컬레이션
            if state.get("coder_retries", 0) < self._coder_retry_limit:
                return "impl_bug_feedback"
            return "escalate"
        if state["replans"] <= self._replan_limit:
            return "infra_feedback"          # 인프라 실패 → Research 피드백
        return "escalate"

    def _after_escalate(self, state: CycleState) -> str:
        return state.get("resume_target") or "__end__"

    # ── 조립 ──────────────────────────────────────────────

    def _build(self):
        builder = StateGraph(CycleState)
        builder.add_node("resource_check", self._resource_check)
        builder.add_node("dispatch_research", self._dispatch_research)
        builder.add_node("critic_research",
                         partial(self._critic_round, draft_key="research_output"))
        builder.add_node("revise_research",
                         partial(self._revise_prep, rounds_key="research_critic_rounds"))
        builder.add_node("residual_research_critic_rounds",
                         partial(self._record_residual, rounds_key="research_critic_rounds"))
        builder.add_node("dispatch_coder", self._dispatch_coder)
        builder.add_node("critic_coder",
                         partial(self._critic_round, draft_key="code_artifact"))
        builder.add_node("revise_coder",
                         partial(self._revise_prep, rounds_key="coder_critic_rounds"))
        builder.add_node("residual_coder_critic_rounds",
                         partial(self._record_residual, rounds_key="coder_critic_rounds"))
        builder.add_node("dispatch_executor", self._dispatch_executor)
        builder.add_node("infra_feedback", self._infra_feedback)
        builder.add_node("impl_bug_feedback", self._impl_bug_feedback)
        builder.add_node("dispatch_analysis", self._dispatch_analysis)
        builder.add_node("critic_analysis",
                         partial(self._critic_round, draft_key="verdict"))
        builder.add_node("revise_analysis",
                         partial(self._revise_prep, rounds_key="analysis_critic_rounds"))
        builder.add_node("residual_analysis_critic_rounds",
                         partial(self._record_residual, rounds_key="analysis_critic_rounds"))
        builder.add_node("check_cycle_log", self._check_cycle_log)
        builder.add_node("decide", self._decide)
        builder.add_node("escalate", self._escalate)

        builder.add_edge(START, "resource_check")
        builder.add_edge("resource_check", "dispatch_research")
        builder.add_conditional_edges("dispatch_research", self._after_research)
        builder.add_conditional_edges("critic_research", partial(
            self._after_critic, rounds_key="research_critic_rounds",
            pass_target="dispatch_coder", revise_target="revise_research"))
        builder.add_edge("revise_research", "dispatch_research")
        builder.add_edge("residual_research_critic_rounds", "dispatch_coder")
        builder.add_conditional_edges("dispatch_coder", self._after_coder)
        builder.add_conditional_edges("critic_coder", partial(
            self._after_critic, rounds_key="coder_critic_rounds",
            pass_target="dispatch_executor", revise_target="revise_coder"))
        builder.add_edge("revise_coder", "dispatch_coder")
        builder.add_edge("residual_coder_critic_rounds", "dispatch_executor")
        builder.add_conditional_edges("dispatch_executor", self._after_executor)
        builder.add_edge("infra_feedback", "dispatch_research")
        builder.add_edge("impl_bug_feedback", "dispatch_coder")
        builder.add_conditional_edges("dispatch_analysis", self._after_analysis)
        builder.add_conditional_edges("critic_analysis", partial(
            self._after_critic, rounds_key="analysis_critic_rounds",
            pass_target="check_cycle_log", revise_target="revise_analysis"))
        builder.add_edge("revise_analysis", "dispatch_analysis")
        builder.add_edge("residual_analysis_critic_rounds", "check_cycle_log")
        builder.add_edge("check_cycle_log", "decide")
        builder.add_edge("decide", END)
        builder.add_conditional_edges("escalate", self._after_escalate, {
            "dispatch_coder": "dispatch_coder",
            "dispatch_research": "dispatch_research",
            "__end__": END,
        })
        return builder.compile()


def count_escalation_events(results: list[dict], window: int) -> int:
    """최근 window 사이클의 에스컬레이션 이벤트 수 — 사람이 재개시킨 것(human_resumes)도 이벤트로 센다.

    최종 outcome만 세면 run_cycles가 첫 escalated에서 중단되므로 트리거 B가 도달 불가(리뷰 지적).
    """
    total = 0
    for r in results[-window:]:
        total += int(r.get("human_resumes", 0) or 0)
        if r.get("outcome") == "escalated":
            total += 1
    return total


async def _query_infra_events() -> list[dict]:
    """공유로그 infra_event 최근 20건 조회 — 실패/예외는 격리(빈 리스트 반환)."""
    try:
        async with mcp_session(SHARED_LOG_SERVER) as session:
            res = await session.call_tool(
                "query_ledger", {"record_type": "infra_event", "limit": 20})
            if res.isError:  # tool 수준 실패도 isError 응답 (스펙 §9 M3 교훈)
                print("[pm] infra_event 조회 실패 — harness 트리거 판단 보류", file=sys.stderr)
                return []
            rows = tool_result_data(res)
            if isinstance(rows, dict):
                rows = rows.get("result", rows)
            return rows if isinstance(rows, list) else []
    except Exception as e:  # noqa: BLE001 — 조회 실패가 사이클을 막지 않게 격리
        print(f"[pm] infra_event 조회 실패(무시): {e}", file=sys.stderr)
        return []


async def _log_harness_action(cycle_id: int, proposal: HarnessProposal) -> None:
    """harness 제안 반영을 공유로그에 diagnosis로 기록 — 실패는 격리(무시)."""
    try:
        async with mcp_session(SHARED_LOG_SERVER) as session:
            res = await session.call_tool("log_append", {
                "cycle_id": cycle_id, "record_type": "diagnosis",
                "owner": "harness_engineer",
                "content": {"agent": proposal.agent, "lesson": proposal.lesson},
            })
            if res.isError:
                print(f"[pm] harness 액션 기록 실패(무시): {res.content[0].text if res.content else ''}",
                      file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — 기록 실패 격리
        print(f"[pm] harness 액션 기록 실패(무시): {e}", file=sys.stderr)


async def run_cycles(n: int) -> list[dict]:
    """외곽 드라이버 — n 사이클 순환, 에스컬레이션 시 중단."""
    cfg = load_config()
    clients = {
        name: PMClient(f"http://127.0.0.1:{agent['port']}")
        for name, agent in cfg["agents"].items() if name != "pm"
    }
    pm = PMCycle(clients)
    results: list[dict] = []
    carry: dict = {}
    fired_triggers: set[tuple[str, str]] = set()  # run당 (kind, category) 1회만 발화(스팸 방지)
    try:
        for cycle_id in range(1, n + 1):
            state: CycleState = {
                "cycle_id": cycle_id, "coder_retries": 0, "replans": 0,
                "research_critic_rounds": 0, "coder_critic_rounds": 0,
                "analysis_critic_rounds": 0, **carry,
            }
            final = await pm.graph.ainvoke(state)
            print(f"[pm] cycle {cycle_id} outcome={final['outcome']}")
            results.append(dict(final))

            # Harness 트리거 감지 (결정적) — 사이클 종료마다 확인
            window = cfg["harness"]["escalation_window"]
            recent_escalations = count_escalation_events(results, window)
            infra_rows = await _query_infra_events()
            for trigger in detect_harness_triggers(
                    infra_rows, recent_escalations,
                    cfg["harness"]["failure_repeat_threshold"],
                    cfg["harness"]["escalation_threshold"]):
                key = (trigger.kind, str(trigger.failure_category or ""))
                if key in fired_triggers:
                    continue
                fired_triggers.add(key)
                # harness는 진단 부채널 — 실패가 사이클 실행을 막지 않는다
                try:
                    proposal_payload = await clients["harness_engineer"].send(to_payload(trigger))
                    if proposal_payload.get("type") != "harness_proposal":
                        continue
                    proposal = HarnessProposal.model_validate(proposal_payload["data"])
                    if append_lesson(proposal.agent, proposal.lesson):
                        print(f"[pm] harness 교훈 반영: {proposal.agent} ← {proposal.lesson}")
                        await _log_harness_action(cycle_id, proposal)
                except Exception as e:  # noqa: BLE001 — harness 호출 실패 격리
                    print(f"[pm] harness 호출 실패(무시): {e}", file=sys.stderr)

            if final["outcome"] == "escalated":
                break
            carry = {"prev_verdict_summary": final.get("prev_verdict_summary", "")}
    finally:
        for client in clients.values():
            await client.aclose()
    return results
