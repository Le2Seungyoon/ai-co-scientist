"""실패 카테고리 — 공유로그 태그이자 Harness Engineer 트리거 키 (스펙 §5)."""
from enum import StrEnum


class FailureCategory(StrEnum):
    IMPL_BUG = "impl_bug"                  # 구현 실패 (Coder 담당)
    INFRA_TIMEOUT = "infra_timeout"        # 인프라 실패 (Executor 담당)
    INFRA_OOM = "infra_oom"
    INFRA_CREDIT = "infra_credit"
    LOGIC_INCONSISTENT = "logic_inconsistent"  # Critic이 잡는 일관성 위반
    UNKNOWN = "unknown"


class AgentTaskFailure(Exception):
    """에이전트가 작업 실패를 구조적으로 신고할 때 raise.

    payload는 to_payload(FailureEvent) 형식. 메시지 모드 서버는 이를
    failure_event payload로 응답하고, task 모드 서버는 TASK_STATE_FAILED의
    status.message에 담는다 (사유 없는 FAILED 방지 — scout §5).
    """

    def __init__(self, payload: dict):
        super().__init__(payload.get("data", {}).get("detail", "agent failure"))
        self.payload = payload


class AgentInputRequired(Exception):
    """에이전트가 사람 입력을 요구할 때 raise. payload는 상황 설명 dict."""

    def __init__(self, payload: dict):
        super().__init__("input required")
        self.payload = payload
