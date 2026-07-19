"""Executor 로직 — 상태 분류와 복구 룰 (실행 유틸은 M5에서 lightning 서버로 이사)."""
from ai_co_scientist.core.failure import FailureCategory

RECOVERY_RULES: dict[FailureCategory, dict] = {
    # 인프라 실패 중 timeout만 자동 복구 시도 — OOM/크레딧은 PM(재설계/대기) 소관
    FailureCategory.INFRA_TIMEOUT: {"retries": 1, "timeout_multiplier": 2.0},
}

_STATUS_MAP = {
    "completed": None,
    "timeout": FailureCategory.INFRA_TIMEOUT,
    "oom": FailureCategory.INFRA_OOM,
    "failed": FailureCategory.IMPL_BUG,   # exit≠0/출력 계약 위반 — Coder 재지시 대상
}


def classify_status(status: str) -> FailureCategory | None:
    return _STATUS_MAP.get(status, FailureCategory.UNKNOWN)
