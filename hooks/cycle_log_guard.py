"""사이클 종료 가드 — 필수 원장 레코드 존재 확인 (LLM 없음, 순수 함수).

조회는 호출자(PM)가 수행해 레코드 목록을 넘긴다 — hook은 판정만 (경계 규율).
"""
REQUIRED_TYPES = ("hypothesis", "diagnosis")


def cycle_log_guard(records: list[dict], cycle_id: int) -> tuple[bool, list[str]]:
    present = {
        r.get("record_type") for r in records if r.get("cycle_id") == cycle_id
    }
    missing = [t for t in REQUIRED_TYPES if t not in present]
    return (not missing, missing)
