"""제출 가드 — 로컬 점수가 개선일 때만 제출 허용 (스펙 §5 hook, LLM 없음)."""


def submission_guard(val_mse: float, best: dict | None) -> tuple[bool, str]:
    if best is None:
        return True, "첫 실험 — 비교 기준 없음, 제출 허용"
    if val_mse < best.get("val_mse", float("inf")):
        return True, "로컬 개선 확인 — 제출 허용"
    return False, "로컬 개선 없음 — 제출 차단"
