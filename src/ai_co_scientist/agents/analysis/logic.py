"""Analysis 판정 로직 — M3: 결정적 룰 (케이스 단위 LLM 판정은 실 LLM 도입 시)."""
from ai_co_scientist.core.schema import RunResult, Verdict


def build_verdict(run_result: RunResult, best: dict | None, overfit_gap: float) -> Verdict:
    metrics = run_result.metrics
    if "val_mse" not in metrics:
        raise ValueError(f"val_mse 없음: {sorted(metrics)}")
    val = metrics["val_mse"]
    bv = best.get("val_mse", float("inf")) if best else float("inf")
    improved = val < bv
    overfit = "train_mse" in metrics and (val - metrics["train_mse"]) > overfit_gap
    findings = [
        f"val_mse={val:.4f} (베스트 {bv:.4f})" if best
        else f"val_mse={val:.4f} (첫 실험 — 비교 기준 없음)",
    ]
    diagnosis = f"{'개선' if improved else '미개선'}: val_mse {val:.4f}"
    if overfit:
        diagnosis += " — 오버피팅 의심(train/val 격차), 제출 확인 필요(M5)"
    return Verdict(
        cycle_id=run_result.cycle_id,
        case_findings=findings,
        improved=improved,
        overfitting_suspected=overfit,
        diagnosis=diagnosis,
    )
