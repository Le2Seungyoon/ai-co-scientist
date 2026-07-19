import pytest

from ai_co_scientist.agents.analysis.logic import build_verdict
from ai_co_scientist.core.schema import RunResult


def _rr(metrics):
    return RunResult(cycle_id=1, metrics=metrics)


def test_first_result_is_improvement():
    v = build_verdict(_rr({"val_mse": 0.3}), best=None, overfit_gap=0.1)
    assert v.improved and not v.overfitting_suspected


def test_worse_than_best_not_improved():
    v = build_verdict(_rr({"val_mse": 0.4}), best={"val_mse": 0.3}, overfit_gap=0.1)
    assert not v.improved
    assert "미개선" in v.diagnosis


def test_overfit_gap_triggers_suspicion():
    v = build_verdict(_rr({"train_mse": 0.05, "val_mse": 0.3}), best=None, overfit_gap=0.1)
    assert v.overfitting_suspected
    assert "M5" in v.diagnosis  # 제출 확인은 M5로 표기


def test_missing_val_mse_raises():
    with pytest.raises(ValueError):
        build_verdict(_rr({"accuracy": 0.9}), best=None, overfit_gap=0.1)


def test_malformed_best_without_val_mse_no_keyerror():
    v = build_verdict(_rr({"val_mse": 0.3}), best={"cycle_id": 1}, overfit_gap=0.1)
    assert v.improved  # 알 수 없는 베스트는 inf 취급 — KeyError 없이 동작
