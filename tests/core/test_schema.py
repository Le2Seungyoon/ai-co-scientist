import pytest

from ai_co_scientist.core.failure import FailureCategory
from ai_co_scientist.core.schema import (
    CycleContext,
    ExperimentDesign,
    Hypothesis,
    ResearchOutput,
    RunResult,
    FailureEvent,
    parse_payload,
    to_payload,
)


def test_payload_roundtrip_cycle_context():
    ctx = CycleContext(cycle_id=1, prev_verdict_summary="첫 사이클")
    payload = to_payload(ctx)
    assert payload["type"] == "cycle_context"
    restored = parse_payload(payload)
    assert restored == ctx


def test_cycle_context_critique_roundtrip():
    ctx = CycleContext(cycle_id=1, critique="근거 부족")
    assert parse_payload(to_payload(ctx)).critique == "근거 부족"


def test_payload_roundtrip_research_output():
    out = ResearchOutput(
        hypothesis=Hypothesis(
            cycle_id=1, statement="lr을 낮추면 val 손실 감소",
            single_variable="learning_rate", rationale="더미",
        ),
        design=ExperimentDesign(
            cycle_id=1, change="lr 0.1→0.01",
            keep_fixed=["model"], expected_effect="val_mse 감소",
        ),
    )
    restored = parse_payload(to_payload(out))
    assert restored.hypothesis.single_variable == "learning_rate"


def test_run_result_with_failure_event():
    rr = RunResult(
        cycle_id=2, metrics={},
        failure=FailureEvent(cycle_id=2, category=FailureCategory.INFRA_OOM, detail="OOM"),
    )
    restored = parse_payload(to_payload(rr))
    assert restored.failure.category is FailureCategory.INFRA_OOM


def test_parse_unknown_type_raises():
    with pytest.raises(KeyError):
        parse_payload({"type": "nope", "data": {}})


def test_run_result_artifacts_roundtrip():
    rr = RunResult(cycle_id=1, metrics={"val_mse": 0.1},
                   artifacts={"predictions_path": "runtime/p.csv"})
    restored = parse_payload(to_payload(rr))
    assert restored.artifacts["predictions_path"] == "runtime/p.csv"
    assert RunResult(cycle_id=1, metrics={}).artifacts == {}  # 하위호환 기본값
