"""M2 배선 검증용 더미 응답 생성기 — M3에서 각 에이전트의 실제 graph로 교체된다.

실패 주입: env COSCIENTIST_INJECT_FAILURE=<FailureCategory값>.
각 스텁은 자기 담당 카테고리일 때만 발화한다 (coder=impl_bug, executor=infra_*).
"""
import os

from ai_co_scientist.core.failure import AgentTaskFailure, FailureCategory
from ai_co_scientist.core.schema import (
    CodeArtifact,
    CritiqueReport,
    FailureEvent,
    RunResult,
    Verdict,
    parse_payload,
    to_payload,
)

_INFRA = {
    FailureCategory.INFRA_OOM,
    FailureCategory.INFRA_TIMEOUT,
    FailureCategory.INFRA_CREDIT,
}


def _injected(cycle_id: int) -> FailureEvent | None:
    mode = os.environ.get("COSCIENTIST_INJECT_FAILURE", "")
    if not mode:
        return None
    return FailureEvent(
        cycle_id=cycle_id, category=FailureCategory(mode), detail=f"주입된 실패: {mode}",
    )


def coder_stub(payload: dict) -> dict:
    design = parse_payload(payload)
    fail = _injected(design.cycle_id)
    if fail and fail.category is FailureCategory.IMPL_BUG:
        raise AgentTaskFailure(to_payload(fail))
    return to_payload(CodeArtifact(
        cycle_id=design.cycle_id,
        entrypoint_path="workspace/dummy_experiment.py",
        lint_passed=True,
        notes="M2 스텁 구현",
    ))


def executor_stub(payload: dict) -> dict:
    artifact = parse_payload(payload)
    fail = _injected(artifact.cycle_id)
    if fail and fail.category in _INFRA:
        raise AgentTaskFailure(to_payload(fail))
    return to_payload(RunResult(cycle_id=artifact.cycle_id, metrics={"val_mse": 0.42}))


def analysis_stub(payload: dict) -> dict:
    rr = parse_payload(payload)
    return to_payload(Verdict(
        cycle_id=rr.cycle_id,
        case_findings=["더미: 이전 실패 케이스 2건 중 1건 개선"],
        improved=True,
        overfitting_suspected=False,
        diagnosis="M2 스텁 진단",
    ))


def critic_stub(payload: dict) -> dict:
    return to_payload(CritiqueReport(
        target=payload.get("type", "unknown"), attacks=[], verdict="pass",
    ))
