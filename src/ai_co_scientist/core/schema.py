"""에이전트 간 메시지 계약. A2A DataPart(JSON)로 to_payload/parse_payload 왕복."""
from pydantic import BaseModel

from ai_co_scientist.core.failure import FailureCategory


class CycleContext(BaseModel):
    """PM→Research 입력: 직전 판정 + 자원 제약 + 컨센서스 요약 + (M4) critic 공격."""
    cycle_id: int
    prev_verdict_summary: str = ""
    resource_constraints: str = ""
    consensus_summary: str = ""
    critique: str = ""


class Hypothesis(BaseModel):
    cycle_id: int
    statement: str
    single_variable: str  # 단일 변인 (스펙 §4 리서치)
    rationale: str


class ExperimentDesign(BaseModel):
    cycle_id: int
    change: str
    keep_fixed: list[str]
    expected_effect: str


class ResearchOutput(BaseModel):
    """Research의 A2A 응답 묶음."""
    hypothesis: Hypothesis
    design: ExperimentDesign


class CodeArtifact(BaseModel):
    cycle_id: int
    entrypoint_path: str
    lint_passed: bool
    notes: str = ""


class FailureEvent(BaseModel):
    cycle_id: int
    category: FailureCategory
    detail: str


class RunResult(BaseModel):
    cycle_id: int
    metrics: dict[str, float]
    failure: FailureEvent | None = None
    artifacts: dict[str, str] = {}


class Verdict(BaseModel):
    cycle_id: int
    case_findings: list[str]
    improved: bool
    overfitting_suspected: bool
    diagnosis: str


class CritiqueReport(BaseModel):
    target: str  # 공격 대상 draft의 payload type (예: research_output, code_artifact, verdict)
    attacks: list[str]
    verdict: str  # "pass" | "revise"


class HarnessTrigger(BaseModel):
    kind: str  # "A"(동일 실패 재발) | "B"(에스컬레이션 빈도 초과)
    failure_category: FailureCategory | None = None
    escalation_count: int = 0
    window: int = 0


class HarnessProposal(BaseModel):
    """Harness Engineer의 응답 — 대상 에이전트 rules에 append할 교훈 한 줄."""
    agent: str
    lesson: str


_PAYLOADS: dict[str, type[BaseModel]] = {
    "cycle_context": CycleContext,
    "hypothesis": Hypothesis,
    "experiment_design": ExperimentDesign,
    "research_output": ResearchOutput,
    "code_artifact": CodeArtifact,
    "failure_event": FailureEvent,
    "run_result": RunResult,
    "verdict": Verdict,
    "critique_report": CritiqueReport,
    "harness_trigger": HarnessTrigger,
    "harness_proposal": HarnessProposal,
}
_TYPE_BY_MODEL = {model: name for name, model in _PAYLOADS.items()}


def to_payload(model: BaseModel) -> dict:
    """A2A DataPart에 실을 dict: {"type": ..., "data": ...}."""
    return {"type": _TYPE_BY_MODEL[type(model)], "data": model.model_dump(mode="json")}


def parse_payload(payload: dict) -> BaseModel:
    """to_payload의 역변환. 미등록 type이면 KeyError."""
    return _PAYLOADS[payload["type"]].model_validate(payload["data"])
