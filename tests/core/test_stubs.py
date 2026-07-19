import pytest

from ai_co_scientist.core.failure import AgentTaskFailure
from ai_co_scientist.core.schema import (
    CodeArtifact,
    ExperimentDesign,
    RunResult,
    parse_payload,
    to_payload,
)
from ai_co_scientist.core import stubs


def _design(cycle_id=1):
    return to_payload(ExperimentDesign(
        cycle_id=cycle_id, change="lr 변경", keep_fixed=["model"], expected_effect="개선",
    ))


def _artifact(cycle_id=1):
    return to_payload(CodeArtifact(
        cycle_id=cycle_id, entrypoint_path="workspace/x.py", lint_passed=True,
    ))


def test_coder_stub_returns_code_artifact(monkeypatch):
    monkeypatch.delenv("COSCIENTIST_INJECT_FAILURE", raising=False)
    out = parse_payload(stubs.coder_stub(_design()))
    assert isinstance(out, CodeArtifact) and out.cycle_id == 1


def test_coder_stub_injected_impl_bug(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_INJECT_FAILURE", "impl_bug")
    with pytest.raises(AgentTaskFailure) as exc:
        stubs.coder_stub(_design())
    assert exc.value.payload["data"]["category"] == "impl_bug"


def test_coder_stub_ignores_infra_injection(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_INJECT_FAILURE", "infra_oom")
    assert stubs.coder_stub(_design())["type"] == "code_artifact"


def test_executor_stub_injected_oom(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_INJECT_FAILURE", "infra_oom")
    with pytest.raises(AgentTaskFailure) as exc:
        stubs.executor_stub(_artifact())
    assert exc.value.payload["data"]["category"] == "infra_oom"


def test_executor_stub_success(monkeypatch):
    monkeypatch.delenv("COSCIENTIST_INJECT_FAILURE", raising=False)
    out = parse_payload(stubs.executor_stub(_artifact()))
    assert isinstance(out, RunResult) and out.metrics


def test_analysis_and_critic_stubs(monkeypatch):
    monkeypatch.delenv("COSCIENTIST_INJECT_FAILURE", raising=False)
    rr = stubs.executor_stub(_artifact())
    verdict = stubs.analysis_stub(rr)
    assert verdict["type"] == "verdict"
    report = stubs.critic_stub(verdict)
    assert report["type"] == "critique_report" and report["data"]["verdict"] == "pass"
