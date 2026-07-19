import pytest

from ai_co_scientist.agents.coder.graph import CoderGraph
from ai_co_scientist.core.failure import AgentTaskFailure
from ai_co_scientist.core.schema import parse_payload
from ai_co_scientist.llm.router import LLMRouter

DESIGN = {"cycle_id": 1, "change": "lr 조정", "keep_fixed": [], "expected_effect": "개선"}


async def test_good_scenario_produces_artifact_first_try(tmp_path, monkeypatch):
    graph = CoderGraph(router=LLMRouter(scenario="default"))
    result = await graph.run(DESIGN, cycle_id=1)
    artifact = parse_payload(result)
    assert artifact.lint_passed and "attempts=1" in artifact.notes
    assert artifact.entrypoint_path.endswith(".py")


async def test_selfcorrect_scenario_recovers_on_second_attempt():
    graph = CoderGraph(router=LLMRouter(scenario="coder_selfcorrect"))
    result = await graph.run(DESIGN, cycle_id=2)
    artifact = parse_payload(result)
    assert "attempts=2" in artifact.notes  # 1차 버그 → self-correct → 2차 성공


async def test_exhausted_attempts_raise_impl_bug():
    class AlwaysBuggy:
        def invoke(self, role, task_input):
            return {"code": "print(undefined_name)\n"}

    graph = CoderGraph(router=AlwaysBuggy())
    with pytest.raises(AgentTaskFailure) as exc:
        await graph.run(DESIGN, cycle_id=3)
    assert exc.value.payload["data"]["category"] == "impl_bug"
    assert exc.value.payload["data"]["detail"]  # 실패 사유(트레이스백/lint)가 비어있지 않음


async def test_rules_text_is_passed_to_llm():
    captured = {}

    class SpyRouter:
        def invoke(self, role, task_input):
            captured.update(task_input)
            return LLMRouter(scenario="default").invoke(role, task_input)

    graph = CoderGraph(router=SpyRouter())
    await graph.run(DESIGN, cycle_id=1)
    assert "## 역할" in captured.get("rules", "")  # rules/coder.md 전문이 주입됨
