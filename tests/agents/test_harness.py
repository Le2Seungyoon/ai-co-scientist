from ai_co_scientist.agents.harness_engineer.server import handle as harness_handle
from ai_co_scientist.agents.pm.cycle import count_escalation_events, detect_harness_triggers
from ai_co_scientist.core.schema import HarnessTrigger, parse_payload, to_payload


async def test_harness_handle_returns_proposal():
    trigger = HarnessTrigger(kind="A", failure_category="infra_oom", escalation_count=0, window=0)
    out = await harness_handle(to_payload(trigger))
    proposal = parse_payload(out)
    assert out["type"] == "harness_proposal"
    assert proposal.agent == "executor" and proposal.lesson


def test_detect_repeat_failure_trigger():
    rows = [{"failure_category": "infra_oom"}] * 3
    triggers = detect_harness_triggers(rows, escalations_in_window=0,
                                       repeat_threshold=2, escalation_threshold=2)
    assert any(t.kind == "A" and t.failure_category == "infra_oom" for t in triggers)


def test_detect_escalation_frequency_trigger():
    triggers = detect_harness_triggers([], escalations_in_window=2,
                                       repeat_threshold=2, escalation_threshold=2)
    assert any(t.kind == "B" for t in triggers)


def test_no_trigger_below_thresholds():
    rows = [{"failure_category": "infra_oom"}]
    assert detect_harness_triggers(rows, escalations_in_window=1,
                                   repeat_threshold=2, escalation_threshold=2) == []


def test_escalation_events_include_resumed():
    results = [{"outcome": "escalated", "human_resumes": 1}]   # 재개 1 + 최종 abort 1
    assert count_escalation_events(results, window=5) == 2      # 단일 사이클로 임계 도달


def test_escalation_events_window():
    results = [{"outcome": "ok", "human_resumes": 1}] * 6
    assert count_escalation_events(results, window=5) == 5      # window 밖 제외
