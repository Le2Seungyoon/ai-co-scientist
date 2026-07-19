from ai_co_scientist.agents.executor import logic
from ai_co_scientist.core.failure import FailureCategory


def test_classify_statuses():
    assert logic.classify_status("completed") is None
    assert logic.classify_status("timeout") is FailureCategory.INFRA_TIMEOUT
    assert logic.classify_status("oom") is FailureCategory.INFRA_OOM
    assert logic.classify_status("failed") is FailureCategory.IMPL_BUG
    assert logic.classify_status("unknown") is FailureCategory.UNKNOWN


def test_recovery_rule_only_for_timeout():
    assert FailureCategory.INFRA_TIMEOUT in logic.RECOVERY_RULES
    assert FailureCategory.INFRA_OOM not in logic.RECOVERY_RULES
    assert FailureCategory.INFRA_CREDIT not in logic.RECOVERY_RULES
