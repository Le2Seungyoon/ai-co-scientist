import pytest

from ai_co_scientist.llm.router import LLMRouter


def test_research_default_rotates_single_variable():
    router = LLMRouter(scenario="default")
    seen = []
    for cycle in range(1, 4):
        out = router.invoke("research", {"cycle_id": cycle, "consensus_summary": ""})
        assert out["hypothesis"]["cycle_id"] == cycle
        assert out["design"]["cycle_id"] == cycle
        var = out["hypothesis"]["single_variable"]
        assert var not in out["design"]["keep_fixed"]  # 단일 변인은 고정 목록과 배타
        seen.append(var)
    assert len(set(seen)) == 3  # 사이클마다 다른 변인


def test_coder_default_improves_metric_each_call():
    router = LLMRouter(scenario="default")
    first = router.invoke("coder", {"design": {}, "error": ""})["code"]
    second = router.invoke("coder", {"design": {}, "error": ""})["code"]
    assert "val_mse" in first and first != second  # 호출마다 다른(개선된) 코드


def test_coder_selfcorrect_first_buggy_then_good():
    router = LLMRouter(scenario="coder_selfcorrect")
    buggy = router.invoke("coder", {"design": {}, "error": ""})["code"]
    fixed = router.invoke("coder", {"design": {}, "error": "NameError"})["code"]
    assert "train_mse =" not in buggy   # 1차: 정의 없이 사용(의도적 버그)
    assert "train_mse =" in fixed       # 2차: 정상


def test_env_scenario_override(monkeypatch):
    monkeypatch.setenv("COSCIENTIST_MOCKLLM_SCENARIO", "coder_slow")
    router = LLMRouter()
    assert "time.sleep" in router.invoke("coder", {"design": {}, "error": ""})["code"]


def test_unknown_role_raises():
    router = LLMRouter(scenario="default")
    with pytest.raises(KeyError):
        router.invoke("nobody", {})


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        LLMRouter(scenario="nope")


def test_critic_default_always_pass():
    router = LLMRouter(scenario="default")
    for _ in range(3):
        out = router.invoke("critic", {"draft_type": "verdict", "draft": {}})
        assert out["verdict"] == "pass" and out["attacks"] == []


def test_critic_revise_scenario_revises_once():
    router = LLMRouter(scenario="critic_revise")
    first = router.invoke("critic", {"draft_type": "research_output", "draft": {}})
    second = router.invoke("critic", {"draft_type": "research_output", "draft": {}})
    assert first["verdict"] == "revise" and first["attacks"]
    assert second["verdict"] == "pass"


def test_gemini_provider_without_key_raises(monkeypatch):
    from ai_co_scientist.core import config as config_module
    cfg = {**config_module.load_config()}
    cfg["llm"] = {**cfg["llm"], "provider": "gemini"}
    monkeypatch.setattr("ai_co_scientist.llm.router.load_config", lambda: cfg)
    # 빈 값으로 선점 — load_dotenv는 기존 키를 덮지 않으므로 .env가 있어도 결정적
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        LLMRouter()


def test_env_provider_override_selects_gemini(monkeypatch):
    # config는 mock이어도 COSCIENTIST_LLM_PROVIDER=gemini가 우선 (실 기동 스위치)
    from ai_co_scientist.llm.gemini import GeminiRouter
    monkeypatch.setenv("COSCIENTIST_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key-no-network")
    router = LLMRouter()
    assert isinstance(router._delegate, GeminiRouter)  # 생성만 — invoke(네트워크)는 안 함


def test_unknown_provider_raises(monkeypatch):
    from ai_co_scientist.core import config as config_module
    cfg = {**config_module.load_config()}
    cfg["llm"] = {**cfg["llm"], "provider": "nope"}
    monkeypatch.setattr("ai_co_scientist.llm.router.load_config", lambda: cfg)
    with pytest.raises(ValueError):
        LLMRouter()
