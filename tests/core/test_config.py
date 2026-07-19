from ai_co_scientist.core.config import load_config, project_root


def test_load_config_has_seven_agent_ports():
    cfg = load_config()
    agents = cfg["agents"]
    assert len(agents) == 7
    assert agents["pm"]["port"] == 9001
    assert agents["research"]["port"] == 9002
    ports = [a["port"] for a in agents.values()]
    assert len(set(ports)) == 7  # 포트 중복 없음


def test_paths_section():
    cfg = load_config()
    assert cfg["paths"]["shared_log_db"].endswith(".sqlite3")
    assert (project_root() / "config.yaml").exists()


def test_pm_section():
    cfg = load_config()
    assert cfg["pm"]["coder_retry_limit"] >= 1
    assert cfg["pm"]["replan_limit"] >= 1
    assert cfg["executor_stub"]["run_delay_s"] >= 0


def test_llm_section():
    cfg = load_config()
    assert cfg["llm"]["provider"] == "mock"
    assert cfg["llm"]["scenario"] == "default"
