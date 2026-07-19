from ai_co_scientist.core.config import project_root
from hooks.import_boundary import check_import_boundary


def test_current_codebase_is_clean():
    violations = check_import_boundary(str(project_root() / "src" / "ai_co_scientist"))
    assert violations == []


def test_detects_cross_agent_import(tmp_path):
    root = tmp_path / "ai_co_scientist"
    bad = root / "agents" / "coder"
    bad.mkdir(parents=True)
    (bad / "server.py").write_text(
        "from ai_co_scientist.agents.research.graph import ResearchGraph\n", encoding="utf-8")
    violations = check_import_boundary(str(root))
    assert violations and "agents.research" in violations[0]


def test_detects_client_import_outside_pm(tmp_path):
    root = tmp_path / "ai_co_scientist"
    bad = root / "agents" / "analysis"
    bad.mkdir(parents=True)
    (bad / "server.py").write_text("from ai_co_scientist.a2a.client import PMClient\n", encoding="utf-8")
    violations = check_import_boundary(str(root))
    assert violations and "a2a.client" in violations[0]


def test_detects_relative_cross_agent_import(tmp_path):
    root = tmp_path / "ai_co_scientist"
    bad = root / "agents" / "coder"
    bad.mkdir(parents=True)
    (bad / "server.py").write_text(
        "from ..research.graph import ResearchGraph\n", encoding="utf-8")
    violations = check_import_boundary(str(root))
    assert violations and "research" in violations[0]


def test_detects_package_then_attribute_import(tmp_path):
    root = tmp_path / "ai_co_scientist"
    bad = root / "agents" / "coder"
    bad.mkdir(parents=True)
    (bad / "server.py").write_text(
        "from ai_co_scientist.agents import research\n", encoding="utf-8")
    violations = check_import_boundary(str(root))
    assert violations and "research" in violations[0]


def test_detects_a2a_client_via_package_import(tmp_path):
    root = tmp_path / "ai_co_scientist"
    bad = root / "agents" / "analysis"
    bad.mkdir(parents=True)
    (bad / "server.py").write_text(
        "from ai_co_scientist.a2a import client\n", encoding="utf-8")
    violations = check_import_boundary(str(root))
    assert violations and "client" in violations[0]
