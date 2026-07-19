"""에이전트 경계 규율의 결정적 검사 (AST) — 스펙 §2 경계 규율의 hook 화.

규칙: agents/<name>/ 은 core/llm/hooks/a2a(base)/자기 패키지만 import.
다른 에이전트 패키지·mcp_servers 직접 import 금지, a2a.client는 pm 전용.

우회 방지: relative import(``from ..research import x``)와 패키지-후-속성
import(``from ai_co_scientist.agents import research``, ``from ai_co_scientist import
mcp_servers`` 등)도 절대 경로로 환산해 동일 규칙을 적용한다.
"""
import ast
from pathlib import Path

# from <base> import <name> 형태에서 <name>이 실제 대상을 가리는 패키지들.
# 이 경로들 자체는 검사하지 않고, import된 각 alias.name을 이어붙여 검사한다.
_PACKAGE_ATTR_BASES = {"ai_co_scientist", "ai_co_scientist.agents", "ai_co_scientist.a2a"}


def _is_ai_co_scientist(mod: str) -> bool:
    return mod == "ai_co_scientist" or mod.startswith("ai_co_scientist.")


def _check_module(mod: str, agent: str, file: Path) -> str | None:
    """완전한(절대) 모듈 경로 하나에 대해 경계 규칙을 적용한다."""
    if mod == "ai_co_scientist.agents":
        return f"{file}: agents 패키지 포괄 import — {mod} (구체 에이전트 지정 필요)"
    if mod.startswith("ai_co_scientist.agents.") and not mod.startswith(
            f"ai_co_scientist.agents.{agent}"):
        return f"{file}: 다른 에이전트 import — {mod} (agents.{agent}에서)"
    if mod.startswith("ai_co_scientist.mcp_servers"):
        return f"{file}: mcp_servers 직접 import — {mod} (spawn 경로 문자열만 허용)"
    if mod.startswith("ai_co_scientist.a2a.client") and agent != "pm":
        return f"{file}: a2a.client import — {mod} (PM 전용)"
    return None


def _file_package(file: Path, src_root: Path) -> str:
    """file이 속한 패키지의 dotted 경로 (예: .../ai_co_scientist/agents/coder/server.py
    → "ai_co_scientist.agents.coder"). __init__.py든 일반 모듈이든 __package__와 동일하다."""
    base = src_root.parent if src_root.name == "ai_co_scientist" else src_root
    rel = file.relative_to(base)
    return ".".join(rel.parts[:-1])


def _resolve_import_from(node: ast.ImportFrom, file_package: str) -> str:
    """relative import(level>0)를 파일의 패키지 위치 기준으로 절대 경로로 환산한다.
    (importlib._bootstrap._resolve_name과 동일한 알고리즘)"""
    if node.level == 0:
        return node.module or ""
    bits = file_package.rsplit(".", node.level - 1)
    base = bits[0]
    return f"{base}.{node.module}" if node.module else base


def _violations_for(file: Path, agent: str, src_root: Path) -> list[str]:
    tree = ast.parse(file.read_text(encoding="utf-8"))
    file_package = _file_package(file, src_root)
    found: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_ai_co_scientist(alias.name):
                    continue
                violation = _check_module(alias.name, agent, file)
                if violation:
                    found.append(violation)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from(node, file_package)
            if not _is_ai_co_scientist(resolved):
                continue
            if resolved in _PACKAGE_ATTR_BASES:
                # from ai_co_scientist[.agents|.a2a] import <name> — <name>을 이어붙여 검사.
                for alias in node.names:
                    full = f"{resolved}.{alias.name}"
                    violation = _check_module(full, agent, file)
                    if violation:
                        found.append(violation)
            else:
                violation = _check_module(resolved, agent, file)
                if violation:
                    found.append(violation)
    return found


def check_import_boundary(src_root: str) -> list[str]:
    agents_dir = Path(src_root) / "agents"
    violations: list[str] = []
    if not agents_dir.exists():
        return violations
    for agent_pkg in sorted(agents_dir.iterdir()):
        if not agent_pkg.is_dir():
            continue
        for file in sorted(agent_pkg.rglob("*.py")):
            violations.extend(_violations_for(file, agent_pkg.name, Path(src_root)))
    return violations
