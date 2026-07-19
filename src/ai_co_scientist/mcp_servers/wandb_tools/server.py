"""wandb mock MCP 서버 (stdio) — 정량 실험 기록 (스펙 §4 지원 인프라).

real 백엔드는 골격만(M5) — 검증은 사람 몫.
"""
import os

from mcp.server.fastmcp import FastMCP

from ai_co_scientist.core.config import load_config, project_root
from ai_co_scientist.mcp_servers.wandb_tools import db

mcp = FastMCP("wandb-tools")


def _db_path() -> str:
    return os.environ.get("COSCIENTIST_WANDB_DB") or str(
        project_root() / load_config()["paths"]["wandb_db"])


def _real_log_metrics(run_id: str, metrics: dict) -> None:
    # TODO(M5+): wandb SDK 연동 — wandb.init(project=...), wandb.log(metrics)
    raise NotImplementedError("real wandb 백엔드는 골격만 (검증은 사람 몫)")


@mcp.tool()
def log_metrics(run_id: str, cycle_id: int, metrics: dict) -> int:
    """실험 메트릭 기록. mock: SQLite / real: TODO."""
    return db.log_metrics(_db_path(), run_id=run_id, cycle_id=cycle_id, metrics=metrics)


@mcp.tool()
def query_runs(cycle_id: int | None = None, limit: int = 50) -> list[dict]:
    """기록된 run 조회(최신순)."""
    return db.query_runs(_db_path(), cycle_id=cycle_id, limit=limit)


if __name__ == "__main__":
    mcp.run()
