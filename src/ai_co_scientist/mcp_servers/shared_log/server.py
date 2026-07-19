"""공유 로그 MCP 서버 (stdio). 모든 에이전트가 이 서버를 서브프로세스로 spawn한다."""
import os

from mcp.server.fastmcp import FastMCP

from ai_co_scientist.core.config import load_config, project_root
from ai_co_scientist.mcp_servers.shared_log import db

mcp = FastMCP("shared-log")


def _db_path() -> str:
    override = os.environ.get("COSCIENTIST_LOG_DB")
    if override:
        return override
    return str(project_root() / load_config()["paths"]["shared_log_db"])


@mcp.tool()
def log_append(
    cycle_id: int, record_type: str, owner: str, content: dict,
    failure_category: str | None = None,
) -> int:
    """원장에 레코드 추가. record_type: hypothesis|result|diagnosis|infra_event. id 반환."""
    return db.append_ledger(
        _db_path(), cycle_id=cycle_id, record_type=record_type, owner=owner,
        content=content, failure_category=failure_category,
    )


@mcp.tool()
def query_ledger(
    cycle_id: int | None = None, record_type: str | None = None,
    owner: str | None = None, limit: int = 50,
) -> list[dict]:
    """원장 조회(최신순). 필터는 모두 선택."""
    return db.query_ledger(
        _db_path(), cycle_id=cycle_id, record_type=record_type, owner=owner, limit=limit,
    )


@mcp.tool()
def get_consensus() -> dict:
    """컨센서스 전체(베스트 파이프라인, 죽은 방향 등) 조회."""
    return db.get_consensus(_db_path())


@mcp.tool()
def update_consensus(key: str, value: dict) -> str:
    """컨센서스 항목 upsert."""
    db.update_consensus(_db_path(), key=key, value=value)
    return "ok"


@mcp.tool()
def update_task_status(
    task_id: str, assignee: str, state: str, blocked_reason: str | None = None,
) -> str:
    """task_status upsert (JIRA류 상태 추적)."""
    db.update_task_status(
        _db_path(), task_id=task_id, assignee=assignee, state=state,
        blocked_reason=blocked_reason,
    )
    return "ok"


if __name__ == "__main__":
    mcp.run()  # stdio transport
