"""공유 로그 저장소 — SQLite(WAL). 원장 + 컨센서스 + task_status (스펙 §4).

여러 에이전트 프로세스의 MCP 인스턴스가 한 파일을 공유하므로,
커넥션은 호출마다 열고 닫는다(WAL + timeout으로 동시성 처리).
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('hypothesis', 'result', 'diagnosis', 'infra_event')),
    owner TEXT NOT NULL,
    failure_category TEXT,
    content TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS consensus (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS task_status (
    task_id TEXT PRIMARY KEY,
    assignee TEXT NOT NULL,
    state TEXT NOT NULL,
    blocked_reason TEXT,
    updated_ts REAL NOT NULL
);
"""


@contextmanager
def _connect(db_path: str):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30.0)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(_SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def append_ledger(
    db_path: str, *, cycle_id: int, record_type: str, owner: str,
    content: dict, failure_category: str | None = None,
) -> int:
    with _connect(db_path) as con:
        cur = con.execute(
            "INSERT INTO ledger (cycle_id, type, owner, failure_category, content, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cycle_id, record_type, owner, failure_category,
             json.dumps(content, ensure_ascii=False), time.time()),
        )
        return cur.lastrowid


def query_ledger(
    db_path: str, *, cycle_id: int | None = None, record_type: str | None = None,
    owner: str | None = None, limit: int = 50,
) -> list[dict]:
    clauses, params = [], []
    if cycle_id is not None:
        clauses.append("cycle_id = ?")
        params.append(cycle_id)
    if record_type is not None:
        clauses.append("type = ?")
        params.append(record_type)
    if owner is not None:
        clauses.append("owner = ?")
        params.append(owner)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(db_path) as con:
        rows = con.execute(
            f"SELECT * FROM ledger {where} ORDER BY id DESC LIMIT ?", (*params, limit)
        ).fetchall()
    results = []
    for r in rows:
        row = dict(r)
        row["record_type"] = row.pop("type")
        row["content"] = json.loads(row["content"])
        results.append(row)
    return results


def update_consensus(db_path: str, *, key: str, value: dict) -> None:
    with _connect(db_path) as con:
        con.execute(
            "INSERT INTO consensus (key, value, updated_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts",
            (key, json.dumps(value, ensure_ascii=False), time.time()),
        )


def get_consensus(db_path: str) -> dict[str, dict]:
    with _connect(db_path) as con:
        rows = con.execute("SELECT key, value FROM consensus").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


def update_task_status(
    db_path: str, *, task_id: str, assignee: str, state: str,
    blocked_reason: str | None = None,
) -> None:
    with _connect(db_path) as con:
        con.execute(
            "INSERT INTO task_status (task_id, assignee, state, blocked_reason, updated_ts) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(task_id) DO UPDATE SET assignee = excluded.assignee, "
            "state = excluded.state, blocked_reason = excluded.blocked_reason, "
            "updated_ts = excluded.updated_ts",
            (task_id, assignee, state, blocked_reason, time.time()),
        )


def get_task_status(db_path: str, *, task_id: str) -> dict | None:
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT * FROM task_status WHERE task_id = ?", (task_id,)
        ).fetchone()
    return dict(row) if row else None
