"""wandb mock 저장소 — 실험 메트릭 기록 (SQLite)."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    cycle_id INTEGER NOT NULL,
    metrics TEXT NOT NULL,
    ts REAL NOT NULL
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


def log_metrics(db_path: str, *, run_id: str, cycle_id: int, metrics: dict) -> int:
    with _connect(db_path) as con:
        cur = con.execute(
            "INSERT INTO runs (run_id, cycle_id, metrics, ts) VALUES (?, ?, ?, ?)",
            (run_id, cycle_id, json.dumps(metrics, ensure_ascii=False), time.time()))
        return cur.lastrowid


def query_runs(db_path: str, *, cycle_id: int | None = None, limit: int = 50) -> list[dict]:
    where = "WHERE cycle_id = ?" if cycle_id is not None else ""
    params = [cycle_id] if cycle_id is not None else []
    with _connect(db_path) as con:
        rows = con.execute(
            f"SELECT * FROM runs {where} ORDER BY id DESC LIMIT ?", (*params, limit)
        ).fetchall()
    return [{**dict(r), "metrics": json.loads(r["metrics"])} for r in rows]
