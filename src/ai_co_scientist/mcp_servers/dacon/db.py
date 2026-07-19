"""dacon mock 저장소 — 제출 이력(public/private 점수) (SQLite)."""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    submission_id TEXT PRIMARY KEY,
    cycle_id INTEGER NOT NULL,
    public REAL,
    private REAL,
    status TEXT NOT NULL DEFAULT 'scored',
    detail TEXT NOT NULL DEFAULT '',
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


def save_submission(
    db_path: str, *, submission_id: str, cycle_id: int,
    public: float | None, private: float | None,
    status: str = "scored", detail: str = "",
) -> None:
    with _connect(db_path) as con:
        con.execute(
            "INSERT OR REPLACE INTO submissions "
            "(submission_id, cycle_id, public, private, status, detail, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (submission_id, cycle_id, public, private, status, detail, time.time()))


def get_submission(db_path: str, submission_id: str) -> dict | None:
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT * FROM submissions WHERE submission_id = ?", (submission_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def list_submissions(db_path: str) -> list[dict]:
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT * FROM submissions ORDER BY ts DESC"
        ).fetchall()
    return [dict(r) for r in rows]
