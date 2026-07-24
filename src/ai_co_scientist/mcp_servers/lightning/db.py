"""lightning mock 저장소 — 잡 실행 결과 + 크레딧 (SQLite)."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    metrics TEXT,
    artifacts TEXT NOT NULL,
    detail TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS credits (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    remaining REAL NOT NULL
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
        try:
            con.execute("ALTER TABLE jobs ADD COLUMN deadline REAL")
        except sqlite3.OperationalError:
            pass  # 이미 마이그레이션됨(컬럼 존재)
        yield con
        con.commit()
    finally:
        con.close()


def init_credits(db_path: str, initial: float) -> None:
    """credits 행이 없을 때만 초기값으로 기록(이미 있으면 그대로 둔다)."""
    with _connect(db_path) as con:
        row = con.execute("SELECT remaining FROM credits WHERE id = 1").fetchone()
        if row is None:
            con.execute("INSERT INTO credits (id, remaining) VALUES (1, ?)", (initial,))


def get_credits(db_path: str) -> float:
    with _connect(db_path) as con:
        row = con.execute("SELECT remaining FROM credits WHERE id = 1").fetchone()
    return row["remaining"] if row is not None else 0.0


def deduct(db_path: str, amount: float) -> bool:
    """크레딧 차감 — 조건부 UPDATE 단일 문으로 프로세스 간 원자성 보장 (잔액 부족 시 False)."""
    with _connect(db_path) as con:
        cur = con.execute(
            "UPDATE credits SET remaining = remaining - ? WHERE id = 1 AND remaining >= ?",
            (amount, amount))
        return cur.rowcount > 0


def save_job(
    db_path: str, job_id: str, status: str, metrics: dict | None, artifacts: dict, detail: str,
    deadline: float | None = None,
) -> None:
    with _connect(db_path) as con:
        con.execute(
            "INSERT OR REPLACE INTO jobs (job_id, status, metrics, artifacts, detail, ts, deadline) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                status,
                json.dumps(metrics, ensure_ascii=False) if metrics is not None else None,
                json.dumps(artifacts, ensure_ascii=False),
                detail,
                time.time(),
                deadline,
            ),
        )


def save_running(db_path: str, job_id: str, deadline: float) -> None:
    """real 백엔드용 — 원격 잡 제출 직후 논터미널 상태로 기록(폴링 대상)."""
    save_job(db_path, job_id, "running", None, {}, "", deadline=deadline)


def get_job(db_path: str, job_id: str) -> dict | None:
    with _connect(db_path) as con:
        row = con.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return {
        "status": row["status"],
        "metrics": json.loads(row["metrics"]) if row["metrics"] is not None else None,
        "artifacts": json.loads(row["artifacts"]),
        "detail": row["detail"],
        "deadline": row["deadline"],
    }
