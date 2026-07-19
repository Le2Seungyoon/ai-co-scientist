import sqlite3

import pytest

from ai_co_scientist.mcp_servers.shared_log import db


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "log.sqlite3")


def test_append_and_query_ledger(db_path):
    rid = db.append_ledger(
        db_path, cycle_id=1, record_type="hypothesis", owner="research",
        content={"statement": "lr을 낮추면 개선"},
    )
    assert rid == 1
    rows = db.query_ledger(db_path, record_type="hypothesis")
    assert len(rows) == 1
    assert rows[0]["owner"] == "research"
    assert rows[0]["content"]["statement"] == "lr을 낮추면 개선"
    assert rows[0]["record_type"] == "hypothesis"
    assert "type" not in rows[0]


def test_query_filters_by_cycle_and_owner(db_path):
    db.append_ledger(db_path, cycle_id=1, record_type="result", owner="executor", content={})
    db.append_ledger(db_path, cycle_id=2, record_type="result", owner="executor", content={})
    db.append_ledger(db_path, cycle_id=2, record_type="diagnosis", owner="analysis", content={})
    assert len(db.query_ledger(db_path, cycle_id=2)) == 2
    assert len(db.query_ledger(db_path, cycle_id=2, owner="analysis")) == 1


def test_invalid_record_type_rejected(db_path):
    with pytest.raises(sqlite3.IntegrityError):
        db.append_ledger(db_path, cycle_id=1, record_type="oops", owner="x", content={})


def test_failure_category_tag(db_path):
    db.append_ledger(
        db_path, cycle_id=3, record_type="infra_event", owner="executor",
        content={"detail": "OOM"}, failure_category="infra_oom",
    )
    rows = db.query_ledger(db_path, record_type="infra_event")
    assert rows[0]["failure_category"] == "infra_oom"


def test_consensus_upsert_and_get(db_path):
    db.update_consensus(db_path, key="best_pipeline", value={"desc": "baseline"})
    db.update_consensus(db_path, key="best_pipeline", value={"desc": "baseline+lr"})
    assert db.get_consensus(db_path)["best_pipeline"]["desc"] == "baseline+lr"


def test_task_status_roundtrip(db_path):
    db.update_task_status(db_path, task_id="t-1", assignee="coder", state="working")
    st = db.get_task_status(db_path, task_id="t-1")
    assert st["state"] == "working"
    assert db.get_task_status(db_path, task_id="none") is None
