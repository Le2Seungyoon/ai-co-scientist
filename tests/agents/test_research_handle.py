import os
from unittest.mock import patch

import pytest

from ai_co_scientist.agents.research.server import handle
from ai_co_scientist.core.schema import CycleContext, Hypothesis, to_payload, parse_payload
from ai_co_scientist.mcp_servers.shared_log import db


async def test_handle_returns_research_output_and_logs(tmp_path):
    db_file = str(tmp_path / "log.sqlite3")
    with patch.dict(os.environ, {"COSCIENTIST_LOG_DB": db_file}):
        result = await handle(to_payload(CycleContext(cycle_id=1)))

    out = parse_payload(result)
    assert result["type"] == "research_output"
    assert out.hypothesis.cycle_id == 1
    assert out.hypothesis.single_variable  # 단일 변인이 비어있지 않음

    rows = db.query_ledger(db_file, record_type="hypothesis")
    assert len(rows) == 1
    assert rows[0]["owner"] == "research"


async def test_handle_rejects_non_cycle_context():
    wrong = Hypothesis(cycle_id=1, statement="x", single_variable="y", rationale="z")
    with pytest.raises(ValueError, match="CycleContext 아님"):
        await handle(to_payload(wrong))
