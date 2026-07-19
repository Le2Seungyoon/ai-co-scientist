from hooks.cycle_log_guard import cycle_log_guard


def _row(cycle_id, record_type):
    return {"cycle_id": cycle_id, "record_type": record_type}


def test_all_present_passes():
    ok, missing = cycle_log_guard([_row(1, "hypothesis"), _row(1, "diagnosis")], cycle_id=1)
    assert ok and missing == []


def test_missing_diagnosis_detected():
    ok, missing = cycle_log_guard([_row(1, "hypothesis")], cycle_id=1)
    assert not ok and missing == ["diagnosis"]


def test_other_cycle_records_ignored():
    ok, missing = cycle_log_guard([_row(2, "hypothesis"), _row(2, "diagnosis")], cycle_id=1)
    assert not ok and set(missing) == {"hypothesis", "diagnosis"}
