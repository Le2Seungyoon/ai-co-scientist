"""실험 기록소 — 선보고 강제와 지표 도메인 일치 판정이 핵심 계약."""
import pytest

from ai_co_scientist import registry

BASE = dict(
    title="effb0 baseline",
    x_domain="sim", x_desc="sim SEM 전체",
    y_source="sim_depth_gt", y_desc="시뮬레이터 depth GT",
    model="smp:unet:efficientnet-b0, l1, e15, bs128",
    method="plain supervised",
    purpose="sim 재현 기준선",
    metric_name="sim_val_rmse", metric_x_domain="sim", metric_y_source="sim_depth_gt",
)


def test_new_report_assigns_sequential_ids(tmp_path):
    p = tmp_path / "reg.jsonl"
    first = registry.new_report(path=p, **BASE)
    second = registry.new_report(path=p, **{**BASE, "title": "두 번째"})
    assert first["report_id"] == "EXP-001"
    assert second["report_id"] == "EXP-002"


def test_new_report_rejects_unknown_x_domain(tmp_path):
    p = tmp_path / "reg.jsonl"
    with pytest.raises(ValueError, match="x_domain"):
        registry.new_report(path=p, **{**BASE, "x_domain": "synthetic"})


def test_new_report_rejects_unknown_y_source(tmp_path):
    p = tmp_path / "reg.jsonl"
    with pytest.raises(ValueError, match="y_source"):
        registry.new_report(path=p, **{**BASE, "y_source": "guess"})


def test_new_report_rejects_empty_purpose(tmp_path):
    p = tmp_path / "reg.jsonl"
    with pytest.raises(ValueError, match="purpose"):
        registry.new_report(path=p, **{**BASE, "purpose": "  "})


def test_sim_metric_is_flagged_as_not_matching_target(tmp_path):
    # 이 프로젝트가 실제로 당한 실패: sim SEM→sim depth 지표를 real validation으로 착각
    p = tmp_path / "reg.jsonl"
    rec = registry.new_report(path=p, **BASE)
    assert rec["metric"]["matches_target"] is False
    assert "sim" in rec["metric"]["warning"]


def test_real_avgdepth_metric_matches_target(tmp_path):
    p = tmp_path / "reg.jsonl"
    rec = registry.new_report(path=p, **{
        **BASE, "metric_name": "real_avgdepth_rmse",
        "metric_x_domain": "real", "metric_y_source": "real_average_depth"})
    assert rec["metric"]["matches_target"] is True
    assert rec["metric"]["warning"] == ""


def test_metric_matches_target_pure_function():
    assert registry.metric_matches_target("real", "real_average_depth") is True
    assert registry.metric_matches_target("sim", "sim_depth_gt") is False
    assert registry.metric_matches_target("real", "pseudo_label") is False  # y가 real GT가 아님


def test_record_result_and_lb_roundtrip(tmp_path):
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE)["report_id"]
    registry.record_result(rid, {"sim_val_rmse": 2.57}, path=p)
    registry.record_lb(rid, public=6.727, private=6.772, path=p)
    registry.set_verdict(rid, "챔피언 기준선", path=p)
    rec = registry.get(rid, path=p)
    assert rec["val"] == {"sim_val_rmse": 2.57}
    assert rec["lb"] == {"public": 6.727, "private": 6.772}
    assert rec["verdict"] == "챔피언 기준선"


def test_record_result_unknown_id_raises(tmp_path):
    p = tmp_path / "reg.jsonl"
    with pytest.raises(KeyError, match="EXP-999"):
        registry.record_result("EXP-999", {"x": 1}, path=p)


def test_render_markdown_contains_ids_and_reset_notice(tmp_path):
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE)["report_id"]
    registry.record_lb(rid, public=6.7, private=6.8, path=p)
    md = registry.render_markdown(path=p)
    assert "EXP-001" in md
    assert "6.7" in md
    assert "2026-07-29 리셋" in md  # 이전 실험 폐기 고지가 항상 상단에 남는다
