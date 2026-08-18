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


def test_real_group_label_metric_matches_target(tmp_path):
    # 폴더명 Depth_110~140은 주최측 실측 라벨이므로 real 도메인 검증으로 인정한다
    p = tmp_path / "reg.jsonl"
    rec = registry.new_report(path=p, **{
        **BASE, "x_domain": "real", "x_desc": "real train SEM hole crop",
        "y_source": "real_group_label", "y_desc": "폴더명 4그룹",
        "metric_name": "site_holdout_accuracy",
        "metric_x_domain": "real", "metric_y_source": "real_group_label"})
    assert rec["metric"]["matches_target"] is True
    assert rec["metric"]["warning"] == ""


def test_leaderboard_metric_declares_real_depth_gt(tmp_path):
    # 리더보드의 y는 숨은 real depth map이다. average_depth(=원본 전체 영상 평균)는 타깃이 아니다
    p = tmp_path / "reg.jsonl"
    rec = registry.new_report(path=p, **{
        **BASE, "metric_name": "leaderboard_rmse",
        "metric_x_domain": "real", "metric_y_source": "real_depth_gt"})
    assert rec["metric"]["matches_target"] is True
    assert rec["metric"]["warning"] == ""


def test_metric_matches_target_pure_function():
    assert registry.metric_matches_target("real", "real_average_depth") is True
    assert registry.metric_matches_target("real", "real_group_label") is True
    assert registry.metric_matches_target("real", "real_depth_gt") is True
    assert registry.metric_matches_target("sim", "sim_depth_gt") is False
    assert registry.metric_matches_target("sim", "real_group_label") is False  # X가 sim
    assert registry.metric_matches_target("real", "pseudo_label") is False  # y가 real GT가 아님


def test_record_result_and_lb_roundtrip(tmp_path):
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE)["report_id"]
    registry.record_result(rid, {"sim_val_rmse": 2.57}, path=p)
    registry.record_lb(rid, public=7.35, private=7.34, path=p)
    registry.set_verdict(rid, "기준선", path=p)
    rec = registry.get(rid, path=p)
    assert rec["val"] == {"sim_val_rmse": 2.57}
    assert rec["lb"] == {"public": 7.35, "private": 7.34}
    assert rec["verdict"] == "기준선"


def test_concurrent_new_report_keeps_both_and_assigns_distinct_ids(tmp_path):
    """sub-agent 병렬 실행 회귀 — 락이 없으면 두 스레드가 같은 EXP-00N을 발급하고
    나중 write가 앞 선보고를 통째로 덮어써 기록이 소실된다 (실측 확인)."""
    import threading
    p = tmp_path / "reg.jsonl"
    errors: list[BaseException] = []

    def register(title):
        try:
            registry.new_report(path=p, **{**BASE, "title": title})
        except BaseException as e:  # noqa: BLE001 - 스레드 예외를 본 스레드로 옮긴다
            errors.append(e)

    threads = [threading.Thread(target=register, args=(f"에이전트{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    records = registry.load_all(p)
    assert len(records) == 8, "선보고가 소실됐다"
    ids = [r["report_id"] for r in records]
    assert len(set(ids)) == 8, f"report_id가 중복됐다: {ids}"
    assert sorted(ids) == [f"EXP-{i:03d}" for i in range(1, 9)]
    assert {r["title"] for r in records} == {f"에이전트{i}" for i in range(8)}


def test_lock_is_exclusive_and_released(tmp_path):
    p = tmp_path / "reg.jsonl"
    with registry.locked(p):
        assert p.with_suffix(".lock").exists()
    assert not p.with_suffix(".lock").exists(), "락이 해제되지 않았다"


def test_lock_times_out_when_held(tmp_path, monkeypatch):
    p = tmp_path / "reg.jsonl"
    monkeypatch.setattr(registry, "LOCK_TIMEOUT", 0.2)
    with registry.locked(p):
        with pytest.raises(TimeoutError, match="락 대기 초과"):
            with registry.locked(p):
                pass


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
