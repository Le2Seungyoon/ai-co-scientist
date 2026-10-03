"""실험 기록소 — 선보고 강제와 지표 도메인 일치 판정이 핵심 계약."""
import json

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
    first = registry.new_report(path=p, **BASE, hypothesis="H0")
    second = registry.new_report(path=p, **{**BASE, "title": "두 번째"}, hypothesis="H0")
    assert first["report_id"] == "EXP-001"
    assert second["report_id"] == "EXP-002"


def test_new_report_rejects_unknown_x_domain(tmp_path):
    p = tmp_path / "reg.jsonl"
    with pytest.raises(ValueError, match="x_domain"):
        registry.new_report(path=p, **{**BASE, "x_domain": "synthetic"}, hypothesis="H0")


def test_new_report_rejects_unknown_y_source(tmp_path):
    p = tmp_path / "reg.jsonl"
    with pytest.raises(ValueError, match="y_source"):
        registry.new_report(path=p, **{**BASE, "y_source": "guess"}, hypothesis="H0")


def test_new_report_rejects_empty_purpose(tmp_path):
    p = tmp_path / "reg.jsonl"
    with pytest.raises(ValueError, match="purpose"):
        registry.new_report(path=p, **{**BASE, "purpose": "  "}, hypothesis="H0")


def test_sim_metric_is_flagged_as_not_matching_target(tmp_path):
    # 이 프로젝트가 실제로 당한 실패: sim SEM→sim depth 지표를 real validation으로 착각
    p = tmp_path / "reg.jsonl"
    rec = registry.new_report(path=p, **BASE, hypothesis="H0")
    assert rec["metric"]["matches_target"] is False
    assert "sim" in rec["metric"]["warning"]


def test_real_avgdepth_metric_matches_target(tmp_path):
    p = tmp_path / "reg.jsonl"
    rec = registry.new_report(path=p, **{
        **BASE, "metric_name": "real_avgdepth_rmse",
        "metric_x_domain": "real", "metric_y_source": "real_average_depth"}, hypothesis="H0")
    assert rec["metric"]["matches_target"] is True
    assert rec["metric"]["warning"] == ""


def test_real_group_label_metric_matches_target(tmp_path):
    # 폴더명 Depth_110~140은 주최측 실측 라벨이므로 real 도메인 검증으로 인정한다
    p = tmp_path / "reg.jsonl"
    rec = registry.new_report(path=p, **{
        **BASE, "x_domain": "real", "x_desc": "real train SEM hole crop",
        "y_source": "real_group_label", "y_desc": "폴더명 4그룹",
        "metric_name": "site_holdout_accuracy",
        "metric_x_domain": "real", "metric_y_source": "real_group_label"}, hypothesis="H0")
    assert rec["metric"]["matches_target"] is True
    assert rec["metric"]["warning"] == ""


def test_leaderboard_metric_declares_real_depth_gt(tmp_path):
    # 리더보드의 y는 숨은 real depth map이다. average_depth(=원본 전체 영상 평균)는 타깃이 아니다
    p = tmp_path / "reg.jsonl"
    rec = registry.new_report(path=p, **{
        **BASE, "metric_name": "leaderboard_rmse",
        "metric_x_domain": "real", "metric_y_source": "real_depth_gt"}, hypothesis="H0")
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
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]
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
            registry.new_report(path=p, **{**BASE, "title": title}, hypothesis="H0")
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
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]
    registry.record_lb(rid, public=6.7, private=6.8, path=p)
    md = registry.render_markdown(path=p)
    assert "EXP-001" in md
    assert "6.7" in md
    assert "2026-07-29 리셋" in md  # 이전 실험 폐기 고지가 항상 상단에 남는다


def test_second_result_call_preserves_first_calls_keys(tmp_path):
    """EXP-020 회귀 — `record_result`가 val을 통째로 교체하던 시절, 제출 후의 두 번째
    호출이 학습 직후 기록한 wall_clock/run/verify_only를 조용히 지웠다."""
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]
    registry.record_result(rid, {"wall_clock": "1h02m", "run": "EXP-020-a"}, path=p)
    up = registry.record_result(rid, {"verify_only": {"files": 25988}}, path=p)
    assert registry.get(rid, path=p)["val"] == {
        "wall_clock": "1h02m", "run": "EXP-020-a", "verify_only": {"files": 25988}}
    assert up.added == ["verify_only"]
    assert up.overwritten == {}
    assert up.dropped == {}


def test_result_merge_is_idempotent_for_equal_values(tmp_path):
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]
    registry.record_result(rid, {"wall_clock": "1h02m"}, path=p)
    up = registry.record_result(rid, {"wall_clock": "1h02m"}, path=p)
    assert up.unchanged == ["wall_clock"]
    assert up.added == []
    assert registry.get(rid, path=p)["val"] == {"wall_clock": "1h02m"}


def test_result_key_collision_with_different_value_raises(tmp_path):
    # 조용한 덮어쓰기는 통째 교체와 같은 결함이라 기본 경로에서 막는다
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]
    registry.record_result(rid, {"wall_clock": "1h02m"}, path=p)
    with pytest.raises(ValueError, match="wall_clock"):
        registry.record_result(rid, {"wall_clock": "2h30m"}, path=p)
    assert registry.get(rid, path=p)["val"] == {"wall_clock": "1h02m"}, "거부됐는데 기록이 변했다"


def test_result_collision_overwrites_only_with_explicit_replace_key(tmp_path):
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]
    registry.record_result(rid, {"wall_clock": "1h02m", "run": "a"}, path=p)
    up = registry.record_result(rid, {"wall_clock": "2h30m"}, replace_keys=["wall_clock"], path=p)
    assert up.overwritten == {"wall_clock": "1h02m"}  # 이전 값이 호출자에게 그대로 보인다
    assert registry.get(rid, path=p)["val"] == {"wall_clock": "2h30m", "run": "a"}


def test_result_replace_reports_every_dropped_key(tmp_path):
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]
    registry.record_result(rid, {"wall_clock": "1h02m", "run": "a"}, path=p)
    up = registry.record_result(rid, {"run": "a"}, replace=True, path=p)
    assert up.dropped == {"wall_clock": "1h02m"}
    assert registry.get(rid, path=p)["val"] == {"run": "a"}


def test_result_rejects_non_dict_val(tmp_path):
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]
    with pytest.raises(ValueError, match="dict"):
        registry.record_result(rid, [1, 2], path=p)


def test_merge_val_pure_function():
    merged, added, unchanged, overwritten = registry.merge_val(None, {"a": 1})
    assert (merged, added, unchanged, overwritten) == ({"a": 1}, ["a"], [], {})
    merged, added, unchanged, overwritten = registry.merge_val({"a": 1}, {"a": 1, "b": 2})
    assert (merged, added, unchanged, overwritten) == ({"a": 1, "b": 2}, ["b"], ["a"], {})
    with pytest.raises(ValueError, match="키 충돌"):
        registry.merge_val({"a": 1}, {"a": 2})
    # 얕은 병합: 중첩 dict는 합치지 않고 충돌로 본다 (같은 결함을 한 단계 아래로 옮기지 않기 위해)
    with pytest.raises(ValueError, match="키 충돌"):
        registry.merge_val({"v": {"files": 1}}, {"v": {"files": 1, "max": 4}})


def test_concurrent_result_calls_keep_every_key(tmp_path):
    """병합은 read-modify-write다 — 락 밖에서 읽으면 선보고 8건 중 2건만 살아남았던
    그 경쟁 조건이 val 안에서 그대로 재현된다."""
    import threading
    p = tmp_path / "reg.jsonl"
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]
    errors: list[BaseException] = []

    def add(i):
        try:
            registry.record_result(rid, {f"k{i}": i}, path=p)
        except BaseException as e:  # noqa: BLE001 - 스레드 예외를 본 스레드로 옮긴다
            errors.append(e)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert registry.get(rid, path=p)["val"] == {f"k{i}": i for i in range(8)}


def _load_exp_cli():
    """scripts/exp.py를 서브프로세스가 아니라 모듈로 적재한다.

    executor가 실제로 쓰는 유일한 경로가 CLI라서 소스 텍스트 검사로는 부족하다. 다만
    실행이 진짜 기록소를 건드리면 안 되므로 호출부에서 `_default_path`를 tmp로 돌린다.
    """
    import importlib.util
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "exp.py"
    spec = importlib.util.spec_from_file_location("exp_cli_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_result_merges_by_default_and_refuses_silent_overwrite(tmp_path, monkeypatch, capsys):
    """`.claude/agents/executor.md` 3단계를 그대로 두 번 따라 해도 매니페스트가 죽지 않아야 한다."""
    p = tmp_path / "reg.jsonl"
    monkeypatch.setattr(registry, "_default_path", lambda: p)
    cli = _load_exp_cli()
    rid = registry.new_report(path=p, **BASE, hypothesis="H0")["report_id"]

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["exp.py", *argv])
        cli.main()
        return capsys.readouterr().out

    run("result", rid, "--val", '{"wall_clock": "1h02m", "run": "a"}')
    run("result", rid, "--val", '{"verify_only": {"files": 25988}}')
    assert registry.get(rid, path=p)["val"] == {
        "wall_clock": "1h02m", "run": "a", "verify_only": {"files": 25988}}

    with pytest.raises(SystemExit, match="wall_clock"):  # 조용히 덮어쓰지 않는다
        run("result", rid, "--val", '{"wall_clock": "2h30m"}')

    out = run("result", rid, "--val", '{"wall_clock": "2h30m"}', "--replace-key", "wall_clock")
    assert "WARNING: 덮어씀 wall_clock" in out and '"1h02m"' in out

    out = run("result", rid, "--val", '{"run": "a"}', "--replace")
    assert "WARNING: 버려짐 wall_clock" in out
    assert registry.get(rid, path=p)["val"] == {"run": "a"}


def test_new_report_records_source_when_given(tmp_path):
    """워커 브랜치에서 돈 실험은 그 브랜치/커밋으로 기록돼야 한다 — 코디네이터의 HEAD가 아니라."""
    path = tmp_path / "r.jsonl"
    rec = registry.new_report(
        title="t", x_domain="real", x_desc="x", y_source="real_group_label", y_desc="y",
        model="m", method="me", purpose="p", metric_name="acc",
        metric_x_domain="real", metric_y_source="real_group_label",
        hypothesis="H0", source_branch="feature/postproc-k", source_commit="deadbeef", path=path)
    assert rec["source"] == {"branch": "feature/postproc-k", "commit": "deadbeef"}
    assert registry.get(rec["report_id"], path)["source"]["branch"] == "feature/postproc-k"


def test_new_report_source_is_none_when_omitted(tmp_path):
    """optional 필드다 — 기존 호출부는 바뀌지 않는다."""
    path = tmp_path / "r.jsonl"
    rec = registry.new_report(
        title="t", x_domain="sim", x_desc="x", y_source="sim_depth_gt", y_desc="y",
        model="m", method="me", purpose="p", metric_name="rmse",
        metric_x_domain="sim", metric_y_source="sim_depth_gt", hypothesis="H0", path=path)
    assert rec["source"] is None


def test_render_markdown_shows_source_when_present(tmp_path):
    """워커 브랜치/커밋이 렌더된 markdown에도 남아야 한다 -- runtime/는 백업이 없어서
    docs/experiment-registry.md가 유일한 사본이다."""
    p = tmp_path / "reg.jsonl"
    registry.new_report(path=p, **{
        **BASE, "source_branch": "feature/postproc-k", "source_commit": "deadbeef"}, hypothesis="H0")
    md = registry.render_markdown(path=p)
    assert "feature/postproc-k" in md
    assert "deadbeef" in md


def test_render_markdown_omits_source_line_when_absent(tmp_path):
    """source가 없는(기존 20건 포함) 레코드는 렌더가 이전과 똑같아야 한다 -- 빈 줄도,
    "None"도 나오면 안 된다."""
    p = tmp_path / "reg.jsonl"
    registry.new_report(path=p, **BASE, hypothesis="H0")
    md = registry.render_markdown(path=p)
    assert "출처" not in md
    assert "None" not in md


def test_records_without_source_still_load(tmp_path):
    """기존 20건에는 source 키가 없다 — 로드가 깨지면 안 된다."""
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({
        "report_id": "EXP-001", "created": "2026-07-30T00:00:00", "title": "old",
        "x": {"domain": "sim", "desc": "d"}, "y": {"source": "sim_depth_gt", "desc": "d"},
        "model": "m", "method": "me", "purpose": "p",
        "metric": {"name": "rmse", "x_domain": "sim", "y_source": "sim_depth_gt",
                   "matches_target": False, "warning": "w"},
        "val": None, "lb": None, "verdict": "",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    assert registry.get("EXP-001", path)["title"] == "old"
    assert registry.load_all(path)[0].get("source") is None


def test_new_report_records_the_hypothesis(tmp_path):
    """가설 ↔ 실행 조인은 손으로 유지하는 표가 아니라 레코드가 들어야 한다."""
    path = tmp_path / "r.jsonl"
    rec = registry.new_report(**BASE, hypothesis="H12", path=path)
    assert rec["hypothesis"] == "H12"
    assert registry.get(rec["report_id"], path)["hypothesis"] == "H12"


def test_new_report_refuses_without_a_hypothesis(tmp_path):
    """조인 없는 레코드는 조인에 보이지 않고 아무도 눈치채지 못한다 —
    조용한 실패이므로 관례가 아니라 거부로 막는다."""
    path = tmp_path / "r.jsonl"
    with pytest.raises(TypeError):
        registry.new_report(**BASE, path=path)


def test_new_report_refuses_a_blank_hypothesis(tmp_path):
    """빈 문자열을 통과시키면 필수 인자가 형식뿐인 것이 된다."""
    path = tmp_path / "r.jsonl"
    with pytest.raises(ValueError):
        registry.new_report(**BASE, hypothesis="   ", path=path)


def test_many_runs_may_answer_one_hypothesis(tmp_path):
    """관계는 다대다다 — EXP-010은 3-arm을 한 항목으로 기록했다."""
    path = tmp_path / "r.jsonl"
    a = registry.new_report(**BASE, hypothesis="H12", path=path)
    b = registry.new_report(**BASE, hypothesis="H12", path=path)
    assert a["report_id"] != b["report_id"]
    ids = [r["report_id"] for r in registry.load_all(path) if r["hypothesis"] == "H12"]
    assert ids == [a["report_id"], b["report_id"]]
