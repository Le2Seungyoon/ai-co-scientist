from ai_co_scientist.config import load_config, project_root


def test_project_root_has_config_yaml():
    assert (project_root() / "config.yaml").exists()


def test_paths_section():
    paths = load_config()["paths"]
    assert paths["registry"].endswith(".jsonl")
    assert paths["registry_doc"].endswith(".md")
    assert paths["data_dir"] == "data"


def test_target_is_real_domain():
    # 판정 타깃은 항상 실측 도메인 — sim 지표를 real validation으로 착각한 과거 실패의 기계적 방지선
    target = load_config()["target"]
    assert target["x_domain"] == "real"
    assert target["y_source"] == "real_depth"
    assert target["leaderboard_best"]["public"] == 6.727


def test_train_defaults():
    train = load_config()["train"]
    assert train["loss"] == "l1"
    assert 0.0 < train["val_fraction"] < 1.0


def test_no_a2a_leftovers():
    cfg = load_config()
    for gone in ("agents", "mock", "pm", "llm", "coder", "executor"):
        assert gone not in cfg, f"A2A 잔재 섹션이 남아있다: {gone}"
