"""train_sem_depth.py의 X/y 매니페스트 계약 — 도메인 라벨이 코드에서 사라지지 않게 고정한다.

실제 학습은 GPU/데이터가 필요해 단위테스트로 돌리지 않는다. 대신 이 스크립트가
(1) standalone(패키지 import 금지)이고 (2) 매니페스트에 X/y 도메인을 남기는지를 검사한다.
"""
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_sem_depth.py"


def test_script_is_standalone():
    # Lightning Studio에 업로드해 단독 실행되어야 하므로 저장소 패키지를 import하면 안 된다
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ai_co_scientist" not in source


def test_manifest_declares_x_and_y_domains():
    source = SCRIPT.read_text(encoding="utf-8")
    for key in ('"x_domain": "sim"', '"y_source": "sim_depth_gt"'):
        assert key in source, f"매니페스트에 {key} 선언이 없다"


def test_real_avgdepth_metric_declares_real_domain():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"x_domain": "real"' in source
    assert '"y_source": "real_average_depth"' in source


def test_void_experiment_features_are_gone():
    source = SCRIPT.read_text(encoding="utf-8")
    for gone in ("fda_transform", "blur_aug", "aug-brightness", "val-case",
                 "histmatch", "clahe", "predict_tta", "skip-inference"):
        assert gone not in source, f"폐기된 실험 기능이 남아있다: {gone}"


def test_infer_submit_is_standalone():
    path = Path(__file__).resolve().parents[1] / "scripts" / "infer_submit.py"
    assert "ai_co_scientist" not in path.read_text(encoding="utf-8")


def test_probe_level_is_standalone_and_declares_real_domain():
    # 레벨 분류 프로브(EXP-004)는 X도 y도 real이라는 점이 핵심 — 도메인 선언이 사라지면 안 된다
    source = (Path(__file__).resolve().parents[1] / "scripts" / "probe_level.py").read_text(
        encoding="utf-8")
    assert "ai_co_scientist" not in source
    assert '"x_domain": "real"' in source
    assert '"y_source": "real_group_label"' in source


def test_probe_level_splits_by_site_not_by_image():
    # 사이트당 crop ~31장이 같은 라벨을 공유한다 — 이미지 단위 split은 정확도를 낙관 편향시킨다
    source = (Path(__file__).resolve().parents[1] / "scripts" / "probe_level.py").read_text(
        encoding="utf-8")
    assert "def site_split(" in source


def _script(name: str) -> str:
    return (Path(__file__).resolve().parents[1] / "scripts" / name).read_text(encoding="utf-8")


def test_structure_scripts_are_standalone():
    for name in ("train_structure.py", "infer_decomposed.py"):
        assert "ai_co_scientist" not in _script(name), f"{name}이 저장소 패키지를 import한다"


def test_infer_decomposed_supports_cnn_level_source():
    # EXP-014: 레벨 축을 QDA → CNN으로 교체할 수 있어야 한다. arm이 사라지면 재현이 깨진다
    source = _script("infer_decomposed.py")
    assert '"cnn"' in source, "--level-source에 cnn arm이 없다"
    assert "def predict_levels_cnn(" in source


def test_level_cnn_gets_no_adabn():
    # 레벨 분류기는 real로 학습해 real에 적용하므로 sim→real 전이가 없다. AdaBN은 구조
    # 모델 전용이며, 레벨에도 걸면 EXP-014의 단일 축 변경이 깨진다
    source = _script("infer_decomposed.py")
    cnn_fn = source.split("def predict_levels_cnn(")[1].split("\ndef ")[0]
    assert "adapt_bn" not in cnn_fn, "레벨 분류기에 AdaBN이 걸렸다 — 축이 하나 더 바뀐다"


def test_train_level_is_standalone_and_declares_real_domain():
    # 레벨 분류기(EXP-013)는 X도 y도 real이라 제출 없이 판정할 수 있다 — 그 선언이 계약이다
    source = _script("train_level.py")
    assert "ai_co_scientist" not in source
    assert '"x_domain": "real"' in source
    assert '"y_source": "real_group_label"' in source


def test_train_level_reuses_site_split():
    # 사이트당 crop ~31장이 같은 라벨을 공유한다. EXP-004와 동일 split이어야 QDA와 비교된다 —
    # 자체 구현하면 조용히 갈라지므로 probe_level의 것을 그대로 import해야 한다
    source = _script("train_level.py")
    assert "site_split" in source
    assert "from probe_level import" in source


def test_train_level_does_not_normalize_per_image():
    # 신호는 밝기가 아니라 절대 intensity 분포 형태다(그룹간 간격 1.7 < 그룹내 std 1.9).
    # 이미지별 표준화나 InstanceNorm은 그 신호를 파괴한다 — 들어오면 성능이 무너진다
    # 산문(경고 문구)이 아니라 **실제 사용**만 잡아야 하므로 `nn.` 접두를 요구한다
    source = _script("train_level.py")
    for banned in ("nn.InstanceNorm", "nn.LayerNorm", "nn.GroupNorm"):
        assert banned not in source, f"이미지별 정규화가 들어왔다: {banned}"


def test_train_structure_declares_sim_domain_and_exact_reparam():
    # s = (L-d)/L 이어야 s in [0,1]이 정확하다. (L-20) 정규화는 s>1을 만들어 클램프 손실을 낳는다
    source = _script("train_structure.py")
    assert '"x_domain": "sim"' in source
    assert '"y_source": "sim_depth_gt"' in source
    assert "(lv - d) / lv" in source, "s 정의가 (L-d)/L에서 벗어났다"


def test_train_structure_backbone_is_swappable():
    # 백본은 갈아끼울 수 있어야 한다 — arch를 고정하면 도메인 갭 실험을 못 돌린다
    source = _script("train_structure.py")
    assert "def make_model(" in source
    for arch in ('"mlp"', '"unet"', '"smp:"'):
        assert arch in source, f"make_model이 {arch}를 다루지 않는다"


def test_train_structure_ckpt_is_backward_compatible():
    # EXP-005는 arch 없이 bare state_dict로 저장됐다 — 재현 가능해야 하므로 로드가 깨지면 안 된다
    source = _script("train_structure.py")
    assert "def load_model(" in source
    assert '"state_dict"' in source


def test_train_structure_does_not_import_wandb_chain():
    # infer_decomposed가 train_structure를 import하므로, 여기서 train_sem_depth를 끌어오면
    # 추론 경로에 wandb(최상단 import)까지 딸려온다
    source = _script("train_structure.py")
    assert "import train_sem_depth" not in source
    assert "from train_sem_depth" not in source


def test_infer_decomposed_declares_leaderboard_target():
    # 리더보드의 y는 숨은 real depth GT다 — average_depth로 잘못 라벨링하면 안 된다
    source = _script("infer_decomposed.py")
    assert '"x_domain": "real"' in source
    assert '"y_source": "real_depth_gt"' in source
    assert "real_average_depth" not in source


def test_infer_submit_has_no_void_postproc():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "infer_submit.py").read_text(
        encoding="utf-8")
    for gone in ("adabn", "tta", "clamp_lo"):
        assert gone not in source.lower(), f"폐기된 후처리가 남아있다: {gone}"
