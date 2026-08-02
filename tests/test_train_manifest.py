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
