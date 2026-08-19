"""현역 스크립트의 **매니페스트** 계약 — 도메인 라벨이 코드에서 조용히 사라지지 않게 한다.

순수 로직(분할·재매개화·평활·QDA)은 `ai_co_scientist.sem`으로 옮겨졌고 `tests/test_sem.py`가
**행동으로** 검사한다. 여기 남은 것은 소스 텍스트로만 확인 가능한 것뿐이다: 학습 스크립트가
매니페스트에 (X, y) 도메인을 남기는가, 그리고 도메인이 섞이면 안 되는 자리에 섞이지 않았는가.
(구)현역 스크립트의 계약 — 도메인 라벨과 설계 결정이 코드에서 조용히 사라지지 않게 고정한다.

실제 학습은 GPU/데이터가 필요해 단위테스트로 돌리지 않으므로 소스 텍스트 계약으로 대신한다.
이건 **행동을 검사하지 못하는 임시 수단**이다 — 로직이 `src/`로 옮겨지면 진짜 단위테스트로
교체할 것 (`.claude/rules/architecture.md` 이행 항목).

`scripts/legacy/`는 검사하지 않는다. 재현 전용으로 얼려둔 파일이라 회귀할 수 없다.
"""
from pathlib import Path





def _script(name: str) -> str:
    return (Path(__file__).resolve().parents[1] / "scripts" / name).read_text(encoding="utf-8")


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



def test_train_level_does_not_normalize_per_image():
    # 신호는 밝기가 아니라 절대 intensity 분포 형태다(그룹간 간격 1.7 < 그룹내 std 1.9).
    # 이미지별 표준화나 InstanceNorm은 그 신호를 파괴한다 — 들어오면 성능이 무너진다
    # 산문(경고 문구)이 아니라 **실제 사용**만 잡아야 하므로 `nn.` 접두를 요구한다
    source = _script("train_level.py")
    for banned in ("nn.InstanceNorm", "nn.LayerNorm", "nn.GroupNorm"):
        assert banned not in source, f"이미지별 정규화가 들어왔다: {banned}"


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


