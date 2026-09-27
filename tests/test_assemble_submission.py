"""CPU 재조립 진입점의 계약 — torch/cv2 없이 돌고, 레벨 후처리 인자가 실제로 결과를 바꾼다.

이 진입점은 워크트리(dev 그룹만 sync — torch도 cv2도 없다)에서 병렬로 도는 것이 존재 이유다.
그래서 **import 계약**이 성능 못지않게 중요하다.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from ai_co_scientist.submission import decode_png_gray8

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "assemble_submission.py"


def _fixture(tmp_path, n=6, h=4, w=3):
    """ŝ 덤프 · 레벨 사후확률 덤프 · 파일명 목록을 합성한다."""
    rng = np.random.default_rng(0)
    structure = rng.random((n, h, w), dtype=np.float32)
    proba = np.full((n, 4), 0.1, dtype=np.float32)
    proba[:, 0] = 0.7
    proba[3, :] = [0.1, 0.7, 0.1, 0.1]  # 런 한가운데의 고립된 한 장 — 평활 대상
    names = [f"{i:06d}.png" for i in range(n)]
    np.save(tmp_path / "s.npy", structure)
    np.save(tmp_path / "p.npy", proba)
    (tmp_path / "names.json").write_text(json.dumps(names), encoding="utf-8")
    return structure, proba, names


def _run(tmp_path, *extra):
    out = tmp_path / "sub.zip"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--structure", str(tmp_path / "s.npy"),
         "--level-proba", str(tmp_path / "p.npy"),
         "--names", str(tmp_path / "names.json"),
         "--submit", str(out), *extra],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return out, json.loads(proc.stdout.strip().splitlines()[-1])


def test_produces_readable_zip_without_torch_or_cv2(tmp_path):
    _fixture(tmp_path)
    out, diag = _run(tmp_path)
    assert diag["n"] == 6
    with zipfile.ZipFile(out) as zf:
        assert len(zf.namelist()) == 6
        assert decode_png_gray8(zf.read("000000.png")).shape == (4, 3)


def test_source_imports_neither_torch_nor_cv2():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import cv2" not in source


def test_level_smooth_changes_the_isolated_image(tmp_path):
    """k=3 최빈값 필터는 런 한가운데 고립된 한 장을 흡수해야 한다."""
    _fixture(tmp_path)
    _, plain = _run(tmp_path)
    _, smoothed = _run(tmp_path, "--level-smooth", "3")
    assert smoothed["level_diag"]["changed"] == 1
    assert plain["level_smooth"] == 0
    assert smoothed["level_smooth"] == 3


def test_smooth_and_hmm_are_mutually_exclusive(tmp_path):
    """평활기를 두 번 겹치면 어느 쪽이 점수를 움직였는지 분리되지 않는다."""
    _fixture(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--structure", str(tmp_path / "s.npy"),
         "--level-proba", str(tmp_path / "p.npy"),
         "--names", str(tmp_path / "names.json"),
         "--submit", str(tmp_path / "x.zip"),
         "--level-smooth", "3", "--level-hmm"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode != 0
    assert "함께 쓸 수 없다" in proc.stderr


def test_declares_leaderboard_target():
    """리더보드의 y는 숨은 real depth GT다 — average_depth로 잘못 라벨링하면 안 된다."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"x_domain": "real"' in source
    assert '"y_source": "real_depth_gt"' in source
    assert "real_average_depth" not in source


def test_two_paths_produce_identical_zip_bytes(tmp_path):
    """GPU 경로의 조립과 CPU 진입점의 조립이 바이트까지 같아야 한다.

    GPU forward 자체는 재현할 수 없으므로, 그 **출력** ŝ를 고정한 뒤 이후 구간만 비교한다.
    조립이 두 벌이 되면(다른 인코더·다른 반올림·다른 zip 타임스탬프) 여기서 즉시 깨진다.
    """
    from ai_co_scientist.sem import LEVELS, assemble_depth
    from ai_co_scientist.submission import write_submission_zip

    structure, proba, names = _fixture(tmp_path)

    # 경로 ① — infer_decomposed.reconstruct_and_zip이 하는 것과 같은 순서
    cls = proba.argmax(1)
    levels = np.array(LEVELS, dtype=np.float32)[cls]
    direct = tmp_path / "direct.zip"
    write_submission_zip(assemble_depth(structure, levels, 0.0), names, direct)

    # 경로 ② — CPU 진입점
    viacli, _ = _run(tmp_path)

    assert direct.read_bytes() == viacli.read_bytes()


def test_structure_dump_roundtrips_bit_exactly(tmp_path):
    """float32 저장·로드가 ŝ를 비트 단위로 보존해야 한다.

    float16(절반 용량)을 버린 근거를 고정한다: d = L*(1-s)에서 L이 최대 170이라 s의 오차
    ~0.0005가 d에서 ~0.085가 되어 round()의 경계를 흔든다.
    """
    from ai_co_scientist.sem import assemble_depth

    rng = np.random.default_rng(7)
    s = rng.random((200, 4, 3), dtype=np.float32)
    path = tmp_path / "s32.npy"
    np.save(path, s)
    assert np.load(path).dtype == np.float32
    assert np.array_equal(np.load(path), s)

    # float16으로 저장했다면 조립 결과가 실제로 달라진다 — 버린 근거를 숫자로 남긴다.
    # L=170은 LEVELS의 최댓값이라 s 오차가 d로 가장 크게 증폭되는 조건이다.
    levels = np.full(len(s), 170.0, dtype=np.float32)
    lossy = s.astype(np.float16).astype(np.float32)
    differing = int((assemble_depth(s, levels) != assemble_depth(lossy, levels)).sum())
    assert differing > 0, (
        "float16 왕복이 이 표본에서 조립 결과를 전혀 바꾸지 않았다 — 표본을 넓히거나, "
        "float32를 고집하는 근거를 다시 재야 한다")


def test_tau_survives_both_paths(tmp_path):
    """τ가 CPU 진입점에서도 같은 클램프를 낸다 — 인자가 한쪽에만 먹으면 조용히 갈린다."""
    from ai_co_scientist.sem import LEVELS, assemble_depth
    from ai_co_scientist.submission import write_submission_zip

    structure, proba, names = _fixture(tmp_path)
    cls = proba.argmax(1)
    levels = np.array(LEVELS, dtype=np.float32)[cls]
    direct = tmp_path / "direct_tau.zip"
    write_submission_zip(assemble_depth(structure, levels, 0.5), names, direct)

    viacli, _ = _run(tmp_path, "--tau", "0.5")
    assert direct.read_bytes() == viacli.read_bytes()
