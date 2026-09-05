"""sem.viterbi_levels — 4상태 Viterbi 레벨 복호 (순수 numpy, 오프라인·합성 열).

정답이 **구성으로** 알려진 열에서만 검사한다: ① 깨끗한 런은 그대로, ② 런 안의 약한 한 장
블립은 흡수, ③ 진짜 경계는 흐려지지 않고 정확한 위치에 선다, ④ 충분히 강한 증거는 전이
비용을 이긴다(무조건 평활기가 아니라는 증거).

`smooth_levels`와 겹쳐 쓰지 못하게 막는 것은 CLI 쪽 계약이라 여기서는 다루지 않는다
(`tests/test_train_manifest.py` 계열의 소스 계약이 아니라 argparse 오류 경로다).
"""
import numpy as np
import pytest

from ai_co_scientist.sem import LEVELS, softmax, viterbi_levels

K = len(LEVELS)


def _confident(labels, p: float = 0.99) -> np.ndarray:
    """각 위치에서 해당 라벨에 p, 나머지에 (1−p)/3을 주는 사후확률 열."""
    proba = np.full((len(labels), K), (1.0 - p) / (K - 1))
    proba[np.arange(len(labels)), np.asarray(labels)] = p
    return proba


def test_clean_run_is_unchanged():
    labels = [2] * 30
    assert np.array_equal(viterbi_levels(_confident(labels)), np.array(labels))


def test_weak_blip_inside_a_run_is_absorbed():
    """런 한가운데 한 장만 다른 클래스를 약하게(0.7 대 0.1) 가리키면 전이 비용에 눌린다.
    argmax는 그 자리를 2로 읽지만 Viterbi는 1로 되돌린다 — 이게 이 기법의 주장이다."""
    labels = [1] * 21
    proba = _confident(labels)
    proba[10] = [0.1, 0.1, 0.7, 0.1]

    assert proba.argmax(1)[10] == 2  # 대조군: 평활 없이는 오분류가 남는다
    decoded = viterbi_levels(proba, a=0.974)
    assert np.array_equal(decoded, np.ones(21, dtype=np.int64))


def test_strong_evidence_is_not_smoothed_away():
    """전이 비용(2회 전환 ≈ 9.44 nat)을 넘는 증거는 살아남아야 한다. 살아남지 않으면 이 함수는
    Viterbi가 아니라 그냥 '런을 상수로 만드는' 장치다."""
    labels = [1] * 21
    proba = _confident(labels)
    proba[10] = [1e-6, 1e-6, 1.0 - 3e-6, 1e-6]
    assert viterbi_levels(proba, a=0.974)[10] == 2


def test_genuine_boundary_stays_sharp_and_in_place():
    labels = [0] * 20 + [3] * 20
    decoded = viterbi_levels(_confident(labels), a=0.974)
    assert np.array_equal(decoded, np.array(labels))
    assert int(np.flatnonzero(np.diff(decoded)).item()) == 19  # 전환점이 정확히 20번째 앞


def test_uniform_transition_reduces_to_argmax():
    """a = 1/K이면 전이행렬이 균등해져 순수 argmax와 같아야 한다 — 방출항이 제대로 들어갔는지
    확인하는 축퇴 검사."""
    rng = np.random.default_rng(0)
    proba = softmax(rng.normal(size=(50, K)))
    assert np.array_equal(viterbi_levels(proba, a=1.0 / K), proba.argmax(1))


def test_higher_a_changes_more_predictions():
    """a가 커질수록(런이 길다는 사전지식이 강할수록) argmax에서 더 많이 벗어난다."""
    rng = np.random.default_rng(1)
    proba = softmax(rng.normal(size=(200, K)) * 1.5)
    base = proba.argmax(1)
    low = int((viterbi_levels(proba, a=0.5) != base).sum())
    high = int((viterbi_levels(proba, a=0.99) != base).sum())
    assert low < high


def test_rejects_bad_transition_probability():
    proba = _confident([0, 1, 2])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="자기전이 확률"):
            viterbi_levels(proba, a=bad)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="N, K"):
        viterbi_levels(np.ones(10))


def test_empty_sequence_returns_empty():
    assert viterbi_levels(np.zeros((0, K))).shape == (0,)


def test_softmax_rows_sum_to_one_and_are_shift_invariant():
    logits = np.array([[1.0, 2.0, 3.0, 4.0]])
    p = softmax(logits)
    assert p.sum() == pytest.approx(1.0)
    assert p == pytest.approx(softmax(logits + 100.0))


def test_cli_refuses_to_stack_hmm_with_smooth():
    """CLI가 두 평활기를 겹쳐 쓰지 못하게 막는지 — 소스 계약 검사다(`.claude/rules/testing.md`의
    stand-in). main()을 실제로 돌리는 것은 실험 스크립트 실행이라 하지 않는다."""
    from pathlib import Path  # noqa: PLC0415
    src = (Path(__file__).resolve().parents[1] / "scripts" / "infer_decomposed.py").read_text(
        encoding="utf-8")
    assert "if args.level_hmm and args.level_smooth > 1:" in src
    assert "--level-hmm과 --level-smooth는 함께 쓸 수 없다" in src
