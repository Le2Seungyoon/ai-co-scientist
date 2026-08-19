"""`ai_co_scientist.sem` — **행동**을 검사한다.

이전에는 이 계약들이 소스 텍스트 grep이었다(`assert "(lv - d) / lv" in source`). 그런 검사는
주석에 있어도 통과하고 변수명만 바꿔도 실패한다. 여기서는 실제 배열로 확인한다.
"""
import numpy as np
import pytest

from ai_co_scientist import sem


# ── 재매개화: s in [0,1]이 정확해야 한다 ──────────────────────

@pytest.mark.parametrize("level", [140.0, 150.0, 160.0, 170.0])
def test_s_is_exactly_in_unit_interval_for_all_valid_depths(level):
    d = np.arange(0, int(level) + 1, dtype=np.float32)  # d in [0, L] 전 범위
    s = sem.depth_to_s(d, level)
    assert s.min() == 0.0 and s.max() == 1.0
    assert ((s >= 0.0) & (s <= 1.0)).all()


def test_roundtrip_depth_s_depth_is_exact():
    for level in sem.LEVELS:
        d = np.linspace(0, level, 500, dtype=np.float64)
        assert np.allclose(sem.s_to_depth(sem.depth_to_s(d, level), level), d, atol=1e-9)


def test_background_maps_to_level_exactly():
    # s=0인 배경은 정확히 L이어야 한다 — 이게 분해 설계의 핵심 보장이다
    for level in sem.LEVELS:
        assert sem.s_to_depth(np.zeros(4), level).tolist() == [level] * 4


def test_l_minus_20_normalisation_would_break_the_bound():
    # 기각된 대안이 왜 기각됐는지 고정한다: 전역 min이 0이라 (L-20)으로 나누면 s>1이 나온다
    level = 140.0
    d = np.array([0.0, 10.0, 20.0])
    assert (((level - d) / (level - 20.0)) > 1.0).any()
    assert (sem.depth_to_s(d, level) <= 1.0).all()


# ── 분할: 누수가 나면 안 된다 ────────────────────────────────

def test_site_split_never_puts_one_site_on_both_sides():
    rng = np.random.default_rng(0)
    site = np.repeat(np.arange(400), 31)              # 사이트당 31장
    y = np.repeat(rng.integers(0, 4, 400), 31)        # 사이트 단위 라벨
    va = sem.site_split(site, y, 0.2, seed=1)
    for s in np.unique(site):
        m = va[site == s]
        assert m.all() or (~m).all(), f"사이트 {s}가 train/val에 걸쳐 있다"


def test_site_split_is_stratified_by_group():
    site = np.repeat(np.arange(400), 10)
    y = np.repeat(np.tile(np.arange(4), 100), 10)     # 그룹당 사이트 100개
    va = sem.site_split(site, y, 0.25, seed=2)
    per_group = [len(np.unique(site[va & (y == c)])) for c in range(4)]
    assert per_group == [25, 25, 25, 25]


def test_map_level_split_keeps_itr_pairs_together():
    # depth map 1장 = SEM 2장(2k, 2k+1). 쌍이 갈리면 사실상 중복 누수다
    case = np.repeat(np.arange(1, 5), 200)            # 각 Case 200장 = depth-map 100개
    va = sem.map_level_split(case, 0.2, seed=3)
    assert (va[0::2] == va[1::2]).all(), "itr0/itr1 쌍이 갈렸다"


def test_map_level_split_holds_out_requested_fraction_per_case():
    case = np.repeat(np.arange(1, 5), 200)
    va = sem.map_level_split(case, 0.25, seed=4)
    for c in range(1, 5):
        maps = va[case == c][0::2]
        assert maps.sum() == 25                        # 100 depth-map의 25퍼센트


# ── 평활: EXP-019의 핵심 후처리 ──────────────────────────────

def test_smooth_is_noop_for_window_le_1():
    p = np.array([0, 3, 1, 2, 0])
    for k in (0, 1):
        assert (sem.smooth_levels(p, k) == p).all()


def test_smooth_removes_isolated_prediction_inside_a_run():
    p = np.array([2] * 10 + [0] + [2] * 10)           # 고립된 오분류 1건
    assert (sem.smooth_levels(p, 9) == 2).all()


def test_smooth_preserves_a_genuine_boundary():
    # 진짜 경계는 지우면 안 된다 — 양쪽 런이 window보다 길면 경계가 유지되어야 한다
    p = np.array([1] * 40 + [3] * 40)
    out = sem.smooth_levels(p, 9)
    assert out[:36].tolist() == [1] * 36
    assert out[44:].tolist() == [3] * 36
    assert len(np.unique(out)) == 2


def test_smooth_does_not_invent_a_class():
    rng = np.random.default_rng(5)
    p = rng.integers(0, 4, 500)
    assert set(np.unique(sem.smooth_levels(p, 9)).tolist()) <= set(np.unique(p).tolist())


# ── 점수 ────────────────────────────────────────────────────

def test_score_counts_adjacent_separately_from_exact():
    truth = np.array([0, 1, 2, 3])
    pred = np.array([0, 2, 2, 1])                     # 정확 2, 인접 3(2칸 오류 1건)
    r = sem.score_classes(pred, truth)
    assert r["accuracy"] == 0.5
    assert r["adjacent_ok"] == 0.75
    assert r["confusion"][3][1] == 1
