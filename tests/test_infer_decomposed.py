"""infer_decomposed.adapt_bn — AdaBN 소스 셔플 계약을 오프라인(GPU/캐시 불필요, 합성 데이터)으로
고정한다. EXP-016: AdaBN 소스 순서가 점수에 영향을 주는지 보려면 셔플이 실제로 배치 소속을
바꿔야 한다 — 안 바꾸면 실험 자체가 무의미해진다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from infer_decomposed import DEVICE, adapt_bn  # noqa: E402


class _FlattenBN(nn.Module):
    """(batch,1,1,1) 입력을 평탄화해 BatchNorm1d에 먹인다. adapt_bn은 `model.modules()`로
    BatchNorm1d/2d만 찾아 리셋하므로 실제 구조 회귀기 대신 최소 모델로 계약을 검사할 수 있다."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        return self.bn(x.reshape(x.shape[0], -1))


def _make_blocked_cache(tmp_path, n_per_block=384) -> Path:
    """블록 A(값 50, uint8) n장 + 블록 B(값 200) n장을 이어붙인 합성 real_sem.npy.
    (N, 1, 1) 형태 — adapt_bn은 픽셀 배열 형태를 몰라도 되므로 1x1이면 충분하다."""
    block_a = np.full((n_per_block, 1, 1), 50, dtype=np.uint8)
    block_b = np.full((n_per_block, 1, 1), 200, dtype=np.uint8)
    sem = np.concatenate([block_a, block_b], axis=0)
    np.save(tmp_path / "real_sem.npy", sem)
    return tmp_path


def test_adabn_shuffle_changes_running_var_not_running_mean(tmp_path):
    """블록 A 384장 + 블록 B 384장, batch=64(=6배치/블록, 768/64로 딱 나뉨 → 배치 크기 동일).

    셔플 없음: 배치 경계가 블록 경계와 정렬돼 있어 배치마다 값이 상수 → running_var == 0.
    셔플(seed=42): 배치가 두 블록을 섞어 배치 내 분산이 생김 → running_var > 0.
    momentum=None 누적평균은 배치 크기가 같으면 그룹 방식과 무관하게 배치 평균들의 등가중
    평균 = 전체 평균이 되므로, 두 경우 모두 running_mean은 전체 평균으로 같아야 한다.
    """
    cache = _make_blocked_cache(tmp_path)

    plain = _FlattenBN().to(DEVICE)
    n_plain = adapt_bn(plain, cache, "real", lut=None, batch=64, shuffle_seed=None)

    shuffled = _FlattenBN().to(DEVICE)
    n_shuffled = adapt_bn(shuffled, cache, "real", lut=None, batch=64, shuffle_seed=42)

    assert n_plain == n_shuffled == 768

    grand_mean = (50 + 200) / 2 / 255.0
    assert plain.bn.running_mean.item() == pytest.approx(grand_mean, abs=1e-5)
    assert shuffled.bn.running_mean.item() == pytest.approx(grand_mean, abs=1e-5), (
        "배치 크기가 같은데 셔플이 running_mean까지 바꾸면 셔플 구현이 배치 소속 크기를 "
        "깨뜨렸다는 신호다")

    assert plain.bn.running_var.item() == pytest.approx(0.0, abs=1e-12), (
        "블록이 배치 경계와 정렬돼 있으면 배치마다 상수값이라 분산이 0이어야 한다")
    assert shuffled.bn.running_var.item() > 1e-4, (
        "셔플이 배치 소속을 실제로 바꾸지 않으면(버그) 분산이 여전히 0으로 남는다"
    )


def test_adabn_shuffle_is_reproducible_with_same_seed(tmp_path):
    """같은 시드는 같은 순열이어야 한다 — 재현 가능한 실험의 최소 조건."""
    cache = _make_blocked_cache(tmp_path)

    a = _FlattenBN().to(DEVICE)
    adapt_bn(a, cache, "real", lut=None, batch=64, shuffle_seed=7)
    b = _FlattenBN().to(DEVICE)
    adapt_bn(b, cache, "real", lut=None, batch=64, shuffle_seed=7)

    assert a.bn.running_var.item() == pytest.approx(b.bn.running_var.item(), abs=0.0)
