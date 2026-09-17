"""adabn.adapt_bn_exact — 층별 순차 전역 통계 추정기 (오프라인·합성 캐시. DEVICE가 cuda면 쓰지만 없어도 통과한다).

무엇을 고정하는가: ① 첫 BN 층의 통계가 적응 집합 **전체**의 정확한 평균/불편분산과 같다,
② 두 번째 BN 층에서 기존 batch 방식과 **다른 값**을 낸다 (같은 값이면 실험이 무의미하다),
③ 통계 덤프가 층 이름 → mean/var로 나온다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# 워크트리는 dev 그룹만 sync하므로 torch가 없다 -- collection 단계에서 ImportError로 죽는
# 대신 스킵으로 넘겨 `uv run pytest -q`가 거기서도 끝까지 돈다.
pytest.importorskip("torch")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ai_co_scientist.adabn import (
    adapt_bn_exact, bn_modules, collect_bn_stats, iter_cache_batches, save_bn_stats,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from infer_decomposed import DEVICE, adapt_bn  # noqa: E402

N_PER_BLOCK = 384
VAL_A, VAL_B = 50, 200


class _TwoBN(nn.Module):
    """BN → ReLU → BN. 비선형이 사이에 있어야 층 2의 입력 분포가 층 1의 통계 방식에 의존한다."""

    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        return self.bn2(torch.relu(self.bn1(x)))


def _blocked_cache(tmp_path: Path) -> Path:
    """블록 A(50) 384장 + 블록 B(200) 384장. 배치 경계가 블록 경계와 어긋나게 만들 수 있다."""
    sem = np.concatenate([np.full((N_PER_BLOCK, 1, 1), VAL_A, dtype=np.uint8),
                          np.full((N_PER_BLOCK, 1, 1), VAL_B, dtype=np.uint8)])
    np.save(tmp_path / "real_sem.npy", sem)
    return tmp_path


def _batches(cache, **kw):
    return lambda: iter_cache_batches(cache, ["real"], None, batch=64, **kw)


def test_exact_first_layer_matches_global_moments(tmp_path):
    """층 1은 앞 층이 없으므로 정답이 손으로 계산된다: 전체 집합의 평균과 **불편**분산."""
    cache = _blocked_cache(tmp_path)
    model = _TwoBN()
    seen = adapt_bn_exact(model, _batches(cache))

    n = 2 * N_PER_BLOCK
    x = np.concatenate([np.full(N_PER_BLOCK, VAL_A), np.full(N_PER_BLOCK, VAL_B)]) / 255.0
    assert seen == n
    assert model.bn1.running_mean.item() == pytest.approx(x.mean(), abs=1e-6)
    assert model.bn1.running_var.item() == pytest.approx(x.var(ddof=1), abs=1e-6)


def test_exact_differs_from_batch_estimator_at_second_layer(tmp_path):
    """층 2에서 두 추정기가 갈리는지 — 갈리지 않으면 --adabn-stats exact는 실험이 아니다.

    층 1은 배치 크기가 같아 누적평균이 전역 평균과 일치하므로 mean은 같고, 분산이 다르다
    (batch 방식은 배치별 불편분산의 평균, exact는 전역 불편분산).
    """
    cache = _blocked_cache(tmp_path)
    exact, batch = _TwoBN().to(DEVICE), _TwoBN().to(DEVICE)
    adapt_bn_exact(exact, _batches(cache), device=DEVICE)
    adapt_bn(batch, cache, "real", lut=None, batch=64)

    assert batch.bn1.running_mean.item() == pytest.approx(exact.bn1.running_mean.item(), abs=1e-6)
    assert batch.bn1.running_var.item() != pytest.approx(exact.bn1.running_var.item(), abs=1e-4)
    assert batch.bn2.running_mean.item() != pytest.approx(exact.bn2.running_mean.item(), abs=1e-4)


def test_exact_visits_every_layer_on_a_fresh_pass(tmp_path):
    """이터레이터를 재생성하지 못하면 두 번째 층이 빈 집합을 보고 통계가 초기값에 머문다."""
    cache = _blocked_cache(tmp_path)
    model = _TwoBN()
    adapt_bn_exact(model, _batches(cache))
    assert model.bn2.running_mean.item() != pytest.approx(0.0, abs=1e-6)
    assert model.bn2.running_var.item() != pytest.approx(1.0, abs=1e-6)


def test_exact_raises_on_empty_adaptation_set(tmp_path):
    model = _TwoBN()
    with pytest.raises(RuntimeError, match="적응 집합이 비었다"):
        adapt_bn_exact(model, lambda: iter([]))


def test_bn_modules_lists_layers_in_definition_order():
    assert [n for n, _ in bn_modules(_TwoBN())] == ["bn1", "bn2"]


def test_dump_json_and_npz(tmp_path):
    cache = _blocked_cache(tmp_path)
    model = _TwoBN()
    adapt_bn_exact(model, _batches(cache))
    stats = collect_bn_stats(model)
    assert set(stats) == {"bn1", "bn2"}

    jp = save_bn_stats(model, tmp_path / "stats.json")
    loaded = json.loads(jp.read_text(encoding="utf-8"))
    assert loaded["bn1"]["mean"] == pytest.approx(stats["bn1"]["mean"])

    np_path = save_bn_stats(model, tmp_path / "stats.npz")
    with np.load(np_path) as z:
        assert set(z.files) == {"bn1.mean", "bn1.var", "bn2.mean", "bn2.var"}
        assert z["bn2.var"] == pytest.approx(stats["bn2"]["var"])


def test_iter_cache_batches_drop_last_and_shuffle(tmp_path):
    """배치 이터레이터가 기존 adapt_bn 루프와 같은 표본 수·같은 값 집합을 낸다."""
    cache = _blocked_cache(tmp_path)
    full = list(iter_cache_batches(cache, ["real"], None, batch=100))
    assert sum(int(b.shape[0]) for b in full) == 768
    dropped = list(iter_cache_batches(cache, ["real"], None, batch=100, drop_last=True))
    assert sum(int(b.shape[0]) for b in dropped) == 700

    plain = torch.cat(full).flatten()
    shuffled = torch.cat(list(iter_cache_batches(cache, ["real"], None, batch=100,
                                                 shuffle_seed=42))).flatten()
    assert plain.sum().item() == pytest.approx(shuffled.sum().item(), abs=1e-4)
    assert not torch.equal(plain, shuffled)  # 배치 소속이 실제로 바뀌었다
