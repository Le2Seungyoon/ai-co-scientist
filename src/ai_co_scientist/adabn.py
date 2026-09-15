"""AdaBN 통계 추정기 — **exact(순차 전역)** 방식과 BN 통계 덤프.

기존 `scripts/infer_decomposed.py:adapt_bn`은 `model.train()`으로 한 번 훑어 모든 BN 층의
통계를 **동시에** 모은다. 그러면 층 k의 통계는 층 1..k−1이 *배치* 통계로 정규화한 활성값에서
모인 것인데, 정작 추론 때 층 1..k−1은 고정된 running stat을 쓴다 — 추정 대상과 사용 조건이
어긋난다. 배치 구성이 점수를 0.51이나 움직인 EXP-016은 이 어긋남의 크기를 보여준다.

여기 있는 `adapt_bn_exact`는 그 어긋남을 없앤다:

    for k in BN 층 순서:
        층 1..k−1 = 이미 확정된 통계로 eval 고정
        적응 집합 **전체**를 순전파하며 층 k 입력의 합/제곱합을 누적
        층 k의 running_mean/var := 전체 집합의 정확한 평균/분산 (배치별 추정의 평균이 아니라)

BN 층 수만큼 순전파가 필요하다 — 층 k의 입력만 있으면 되므로 그 지점에서 순전파를 끊어
비용을 절반가량으로 줄인다. 그래도 MLP(BN 7층)는 싸고, effb0(BN 59층)은 비싸다.

**이 모듈은 새 경로다.** 기존 `adapt_bn`은 손대지 않는다 — 지금까지 점수를 만든 레시피가
비트 단위로 재현되어야 하기 때문이다.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class _StopForward(Exception):
    """층 k의 입력을 받은 순간 순전파를 중단하기 위한 내부 신호."""


def bn_modules(model) -> list[tuple[str, nn.Module]]:
    """BN 층을 **정의 순서**로 나열한다. `named_modules()`의 순서가 곧 전방 순서라는 가정은
    표준 Sequential/블록 구조에서 성립한다."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _BN_TYPES)]


class _ChannelStats:
    """채널축(dim=1)만 남기고 전 원소에 대한 합/제곱합을 누적한다."""

    def __init__(self, n_channels: int, device):
        self.s = torch.zeros(n_channels, dtype=torch.float64, device=device)
        self.ss = torch.zeros(n_channels, dtype=torch.float64, device=device)
        self.n = 0

    def update(self, x: torch.Tensor) -> None:
        a = x.detach().to(torch.float64)
        dims = [d for d in range(a.dim()) if d != 1]
        self.s += a.sum(dim=dims)
        self.ss += (a * a).sum(dim=dims)
        self.n += int(np.prod([a.shape[d] for d in dims]))

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(평균, **불편**분산). PyTorch의 running_var가 불편분산이라 그 관례를 따른다."""
        mean = self.s / self.n
        var = (self.ss / self.n - mean * mean) * (self.n / max(self.n - 1, 1))
        return mean, var.clamp_min(0.0)


@torch.no_grad()
def adapt_bn_exact(model, batches, device=None) -> int:
    """BN 통계를 층별 순차 정확 추정으로 재계산한다. 반환: 마지막 패스에서 본 표본 수.

    batches: **매번 새 이터레이터를 돌려주는 콜러블**. BN 층 수만큼 적응 집합을 다시 훑기
    때문에 이터레이터 하나를 받으면 두 번째 층부터 빈 집합을 보게 된다.
    """
    layers = bn_modules(model)
    model.eval()  # 모든 층이 running stat으로 정규화 — 배치 통계는 어디에도 쓰이지 않는다
    seen = 0
    for _, m in layers:
        acc = _ChannelStats(m.num_features, m.running_mean.device)

        def _hook(_mod, inp, acc=acc):
            acc.update(inp[0])
            raise _StopForward  # 이 층 입력이면 충분하다 — 뒤쪽 층은 이번 패스와 무관

        handle = m.register_forward_pre_hook(_hook)
        seen = 0
        try:
            for x in batches():
                if device is not None:
                    x = x.to(device)
                try:
                    model(x)
                except _StopForward:
                    pass
                seen += int(x.shape[0])
        finally:
            handle.remove()
        if acc.n == 0:
            raise RuntimeError("적응 집합이 비었다 — BN 통계를 추정할 표본이 없다")
        mean, var = acc.finalize()
        m.running_mean.copy_(mean.to(m.running_mean.dtype))
        m.running_var.copy_(var.to(m.running_var.dtype))
        if m.num_batches_tracked is not None:
            m.num_batches_tracked.fill_(1)
    return seen


def collect_bn_stats(model) -> dict[str, dict[str, list[float]]]:
    """층 이름 → {mean, var}. 실험 1단계에서 batch 방식 통계와 층별로 비교하는 대상이다."""
    return {name: {"mean": m.running_mean.detach().cpu().tolist(),
                   "var": m.running_var.detach().cpu().tolist()}
            for name, m in bn_modules(model)}


def save_bn_stats(model, path: str | Path) -> Path:
    """BN 통계를 파일로 덤프한다 (`.npz`면 배열, 그 외에는 JSON). 덤프가 없으면 batch 방식과의
    층별 비교 자체가 불가능하므로 이것도 실험의 일부다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stats = collect_bn_stats(model)
    if p.suffix == ".npz":
        flat = {f"{n}.{k}": np.asarray(v, dtype=np.float64)
                for n, d in stats.items() for k, v in d.items()}
        np.savez(p, **flat)
    else:
        p.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    return p


def iter_cache_batches(cache: Path, names, lut=None, batch: int = 512,
                       drop_last: bool = False, shuffle_seed: int | None = None):
    """`<name>_sem.npy` 캐시들을 배치 텐서로 흘려보낸다 (uint8 → /255, (B,1,H,W)).

    `adapt_bn`이 자기 루프 안에서 하던 일과 같지만, exact 추정기는 적응 집합을 여러 번
    돌려야 해서 **다시 만들 수 있는** 형태가 필요하다. 기존 함수 본문은 건드리지 않는다 —
    점수를 만든 경로를 바꾸지 않는 것이 이 변경의 조건이다.
    """
    for name in names:
        sem = np.load(Path(cache) / f"{name}_sem.npy", mmap_mode="r")
        end = len(sem) - (len(sem) % batch) if drop_last else len(sem)
        order = (np.random.default_rng(shuffle_seed).permutation(len(sem))
                 if shuffle_seed is not None else None)
        for s in range(0, end, batch):
            idx = np.sort(order[s:s + batch]) if order is not None else slice(s, s + batch)
            a = np.asarray(sem[idx])
            if lut is not None:
                a = lut[a]
            yield torch.from_numpy(a.astype(np.float32)[:, None] / 255.0)
