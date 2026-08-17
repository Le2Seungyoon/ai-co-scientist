"""avg-조건 생성기 f(SEM, average_depth) → depth map. standalone (Lightning Studio 업로드 가능).

용도: sim에서 `f(sim SEM, sim avg) → sim depth GT`를 학습해 두고, 나중에
`f(real SEM, real average_depth)` 로 실측 의사 GT를 만든다. avg는 real에서 유일하게
주어지는 진짜 GT(사이트별 평균 깊이)라, 전역 레벨을 실측값으로 못박는 앵커 역할을 한다.

**2026-08-02: 이 전제는 무효다 — `docs/data-facts.md` §4를 먼저 읽을 것.**
운영자 답변으로 `average_depth`가 확정됐다: depth map intensity의 단순 평균이며, 그것도
hole crop이 아니라 **원본 전체 SEM 영상** 기준이다. 즉
  (a) 기본값 `hole_mean`은 틀린 통계량이고 (corr(image_mean, hole_mean)=0.884, 동일 아님),
  (b) 애초에 avg는 우리가 예측하는 crop depth map의 평균이 아니라서 "전역 레벨 앵커"가 될 수
      없다 — 위 문단의 앵커 서술은 히스토리로만 유효하다.
레벨은 avg가 아니라 **배경 레벨 L ∈ {140,150,160,170} 4택1**로 다뤄야 하며, 조건 입력이 아니라
타깃 재매개화 `d = L − s·(L−20)`로 넣는다 (조건 입력은 출력을 구속하지 못한다 — EXP-002).

기본값 `hole_mean`은 **EXP-001 재현을 위해 그대로 둔다.** 새 실험에 이 스크립트를 쓰지 말 것.
아래는 당시 근거 기록: real 사이트 단위(2,059 사이트, 사이트당 SEM ~29.5장이 값 하나를 공유),
실측 105.3~142.2 / sim depth 배경 {140,150,160,170}, 구멍 바닥 ≈ 0~30 / depth map 1장 ↔
SEM 2장(itr0/itr1). 후보 통계량을 real 사분위(111.8/121.5/132.0)와 대조해 오차가 가장 작은
`hole_mean`(8.9)을 골랐으나, 그 대조 자체가 crop↔전체영상 차이를 무시한 것이었다.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

H, W = 72, 48
AVG_DEFS = ("hole_mean", "max_depth", "image_mean")


def ensure_utf8_console() -> None:
    """Windows cp949 콘솔에서 유니코드(—, → 등) print가 UnicodeEncodeError로 죽는 걸 방지.

    standalone 스크립트라 `ai_co_scientist.config`의 동명 함수를 import할 수 없어 여기에 둔다
    (Lightning Studio에 파일만 올려 돌리기 때문). 이 파일을 import하는 standalone 스크립트들이
    공유한다 — 각 진입점에서 최초 1회 호출할 것.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ── avg 정의 ────────────────────────────────────────────────

def compute_avg(depth: np.ndarray, how: str) -> np.ndarray:
    """depth (N,H,W) uint8 → avg 스칼라 (N,). 배경은 테두리 중앙값으로 잡는다."""
    d = depth.astype(np.float32)
    flat = d.reshape(len(d), -1)
    if how == "image_mean":
        return flat.mean(1)
    border = np.concatenate([d[:, 0, :], d[:, -1, :], d[:, :, 0], d[:, :, -1]], axis=1)
    bg = np.median(border, axis=1)[:, None]
    depth_px = bg - flat  # 배경=0, 구멍=양수
    if how == "max_depth":
        return depth_px.max(1)
    if how == "hole_mean":  # 구멍 내부(최대깊이의 50% 초과)만 평균 → "평균 깊이"
        mask = depth_px > (depth_px.max(1, keepdims=True) * 0.5)
        return (depth_px * mask).sum(1) / np.maximum(mask.sum(1), 1)
    raise ValueError(f"unknown avg-def: {how}")


def build_sim_case_cache(data_dir: Path, cache_dir: Path) -> None:
    """sim_sem.npy와 동일 순서(sorted glob)로 Case 라벨(1~4)을 파생 저장 — 이미지 로드 없음."""
    sem = sorted(glob.glob(str(data_dir / "simulation_data" / "SEM" / "*" / "*" / "*.png")))
    case = np.array([int(Path(p).parts[-3].split("_")[1]) for p in sem], dtype=np.int8)
    np.save(cache_dir / "sim_case.npy", case)
    print(f"sim_case cache: {len(case)} imgs", flush=True)


# ── 데이터 ──────────────────────────────────────────────────

class AvgCondDataset(Dataset):
    """(SEM 3456 + avg 1) → depth 3456. avg는 real에서 사이트 평균이라 개별 이미지의
    정확값이 아니다 → 학습 시 지터를 섞어 정확값 의존을 막는다."""

    def __init__(self, sem, depth, avg, avg_jitter: float = 0.0):
        self.sem, self.depth, self.avg, self.jitter = sem, depth, avg, avg_jitter

    def __len__(self):
        return len(self.sem)

    def __getitem__(self, i):
        x = np.ascontiguousarray(self.sem[i]).astype(np.float32).ravel() / 255.0
        a = float(self.avg[i])
        if self.jitter > 0:
            a += random.gauss(0.0, self.jitter)
        x = np.concatenate([x, [a / 255.0]]).astype(np.float32)
        y = np.ascontiguousarray(self.depth[i]).astype(np.float32).ravel() / 255.0
        return torch.from_numpy(x), torch.from_numpy(y)


class AvgCondMLP(nn.Module):
    """baseline MLP AE와 동일 구조에 입력만 +1 (avg 스칼라)."""

    def __init__(self, in_dim: int = H * W + 1):
        super().__init__()
        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]
        self.encoder = nn.Sequential(*block(in_dim, 1024), *block(1024, 512),
                                     *block(512, 256), *block(256, 128))
        self.decoder = nn.Sequential(*block(128, 256), *block(256, 512),
                                     *block(512, 1024), nn.Linear(1024, H * W))

    def forward(self, x):
        return self.decoder(self.encoder(x))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mse = nn.MSELoss().to(device)
    rmses = []
    for x, y in loader:
        pred = (model(x.to(device)) * 255.0).round().clamp(0, 255)
        true = (y.to(device) * 255.0).round().clamp(0, 255)
        rmses.append(torch.sqrt(mse(pred, true)).item())
    return float(np.mean(rmses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--output-dir", default="runtime/ckpt")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--avg-def", default="hole_mean", choices=AVG_DEFS)
    ap.add_argument("--avg-jitter", type=float, default=2.0, help="학습 시 avg 가우시안 지터 σ(0-255 스케일)")
    ap.add_argument("--val-case", type=int, default=4, help="통째로 격리할 Case 번호")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--save-epochs", type=int, nargs="+", default=[1, 3, 10, 30],
                    help="체크포인트를 남길 epoch들 (sim_val_rmse가 다른 모델 확보용)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-subsample", type=int, default=0)
    ap.add_argument("--val-subsample", type=int, default=8000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_utf8_console()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir, out_dir = Path(args.cache_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | avg-def: {args.avg_def}", flush=True)

    sem = np.load(cache_dir / "sim_sem.npy", mmap_mode="r")
    depth = np.load(cache_dir / "sim_depth.npy", mmap_mode="r")
    if not (cache_dir / "sim_case.npy").exists():
        build_sim_case_cache(Path(args.data_dir), cache_dir)
    case = np.load(cache_dir / "sim_case.npy")

    def pick(pool, cap):
        if cap <= 0 or cap >= len(pool):
            return np.sort(pool)
        rng = np.random.default_rng(args.seed)
        return np.sort(rng.choice(pool, size=cap, replace=False))

    tr_idx = pick(np.where(case != args.val_case)[0], args.train_subsample)
    va_idx = pick(np.where(case == args.val_case)[0], args.val_subsample)
    print(f"train {len(tr_idx)} (Case!={args.val_case}) / val {len(va_idx)} (Case=={args.val_case})",
          flush=True)

    tr_sem, tr_dep = np.asarray(sem[tr_idx]), np.asarray(depth[tr_idx])
    va_sem, va_dep = np.asarray(sem[va_idx]), np.asarray(depth[va_idx])
    tr_avg, va_avg = compute_avg(tr_dep, args.avg_def), compute_avg(va_dep, args.avg_def)
    print(f"avg({args.avg_def}) train 분포: {tr_avg.min():.1f}~{tr_avg.max():.1f} "
          f"(중앙 {np.median(tr_avg):.1f})", flush=True)

    tl = DataLoader(AvgCondDataset(tr_sem, tr_dep, tr_avg, args.avg_jitter),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    vl = DataLoader(AvgCondDataset(va_sem, va_dep, va_avg),  # val엔 지터 없음
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AvgCondMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.L1Loss().to(device)

    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(tl, desc=f"epoch {ep}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        rmse = validate(model, vl, device)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "sim_val_rmse": rmse})
        print(f"epoch {ep}: train_loss={np.mean(losses):.5f} sim_val_rmse={rmse:.4f}", flush=True)
        if ep in args.save_epochs:
            path = out_dir / f"{args.run_name}-e{ep}.pt"
            torch.save(model.state_dict(), path)
            print(f"  saved {path}", flush=True)

    saved = [{"epoch": h["epoch"], "sim_val_rmse": h["sim_val_rmse"],
              "ckpt": str(out_dir / f"{args.run_name}-e{h['epoch']}.pt")}
             for h in history if h["epoch"] in args.save_epochs]
    print(json.dumps({
        "run_name": args.run_name,
        "x_domain": "sim", "y_source": "sim_depth_gt",
        "avg_def": args.avg_def, "avg_jitter": args.avg_jitter,
        "val": {"name": "sim_val_rmse", "x_domain": "sim", "y_source": "sim_depth_gt",
                "best": min(h["sim_val_rmse"] for h in history),
                "by_epoch": {h["epoch"]: round(h["sim_val_rmse"], 4) for h in history}},
        "checkpoints": saved,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
