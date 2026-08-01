"""avg-조건 pseudo-labeling 실험 (가벼운 검증).

아이디어(사용자 제안): average_depth를 '추가 입력'으로 쓴다.
  f(SEM, avg) → depth 를 sim에서 학습(avg=depth GT 평균) →
  real 의사라벨 생성 f(SEM_real, real_avg) [real GT 평균으로 전역 레벨 앵커] →
  g(SEM) → depth 를 real 의사라벨로 학습 → test는 g(SEM).

검증: g의 real_proxy(예측 평균 vs 진짜 average_depth) — sim-val이 못 보는 진짜 real 신호.
챔피언(sim-only)의 proxy를 기준선으로 비교. sim-val↔리더보드 상관과 별개로 real GT 기반 판정.

가볍게: 저스케일·소에폭. 개선 신호가 있으면 full-data로 확대(Lightning).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ai_co_scientist.core.config import ensure_utf8_console  # noqa: E402

ensure_utf8_console()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sem_depth import (  # noqa: E402
    H, W, apply_preproc, build_realproxy_cache, eval_realproxy, make_model,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731


class AvgCondDataset(Dataset):
    """2채널 입력 [SEM, avg(broadcast)] → depth. avg는 depth GT 평균(/255) 또는 명시값."""

    def __init__(self, sem, depth, avg=None, preproc="none"):
        self.sem, self.depth, self.avg, self.preproc = sem, depth, avg, preproc

    def __len__(self):
        return len(self.sem)

    def __getitem__(self, i):
        d = np.ascontiguousarray(self.depth[i]).astype(np.float32) / 255.0
        a = float(d.mean()) if self.avg is None else float(self.avg[i])
        sem = apply_preproc(np.ascontiguousarray(self.sem[i]), self.preproc)
        x = np.stack([sem, np.full((H, W), a, dtype=np.float32)])  # 2ch
        return torch.from_numpy(x), torch.from_numpy(d).unsqueeze(0)


class PlainDataset(Dataset):
    """1채널 SEM → depth (최종 g 학습용)."""

    def __init__(self, sem, depth, preproc="none"):
        self.sem, self.depth, self.preproc = sem, depth, preproc

    def __len__(self):
        return len(self.sem)

    def __getitem__(self, i):
        sem = apply_preproc(np.ascontiguousarray(self.sem[i]), self.preproc)
        d = np.ascontiguousarray(self.depth[i]).astype(np.float32) / 255.0
        return torch.from_numpy(sem).unsqueeze(0), torch.from_numpy(d).unsqueeze(0)


def train(model, train_ds, val_ds, epochs, bs, lr, tag):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.L1Loss()
    mse = nn.MSELoss()
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    best = 1e9
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        model.eval()
        rmses = []
        with torch.no_grad():
            for x, y in vl:
                p = (model(x.to(DEVICE)) * 255).round().clamp(0, 255)
                t = (y.to(DEVICE) * 255).round().clamp(0, 255)
                rmses.append(torch.sqrt(mse(p, t)).item())
        r = float(np.mean(rmses))
        best = min(best, r)
        log(f"[{tag}] epoch {ep}/{epochs} sim-val={r:.4f}")
    return best


@torch.no_grad()
def gen_pseudo(f, sem, avg_per_img, bs=256):
    """f(SEM_real, real_avg) → 의사 depth map (uint8). avg_per_img: 이미지별 avg(/255)."""
    f.eval()
    out = np.empty((len(sem), H, W), dtype=np.uint8)
    for s in range(0, len(sem), bs):
        imgs = np.asarray(sem[s:s + bs])
        a = avg_per_img[s:s + bs]
        x = np.stack([np.stack([apply_preproc(im, "none"),
                                np.full((H, W), av, dtype=np.float32)])
                      for im, av in zip(imgs, a)])
        p = (f(torch.from_numpy(x).to(DEVICE)) * 255).round().clamp(0, 255)
        out[s:s + bs] = p.squeeze(1).cpu().numpy().astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--arch", default="smp:unet:efficientnet-b0")
    ap.add_argument("--champion", default="runtime/final/L-effb0-l1-e40.pt")
    ap.add_argument("--f-subsample", type=int, default=20000)
    ap.add_argument("--g-subsample", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mix-sim", type=int, default=0, help=">0이면 g 학습에 sim 데이터 N장 혼합")
    args = ap.parse_args()
    cache = Path(args.cache_dir)

    if not (cache / "real_site_idx.npy").exists():
        build_realproxy_cache(Path(args.data_dir), cache)

    sim_sem = np.load(cache / "sim_sem.npy", mmap_mode="r")
    sim_depth = np.load(cache / "sim_depth.npy", mmap_mode="r")
    n = len(sim_sem)
    n_val = min(8000, n // 5)
    n_train = n - n_val
    real_sem = np.load(cache / "real_sem.npy", mmap_mode="r")
    real_site_idx = np.load(cache / "real_site_idx.npy")
    real_site_depth = np.load(cache / "real_site_depth.npy")  # 0-255
    real_avg_img = (real_site_depth[real_site_idx] / 255.0).astype(np.float32)
    log(f"sim train={n_train} val={n_val} | real={len(real_sem)} | avg 범위 "
        f"{real_site_depth.min():.1f}-{real_site_depth.max():.1f}")

    # ── 기준선: 챔피언(sim-only)의 real_proxy ──
    champ = make_model(args.arch, 32, in_channels=1).to(DEVICE)
    champ.load_state_dict(torch.load(args.champion, map_location=DEVICE))
    base_proxy = eval_realproxy(champ, cache, DEVICE, "none")
    log(f"[기준선] 챔피언 sim-only real_proxy = {base_proxy:.4f}")

    # ── Step 1: f(SEM,avg)→depth on sim ──
    tr = np.arange(0, n_train)[:: max(1, n_train // args.f_subsample)][:args.f_subsample]
    va = np.arange(n_train, n)[:n_val]
    f = make_model(args.arch, 32, in_channels=2)
    f_val = train(f, AvgCondDataset(sim_sem[tr], sim_depth[tr]),
                  AvgCondDataset(sim_sem[va], sim_depth[va]),
                  args.epochs, args.batch_size, args.lr, "f(avg-cond)")
    log(f"[Step1] f sim-val(avg-cond) = {f_val:.4f}")

    # ── Step 2: real 의사라벨 생성 (real_avg 앵커) ──
    real_pseudo = gen_pseudo(f, real_sem, real_avg_img)
    pm = real_pseudo.reshape(len(real_pseudo), -1).mean(1)
    anchor_rmse = float(np.sqrt(np.mean((pm - real_site_depth[real_site_idx]) ** 2)))
    log(f"[Step2] 의사라벨 생성 {real_pseudo.shape} | 평균 앵커 오차(should ~0)={anchor_rmse:.3f}")

    # ── 대조군: 같은 예산의 sim-only g (메커니즘 격리 — 챔피언은 full/e40이라 예산 다름) ──
    g_ctrl = make_model(args.arch, 32, in_channels=1)
    train(g_ctrl, PlainDataset(sim_sem[tr], sim_depth[tr]),
          PlainDataset(sim_sem[va], sim_depth[va]),
          args.epochs, args.batch_size, args.lr, "g-ctrl(sim-only)")
    ctrl_proxy = eval_realproxy(g_ctrl, cache, DEVICE, "none")
    log(f"[대조군] sim-only(같은예산) real_proxy = {ctrl_proxy:.4f}")

    # ── Step 3: g(SEM)→depth on real 의사라벨 (+옵션 sim 혼합) ──
    ridx = np.arange(len(real_sem))[:: max(1, len(real_sem) // args.g_subsample)][:args.g_subsample]
    g_sem, g_depth = real_sem[ridx], real_pseudo[ridx]
    if args.mix_sim > 0:
        midx = tr[:args.mix_sim]
        g_sem = np.concatenate([np.asarray(g_sem), np.asarray(sim_sem[midx])])
        g_depth = np.concatenate([np.asarray(g_depth), np.asarray(sim_depth[midx])])
        log(f"[Step3] sim {len(midx)}장 혼합 → g train={len(g_sem)}")
    g = make_model(args.arch, 32, in_channels=1)
    g_val = train(g, PlainDataset(g_sem, g_depth),
                  PlainDataset(sim_sem[va], sim_depth[va]),  # sim clean val (챔피언과 동일 축)
                  args.epochs, args.batch_size, args.lr, "g(SEM)")

    # ── Step 4: 판정 ──
    g_proxy = eval_realproxy(g, cache, DEVICE, "none")
    log("=" * 60)
    log(f"[결과] 챔피언 sim-only(full/e40)  real_proxy = {base_proxy:.4f}")
    log(f"[결과] 대조군 sim-only(같은예산)   real_proxy = {ctrl_proxy:.4f}")
    log(f"[결과] g(avg-pseudo)             real_proxy = {g_proxy:.4f}  "
        f"(대조군 대비 {'개선' if g_proxy < ctrl_proxy else '악화'} {ctrl_proxy - g_proxy:+.4f})")
    log(f"[결과] g sim-val = {g_val:.4f} | 평균앵커 오차 = {anchor_rmse:.3f}(f가 avg 사용하면 ~0)")
    torch.save(g.state_dict(), "runtime/final/g_avgpseudo.pt")
    print(json.dumps({"base_proxy": base_proxy, "ctrl_proxy": ctrl_proxy, "g_proxy": g_proxy,
                      "f_simval": f_val, "g_simval": g_val, "anchor_rmse": anchor_rmse}))


if __name__ == "__main__":
    main()
