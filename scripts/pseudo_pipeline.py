"""avg-앵커 pseudo-labeling 파이프라인: f → real 의사GT → g → test 제출본. standalone.

    f(real SEM, real average_depth) → real depth 의사라벨      [전역 레벨을 실측 GT로 고정]
    g(real SEM) → depth  를 그 의사라벨로 학습                  [test엔 avg가 없으므로 1채널]
    g(test SEM) → 제출 zip

real average_depth는 **사이트 단위**(2,059 사이트, 사이트당 SEM ~29.5장이 값 하나 공유)라,
같은 사이트의 이미지들은 같은 avg를 입력으로 받는다 — 학습 때 지터를 넣은 이유가 이것이다.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_avgcond import H, W, AvgCondMLP, ensure_utf8_console, seed_everything  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlainMLP(nn.Module):
    """g: SEM(3456) → depth(3456). test엔 average_depth가 없으므로 avg 입력을 받지 않는다."""

    def __init__(self):
        super().__init__()
        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]
        self.encoder = nn.Sequential(*block(H * W, 1024), *block(1024, 512),
                                     *block(512, 256), *block(256, 128))
        self.decoder = nn.Sequential(*block(128, 256), *block(256, 512),
                                     *block(512, 1024), nn.Linear(1024, H * W))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ArrayDataset(Dataset):
    """(N,H,W) uint8 SEM → (N,H,W) uint8 depth, 둘 다 평탄화 + /255."""

    def __init__(self, sem, depth):
        self.sem, self.depth = sem, depth

    def __len__(self):
        return len(self.sem)

    def __getitem__(self, i):
        x = np.ascontiguousarray(self.sem[i]).astype(np.float32).ravel() / 255.0
        y = np.ascontiguousarray(self.depth[i]).astype(np.float32).ravel() / 255.0
        return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def gen_pseudo(f, sem, avg_per_img, batch_size=512):
    """f(real SEM, real average_depth) → 의사 depth map (uint8)."""
    f.eval()
    out = np.empty((len(sem), H, W), dtype=np.uint8)
    for s in tqdm(range(0, len(sem), batch_size), desc="의사GT 생성", leave=False):
        imgs = np.asarray(sem[s:s + batch_size]).astype(np.float32).reshape(-1, H * W) / 255.0
        a = (avg_per_img[s:s + batch_size] / 255.0).astype(np.float32)[:, None]
        x = torch.from_numpy(np.concatenate([imgs, a], axis=1)).to(DEVICE)
        pred = (f(x) * 255.0).round().clamp(0, 255)
        out[s:s + len(imgs)] = pred.cpu().numpy().reshape(-1, H, W).astype(np.uint8)
    return out


def train_g(sem, pseudo, epochs, batch_size, lr, workers=0):
    g = PlainMLP().to(DEVICE)
    opt = torch.optim.AdamW(g.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.L1Loss().to(DEVICE)
    dl = DataLoader(ArrayDataset(sem, pseudo), batch_size=batch_size, shuffle=True,
                    num_workers=workers)
    for ep in range(1, epochs + 1):
        g.train()
        losses = []
        for x, y in tqdm(dl, desc=f"g epoch {ep}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(g(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        print(f"  g epoch {ep}: train_loss={np.mean(losses):.5f}", flush=True)
    return g


@torch.no_grad()
def predict_and_zip(g, cache_dir: Path, zip_path: Path, batch_size=512):
    g.eval()
    sem = np.load(cache_dir / "test_sem.npy", mmap_mode="r")
    names = json.loads((cache_dir / "test_names.json").read_text(encoding="utf-8"))
    work = cache_dir.parent / "submission_work"
    work.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for s in tqdm(range(0, len(sem), batch_size), desc="test 추론", leave=False):
            imgs = np.asarray(sem[s:s + batch_size]).astype(np.float32).reshape(-1, H * W) / 255.0
            pred = (g(torch.from_numpy(imgs).to(DEVICE)) * 255.0).round().clamp(0, 255)
            arr = pred.cpu().numpy().reshape(-1, H, W).astype(np.uint8)
            for j, img in enumerate(arr):
                name = names[s + j]
                png = work / name
                cv2.imwrite(str(png), img)
                zf.write(png, arcname=name)
    return len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f-ckpt", required=True, help="EXP-001 생성기 체크포인트")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--submit", required=True, help="생성할 제출 zip 경로")
    ap.add_argument("--g-epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_utf8_console()
    seed_everything(args.seed)
    cache_dir = Path(args.cache_dir)
    print(f"device: {DEVICE} | f: {args.f_ckpt}", flush=True)

    real_sem = np.load(cache_dir / "real_sem.npy", mmap_mode="r")
    site_idx = np.load(cache_dir / "real_site_idx.npy")
    site_depth = np.load(cache_dir / "real_site_depth.npy")
    avg_per_img = site_depth[site_idx].astype(np.float32)  # 사이트 값을 그 사이트 이미지들에 배포
    print(f"real {len(real_sem)}장 / {len(site_depth)}사이트 | "
          f"avg {avg_per_img.min():.1f}~{avg_per_img.max():.1f}", flush=True)

    f = AvgCondMLP().to(DEVICE)
    f.load_state_dict(torch.load(args.f_ckpt, map_location=DEVICE))
    pseudo = gen_pseudo(f, real_sem, avg_per_img)
    pm = pseudo.reshape(len(pseudo), -1).mean(1)
    print(f"의사GT: 픽셀 {pseudo.min()}~{pseudo.max()} | 이미지평균 {pm.min():.1f}~{pm.max():.1f}",
          flush=True)

    g = train_g(real_sem, pseudo, args.g_epochs, args.batch_size, args.lr, args.num_workers)
    n = predict_and_zip(g, cache_dir, Path(args.submit))
    print(json.dumps({"f_ckpt": args.f_ckpt, "zip": args.submit, "n": n,
                      "pseudo_mean_range": [float(pm.min()), float(pm.max())]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
