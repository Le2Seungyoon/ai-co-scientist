"""SEM -> Depth 개선 학습기 — baseline(MLP AE) 대비 Conv U-Net + 스윕 가능한 하이퍼파라미터.

Lightning Studio에서 단독 실행 가능해야 하므로 repo 패키지를 import하지 않는다 (standalone).
baseline_sem_depth.py와 동일한 데이터 규약(경로·짝맞춤·48x72·[0,1] 스케일·80/20 split)을
유지해 val_rmse가 직접 비교 가능하다. 최초 실행 시 PNG 35만 장을 uint8 npy 캐시로 굽고
(IO 병목 제거), 이후 실행은 캐시를 mmap으로 읽는다.

사용 예:
  python train_sem_depth.py --arch unet --width 32 --lr 1e-3 --loss l1 --epochs 30
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

H, W = 72, 48


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ── 데이터 캐시 ──────────────────────────────────────────────

def _load_png_stack(paths: list[str], desc: str) -> np.ndarray:
    arr = np.empty((len(paths), H, W), dtype=np.uint8)
    for i, p in enumerate(tqdm(paths, desc=desc)):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        arr[i] = img
    return arr


def build_cache(data_dir: Path, cache_dir: Path) -> None:
    """baseline과 동일한 정렬·짝맞춤(SEM itr0/itr1 x2 depth 복제) 후 npy로 저장."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sem = sorted(glob.glob(str(data_dir / "simulation_data" / "SEM" / "*" / "*" / "*.png")))
    dep = sorted(
        glob.glob(str(data_dir / "simulation_data" / "Depth" / "*" / "*" / "*.png")) * 2)
    assert len(sem) == len(dep), f"SEM/Depth 짝 불일치: {len(sem)} vs {len(dep)}"
    np.save(cache_dir / "sim_sem.npy", _load_png_stack(sem, "cache sim SEM"))
    np.save(cache_dir / "sim_depth.npy", _load_png_stack(dep, "cache sim Depth"))

    test = sorted(glob.glob(str(data_dir / "test" / "SEM" / "*.png")))
    np.save(cache_dir / "test_sem.npy", _load_png_stack(test, "cache test SEM"))
    (cache_dir / "test_names.json").write_text(
        json.dumps([os.path.basename(p) for p in test]), encoding="utf-8")
    print(f"cache built at {cache_dir}: sim={len(sem)}, test={len(test)}")


def build_realproxy_cache(data_dir: Path, cache_dir: Path) -> None:
    """실측 train SEM + 사이트별 평균 depth를 캐시 (도메인 매칭 프록시용).

    경로 train/SEM/Depth_<d>/site_<n>/SEM_*.png → csv 키 'depth_<d>_site_<n>'로 평균 depth 조회.
    """
    import csv as _csv
    avg: dict[str, float] = {}
    with open(data_dir / "train" / "average_depth.csv", encoding="utf-8") as f:
        r = _csv.reader(f)
        next(r)
        for name, v in r:
            avg[name.strip().lower()] = float(v)
    paths = sorted(glob.glob(str(data_dir / "train" / "SEM" / "*" / "site_*" / "*.png")))
    site_idx = np.empty(len(paths), dtype=np.int32)
    key_to_idx: dict[str, int] = {}
    depths: list[float] = []
    for i, p in enumerate(paths):
        parts = Path(p).parts  # split('/') 금지 (Windows 역슬래시) — Path.parts 사용
        key = f"{parts[-3]}_{parts[-2]}".lower()  # depth_140_site_00233
        if key not in key_to_idx:
            if key not in avg:
                raise KeyError(f"average_depth.csv에 없는 사이트: {key}")
            key_to_idx[key] = len(depths)
            depths.append(avg[key])
        site_idx[i] = key_to_idx[key]
    np.save(cache_dir / "real_sem.npy", _load_png_stack(paths, "cache real train SEM"))
    np.save(cache_dir / "real_site_idx.npy", site_idx)
    np.save(cache_dir / "real_site_depth.npy", np.array(depths, dtype=np.float32))
    print(f"realproxy cache: {len(paths)} imgs, {len(depths)} sites", flush=True)


@torch.no_grad()
def validate_real_avgdepth(model, cache_dir: Path, device, batch_size: int = 256) -> float:
    """X = 실측 train SEM, y = average_depth.csv(사이트별 평균, 실측 GT) → RMSE.

    도메인은 test와 같은 real이지만 **평균(레벨)만** 재고 픽셀 구조는 못 본다.
    구조가 이 태스크의 병목이므로 이 값 하나로 리더보드를 대신 판정하지 말 것 —
    기록소의 metric 필드에 그대로 남겨 해석은 사람/critic이 한다.
    """
    sem = np.load(cache_dir / "real_sem.npy", mmap_mode="r")
    site_idx = np.load(cache_dir / "real_site_idx.npy")
    true_depth = np.load(cache_dir / "real_site_depth.npy")
    sum_d = np.zeros(len(true_depth), dtype=np.float64)
    cnt = np.zeros(len(true_depth), dtype=np.float64)
    model.eval()
    for start in range(0, len(sem), batch_size):
        imgs = np.asarray(sem[start:start + batch_size])
        batch = torch.from_numpy(
            np.stack([im.astype(np.float32) / 255.0 for im in imgs])).unsqueeze(1)
        pred = (model(batch.to(device)) * 255.0).clamp(0, 255)
        per_img = pred.mean(dim=(1, 2, 3)).cpu().numpy()
        sites = site_idx[start:start + len(per_img)]
        np.add.at(sum_d, sites, per_img)
        np.add.at(cnt, sites, 1)
    pred_site = sum_d / np.maximum(cnt, 1)
    return float(np.sqrt(np.mean((pred_site - true_depth) ** 2)))


class AugmentDataset(Dataset):
    """uint8 스택 -> SEM은 [0,1] 정규화, depth는 /255 + (옵션) SEM/Depth 동기 좌우반전."""

    def __init__(self, sem: np.ndarray, depth: np.ndarray, hflip: bool):
        self.sem, self.depth, self.hflip = sem, depth, hflip

    def __len__(self):
        return len(self.sem)

    def __getitem__(self, i):
        sem, depth = self.sem[i], self.depth[i]
        if self.hflip and random.random() < 0.5:
            sem, depth = sem[:, ::-1], depth[:, ::-1]
        sem = np.ascontiguousarray(sem).astype(np.float32) / 255.0
        return (torch.from_numpy(sem).unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0).float() / 255.0)


# ── 모델 ────────────────────────────────────────────────────

class BaselineMLP(nn.Module):
    """baseline_sem_depth.py의 BaseModel 동일 구조 (비교 대조군)."""

    def __init__(self):
        super().__init__()
        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]
        self.encoder = nn.Sequential(*block(H * W, 1024), *block(1024, 512),
                                     *block(512, 256), *block(256, 128))
        self.decoder = nn.Sequential(*block(128, 256), *block(256, 512),
                                     *block(512, 1024), nn.Linear(1024, H * W))

    def forward(self, x):
        b = x.shape[0]
        return self.decoder(self.encoder(x.view(b, -1))).view(b, 1, H, W)


def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNetSmall(nn.Module):
    """3단 U-Net — 72x48은 2회 다운샘플(18x12)까지 나누어떨어짐."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.enc3 = conv_block(1, w), conv_block(w, w * 2), conv_block(w * 2, w * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.dec2 = conv_block(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.dec1 = conv_block(w * 2, w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class SmpModel(nn.Module):
    """일반 segmentation_models_pytorch dense-prediction 모델.

    72×48을 인코더 요구 크기로 리사이즈 후 출력을 72×48로 원복. 1채널 SEM(in_channels=1,
    smp가 사전학습 1st-conv를 채널평균으로 초기화). arch: unet/segformer/dpt/fpn/deeplabv3plus/
    unetpp/manet. DPT는 ViT 고정 크기(224) 필요, 나머지는 96×64(32 배수).
    """

    def __init__(self, arch: str, encoder: str, in_size: tuple[int, int]):
        super().__init__()
        import segmentation_models_pytorch as smp
        archs = {"unet": smp.Unet, "segformer": smp.Segformer, "dpt": smp.DPT,
                 "fpn": smp.FPN, "deeplabv3plus": smp.DeepLabV3Plus,
                 "unetpp": smp.UnetPlusPlus, "manet": smp.MAnet}
        self.net = archs[arch](encoder_name=encoder, encoder_weights="imagenet",
                               in_channels=1, classes=1)
        self._in = in_size

    def forward(self, x):
        h, w = x.shape[-2:]
        x = nn.functional.interpolate(x, size=self._in, mode="bilinear", align_corners=False)
        x = self.net(x)
        return nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


def make_model(arch: str, width: int) -> nn.Module:
    if arch == "mlp":
        return BaselineMLP()
    if arch == "unet":
        return UNetSmall(width)
    if arch.startswith("pretrained:"):  # 하위호환: pretrained:resnet18 → smp Unet
        return SmpModel("unet", arch.split(":", 1)[1], (96, 64))
    if arch.startswith("smp:"):  # smp:<arch>:<encoder> (예: smp:segformer:mit_b0)
        _, a, enc = arch.split(":", 2)
        return SmpModel(a, enc, (224, 224) if a == "dpt" else (96, 64))
    raise ValueError(arch)


# ── 학습/평가 ────────────────────────────────────────────────

@torch.no_grad()
def validate(model, criterion, loader, device):
    model.eval()
    mse = nn.MSELoss().to(device)
    losses, rmses = [], []
    for sem, depth in loader:
        sem, depth = sem.to(device), depth.to(device)
        pred = model(sem)
        losses.append(criterion(pred, depth).item())
        p255 = (pred * 255.0).round().clamp(0, 255)
        t255 = (depth * 255.0).round().clamp(0, 255)
        rmses.append(torch.sqrt(mse(p255, t255)).item())
    return float(np.mean(losses)), float(np.mean(rmses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="runtime/cache")  # config.yaml paths.cache_dir와 동일
    ap.add_argument("--output-dir", default="runtime/ckpt")
    ap.add_argument("--arch", default="unet",
                    help="mlp | unet | pretrained:<encoder> (예: pretrained:resnet18)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["l1", "l2", "huber"], default="l1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=0, help="선형 warmup (트랜스포머 안정화)")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="grad norm 클리핑 (0=off, 트랜스포머 권장 1.0)")
    ap.add_argument("--no-amp", action="store_true", help="AMP 끄기 (트랜스포머 fp16 발산 회피)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-subsample", type=int, default=0,
                    help="학습 표본 상한 (0=전체). 저비용 스크리닝용 데이터 축소")
    ap.add_argument("--val-subsample", type=int, default=0,
                    help="검증 표본 상한 (0=전체). 스크리닝 시 검증도 빠르게")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--real-avgdepth", action="store_true",
                    help="학습 후 실측 SEM→average_depth RMSE도 계산 (real 도메인, 레벨만 측정)")
    ap.add_argument("--build-cache-only", action="store_true")
    args = ap.parse_args()

    data_dir, cache_dir = Path(args.data_dir), Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (cache_dir / "sim_sem.npy").exists():
        build_cache(data_dir, cache_dir)
    if args.real_avgdepth and not (cache_dir / "real_sem.npy").exists():
        build_realproxy_cache(data_dir, cache_dir)
    if args.build_cache_only:
        return

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    sem = np.load(cache_dir / "sim_sem.npy", mmap_mode="r")
    depth = np.load(cache_dir / "sim_depth.npy", mmap_mode="r")
    n_train = int(len(sem) * 0.8)

    def subsample(lo, hi, cap):
        """[lo,hi) 구간에서 최대 cap개를 고정 시드로 무작위 추출(정렬해 mmap 효율 유지). cap=0이면 전체."""
        if cap <= 0 or cap >= (hi - lo):
            return slice(lo, hi)
        rng = np.random.default_rng(args.seed)
        return np.sort(rng.choice(np.arange(lo, hi), size=cap, replace=False))

    tr_idx = subsample(0, n_train, args.train_subsample)
    va_idx = subsample(n_train, len(sem), args.val_subsample)
    # mmap fancy-index는 메모리로 로드됨 — 스크리닝의 작은 cap에서만 쓰고, 전체(slice)는 mmap 유지
    train_ds = AugmentDataset(sem[tr_idx], depth[tr_idx], hflip=args.hflip)
    val_ds = AugmentDataset(sem[va_idx], depth[va_idx], hflip=False)
    print(f"train={len(train_ds)} val={len(val_ds)} "
          f"(subsample tr={args.train_subsample or 'full'} va={args.val_subsample or 'full'})", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    run_name = args.run_name or (
        f"{args.arch}-w{args.width}-lr{args.lr:g}-{args.loss}"
        f"{'-hflip' if args.hflip else ''}-e{args.epochs}")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ai-co-scientist"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_name,
        config={"arch": args.arch, "width": args.width, "lr": args.lr, "loss": args.loss,
                "epochs": args.epochs, "batch_size": args.batch_size, "hflip": args.hflip,
                "seed": args.seed, "train_subsample": args.train_subsample,
                "val_subsample": args.val_subsample, "warmup_epochs": args.warmup_epochs,
                "grad_clip": args.grad_clip},
        mode="online" if os.environ.get("WANDB_API_KEY") else "disabled",
    )

    model = make_model(args.arch, args.width).to(device)
    criterion = {"l1": nn.L1Loss(), "l2": nn.MSELoss(),
                 "huber": nn.HuberLoss(delta=0.1)}[args.loss].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cosine_t = max(1, args.epochs - args.warmup_epochs)
    if args.warmup_epochs > 0:  # 선형 warmup → cosine (트랜스포머 발산 방지)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                               total_iters=args.warmup_epochs),
             torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t)],
            milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_rmse, best_path = float("inf"), output_dir / f"{run_name}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for sem_b, depth_b in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            sem_b = sem_b.to(device, non_blocking=True)
            depth_b = depth_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(sem_b), depth_b)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        val_loss, val_rmse = validate(model, criterion, val_loader, device)
        train_loss = float(np.mean(losses))
        print(f"epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
              f"val_rmse={val_rmse:.5f}", flush=True)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_rmse": val_rmse,
                   "lr": scheduler.get_last_lr()[0]}, step=epoch)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_path)

    wandb.summary["best_val_rmse"] = best_rmse
    print(f"best val_rmse: {best_rmse:.5f}")

    manifest = {
        "run_name": run_name,
        "x_domain": "sim",              # 학습 입력 = 시뮬레이션 SEM
        "y_source": "sim_depth_gt",     # 학습 정답 = 시뮬레이터 depth GT
        "val": {"name": "sim_val_rmse", "value": best_rmse,
                "x_domain": "sim", "y_source": "sim_depth_gt"},
        "real_avgdepth": None,
        "ckpt": str(best_path),
    }
    if args.real_avgdepth:
        model.load_state_dict(torch.load(best_path, map_location=device))
        value = validate_real_avgdepth(model, cache_dir, device, args.batch_size)
        manifest["real_avgdepth"] = {"name": "real_avgdepth_rmse", "value": value,
                                     "x_domain": "real", "y_source": "real_average_depth"}
        print(f"real_avgdepth_rmse: {value:.5f}", flush=True)

    wandb.finish()
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
