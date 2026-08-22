"""뼈대 1/2 — 정규화 구조 회귀기 s: sim SEM → s. standalone.

    s = (L − d) / L        재구성  d = L·(1 − s)          [docs/data-facts.md §2]

L은 배경 레벨 {140,150,160,170}이고 Case가 결정한다. `d ∈ [0,L]`이라 `s ∈ [0,1]`이 정확히
성립하므로 sigmoid 출력이면 클램프가 필요 없다 — `(L−20)` 정규화를 쓰지 않는 이유가 이것이다
(전역 min이 0이라 s>1이 1퍼센트 발생하고, 클램프 손실이 RMSE 2.0에 달한다).

**레벨을 조건 입력이 아니라 타깃 변환으로 넣는 게 핵심이다.** 조건 입력은 출력을 구속하지
못한다(EXP-002: avg 조건화에도 출력 max가 std 8.1로 연속 분포). 재매개화하면 ŝ=0인 배경이
구조적으로 정확히 L̂이 된다.

백본은 `--arch`로 갈아끼운다 (`scripts/legacy/train_sem_depth.py`의 make_model 규약과 동일):
    mlp                     EXP-005 기준선. dense라 공간 귀납편향이 없다 — real 전이가 약하다
    unet[--width N]         3단 U-Net. 72×48은 2회 다운샘플(18×12)까지 나누어떨어진다
    smp:<arch>:<encoder>    예) smp:unet:efficientnet-b0 (`uv run --group baseline` 필요)
아키텍처 정의를 legacy/train_sem_depth.py에서 import하지 않는 이유: 그 파일은 최상단에서 wandb를
import하므로 추론 경로(infer_decomposed.py)까지 끌려온다.

누수: depth map 1장 ↔ SEM 2장(itr0/itr1)이고 캐시에서 인접 쌍(2k, 2k+1)이다. 따라서 분할은
**depth-map id 단위**여야 한다. 이미지 random split은 itr0/itr1을 갈라 중복 누수를 만든다.
"""
import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ai_co_scientist.config import ensure_utf8_console  # noqa: E402
from ai_co_scientist.sem import CASE_LEVEL, depth_to_s, map_level_split  # noqa: E402

H, W = 72, 48
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True는 재현성을 조금 희생하지만 **빼면 안 된다**: False면 cuDNN이 호출마다
    # 워크스페이스를 가변 요청해 PyTorch 할당자 밖에서 OOM이 난다 (smp 백본은 1.3GiB에서
    # 200 step 근처 실패).
    torch.backends.cudnn.benchmark = True


# ── 데이터 ──────────────────────────────────────────────────

class StructureDataset(Dataset):
    """SEM (1,H,W) → s (1,H,W). s는 depth GT와 Case에서 즉석 계산한다 (2.4GB 사전계산 회피).

    level·blur_sigma는 module global이 아니라 인스턴스 속성이어야 한다 — Windows spawn 워커는
    globals를 상속하지 않는다 (coding-patterns.md).

    blur_sigma > 0이면 입력 SEM에 고정 시그마 가우시안 블러를 건다 (H4: sim→real 공간통계
    격차, docs/data-facts.md §7). `/255` 스케일링 **전** float 이미지에 적용한다.
    """

    def __init__(self, sem, depth, case, blur_sigma: float = 0.0):
        self.sem, self.depth, self.case = sem, depth, case
        self.level = CASE_LEVEL
        self.blur_sigma = blur_sigma

    def __len__(self) -> int:
        return len(self.sem)

    def __getitem__(self, i):
        x = np.ascontiguousarray(self.sem[i]).astype(np.float32)
        if self.blur_sigma > 0:
            x = cv2.GaussianBlur(x, (0, 0), sigmaX=self.blur_sigma, sigmaY=self.blur_sigma,
                                 borderType=cv2.BORDER_REFLECT101)
        x = x[None] / 255.0
        d = np.ascontiguousarray(self.depth[i]).astype(np.float32)[None]
        lv = self.level[int(self.case[i])]
        s = depth_to_s(d, lv)  # d in [0, L] → s in [0, 1] (정확)
        return torch.from_numpy(x), torch.from_numpy(s)



# ── 백본 (모두 (B,1,H,W) → (B,1,H,W), 출력은 sigmoid로 s in [0,1]) ──

class PlainMLP(nn.Module):
    """EXP-003의 g와 동일 구조 + sigmoid. 파라미터 이름을 유지해 EXP-005 ckpt가 그대로 로드된다."""

    def __init__(self):
        super().__init__()
        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]
        self.encoder = nn.Sequential(*block(H * W, 1024), *block(1024, 512),
                                     *block(512, 256), *block(256, 128))
        self.decoder = nn.Sequential(*block(128, 256), *block(256, 512),
                                     *block(512, 1024), nn.Linear(1024, H * W))
        self.out = nn.Sigmoid()

    def forward(self, x):
        b = x.shape[0]
        return self.out(self.decoder(self.encoder(x.view(b, -1)))).view(b, 1, H, W)


def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNetSmall(nn.Module):
    """3단 U-Net — 72×48은 2회 다운샘플(18×12)까지 나누어떨어진다."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.enc1, self.enc2 = conv_block(1, w), conv_block(w, w * 2)
        self.enc3 = conv_block(w * 2, w * 4)
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
        return torch.sigmoid(self.head(d1))


class SmpModel(nn.Module):
    """segmentation_models_pytorch dense-prediction 백본. 72×48 → 인코더 크기 → 72×48 원복."""

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
        z = nn.functional.interpolate(x, size=self._in, mode="bilinear", align_corners=False)
        z = self.net(z)
        z = nn.functional.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(z)


def make_model(arch: str, width: int = 32) -> nn.Module:
    """백본만 갈아끼우는 진입점. legacy/train_sem_depth.py의 make_model 규약과 동일."""
    if arch == "mlp":
        return PlainMLP()
    if arch == "unet":
        return UNetSmall(width)
    if arch.startswith("smp:"):  # smp:<arch>:<encoder> (예: smp:unet:efficientnet-b0)
        _, a, enc = arch.split(":", 2)
        return SmpModel(a, enc, (224, 224) if a == "dpt" else (96, 64))
    raise ValueError(f"unknown arch: {arch}")


def load_model(ckpt_path: str, arch: str = "", width: int = 32) -> tuple[nn.Module, str]:
    """체크포인트에 arch가 박혀 있으면 그것으로, 아니면 --arch로 만든다.

    EXP-005는 bare state_dict로 저장돼 있어 하위호환이 필요하다 (그때는 mlp뿐이었다).
    """
    obj = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(obj, dict) and "state_dict" in obj:
        arch, width = obj.get("arch", arch), obj.get("width", width)
        state = obj["state_dict"]
    else:
        arch, state = arch or "mlp", obj
    model = make_model(arch, width).to(DEVICE)
    model.load_state_dict(state)
    return model, arch


# ── 평가 ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, levels: torch.Tensor, taus) -> dict:
    """s RMSE + **참 L로 재구성한 depth RMSE**(=구조 성분 오차, LB와 같은 척도).

    taus: 배경 클램프 임계 후보. ŝ<τ → 0 으로 눌러 배경을 정확히 L에 붙인다.
    """
    model.eval()
    se_s = n = 0.0
    se_d = {t: 0.0 for t in taus}
    off = 0
    for x, s in tqdm(loader, desc="eval", leave=False):
        x, s = x.to(DEVICE), s.to(DEVICE)
        lv = levels[off:off + len(x)].to(DEVICE).view(-1, 1, 1, 1)
        off += len(x)
        pred = model(x)
        se_s += ((pred - s) ** 2).sum().item()
        n += s.numel()
        true_d = lv * (1.0 - s)
        for t in taus:
            p = torch.where(pred < t, torch.zeros_like(pred), pred)
            rec = (lv * (1.0 - p)).round().clamp(0, 255)
            se_d[t] += ((rec - true_d) ** 2).sum().item()
    return {"s_rmse": (se_s / n) ** 0.5,
            "depth_rmse_by_tau": {round(t, 4): (se_d[t] / n) ** 0.5 for t in taus}}


def main():
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet",
                    help="mlp | unet | smp:<arch>:<encoder> (smp는 --group baseline 필요)")
    ap.add_argument("--width", type=int, default=32, help="unet 채널 폭")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--out", default="")
    ap.add_argument("--val-frac", type=float, default=0.2, help="홀드아웃할 depth-map 비율")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0, help="Windows는 0 권장")
    ap.add_argument("--amp", action="store_true",
                    help="혼합정밀. 큰 백본(smp:*)은 bs128에서 8GB를 넘긴다 — 배치를 줄이면 "
                         "BN 통계 조건이 달라져 교란이 생기므로 AMP로 메모리만 줄인다")
    ap.add_argument("--resume", action="store_true",
                    help="<out>.resume.pt가 있으면 그 다음 에폭부터 이어서 학습")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blur-sigma", type=float, default=0.0,
                    help="학습 입력에 거는 고정 가우시안 블러 시그마 (0=off). H4")
    ap.add_argument("--save-epoch", type=int, default=0,
                    help="0보다 크면 이 에폭의 체크포인트를 sim 홀드아웃 순위와 무관하게 "
                         "--out에 강제 저장한다 (사전등록 체크포인트 선택, H4)")
    args = ap.parse_args()

    seed_everything(args.seed)
    out = args.out or f"runtime/ckpt/structure-{args.arch.replace(':', '_')}.pt"
    cache = Path(args.cache_dir)
    sem = np.load(cache / "sim_sem.npy", mmap_mode="r")
    depth = np.load(cache / "sim_depth.npy", mmap_mode="r")
    case = np.load(cache / "sim_case.npy")

    model = make_model(args.arch, args.width).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | arch={args.arch} params={n_par:,} | sim {len(sem)}장", flush=True)

    va = map_level_split(case, args.val_frac, args.seed)
    tr = ~va
    print(f"split(depth-map 단위): train {tr.sum()} / val {va.sum()}", flush=True)

    ds = StructureDataset(sem, depth, case, blur_sigma=args.blur_sigma)
    tl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    sampler=torch.utils.data.SubsetRandomSampler(np.where(tr)[0].tolist()))
    va_idx = np.where(va)[0]
    vl = DataLoader(torch.utils.data.Subset(ds, va_idx.tolist()), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers)
    va_levels = torch.tensor([CASE_LEVEL[int(c)] for c in case[va_idx]], dtype=torch.float32)
    # raw(무블러) sim 홀드아웃 — blur_sigma>0일 때만 별도 필요. 같은 va_idx, 별도 Dataset
    # 인스턴스(blur_sigma=0)라 학습에는 관여하지 않고 평가 전용이다.
    raw_vl = None
    if args.blur_sigma > 0:
        raw_ds = StructureDataset(sem, depth, case, blur_sigma=0.0)
        raw_vl = DataLoader(torch.utils.data.Subset(raw_ds, va_idx.tolist()),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.L1Loss().to(DEVICE)
    taus = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # 에폭별 재개점. best ckpt(`out`)와 **별도 파일**이어야 한다 — out은 추론이 읽는 계약
    # (arch + state_dict)이고 여기에 optimizer/scaler를 섞으면 load_model 하위호환이 깨진다.
    resume_path = Path(out).with_suffix(".resume.pt")
    best, start_ep = None, 1
    saved_state = None  # --save-epoch가 강제 저장한 가중치 (사전등록 선택, holdout 무관)
    if args.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, start_ep = ck["best"], ck["epoch"] + 1
        print(f"resume: {resume_path} → epoch {start_ep}부터 (best={best})", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        losses = []
        for x, s in tqdm(tl, desc=f"epoch {ep}", leave=False):
            x, s = x.to(DEVICE), s.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss = crit(model(x), s)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        m = evaluate(model, vl, va_levels, taus)
        bt = min(m["depth_rmse_by_tau"], key=m["depth_rmse_by_tau"].get)
        print(f"epoch {ep}: train_l1={np.mean(losses):.5f} s_rmse={m['s_rmse']:.5f} "
              f"depth_rmse={m['depth_rmse_by_tau'][bt]:.4f} (tau={bt})", flush=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if args.save_epoch:
            # 사전등록 체크포인트 선택: sim 홀드아웃 순위와 무관하게 지정된 에폭만 저장한다.
            # H4 가설상 이 선택기는 sim 단서를 보존하는 쪽으로 기운다 — best-by-holdout을
            # 쓰지 않는다.
            if ep == args.save_epoch:
                saved_state = {k: v.clone() for k, v in model.state_dict().items()}
                best = {"epoch": ep, "s_rmse": m["s_rmse"], "tau": 0.0,
                        "depth_rmse": m["depth_rmse_by_tau"][0.0],
                        "depth_rmse_by_tau": m["depth_rmse_by_tau"],
                        "note": "pre-registered save-epoch, not holdout-selected"}
                torch.save({"arch": args.arch, "width": args.width,
                            "blur_sigma": args.blur_sigma,
                            "state_dict": saved_state}, out)
        elif best is None or m["depth_rmse_by_tau"][bt] < best["depth_rmse"]:
            best = {"epoch": ep, "s_rmse": m["s_rmse"], "tau": bt,
                    "depth_rmse": m["depth_rmse_by_tau"][bt],
                    "depth_rmse_by_tau": m["depth_rmse_by_tau"]}
            torch.save({"arch": args.arch, "width": args.width,
                        "blur_sigma": args.blur_sigma,
                        "state_dict": model.state_dict()}, out)
        # 에폭마다 재개점을 덮어쓴다. 이 머신은 학습 중 0x10E(비디오 메모리 관리자) BSOD가
        # 재발하므로 크래시 손실을 1에폭으로 묶는다. 샘플러 RNG는 복원하지 않으므로 재개 후
        # 배치 순서는 달라진다 — 재현성이 필요한 실행은 처음부터 돌릴 것.
        torch.save({"arch": args.arch, "width": args.width, "epoch": ep, "best": best,
                    "blur_sigma": args.blur_sigma,
                    "state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict()}, resume_path)

    # --save-epoch 강제 저장이 학습 마지막 에폭보다 먼저 걸렸을 수 있으니(예: epochs>save-epoch),
    # 이후 raw-sim 평가는 반드시 저장된 그 가중치로 한다 — 루프 종료 시점의 model이 아니라.
    if args.save_epoch and saved_state is not None:
        model.load_state_dict(saved_state)

    raw_sim_holdout = None
    if raw_vl is not None:
        raw_sim_holdout = evaluate(model, raw_vl, va_levels, taus)
        print(f"raw-sim(무블러) 홀드아웃: s_rmse={raw_sim_holdout['s_rmse']:.5f} "
              f"depth_rmse(tau=0)={raw_sim_holdout['depth_rmse_by_tau'][0.0]:.4f}", flush=True)

    print(json.dumps({
        "x_domain": "sim", "y_source": "sim_depth_gt",
        "metric": {"name": "sim_holdout_s_rmse", "x_domain": "sim", "y_source": "sim_depth_gt"},
        "target": "s = (L - d) / L", "levels": CASE_LEVEL,
        "arch": args.arch, "width": args.width, "params": n_par,
        "split": "depth-map id 단위", "val_frac": args.val_frac, "seed": args.seed,
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "amp": args.amp,
        "blur_sigma": args.blur_sigma, "save_epoch": args.save_epoch or None,
        "ckpt": out, "best": best,
        "raw_sim_holdout": raw_sim_holdout,
        "note": "depth_rmse는 **참 L로 재구성**한 값 = 구조 성분 오차. 레벨 오차는 포함되지 않는다. "
                "blur_sigma>0이면 best/depth_rmse는 **블러 입력** 기준이고 raw_sim_holdout이 "
                "무블러 기준이다",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
