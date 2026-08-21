"""레벨 분류기 — real SEM hole crop → depth 그룹(=배경 레벨 L). standalone.

    Depth_{110,120,130,140}  →  L ∈ {140,150,160,170}   [docs/data-facts.md §2]

EXP-004의 무학습 QDA(픽셀 통계 13개, 사이트홀드아웃 89.83%)를 CNN으로 교체한다. QDA는 공간
정보를 **전부 버린 하한**이므로 차이가 곧 공간 정보의 기여분이다. 데이터·split·seed·라벨은
`ai_co_scientist.sem`의 것을 그대로 재사용한다 — 분류기만 바뀌는 단일 축 변경이어야 비교가 성립한다.

**X도 y도 real이라 도메인 정합 검증이다.** 이 저장소에서 리더보드를 쓰지 않고 판정할 수 있는
드문 실험이며(구조 회귀기는 전부 sim 학습이라 불가), 그래서 제출 전에 p를 확정할 수 있다.

**정규화 금지 — 이 파일의 핵심 함정.** 그룹 평균 intensity는 118.227/116.388/114.660/113.036로
간격이 1.7인데 **그룹내 std가 1.9**라 겹친다(EXP-004). 평균만 쓰면 50.41%뿐이고 신호는 밝기가
아니라 **intensity 분포 형태**에 있다. 따라서 이미지별 표준화나 InstanceNorm을 넣으면 절대
intensity가 파괴돼 성능이 무너진다. 입력은 `train_structure.py`와 동일하게 `/255.0`만 한다.
BatchNorm은 배치 축 정규화라 이미지 간 상대차를 보존하므로 무해하다.

누수: 사이트당 crop ~31장이 같은 라벨을 공유하므로 split은 **사이트 단위**여야 한다
(`sem.site_split`). 이미지 random split은 정확도를 낙관 편향시킨다.
"""
import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.sem import GROUPS, LEVELS, load_labels, site_split
from ai_co_scientist.sem import score_classes as score

H, W = 72, 48
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True를 **빼면 안 된다** — False면 cuDNN이 호출마다 워크스페이스를 가변 요청해
    # PyTorch 할당자 밖에서 OOM이 난다 (coding-patterns.md). train_structure.py와 동일 설정.
    torch.backends.cudnn.benchmark = True


class LevelDataset(Dataset):
    """real SEM (1,H,W) → 그룹 라벨 0..3. 라벨은 인스턴스 속성이어야 한다 —
    Windows spawn 워커는 module global을 상속하지 않는다 (coding-patterns.md)."""

    def __init__(self, sem, y):
        self.sem, self.y = sem, y

    def __len__(self) -> int:
        return len(self.sem)

    def __getitem__(self, i):
        x = np.ascontiguousarray(self.sem[i]).astype(np.float32)[None] / 255.0
        return torch.from_numpy(x), int(self.y[i])


class LevelCNN(nn.Module):
    """1→32→64→128→256 stride-2 conv → GAP → FC(4). 약 39만 params.

    GAP는 위치를 버리지만 채널 응답은 남기므로 intensity 분포 형태를 표현할 수 있다.
    """

    def __init__(self, n_class: int = len(GROUPS), width: int = 32):
        super().__init__()
        w = width
        chans = [1, w, w * 2, w * 4, w * 8]
        self.features = nn.Sequential(*[
            layer
            for i in range(4)
            for layer in (nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1),
                          nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True))
        ])
        self.head = nn.Linear(chans[-1], n_class)

    def forward(self, x):
        z = self.features(x)
        return self.head(z.mean((2, 3)))


@torch.no_grad()
def evaluate(model, loader) -> np.ndarray:
    """val 예측 클래스를 이어붙여 반환한다."""
    model.eval()
    out = []
    for x, _ in tqdm(loader, desc="eval", leave=False):
        out.append(model(x.to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--out", default="runtime/ckpt/level-cnn.pt")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.2, help="홀드아웃할 **사이트** 비율")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0, help="Windows는 0 권장")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="<out>.resume.pt가 있으면 그 다음 에폭부터 이어서 학습")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    cache = Path(args.cache_dir)
    sem = np.load(cache / "real_sem.npy", mmap_mode="r")
    y, site = load_labels(Path(args.data_dir), len(sem))

    model = LevelCNN(width=args.width).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | params={n_par:,} | real {len(sem)}장 "
          f"/ 사이트 {int(site.max()) + 1}", flush=True)

    va = site_split(site, y, args.val_frac, args.seed)
    tr = ~va
    print(f"split(사이트 단위): train {tr.sum()} / val {va.sum()}", flush=True)

    ds = LevelDataset(sem, y)
    tl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    sampler=torch.utils.data.SubsetRandomSampler(np.where(tr)[0].tolist()))
    va_idx = np.where(va)[0]
    vl = DataLoader(torch.utils.data.Subset(ds, va_idx.tolist()), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers)
    y_va = y[va_idx]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss().to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # 에폭별 재개점 — best ckpt(`out`)와 별도 파일이어야 한다 (train_structure.py와 동일 계약)
    resume_path = Path(args.out).with_suffix(".resume.pt")
    best, start_ep = None, 1
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
        for x, t in tqdm(tl, desc=f"epoch {ep}", leave=False):
            x, t = x.to(DEVICE), t.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss = crit(model(x), t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        m = score(evaluate(model, vl), y_va)
        print(f"epoch {ep}: train_ce={np.mean(losses):.5f} acc={m['accuracy'] * 100:.2f}% "
              f"(인접허용 {m['adjacent_ok'] * 100:.2f}%)", flush=True)

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        if best is None or m["accuracy"] > best["accuracy"]:
            best = {"epoch": ep, **m}
            torch.save({"width": args.width, "state_dict": model.state_dict()}, args.out)
        torch.save({"width": args.width, "epoch": ep, "best": best,
                    "state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict()}, resume_path)

    print(json.dumps({
        "x_domain": "real", "y_source": "real_group_label",
        "metric": {"name": "site_holdout_accuracy",
                   "x_domain": "real", "y_source": "real_group_label"},
        "groups": list(GROUPS), "levels": list(LEVELS),
        "params": n_par, "width": args.width,
        "split": "사이트 단위", "val_frac": args.val_frac, "seed": args.seed,
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "amp": args.amp,
        "ckpt": args.out, "best": best,
        "baseline_exp004_qda": {"accuracy": 0.8983, "adjacent_ok": 0.9987},
        "note": "인접허용이 EXP-004의 99.87%보다 크게 낮으면 예산 계수 63.77이 과소추정이 된다",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
