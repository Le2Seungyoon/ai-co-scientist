"""PlainMLP g를 sim에 직접 학습 (case holdout) → 홀드아웃 RMSE 측정 + 체크포인트별 제출본. standalone.

두 용도:
  1) 대조군 — pseudo-labeling 파이프라인(EXP-002)과 모델·하이퍼가 동일하고 학습 타깃만 다르다
     (여기: sim SEM→sim GT / 저기: real SEM→의사라벨). 두 LB 차이 = pseudo-labeling 순효과.
  2) 지표 검증 — Case_N을 통째로 홀드아웃해 그 RMSE를 재고, 같은 모델의 리더보드 점수와 대조한다.
     "holdout RMSE가 LB를 예측하는가"를 fold를 바꿔가며 확인하기 위한 것.

주의: 홀드아웃 RMSE는 **sim 도메인**(X=sim SEM, y=sim depth GT) 지표라 그 자체로 real 성능을
검증하지 못한다. 최종 판정은 리더보드로만 한다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pseudo_pipeline import DEVICE, ArrayDataset, PlainMLP, predict_and_zip  # noqa: E402
from train_avgcond import build_sim_case_cache, ensure_utf8_console, seed_everything  # noqa: E402


@torch.no_grad()
def holdout_rmse(model, loader) -> float:
    """홀드아웃 case의 RMSE (0-255 스케일). X=sim SEM, y=sim depth GT — sim 도메인 지표."""
    model.eval()
    mse = nn.MSELoss().to(DEVICE)
    rmses = []
    for x, y in loader:
        pred = (model(x.to(DEVICE)) * 255.0).round().clamp(0, 255)
        true = (y.to(DEVICE) * 255.0).round().clamp(0, 255)
        rmses.append(torch.sqrt(mse(pred, true)).item())
    return float(np.mean(rmses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--submit-prefix", required=True, help="제출 zip 경로 접두사 (-e<N>.zip 붙음)")
    ap.add_argument("--val-case", type=int, default=4, help="통째로 홀드아웃할 Case")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--save-epochs", type=int, nargs="+", default=[3, 15])
    ap.add_argument("--val-subsample", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_utf8_console()
    seed_everything(args.seed)
    cache_dir = Path(args.cache_dir)
    print(f"device: {DEVICE} | holdout Case_{args.val_case}", flush=True)

    sem = np.load(cache_dir / "sim_sem.npy", mmap_mode="r")
    depth = np.load(cache_dir / "sim_depth.npy", mmap_mode="r")
    if not (cache_dir / "sim_case.npy").exists():
        build_sim_case_cache(Path(args.data_dir), cache_dir)
    case = np.load(cache_dir / "sim_case.npy")

    tr = np.where(case != args.val_case)[0]
    va = np.where(case == args.val_case)[0]
    if 0 < args.val_subsample < len(va):
        va = np.sort(np.random.default_rng(args.seed).choice(va, args.val_subsample, replace=False))
    print(f"train {len(tr)} / holdout {len(va)} (Case_{args.val_case})", flush=True)

    tl = DataLoader(ArrayDataset(np.asarray(sem[tr]), np.asarray(depth[tr])),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    vl = DataLoader(ArrayDataset(np.asarray(sem[va]), np.asarray(depth[va])),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    g = PlainMLP().to(DEVICE)
    opt = torch.optim.AdamW(g.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.L1Loss().to(DEVICE)

    results = []
    for ep in range(1, args.epochs + 1):
        g.train()
        losses = []
        for x, y in tqdm(tl, desc=f"epoch {ep}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(g(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        rmse = holdout_rmse(g, vl)
        print(f"epoch {ep}: train_loss={np.mean(losses):.5f} holdout_rmse={rmse:.4f}", flush=True)
        if ep in args.save_epochs:
            zip_path = Path(f"{args.submit_prefix}-e{ep}.zip")
            predict_and_zip(g, cache_dir, zip_path)
            results.append({"epoch": ep, "holdout_rmse": round(rmse, 4), "zip": str(zip_path)})
            print(f"  wrote {zip_path}", flush=True)

    print(json.dumps({
        "val_case": args.val_case, "train_n": len(tr), "holdout_n": len(va),
        "x_domain": "sim", "y_source": "sim_depth_gt",
        "metric": {"name": f"case{args.val_case}_holdout_rmse",
                   "x_domain": "sim", "y_source": "sim_depth_gt"},
        "checkpoints": results,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
