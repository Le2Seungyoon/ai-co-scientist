"""체크포인트(들)로 sim-val 후처리 스크린 + 앙상블 submission 생성 (재학습 불필요).

--eval: 홀드아웃 sim-val에서 {clamp, TTA} 조합의 val_rmse를 출력(후처리 판정, 도메인 중립).
--submit: 체크포인트 예측을 평균(앙상블) + clamp/TTA 적용해 submission zip 생성.

train_sem_depth.py의 make_model/apply_preproc/predict_tta 재사용 (계약 일치). 로컬 GPU.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

# train_sem_depth.py와 같은 디렉토리에서 import (로컬 scripts/ 또는 studio 홈 둘 다 동작)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sem_depth import apply_preproc, make_model, predict_tta  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(specs: list[str]):
    """["arch:path", ...] → [model, ...]. arch는 make_model 계약(smp:unet:efficientnet-b0 등)."""
    models = []
    for spec in specs:
        arch, path = spec.split("=", 1)
        m = make_model(arch, 32).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))  # cwd 기준 경로
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def adapt_bn(model, cache, preproc, batch_size=128):
    """AdaBN: 실측 SEM으로 BN running stats 재계산 (파라미터 학습 없음, UDA 최경량).
    sim으로 배운 관계는 유지한 채 정규화 통계만 real 도메인에 맞춘다."""
    real = np.load(cache / "real_sem.npy", mmap_mode="r")
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None  # 누적 이동평균 → 전체 real의 참 통계
            n += 1
    model.train()  # BN update 모드 (grad 없음)
    for s in range(0, len(real), batch_size):
        imgs = np.asarray(real[s:s + batch_size])
        batch = torch.from_numpy(np.stack(
            [apply_preproc(im, preproc) for im in imgs])).unsqueeze(1).to(DEVICE)
        model(batch)
    model.eval()
    print(f"[adabn] {n} BN layers 재계산 (real {len(real)}장)", flush=True)


@torch.no_grad()
def ensemble_pred(models, batch, tta: bool):
    """모델별 예측 평균 (각 모델 내부에서 TTA)."""
    acc = None
    for m in models:
        p = predict_tta(m, batch, tta)
        acc = p if acc is None else acc + p
    return acc / len(models)


def val_split(cache: Path, n_val: int):
    sim_sem = np.load(cache / "sim_sem.npy", mmap_mode="r")
    sim_depth = np.load(cache / "sim_depth.npy", mmap_mode="r")
    n_train = int(len(sim_sem) * 0.8)
    rng = np.random.default_rng(41)  # 학습 val split과 동일 시드
    idx = np.sort(rng.choice(np.arange(n_train, len(sim_sem)), n_val, replace=False))
    return np.asarray(sim_sem[idx]), np.asarray(sim_depth[idx]).astype(np.float32)


@torch.no_grad()
def eval_simval(models, cache, n_val, preproc):
    sem, depth = val_split(cache, n_val)
    combos = [(False, 0, 255), (False, 25, 170), (True, 0, 255), (True, 25, 170)]
    print(f"{'TTA':>5}{'clamp':>12}{'sim-val RMSE':>14}")
    for tta, lo, hi in combos:
        rmses = []
        for s in range(0, len(sem), 256):
            imgs = sem[s:s + 256]
            batch = torch.from_numpy(np.stack(
                [apply_preproc(im, preproc) for im in imgs])).unsqueeze(1).to(DEVICE)
            pred = (ensemble_pred(models, batch, tta) * 255.0).round().clamp(lo, hi)
            true = torch.from_numpy(depth[s:s + 256]).unsqueeze(1).to(DEVICE)
            rmses.append(torch.sqrt(((pred - true) ** 2).mean()).item())
        print(f"{str(tta):>5}{f'[{lo},{hi}]':>12}{np.mean(rmses):>14.4f}", flush=True)


@torch.no_grad()
def make_submission(models, cache, out_zip, tta, lo, hi, preproc, batch_size=128):
    test_sem = np.load(cache / "test_sem.npy", mmap_mode="r")
    names = json.loads((cache / "test_names.json").read_text(encoding="utf-8"))
    test_preproc = "none" if preproc == "histmatch" else preproc
    out_zip = Path(out_zip)
    sub_dir = out_zip.parent / (out_zip.stem + "_imgs")
    sub_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w") as zf:
        for s in range(0, len(test_sem), batch_size):
            imgs = np.asarray(test_sem[s:s + batch_size])
            batch = torch.from_numpy(np.stack(
                [apply_preproc(im, test_preproc) for im in imgs])).unsqueeze(1).to(DEVICE)
            pred = (ensemble_pred(models, batch, tta) * 255.0).round().clamp(lo, hi)
            pred = pred.cpu().numpy().astype(np.uint8)
            for j in range(len(imgs)):
                p = sub_dir / names[s + j]
                cv2.imwrite(str(p), pred[j, 0])
                zf.write(p, arcname=names[s + j])
    print(f"submission → {out_zip}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="arch=path (예: smp:unet:efficientnet-b0=runtime/final/x.pt)")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--preproc", default="none")
    ap.add_argument("--eval", action="store_true", help="sim-val 후처리 스크린")
    ap.add_argument("--val-subsample", type=int, default=8000)
    ap.add_argument("--submit", default=None, help="submission zip 경로 (생성 시)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--adabn", action="store_true", help="추론 전 실측 SEM으로 BN 통계 재계산(AdaBN)")
    ap.add_argument("--clamp-lo", type=float, default=0.0)
    ap.add_argument("--clamp-hi", type=float, default=255.0)
    args = ap.parse_args()

    cache = Path(args.cache_dir)  # cwd 기준
    models = load_models(args.ckpts)
    print(f"loaded {len(models)} model(s)", flush=True)
    if args.adabn:
        for m in models:
            adapt_bn(m, cache, args.preproc)
    if args.eval:
        eval_simval(models, cache, args.val_subsample, args.preproc)
    if args.submit:
        make_submission(models, cache, args.submit, args.tta,
                        args.clamp_lo, args.clamp_hi, args.preproc)


if __name__ == "__main__":
    main()
