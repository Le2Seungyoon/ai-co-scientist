"""체크포인트 → test 예측 → 제출 zip. standalone (Lightning Studio 업로드 가능).

체크포인트를 여러 개 주면 예측을 단순 평균한다(앙상블). 후처리는 하지 않는다 —
후처리류(플립 평균·범위 클램프)는 폐기된 실험 기능이라 제거했고, 새로 필요하면 기록소에 선보고 후 다시 붙인다.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sem_depth import H, W, make_model  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(specs: list[str]) -> list:
    """spec 형식: '<arch>=<ckpt 경로>' (예: smp:unet:efficientnet-b0=runtime/ckpt/a.pt)."""
    models = []
    for spec in specs:
        arch, _, ckpt = spec.partition("=")
        model = make_model(arch, 32).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def predict(models, cache_dir: Path, batch_size: int = 256) -> np.ndarray:
    sem = np.load(cache_dir / "test_sem.npy", mmap_mode="r")
    out = np.empty((len(sem), H, W), dtype=np.uint8)
    for start in range(0, len(sem), batch_size):
        imgs = np.asarray(sem[start:start + batch_size]).astype(np.float32) / 255.0
        batch = torch.from_numpy(imgs).unsqueeze(1).to(DEVICE)
        pred = torch.stack([m(batch) for m in models]).mean(0)
        pred = (pred * 255.0).round().clamp(0, 255)
        out[start:start + len(imgs)] = pred.squeeze(1).cpu().numpy().astype(np.uint8)
    return out


def write_zip(preds: np.ndarray, names: list[str], zip_path: Path, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for pred, name in zip(preds, names):
            png = work_dir / name
            cv2.imwrite(str(png), pred)
            zf.write(png, arcname=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="'<arch>=<ckpt>' 형식, 여러 개면 평균")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--submit", required=True, help="생성할 zip 경로")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    names = json.loads((cache_dir / "test_names.json").read_text(encoding="utf-8"))
    models = load_models(args.ckpts)
    preds = predict(models, cache_dir, args.batch_size)
    zip_path = Path(args.submit)
    write_zip(preds, names, zip_path, cache_dir.parent / "submission_work")
    print(json.dumps({"zip": str(zip_path), "n": len(names), "n_models": len(models)}))


if __name__ == "__main__":
    main()
