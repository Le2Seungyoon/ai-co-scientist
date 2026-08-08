"""뼈대 2/2 — 분해 추론: L̂(분류) + ŝ(회귀) → depth map → 제출 zip.

    d̂ = L̂ · (1 − ŝ)        L̂ ∈ {140,150,160,170}      [docs/data-facts.md §2]

레벨은 **real 라벨로 학습한 독립 분류기**에서 오고, 구조는 sim으로 학습한 회귀기에서 온다.
두 성분이 분리돼 있어 리더보드 한 점에서 `구조오차_real = √(LB² − (1−p)·61.6)`을 역산할 수
있다 — p는 EXP-004의 real 사이트홀드아웃 정확도라 도메인 정합 추정치다.

`--level-source`로 arm을 만든다:
  qda        13개 픽셀 통계 QDA, 사이트홀드아웃 89.8퍼센트 (EXP-004)
  mean_only  평균 intensity 1개, 50.4퍼센트 — 예산 공식 LB ≈ √(구조² + (1−p)·61.6)의 검증용.
             p가 크게 다른 두 점이 있어야 공식이 맞는지 확인된다.

standalone(패키지 import 없음). 형제 스크립트 import는 기존 패턴을 따른다
(pseudo_pipeline.py → train_avgcond.py).
"""
import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_level import (  # noqa: E402
    GROUPS, LEVELS, ensure_utf8_console, load_labels, pixel_features, qda_log_posterior,
)
from train_level import LevelCNN  # noqa: E402
from train_structure import DEVICE, H, W, load_model  # noqa: E402


@torch.no_grad()
def predict_levels_cnn(cache: Path, ckpt: str, batch: int = 512) -> np.ndarray:
    """EXP-013 CNN으로 test 클래스를 예측한다 (적합 불필요 — 이미 학습된 모델이다).

    구조 회귀기와 달리 **AdaBN을 걸지 않는다.** 이 분류기는 real로 학습해 real에 적용하므로
    sim→real 전이가 없다. real train ↔ test 간 잔여 이동에 AdaBN을 거는 것은 별개 축이며,
    걸면 단일 축 변경이 깨진다.
    """
    obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = LevelCNN(width=obj.get("width", 32)).to(DEVICE)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    sem = np.load(cache / "test_sem.npy", mmap_mode="r")
    out = []
    for s in range(0, len(sem), batch):
        x = np.asarray(sem[s:s + batch]).astype(np.float32)[:, None] / 255.0
        out.append(model(torch.from_numpy(x).to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def _diag(source: str, pred: np.ndarray) -> dict:
    dist = Counter(pred.tolist())
    return {"source": source,
            "test_class_dist": {GROUPS[c]: dist.get(c, 0) for c in range(len(GROUPS))},
            "test_class_frac": {GROUPS[c]: round(dist.get(c, 0) / len(pred), 4)
                                for c in range(len(GROUPS))}}


def fit_predict_levels(data_dir: Path, cache: Path, source: str,
                       level_ckpt: str = "") -> tuple[np.ndarray, dict]:
    """real train 전량으로 분류기를 적합해 test 클래스를 예측한다. 반환: (클래스, 진단)."""
    if source == "cnn":  # 픽셀 통계가 필요 없으므로 먼저 분기한다 (60,664장 계산 회피)
        pred = predict_levels_cnn(cache, level_ckpt)
        return pred, _diag(source, pred)

    real = np.load(cache / "real_sem.npy", mmap_mode="r")
    y, _ = load_labels(data_dir, len(real))
    x_tr = pixel_features(real)
    x_te = pixel_features(np.load(cache / "test_sem.npy", mmap_mode="r"))
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd

    if source == "qda":
        pred = qda_log_posterior(x_tr, y, x_te).argmax(1)
    elif source == "mean_only":  # 평균 intensity 1개 → 최근접 클래스 중심
        cent = np.array([x_tr[y == c][:, 0].mean() for c in range(len(GROUPS))])
        pred = np.abs(x_te[:, [0]] - cent).argmin(1)
    else:
        raise ValueError(f"unknown level source: {source}")

    return pred, _diag(source, pred)


def _hist(arr, cache: Path, n: int = 20000) -> np.ndarray:
    """256-bin 정규화 히스토그램. 전량 대신 표본으로 충분하다 (256 bin CDF)."""
    ix = np.sort(np.random.default_rng(0).choice(len(arr), min(n, len(arr)), replace=False))
    h = np.zeros(256, dtype=np.float64)
    for s in range(0, len(ix), 2000):
        h += np.bincount(np.asarray(arr[ix[s:s + 2000]]).ravel(), minlength=256)
    return h / h.sum()


def build_histmatch_lut(cache: Path) -> np.ndarray:
    """test 픽셀 CDF → sim 픽셀 CDF 매칭 LUT (uint8 256).

    구조 회귀기가 sim에서 학습됐으므로 **입력을 sim 쪽으로** 옮긴다. 아핀 정렬로는 부족하다 —
    평균·std를 맞춰도 분위 잔차가 최대 15.1 남는다 (docs/data-facts.md §7).
    """
    c_src = _hist(np.load(cache / "test_sem.npy", mmap_mode="r"), cache).cumsum()
    c_ref = _hist(np.load(cache / "sim_sem.npy", mmap_mode="r"), cache).cumsum()
    return np.searchsorted(c_ref, c_src).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def adapt_bn(model, cache: Path, source: str, lut, batch: int = 512) -> int:
    """AdaBN — BatchNorm running stat을 타깃 도메인으로 재계산. 역전파도 라벨도 없다.

    모델은 sim 통계로 정규화하도록 학습됐는데 real 입력은 통계가 다르다(§7). momentum=None은
    지수이동평균 대신 **누적 평균**이라 통계가 정확해진다.

    위 `no_grad` 데코레이터를 **빼지 말 것**. running stat 갱신은 autograd와 무관한 버퍼
    연산이라 결과는 같은데, 빼면 batch=512짜리 그래프가 쌓인다 — effb0(BN 59층)에서 실측
    peak 8.30 GiB로 8GB 카드를 넘겨 공유메모리로 흘렀다(no_grad면 1.01 GiB, 8.2배).
    MLP(BN 8층)에서는 작아서 드러나지 않으므로 큰 백본에서만 터진다.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    sem = np.load(cache / f"{source}_sem.npy", mmap_mode="r")
    model.train()
    for s in range(0, len(sem), batch):
        a = np.asarray(sem[s:s + batch])
        if lut is not None:
            a = lut[a]
        x = a.astype(np.float32)[:, None] / 255.0
        model(torch.from_numpy(x).to(DEVICE))
    model.eval()
    return len(sem)


@torch.no_grad()
def reconstruct_and_zip(model, cache: Path, cls: np.ndarray, tau: float, zip_path: Path,
                        lut=None, batch: int = 512) -> int:
    """d̂ = L̂·(1 − ŝ). ŝ<τ → 0 클램프로 배경을 정확히 L̂에 붙인다 (τ=0이면 클램프 없음)."""
    model.eval()
    sem = np.load(cache / "test_sem.npy", mmap_mode="r")
    names = json.loads((cache / "test_names.json").read_text(encoding="utf-8"))
    levels = np.array(LEVELS, dtype=np.float32)[cls]
    work = cache.parent / "submission_work"
    work.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for s in range(0, len(sem), batch):
            a = np.asarray(sem[s:s + batch])
            if lut is not None:
                a = lut[a]  # 구조 모델 입력만 변환 — 레벨 분류기는 원본 real 특징을 쓴다
            x = a.astype(np.float32)[:, None] / 255.0
            sp = model(torch.from_numpy(x).to(DEVICE))
            if tau > 0:
                sp = torch.where(sp < tau, torch.zeros_like(sp), sp)
            lv = torch.from_numpy(levels[s:s + len(x)]).to(DEVICE).view(-1, 1, 1, 1)
            d = (lv * (1.0 - sp)).round().clamp(0, 255).cpu().numpy()
            for j, img in enumerate(d.reshape(-1, H, W).astype(np.uint8)):
                name = names[s + j]
                cv2.imwrite(str(work / name), img)
                zf.write(work / name, arcname=name)
    return len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runtime/ckpt/EXP-005-structure.pt")
    ap.add_argument("--arch", default="", help="ckpt에 arch가 없을 때만 사용 (구 ckpt 하위호환)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--submit", required=True, help="생성할 제출 zip 경로")
    ap.add_argument("--level-source", default="qda", choices=("qda", "mean_only", "cnn"))
    ap.add_argument("--level-ckpt", default="runtime/ckpt/EXP-013-level-cnn.pt",
                    help="--level-source cnn일 때 쓸 LevelCNN 체크포인트")
    ap.add_argument("--tau", type=float, default=0.0, help="배경 클램프 임계 (sim 홀드아웃에서 결정)")
    ap.add_argument("--histmatch", action="store_true",
                    help="구조 모델 입력을 test→sim CDF 매칭 (레벨 분류기는 원본 유지)")
    ap.add_argument("--adabn", default="none", choices=("none", "test", "real"),
                    help="BatchNorm running stat을 해당 도메인으로 재계산")
    args = ap.parse_args()

    ensure_utf8_console()
    cache = Path(args.cache_dir)
    model, arch = load_model(args.ckpt, args.arch, args.width)
    print(f"구조 모델: {args.ckpt} (arch={arch})", flush=True)

    lut = None
    if args.histmatch:
        lut = build_histmatch_lut(cache)
        shift = lut.astype(int) - np.arange(256)
        print(f"histmatch LUT: 이동량 평균 {shift.mean():+.1f} "
              f"범위 [{shift.min():+d},{shift.max():+d}]", flush=True)
    if args.adabn != "none":
        n_bn = adapt_bn(model, cache, args.adabn, lut)
        print(f"AdaBN: {args.adabn} {n_bn}장으로 BN 통계 재계산", flush=True)

    cls, diag = fit_predict_levels(Path(args.data_dir), cache, args.level_source, args.level_ckpt)
    print(f"레벨 분류({args.level_source}) test 분포: {diag['test_class_frac']}", flush=True)
    print("  ↑ 4그룹이 균등(약 0.25)에서 크게 벗어나면 경고 신호", flush=True)

    n = reconstruct_and_zip(model, cache, cls, args.tau, Path(args.submit), lut)
    print(f"제출본 {n}장 → {args.submit}", flush=True)

    print(json.dumps({
        "x_domain": "real", "y_source": "real_depth_gt",
        "metric": {"name": "leaderboard_rmse", "x_domain": "real", "y_source": "real_depth_gt"},
        "ckpt": args.ckpt, "arch": arch, "level_source": args.level_source, "tau": args.tau,
        "level_ckpt": args.level_ckpt if args.level_source == "cnn" else None,
        "histmatch": bool(args.histmatch), "adabn": args.adabn,
        "reconstruct": "d = L * (1 - s)", "levels": list(LEVELS),
        "n": n, "zip": args.submit, "level_diag": diag,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
