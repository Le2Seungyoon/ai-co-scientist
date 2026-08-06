"""hole crop SEM 한 장으로 depth 그룹(=배경 레벨 L)이 갈리는가 — 무학습 프로브. standalone.

**이것이 파이프라인 전체의 전제다.** depth map은 픽셀의 51%가 정확히 배경 레벨
L ∈ {140,150,160,170}이고, L은 4개 그룹과 1:1이다(`docs/data-facts.md`). 따라서 test 추론은
"SEM → L 분류" + "SEM → 정규화 구조 s" 로 분해되며, 분류가 안 되면 그 분해가 무너진다.

여기서 재는 것은 **분류 가능성의 하한**이다: CNN 없이 픽셀 통계 13개 + 가우시안 QDA만 쓴다.
이 하한이 충분히 높으면 CNN에 투자할 근거가 되고, 낮으면 설계를 다시 짠다.

X = real train SEM hole crop / y = 폴더명 Depth_{110,120,130,140} → 둘 다 real 도메인이라
이 프로젝트에서 드물게 **도메인 정합 검증**이 성립한다 (리더보드 대리는 여전히 아님).

누수 방지: 사이트당 crop ~31장이 같은 라벨을 공유하므로 split은 **사이트 단위**여야 한다.
이미지 단위 random split은 같은 사이트를 train/val 양쪽에 넣어 정확도를 낙관 편향시킨다.
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

GROUPS = ("Depth_110", "Depth_120", "Depth_130", "Depth_140")
LEVELS = (140, 150, 160, 170)  # 그룹 → sim depth GT의 배경 레벨 (docs/data-facts.md)
QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def ensure_utf8_console() -> None:
    """Windows cp949 콘솔에서 유니코드 print가 UnicodeEncodeError로 죽는 걸 방지."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_labels(data_dir: Path, n_expected: int) -> tuple[np.ndarray, np.ndarray]:
    """real_sem.npy와 **동일한 sorted glob 순서**로 (그룹 라벨, 사이트 인덱스)를 복원한다.

    캐시는 train_sem_depth.py가 같은 glob으로 만들므로 행 순서가 일치한다 — 어긋나면 예외.
    """
    paths = sorted(glob.glob(str(data_dir / "train" / "SEM" / "*" / "site_*" / "*.png")))
    if len(paths) != n_expected:
        raise RuntimeError(f"경로 {len(paths)}장 != 캐시 {n_expected}장 — 캐시를 다시 만들 것")
    gid = {g: i for i, g in enumerate(GROUPS)}
    parts = [Path(p).parts for p in paths]  # split('/') 금지 (Windows 역슬래시)
    y = np.array([gid[q[-3]] for q in parts], dtype=np.int64)
    keys = [f"{q[-3]}/{q[-2]}" for q in parts]
    sidx = {s: i for i, s in enumerate(sorted(set(keys)))}
    return y, np.array([sidx[s] for s in keys], dtype=np.int32)


def pixel_features(sem, chunk: int = 5000) -> np.ndarray:
    """crop → 픽셀 통계 13개. 공간 정보를 버리므로 CNN 성능의 하한이 된다."""
    feats = []
    for s in range(0, len(sem), chunk):
        a = np.asarray(sem[s:s + chunk]).reshape(len(sem[s:s + chunk]), -1).astype(np.float32)
        feats.append(np.column_stack([a.mean(1), a.std(1), a.min(1), a.max(1),
                                      np.percentile(a, QUANTILES, axis=1).T]))
    return np.concatenate(feats)


def site_split(site: np.ndarray, y: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """그룹별로 사이트를 val_frac만큼 떼어낸다. 반환: 이미지 단위 val 마스크."""
    n_sites = int(site.max()) + 1
    site_y = np.zeros(n_sites, dtype=np.int64)
    site_y[site] = y
    rng = np.random.default_rng(seed)
    hold = np.zeros(n_sites, dtype=bool)
    for c in range(len(GROUPS)):
        ids = np.where(site_y == c)[0]
        hold[rng.choice(ids, int(val_frac * len(ids)), replace=False)] = True
    return hold[site]


def qda_log_posterior(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray,
                      ridge: float = 1e-3) -> np.ndarray:
    """클래스별 가우시안 적합 → 로그사후확률 (N, 4). 학습이라 부를 것도 없는 폐형식 해."""
    out = np.zeros((len(x_va), len(GROUPS)))
    for c in range(len(GROUPS)):
        a = x_tr[y_tr == c]
        cov = np.cov(a.T) + np.eye(a.shape[1]) * ridge
        inv = np.linalg.inv(cov)
        _, logdet = np.linalg.slogdet(cov)
        d = x_va - a.mean(0)
        out[:, c] = -0.5 * (np.einsum("ij,jk,ik->i", d, inv, d) + logdet)
    return out


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    """정확도 + 인접허용 정확도 + 혼동행렬. 오차가 인접 그룹에 몰리는지가 중요하다
    (인접 오분류는 레벨 오차 10 → LB 기여가 계산 가능하지만, 2칸 이상은 구조적 실패)."""
    cm = np.zeros((len(GROUPS), len(GROUPS)), dtype=int)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    return {"accuracy": round(float((pred == truth).mean()), 4),
            "adjacent_ok": round(float((np.abs(pred - truth) <= 1).mean()), 4),
            "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--val-frac", type=float, default=0.2, help="홀드아웃할 사이트 비율")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_utf8_console()
    sem = np.load(Path(args.cache_dir) / "real_sem.npy", mmap_mode="r")
    y, site = load_labels(Path(args.data_dir), len(sem))
    print(f"이미지 {len(y)} / 사이트 {int(site.max()) + 1}", flush=True)
    print(f"  그룹별 이미지: {dict(Counter(GROUPS[c] for c in y))}", flush=True)

    x = pixel_features(sem)
    group_mean = [round(float(x[y == c, 0].mean()), 3) for c in range(len(GROUPS))]
    group_std = [round(float(x[y == c, 0].std()), 3) for c in range(len(GROUPS))]
    print(f"  그룹별 평균 intensity {group_mean} (그룹내 std {group_std})", flush=True)

    va = site_split(site, y, args.val_frac, args.seed)
    tr = ~va
    print(f"split: train {tr.sum()}장 / val {va.sum()}장 (사이트 단위)", flush=True)

    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    xn = (x - mu) / sd

    # (1) 평균 intensity 1개 — 최근접 클래스 중심. 밝기만으로 되는지 확인용 대조군
    cent = np.array([xn[tr & (y == c)][:, 0].mean() for c in range(len(GROUPS))])
    mean_only = score(np.abs(xn[va][:, [0]] - cent).argmin(1), y[va])

    # (2) 통계 13개 QDA — 이 프로브의 본체
    logp = qda_log_posterior(xn[tr], y[tr], xn[va])
    pred = logp.argmax(1)
    qda = score(pred, y[va])

    # (3) 사이트 다수결 — test엔 site 그룹이 없으므로 **상한 참고값**이다 (직접 적용 불가)
    vsite = site[va]
    vote = {s: Counter(pred[vsite == s].tolist()).most_common(1)[0][0] for s in np.unique(vsite)}
    site_major = score(np.array([vote[s] for s in vsite]), y[va])

    for tag, r in (("평균 intensity 1개", mean_only), ("통계 13개 QDA", qda),
                   ("QDA + 사이트 다수결(상한)", site_major)):
        print(f"  {tag}: acc={r['accuracy'] * 100:.2f}% "
              f"(인접허용 {r['adjacent_ok'] * 100:.2f}%)", flush=True)

    print(json.dumps({
        "x_domain": "real", "y_source": "real_group_label",
        "metric": {"name": "site_holdout_accuracy",
                   "x_domain": "real", "y_source": "real_group_label"},
        "groups": list(GROUPS), "levels": list(LEVELS),
        "n_images": int(len(y)), "n_sites": int(site.max()) + 1,
        "val_frac": args.val_frac, "seed": args.seed,
        "group_mean_intensity": group_mean, "group_std_intensity": group_std,
        "mean_only": mean_only, "qda": qda, "site_majority_upper_bound": site_major,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
