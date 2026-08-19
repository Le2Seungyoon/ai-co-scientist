"""SEM→Depth의 순수 로직 — 분할·라벨·특징·평활·재매개화.

**이 모듈은 torch를 import하지 않는다.** 패키지 본 의존성은 numpy까지이고 torch/cv2는
`baseline` 그룹이다. 모델과 학습 루프는 `scripts/`에 남는다 — 여기 있는 것은 라벨이 어떻게
흐르고 무엇이 누수인지를 정하는 부분이라, **행동으로 테스트되어야 하는** 코드다.

경계가 이렇게 그어진 이유: 예전에는 스크립트가 standalone이라 import가 불가능했고, 그래서
`tests/`가 소스 텍스트를 grep하는 계약 검사에 머물렀다(`assert "(lv - d) / lv" in source`).
그런 검사는 주석에 있어도 통과하고 변수명만 바꿔도 실패한다 — 행동을 전혀 보지 못한다.
"""
import glob
from collections import Counter
from pathlib import Path

import numpy as np

# real 폴더명 → sim Case → sim depth GT의 배경 레벨 L. 1:1 대응 (docs/data-facts.md §3)
GROUPS = ("Depth_110", "Depth_120", "Depth_130", "Depth_140")
LEVELS = (140, 150, 160, 170)
CASE_LEVEL = {1: 140.0, 2: 150.0, 3: 160.0, 4: 170.0}
QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


# ── 재매개화 ────────────────────────────────────────────────

def depth_to_s(depth: np.ndarray, level: float) -> np.ndarray:
    """depth → 정규화 구조 `s = (L − d) / L`.

    `d ∈ [0, L]`이므로 `s ∈ [0, 1]`이 **정확히** 성립한다. `(L−20)`으로 나누면 전역 min이 0인
    탓에 픽셀의 약 1퍼센트가 s>1이 되고, 클램프 손실이 RMSE 2.0에 달한다 (docs/data-facts.md §2).
    """
    return (level - depth) / level


def s_to_depth(s: np.ndarray, level: float) -> np.ndarray:
    """재구성 `d = L·(1 − s)`. `s=0`인 배경이 구조적으로 정확히 L이 된다."""
    return level * (1.0 - s)


# ── 라벨 / 분할 ─────────────────────────────────────────────

def load_labels(data_dir: Path, n_expected: int) -> tuple[np.ndarray, np.ndarray]:
    """`real_sem.npy`와 **동일한 sorted glob 순서**로 (그룹 라벨, 사이트 인덱스)를 복원한다."""
    paths = sorted(glob.glob(str(data_dir / "train" / "SEM" / "*" / "site_*" / "*.png")))
    if len(paths) != n_expected:
        raise RuntimeError(f"경로 {len(paths)}장 != 캐시 {n_expected}장 — 캐시를 다시 만들 것")
    gid = {g: i for i, g in enumerate(GROUPS)}
    parts = [Path(p).parts for p in paths]  # split('/') 금지 (Windows 역슬래시)
    y = np.array([gid[q[-3]] for q in parts], dtype=np.int64)
    keys = [f"{q[-3]}/{q[-2]}" for q in parts]
    sidx = {s: i for i, s in enumerate(sorted(set(keys)))}
    return y, np.array([sidx[s] for s in keys], dtype=np.int32)


def site_split(site: np.ndarray, y: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """**사이트 단위** 층화 홀드아웃. 반환: 이미지 단위 val 마스크.

    사이트당 crop이 ~31장이고 모두 같은 라벨을 공유한다 — 이미지 단위 random split은 같은
    사이트를 train/val 양쪽에 넣어 정확도를 낙관 편향시킨다.
    """
    n_sites = int(site.max()) + 1
    site_y = np.zeros(n_sites, dtype=np.int64)
    site_y[site] = y
    rng = np.random.default_rng(seed)
    hold = np.zeros(n_sites, dtype=bool)
    for c in range(len(GROUPS)):
        ids = np.where(site_y == c)[0]
        hold[rng.choice(ids, int(val_frac * len(ids)), replace=False)] = True
    return hold[site]


def map_level_split(case: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """sim을 **depth-map id 단위**로 Case별 층화 홀드아웃. 반환: 이미지 단위 val 마스크.

    depth map 1장 ↔ SEM 2장(itr0/itr1)이고 캐시에서 인접 쌍(2k, 2k+1)이다. 이미지 단위
    random split은 그 쌍을 갈라 사실상 중복 누수를 만든다.
    """
    n_maps = len(case) // 2
    map_case = case[::2]
    rng = np.random.default_rng(seed)
    hold = np.zeros(n_maps, dtype=bool)
    for c in np.unique(map_case):
        ids = np.where(map_case == c)[0]
        hold[rng.choice(ids, int(val_frac * len(ids)), replace=False)] = True
    return np.repeat(hold, 2)


# ── 특징 / 분류 ─────────────────────────────────────────────

def pixel_features(sem, chunk: int = 5000) -> np.ndarray:
    """crop → 픽셀 통계 13개. 공간 정보를 버리므로 CNN 성능의 하한이 된다 (EXP-004)."""
    feats = []
    for s in range(0, len(sem), chunk):
        a = np.asarray(sem[s:s + chunk]).reshape(len(sem[s:s + chunk]), -1).astype(np.float32)
        feats.append(np.column_stack([a.mean(1), a.std(1), a.min(1), a.max(1),
                                      np.percentile(a, QUANTILES, axis=1).T]))
    return np.concatenate(feats)


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


def smooth_levels(pred: np.ndarray, window: int) -> np.ndarray:
    """파일 순서를 따라 예측 클래스를 최빈값 필터로 평활한다 (window<=1이면 no-op).

    **test 파일 순서는 내용 순서다.** 분류기는 이미지를 독립적으로 보는데도 인접 파일이 같은
    클래스일 확률이 0.9598이다(무작위면 0.25) — 긴 런 안의 고립된 예측은 거의 확실히 오분류다.
    라벨을 쓰지 않으므로 transductive하지만 합법이다.

    window: real을 test보다 잘게 쪼갠 배열(런 2,836개 대 test 1,046개)로 골랐다 — k=9에서
    0.9806 → 0.9975, k=31은 0.9897로 **경계 손상 때문에 되레 나빠진다**.
    """
    if window <= 1:
        return pred
    half = window // 2
    out = pred.copy()
    for i in range(len(pred)):
        out[i] = Counter(pred[max(0, i - half):i + half + 1].tolist()).most_common(1)[0][0]
    return out


def score_classes(pred: np.ndarray, truth: np.ndarray) -> dict:
    """정확도 + 인접허용 + 혼동행렬.

    인접 여부가 중요하다: 예산 계수 63.77 = `100·E[(1−s)²]`은 **인접 오분류 전용**이라
    2칸 이상이 많아지면 기대 이득 계산이 무효가 된다 (EXP-006 armB).
    """
    cm = np.zeros((len(GROUPS), len(GROUPS)), dtype=int)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    return {"accuracy": round(float((pred == truth).mean()), 4),
            "adjacent_ok": round(float((np.abs(pred - truth) <= 1).mean()), 4),
            "confusion": cm.tolist()}
