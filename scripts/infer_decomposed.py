"""뼈대 2/2 — 분해 추론: L̂(분류) + ŝ(회귀) → depth map → 제출 zip.

    d̂ = L̂ · (1 − ŝ)        L̂ ∈ {140,150,160,170}      [docs/data-facts.md §2]

레벨은 **real 라벨로 학습한 독립 분류기**에서 오고, 구조는 sim으로 학습한 회귀기에서 온다.
두 성분이 분리돼 있어 리더보드 한 점에서 `구조오차_real = √(LB² − (1−p)·61.6)`을 역산할 수
있다 — p는 EXP-004의 real 사이트홀드아웃 정확도라 도메인 정합 추정치다.

`--level-source`로 arm을 만든다:
  qda        13개 픽셀 통계 QDA, 사이트홀드아웃 89.8퍼센트 (EXP-004)
  mean_only  평균 intensity 1개, 50.4퍼센트 — 예산 공식 LB ≈ √(구조² + (1−p)·61.6)의 검증용.
             p가 크게 다른 두 점이 있어야 공식이 맞는지 확인된다.
  cnn        LevelCNN, 사이트홀드아웃 98.06퍼센트 (EXP-013) — QDA 대비 +8.23pp로 리더보드
             신기록에 쓰인 arm (EXP-014, EXP-019).

`ai_co_scientist`를 import한다(adabn·config·sem·submission) — standalone이 아니다. 형제
스크립트(train_level.py, train_structure.py) import는 기존 패턴을 따른다
(pseudo_pipeline.py → train_avgcond.py).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_co_scientist.adabn import adapt_bn_exact, iter_cache_batches, save_bn_stats
from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.locks import GPU_LOCK, ResourceBusy, resource_lock
from ai_co_scientist.sem import (
    GROUPS, LEVELS, assemble_depth, load_labels, pixel_features, qda_log_posterior,
    smooth_levels, softmax, viterbi_levels,
)
from ai_co_scientist.submission import write_submission_zip
from train_level import LevelCNN  # noqa: E402
from train_structure import DEVICE, H, W, load_model  # noqa: E402

# AdaBN 통계를 뽑을 캐시 소스. realtest는 60,664 + 25,988 = 86,652장으로 표본을 최대화한다 —
# EXP-010에서 test(25,988) → real(60,664)만으로 -0.522가 나와 표본 민감도가 확인됐다.
SOURCES = {"test": ["test"], "real": ["real"], "realtest": ["real", "test"]}


@torch.no_grad()
def predict_levels_cnn(cache: Path, ckpt: str, batch: int = 512,
                       return_proba: bool = False,
                       npy: str = "test_sem.npy") -> np.ndarray | tuple:
    """EXP-013 CNN으로 test 클래스를 예측한다 (적합 불필요 — 이미 학습된 모델이다).

    return_proba=True면 (클래스, 사후확률 (N,4))를 준다 — Viterbi 복호(`--level-hmm`)에는
    argmax가 아니라 방출확률이 필요하다. 기본값은 기존 호출부를 위해 argmax 배열 그대로다.

    npy로 소스 배열을 고른다 — real train(real_sem.npy)에서 뽑으면 레벨 축 후처리의 사전
    선별 지표가 된다.

    구조 회귀기와 달리 **AdaBN을 걸지 않는다.** 이 분류기는 real로 학습해 real에 적용하므로
    sim→real 전이가 없다. real train ↔ test 간 잔여 이동에 AdaBN을 거는 것은 별개 축이며,
    걸면 단일 축 변경이 깨진다.
    """
    obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = LevelCNN(width=obj.get("width", 32)).to(DEVICE)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    sem = np.load(cache / npy, mmap_mode="r")
    out = []
    for s in range(0, len(sem), batch):
        x = np.asarray(sem[s:s + batch]).astype(np.float32)[:, None] / 255.0
        out.append(model(torch.from_numpy(x).to(DEVICE)).cpu().numpy())
    logits = np.concatenate(out)
    pred = logits.argmax(1)
    return (pred, softmax(logits)) if return_proba else pred


def _diag(source: str, pred: np.ndarray) -> dict:
    dist = Counter(pred.tolist())
    return {"source": source,
            "test_class_dist": {GROUPS[c]: dist.get(c, 0) for c in range(len(GROUPS))},
            "test_class_frac": {GROUPS[c]: round(dist.get(c, 0) / len(pred), 4)
                                for c in range(len(GROUPS))}}


def fit_predict_levels(data_dir: Path, cache: Path, source: str, level_ckpt: str = "",
                       return_proba: bool = False) -> tuple:
    """real train 전량으로 분류기를 적합해 test 클래스를 예측한다. 반환: (클래스, 진단).

    return_proba=True면 (클래스, 진단, 사후확률)을 준다 (기본 False — 기존 호출부 유지).
    mean_only는 사후확률이 정의되지 않으므로 거부한다.
    """
    if source == "cnn":  # 픽셀 통계가 필요 없으므로 먼저 분기한다 (60,664장 계산 회피)
        if return_proba:
            pred, proba = predict_levels_cnn(cache, level_ckpt, return_proba=True)
            return pred, _diag(source, pred), proba
        pred = predict_levels_cnn(cache, level_ckpt)
        return pred, _diag(source, pred)

    real = np.load(cache / "real_sem.npy", mmap_mode="r")
    y, _ = load_labels(data_dir, len(real))
    x_tr = pixel_features(real)
    x_te = pixel_features(np.load(cache / "test_sem.npy", mmap_mode="r"))
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd

    proba = None
    if source == "qda":
        logp = qda_log_posterior(x_tr, y, x_te)
        pred, proba = logp.argmax(1), softmax(logp)
    elif source == "mean_only":  # 평균 intensity 1개 → 최근접 클래스 중심
        cent = np.array([x_tr[y == c][:, 0].mean() for c in range(len(GROUPS))])
        pred = np.abs(x_te[:, [0]] - cent).argmin(1)
    else:
        raise ValueError(f"unknown level source: {source}")

    if return_proba:
        if proba is None:
            raise ValueError(f"--level-source {source}는 사후확률을 내지 않는다 — "
                             "--level-hmm / --dump-level-proba는 cnn 또는 qda에서만 쓸 수 있다")
        return pred, _diag(source, pred), proba
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
def adapt_bn(model, cache: Path, source: str, lut, batch: int = 512,
             drop_last: bool = False, shuffle_seed: int | None = None) -> int:
    """AdaBN — BatchNorm running stat을 타깃 도메인으로 재계산. 역전파도 라벨도 없다.

    모델은 sim 통계로 정규화하도록 학습됐는데 real 입력은 통계가 다르다(§7). momentum=None은
    지수이동평균 대신 **누적 평균**이라 통계가 정확해진다.

    위 `no_grad` 데코레이터를 **빼지 말 것**. running stat 갱신은 autograd와 무관한 버퍼
    연산이라 결과는 같은데, 빼면 batch=512짜리 그래프가 쌓인다 — effb0(BN 59층)에서 실측
    peak 8.30 GiB로 8GB 카드를 넘겨 공유메모리로 흘렀다(no_grad면 1.01 GiB, 8.2배).
    MLP(BN 7층: 1024/512/256/128/256/512/1024)에서는 작아서 드러나지 않으므로 큰 백본에서만 터진다.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    used = 0
    for name in SOURCES[source]:
        sem = np.load(cache / f"{name}_sem.npy", mmap_mode="r")
        # momentum=None 누적평균은 배치 **크기와 무관하게** 배치마다 같은 가중을 준다. 따라서
        # 마지막 부분배치가 제 몫보다 크게 반영된다 — 배치 수가 적을수록 왜곡이 크다
        # (test 51배치 대 real 119배치). drop_last는 그 왜곡을 제거해 가설을 분리한다.
        end = len(sem) - (len(sem) % batch) if drop_last else len(sem)
        if shuffle_seed is not None:
            # 배치 소속(어떤 인덱스들이 같은 배치에 묶이는지)을 섞으면 순차 누적평균에서
            # 배치별 국소 통계가 바뀐다 — 배치 내부는 정렬해 순차 mmap 읽기를 유지한다
            # (소속은 그대로, 읽기 순서만 최적화되므로 통계에는 영향이 없다).
            order = np.random.default_rng(shuffle_seed).permutation(len(sem))
            for s in range(0, end, batch):
                idx = np.sort(order[s:s + batch])
                a = np.asarray(sem[idx])
                if lut is not None:
                    a = lut[a]
                x = a.astype(np.float32)[:, None] / 255.0
                model(torch.from_numpy(x).to(DEVICE))
        else:
            for s in range(0, end, batch):
                a = np.asarray(sem[s:s + batch])
                if lut is not None:
                    a = lut[a]
                x = a.astype(np.float32)[:, None] / 255.0
                model(torch.from_numpy(x).to(DEVICE))
        used += end
    model.eval()
    return used


@torch.no_grad()
def predict_structure(model, cache: Path, lut=None, batch: int = 512) -> np.ndarray:
    """test SEM 전량 → ŝ (N, H, W) float32. **여기까지가 GPU 구간이다.**

    이후의 레벨 결정·τ 클램프·조립·zip은 전부 ŝ를 읽기만 하는 CPU 연산이라
    `scripts/assemble_submission.py`가 GPU 없이 되풀이할 수 있다.

    배치 스트리밍 대신 전량을 메모리에 들고 있는다 — ŝ 자체가 덤프·재생될 산출물이기
    때문이다. real test 규모(25,988 × 72 × 48 float32 ≈ 359MB)에서 `assemble_depth`의
    임시 배열(1−ŝ, 곱셈, clip)까지 합치면 피크 RSS는 대략 1~1.5GB다.
    """
    model.eval()
    sem = np.load(cache / "test_sem.npy", mmap_mode="r")
    out = np.empty((len(sem), H, W), dtype=np.float32)
    for s in range(0, len(sem), batch):
        a = np.asarray(sem[s:s + batch])
        if lut is not None:
            a = lut[a]  # 구조 모델 입력만 변환 — 레벨 분류기는 원본 real 특징을 쓴다
        x = a.astype(np.float32)[:, None] / 255.0
        sp = model(torch.from_numpy(x).to(DEVICE))
        out[s:s + len(x)] = sp.reshape(-1, H, W).cpu().numpy()
    return out


def reconstruct_and_zip(structure: np.ndarray, cache: Path, cls: np.ndarray, tau: float,
                        zip_path: Path) -> int:
    """ŝ + 레벨 → 제출 zip. GPU를 쓰지 않는다 (조립은 `assemble_depth`가 한다)."""
    names = json.loads((cache / "test_names.json").read_text(encoding="utf-8"))
    levels = np.array(LEVELS, dtype=np.float32)[cls]
    depth = assemble_depth(structure, levels, tau)
    return write_submission_zip(depth, names, zip_path,
                                work_dir=cache.parent / "submission_work" / zip_path.stem)


def main():
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
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
    ap.add_argument("--adabn", default="none", choices=("none", "test", "real", "realtest"),
                    help="BatchNorm running stat을 해당 도메인으로 재계산")
    ap.add_argument("--level-smooth", type=int, default=0,
                    help="파일 순서를 따라 레벨 예측을 최빈값 필터로 평활 (0=끕, 권장 9)")
    ap.add_argument("--adabn-drop-last", action="store_true",
                    help="AdaBN에서 마지막 부분배치를 버린다 — momentum=None 누적평균이 배치마다 "
                         "같은 가중을 주어 생기는 왜곡을 분리한다 (H3)")
    ap.add_argument("--adabn-shuffle", type=int, default=None,
                    help="AdaBN 소스의 배치 소속을 이 시드로 섞는다 (기본: 원본 순서 유지). "
                         "순서가 점수에 영향을 주는지 보는 단일 변수 실험용")
    ap.add_argument("--adabn-stats", default="batch", choices=("batch", "exact"),
                    help="BN 통계 추정기. batch=기존(한 번 훑어 배치 통계로 누적, 기본) · "
                         "exact=층별 순차 전역 통계 (앞 층을 확정 통계로 고정하고 전체 집합의 "
                         "정확한 평균/분산을 누적). BN 층 수만큼 순전파한다")
    ap.add_argument("--level-hmm", action="store_true",
                    help="레벨 예측을 4상태 Viterbi로 복호한다 (--level-smooth의 대안, 동시 사용 불가)")
    ap.add_argument("--level-hmm-a", type=float, default=0.974,
                    help="Viterbi 자기전이 확률 a (다른 상태는 (1-a)/3씩). 기본 0.974")
    ap.add_argument("--adabn-dump", default="",
                    help="재계산된 BN 층별 통계를 이 경로에 덤프 (.npz면 배열, 그 외 JSON). "
                         "batch 방식 통계와 층별로 비교하려면 필요하다")
    ap.add_argument("--dump-structure", default="",
                    help="구조 성분 ŝ를 이 경로에 .npy로 덤프한다 (N,H,W) float32. "
                         "CPU 재조립(scripts/assemble_submission.py)의 입력")
    ap.add_argument("--dump-level-proba", default="",
                    help="레벨 사후확률을 이 경로에 .npy로 덤프한다 (N,4) float32. "
                         "--level-source mean_only는 사후확률이 없어 거부된다")
    args = ap.parse_args()
    if args.level_hmm and args.level_smooth > 1:
        ap.error("--level-hmm과 --level-smooth는 함께 쓸 수 없다 — 평활기를 두 번 겹치면 "
                 "어느 쪽이 점수를 움직였는지 분리되지 않는다. 하나만 고를 것")
    if (args.level_hmm or args.dump_level_proba) and args.level_source == "mean_only":
        ap.error("--level-source mean_only는 사후확률을 내지 않는다 — "
                 "--level-hmm / --dump-level-proba는 cnn 또는 qda에서만 쓸 수 있다")

    try:
        with resource_lock(GPU_LOCK):
            return _run_locked(args)
    except ResourceBusy as e:
        ap.error(f"{GPU_LOCK} 사용 중: {e}")


def _run_locked(args):
    """모델 로딩부터 덤프·zip 출력까지 같은 GPU 락 안에서 실행한다."""
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
        if args.adabn_stats == "exact":
            n_bn = adapt_bn_exact(model, lambda: iter_cache_batches(
                cache, SOURCES[args.adabn], lut, drop_last=args.adabn_drop_last,
                shuffle_seed=args.adabn_shuffle), device=DEVICE)
        else:  # 기존 경로 — 지금까지의 레시피가 비트 단위로 재현되어야 한다
            n_bn = adapt_bn(model, cache, args.adabn, lut, drop_last=args.adabn_drop_last,
                            shuffle_seed=args.adabn_shuffle)
        shuffle_note = f", shuffle_seed={args.adabn_shuffle}" if args.adabn_shuffle is not None else ""
        print(f"AdaBN({args.adabn_stats}): {args.adabn} {n_bn}장으로 BN 통계 재계산{shuffle_note}",
              flush=True)
        if args.adabn_dump:
            print(f"BN 통계 덤프 → {save_bn_stats(model, args.adabn_dump)}", flush=True)

    want_proba = bool(args.level_hmm or args.dump_level_proba)
    if want_proba:
        cls, diag, proba = fit_predict_levels(Path(args.data_dir), cache, args.level_source,
                                              args.level_ckpt, return_proba=True)
    else:
        cls, diag = fit_predict_levels(Path(args.data_dir), cache, args.level_source,
                                       args.level_ckpt)
        proba = None
    if args.dump_level_proba:
        Path(args.dump_level_proba).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_level_proba, proba.astype(np.float32))
        print(f"레벨 사후확률 덤프 → {args.dump_level_proba} {proba.shape}", flush=True)
    if args.level_hmm:
        before = cls.copy()
        cls = viterbi_levels(proba, a=args.level_hmm_a)
        changed = int((before != cls).sum())
        diag = {**_diag(args.level_source, cls), "hmm_a": args.level_hmm_a,
                "changed": changed, "changed_frac": round(changed / len(cls), 4)}
        print(f"레벨 Viterbi(a={args.level_hmm_a}): {changed}장 변경 "
              f"({100 * changed / len(cls):.2f} percent)", flush=True)
    if args.level_smooth > 1:
        before = cls.copy()
        cls = smooth_levels(cls, args.level_smooth)
        changed = int((before != cls).sum())
        diag = {**_diag(args.level_source, cls), "smoothed": args.level_smooth,
                "changed": changed, "changed_frac": round(changed / len(cls), 4)}
        print(f"레벨 평활(k={args.level_smooth}): {changed}장 변경 "
              f"({100 * changed / len(cls):.2f} percent)", flush=True)
    print(f"레벨 분류({args.level_source}) test 분포: {diag['test_class_frac']}", flush=True)
    print("  ↑ 4그룹이 균등(약 0.25)에서 크게 벗어나면 경고 신호", flush=True)

    structure = predict_structure(model, cache, lut)
    if args.dump_structure:
        Path(args.dump_structure).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_structure, structure)
        print(f"구조 성분 덤프 → {args.dump_structure} {structure.shape}", flush=True)
    n = reconstruct_and_zip(structure, cache, cls, args.tau, Path(args.submit))
    print(f"제출본 {n}장 → {args.submit}", flush=True)

    print(json.dumps({
        "x_domain": "real", "y_source": "real_depth_gt",
        "metric": {"name": "leaderboard_rmse", "x_domain": "real", "y_source": "real_depth_gt"},
        "ckpt": args.ckpt, "arch": arch, "level_source": args.level_source, "tau": args.tau,
        "level_ckpt": args.level_ckpt if args.level_source == "cnn" else None,
        "histmatch": bool(args.histmatch), "adabn": args.adabn,
        "level_smooth": args.level_smooth, "level_hmm": bool(args.level_hmm),
        "level_hmm_a": args.level_hmm_a if args.level_hmm else None,
        "adabn_drop_last": bool(args.adabn_drop_last), "adabn_shuffle": args.adabn_shuffle,
        "adabn_stats": args.adabn_stats, "adabn_dump": args.adabn_dump or None,
        "dump_structure": args.dump_structure or None,
        "dump_level_proba": args.dump_level_proba or None,
        "reconstruct": "d = L * (1 - s)", "levels": list(LEVELS),
        "n": n, "zip": args.submit, "level_diag": diag,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
