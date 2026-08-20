"""hole crop SEM 한 장으로 depth 그룹(=배경 레벨 L)이 갈리는가 — 무학습 프로브 (EXP-004).

**이것이 파이프라인 전체의 전제다.** depth map은 픽셀의 51%가 정확히 배경 레벨
L ∈ {140,150,160,170}이고, L은 4개 그룹과 1:1이다(`docs/data-facts.md`). 따라서 test 추론은
"SEM → L 분류" + "SEM → 정규화 구조 s" 로 분해되며, 분류가 안 되면 그 분해가 무너진다.

여기서 재는 것은 **분류 가능성의 하한**이다: CNN 없이 픽셀 통계 13개 + 가우시안 QDA만 쓴다.

X = real train SEM hole crop / y = 폴더명 Depth_{110,120,130,140} → 둘 다 real 도메인이라
이 프로젝트에서 드물게 **도메인 정합 검증**이 성립한다 (리더보드 대리는 여전히 아님).

로직은 `ai_co_scientist.sem`에 있다 — 분할·특징·QDA는 행동으로 테스트된다(`tests/test_sem.py`).
이 파일은 그것들을 엮어 한 번 돌리고 결과를 JSON으로 뱉는 얇은 CLI다.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.sem import (
    GROUPS, LEVELS, load_labels, pixel_features, qda_log_posterior, score_classes, site_split,
)


def main():
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--val-frac", type=float, default=0.2, help="홀드아웃할 사이트 비율")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
    mean_only = score_classes(np.abs(xn[va][:, [0]] - cent).argmin(1), y[va])

    # (2) 통계 13개 QDA — 이 프로브의 본체
    pred = qda_log_posterior(xn[tr], y[tr], xn[va]).argmax(1)
    qda = score_classes(pred, y[va])

    # (3) 사이트 다수결 — test엔 site 그룹이 없으므로 **상한 참고값**이다 (직접 적용 불가)
    vsite = site[va]
    vote = {s: Counter(pred[vsite == s].tolist()).most_common(1)[0][0] for s in np.unique(vsite)}
    site_major = score_classes(np.array([vote[s] for s in vsite]), y[va])

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
