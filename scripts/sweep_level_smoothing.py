"""레벨 평활 스윕 — 덤프된 사후확률에 창 k 또는 Viterbi a를 걸고 홀드아웃 정확도를 잰다.

**GPU를 쓰지 않는다.** 레벨 축 후처리는 사후확률만 읽으므로, 사후확률을 한 번 덤프해 두면
값마다 다른 레인이 동시에 돌 수 있다 — 이 스크립트가 그 레인이 실행하는 것이다.
torch도 cv2도 import하지 않는다(워크트리는 dev 그룹만 sync한다).

**이것은 사전 선별 지표이지 판정이 아니다.** 리더보드가 유일한 판정이고, 이 프록시는 두 번 다
낙관이었다(EXP-013→014에서 2.55pp, EXP-019에서 3.55pp 할인). 읽는 것은 **순위**이지 절대값이
아니다.

**런 구조는 만들어진 것이다.** test는 파일 순서가 곧 내용 순서라 런이 생기지만, real train은
그렇지 않다. 그래서 사이트를 시드로 섞어 이어붙여 런을 만든다 — 사이트 하나는 라벨 하나를
공유하므로 사이트 길이만큼의 런이 생긴다. 결과의 `run_count`를 test의 런 수와 나란히 읽어야
비교 가능성이 판단된다.

**예측은 in-sample이다.** 사후확률은 real train으로 학습된 분류기가 real train에 낸 것이라
절대 정확도는 낙관 편향돼 있다. 두 arm이 같은 편향을 공유하므로 **arm 사이의 순위**만 읽는다.

**입력 출처를 결과에 찍는다.** 길이 일치 검사만으로는 "같은 덤프"를 보장하지 못한다 — 같은
길이의 스테일하거나 재생성된 덤프도 통과한다. 세 입력 각각의 경로·바이트 크기·sha256을
`inputs`에 남겨, 두 레인이 정말 같은 파일을 읽었는지 사후에 대조할 수 있게 한다.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.sem import score_classes, site_split, smooth_levels, viterbi_levels

_HASH_CHUNK = 1024 * 1024  # 스트리밍 해시 청크 — 입력이 ~200MB라 통째로 읽지 않는다


def input_provenance(path: str) -> dict:
    """입력 파일 하나의 출처 — 이관된 경로·바이트 크기·sha256.

    두 레인이 "같은 덤프를 읽는다"는 것은 주장일 뿐 결과 JSON에는 아무 흔적도 남지 않았다.
    같은 길이의 다른(스테일/재생성) 덤프를 읽어도 비교 가능성 검사(길이 일치)를 통과하므로,
    파일별 해시가 유일한 사후 증거다.
    """
    resolved = Path(path).resolve()
    h = hashlib.sha256()
    with open(resolved, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": h.hexdigest()}


def build_run_order(site: np.ndarray, seed: int) -> np.ndarray:
    """사이트를 시드로 섞어 이어붙인 인덱스 순열. 사이트 내부 순서는 보존한다."""
    rng = np.random.default_rng(seed)
    ids = np.unique(site)
    rng.shuffle(ids)
    return np.concatenate([np.flatnonzero(site == s) for s in ids])


def count_runs(labels_in_order: np.ndarray) -> int:
    """인접 라벨이 바뀌는 지점의 수 + 1 = 런의 개수."""
    if len(labels_in_order) == 0:
        return 0
    return int((np.diff(labels_in_order) != 0).sum()) + 1


def main() -> int:
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba", required=True,
                    help="레벨 사후확률 .npy (N,4). scripts/dump_level_proba.py의 산출물. "
                         "기본값을 두지 않는다 — 워크트리에서 상대 기본값은 빈 runtime/으로 "
                         "조용히 풀린다")
    ap.add_argument("--labels", required=True, help="정답 클래스 .npy (N,) int")
    ap.add_argument("--site", required=True, help="사이트 id .npy (N,) int")
    ap.add_argument("--arm", required=True, choices=("smooth", "hmm"))
    ap.add_argument("--values", required=True,
                    help="쉼표로 구분한 값 — smooth면 창 k(정수), hmm이면 자기전이 a(실수)")
    ap.add_argument("--seed", type=int, default=0, help="사이트 섞기 시드")
    ap.add_argument("--val-frac", type=float, default=0.2, help="사이트 단위 홀드아웃 비율")
    args = ap.parse_args()

    # 길이 일치 검사보다 먼저 해시를 찍는다 — 검사를 통과해도 "같은 덤프"라는 보장은 없다.
    provenance = {
        "proba": input_provenance(args.proba),
        "labels": input_provenance(args.labels),
        "site": input_provenance(args.site),
    }

    proba = np.load(args.proba)
    labels = np.load(args.labels)
    site = np.load(args.site)
    if not (len(proba) == len(labels) == len(site)):
        raise SystemExit(f"장수 불일치 — 사후확률 {len(proba)} · 라벨 {len(labels)} · "
                         f"사이트 {len(site)}")

    order = build_run_order(site, args.seed)
    held = site_split(site, labels, args.val_frac, args.seed)   # 이미지 단위 마스크
    base_cls = proba.argmax(1)

    arms = []
    for raw in args.values.split(","):
        raw = raw.strip()
        value = int(raw) if args.arm == "smooth" else float(raw)
        ordered = (smooth_levels(base_cls[order], value) if args.arm == "smooth"
                   else viterbi_levels(proba[order], a=value))
        cls = np.empty_like(base_cls)
        cls[order] = ordered                      # 원래 인덱스로 되돌린다
        changed = int((cls != base_cls).sum())
        scored = score_classes(cls[held], labels[held])
        arms.append({
            "value": value,
            "holdout_accuracy": scored["accuracy"],
            "holdout_adjacent_ok": scored["adjacent_ok"],
            "changed": changed,
            "changed_frac": round(changed / len(cls), 4),
        })

    print(json.dumps({
        "x_domain": "real", "y_source": "real_group_label",
        "is_leaderboard_proxy": False,
        "arm": args.arm, "seed": args.seed, "val_frac": args.val_frac,
        "n": int(len(labels)), "n_holdout": int(held.sum()),
        "run_count": count_runs(labels[order]),
        "baseline_holdout_accuracy": score_classes(base_cls[held], labels[held])["accuracy"],
        "arms": arms,
        "inputs": provenance,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
