"""CPU 재조립 — 덤프된 ŝ와 레벨 사후확률로 제출 zip을 다시 만든다 (GPU 불필요).

`scripts/infer_decomposed.py`가 GPU로 한 번 뽑아 둔 ŝ를 읽기만 하므로, 레벨 축 후처리 가설
(`--level-smooth` k · `--level-hmm` a)과 배경 클램프 τ를 **몇 개든 병렬로** 되풀이할 수 있다.
구조 성분 ŝ는 이 인자들 중 어느 것에도 영향받지 않는다 — 그것이 분리가 성립하는 이유다.

**torch도 cv2도 import하지 않는다.** 워크트리는 dev 그룹만 sync하므로 둘 다 없고, 이 진입점이
거기서 도는 것이 존재 이유다. 조립·인코딩은 `ai_co_scientist`의 공유 함수가 한다 — GPU 경로와
같은 코드다.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.sem import GROUPS, LEVELS, assemble_depth, smooth_levels, viterbi_levels
from ai_co_scientist.submission import verify_submission, write_submission_zip

_HASH_CHUNK = 1 << 20  # 1MiB -- 전체를 메모리에 올리지 않고 스트리밍으로 해시한다


def _sha256_and_size(path: Path) -> dict:
    """입력 파일이 실제로 무엇이었는지 남긴다 -- 같은 경로에 다른 실행이 다른 파일을
    다시 써넣어도(디스패치와 워커 실행 사이의 재실행 등) 진단 JSON이 그 사실을 드러내도록."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": size}


def _diag(pred: np.ndarray) -> dict:
    dist = Counter(pred.tolist())
    return {"test_class_dist": {GROUPS[c]: dist.get(c, 0) for c in range(len(GROUPS))},
            "test_class_frac": {GROUPS[c]: round(dist.get(c, 0) / len(pred), 4)
                                for c in range(len(GROUPS))}}


def main() -> int:
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True,
                    help="infer_decomposed --dump-structure가 남긴 ŝ (.npy, (N,H,W) float32). "
                         "기본값을 두지 않는다 — 워크트리에서 상대 기본값은 빈 runtime/으로 "
                         "조용히 풀린다")
    ap.add_argument("--level-proba", required=True,
                    help="infer_decomposed --dump-level-proba가 남긴 사후확률 (.npy, (N,4))")
    ap.add_argument("--names", required=True, help="runtime/cache/test_names.json 경로")
    ap.add_argument("--submit", required=True, help="생성할 제출 zip 경로")
    ap.add_argument("--work-dir", default="", help="PNG 사본을 남길 디렉터리 (비우면 남기지 않음)")
    ap.add_argument("--tau", type=float, default=0.0, help="배경 클램프 임계 (0=끕)")
    ap.add_argument("--level-smooth", type=int, default=0,
                    help="파일 순서를 따라 레벨 예측을 최빈값 필터로 평활 (0=끕, EXP-019는 9)")
    ap.add_argument("--level-hmm", action="store_true",
                    help="레벨 예측을 4상태 Viterbi로 복호한다 (--level-smooth의 대안)")
    ap.add_argument("--level-hmm-a", type=float, default=0.974, help="Viterbi 자기전이 확률")
    ap.add_argument("--verify", action="store_true",
                    help="조립한 zip을 verify_submission()으로 검증한다 (architecture.md: "
                         "워커가 자기 트리에서 실행하는 그 검증 — 여기가 그 진입점이다). "
                         "실패하면 비정상 종료한다")
    ap.add_argument("--verify-expected-files", type=int, default=None,
                    help="--verify가 기대할 파일 수 (기본: verify_submission의 기본값 25,988). "
                         "합성 zip처럼 장수가 다르면 명시해야 한다")
    args = ap.parse_args()

    if args.level_hmm and args.level_smooth > 1:
        ap.error("--level-hmm과 --level-smooth는 함께 쓸 수 없다 — 평활기를 두 번 겹치면 "
                 "어느 쪽이 점수를 움직였는지 분리되지 않는다. 하나만 고를 것")

    structure = np.load(args.structure, mmap_mode="r")
    proba = np.load(args.level_proba)
    names = json.loads(Path(args.names).read_text(encoding="utf-8"))
    if not (len(structure) == len(proba) == len(names)):
        raise SystemExit(f"장수 불일치 — 구조 {len(structure)} · 사후확률 {len(proba)} · "
                         f"파일명 {len(names)}")

    cls = proba.argmax(1)
    diag = _diag(cls)
    if args.level_hmm:
        before = cls.copy()
        cls = viterbi_levels(proba, a=args.level_hmm_a)
        changed = int((before != cls).sum())
        diag = {**_diag(cls), "hmm_a": args.level_hmm_a, "changed": changed,
                "changed_frac": round(changed / len(cls), 4)}
        print(f"레벨 Viterbi(a={args.level_hmm_a}): {changed}장 변경", flush=True)
    elif args.level_smooth > 1:
        before = cls.copy()
        cls = smooth_levels(cls, args.level_smooth)
        changed = int((before != cls).sum())
        diag = {**_diag(cls), "smoothed": args.level_smooth, "changed": changed,
                "changed_frac": round(changed / len(cls), 4)}
        print(f"레벨 평활(k={args.level_smooth}): {changed}장 변경", flush=True)

    levels = np.array(LEVELS, dtype=np.float32)[cls]
    depth = assemble_depth(structure, levels, args.tau)
    n = write_submission_zip(depth, names, Path(args.submit),
                             work_dir=Path(args.work_dir) if args.work_dir else None)
    print(f"제출본 {n}장 → {args.submit}", flush=True)

    # 입력 파일 해시 -- ŝ 덤프와 레벨 사후확률이 실제로 무엇이었는지 진단에 남긴다. 두 덤프
    # 모두 25,988장으로 나올 수 있으니 장수 일치만으로는 다른 run에서 온 짝을 구분 못한다.
    structure_file = _sha256_and_size(Path(args.structure))
    level_proba_file = _sha256_and_size(Path(args.level_proba))

    verify_result = None
    verify_ok = True
    if args.verify:
        verify_kwargs = {}
        if args.verify_expected_files is not None:
            verify_kwargs["expected_files"] = args.verify_expected_files
        check = verify_submission(Path(args.submit), **verify_kwargs)
        print(check.summary(), flush=True)
        verify_result = check.to_dict()
        verify_ok = check.ok

    print(json.dumps({
        "x_domain": "real", "y_source": "real_depth_gt",
        "metric": {"name": "leaderboard_rmse", "x_domain": "real", "y_source": "real_depth_gt"},
        "structure": args.structure, "level_proba": args.level_proba,
        "structure_file": structure_file, "level_proba_file": level_proba_file,
        "tau": args.tau, "level_smooth": args.level_smooth, "level_hmm": bool(args.level_hmm),
        "level_hmm_a": args.level_hmm_a if args.level_hmm else None,
        "reconstruct": "d = L * (1 - s)", "levels": list(LEVELS),
        "n": n, "zip": args.submit, "level_diag": diag,
        "verify": verify_result,
    }, ensure_ascii=False))
    return 0 if verify_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
