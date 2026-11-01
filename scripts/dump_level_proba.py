"""레벨 사후확률 덤프 — CPU 레인이 평활 스윕에 쓸 입력을 GPU에서 한 번 뽑는다.

`2026-09-13`의 ŝ 덤프와 같은 패턴이다: GPU가 필요한 부분을 한 번만 치르고, 그 산출물을
읽기 전용으로 공유해 후처리 가설을 병렬로 돌린다. LevelCNN은 torch라 워크트리에서 돌지
않으므로, 이 스크립트는 **main 체크아웃 전용**이다.

X = real train SEM · y = 폴더명 Depth_{110,120,130,140}(주최측 실측 라벨). 둘 다 real이라
이 프로젝트에서 드물게 도메인 정합 검증이 성립한다 — 다만 리더보드 대리는 아니다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_co_scientist.config import ensure_utf8_console  # noqa: E402
from ai_co_scientist.locks import GPU_LOCK, ResourceBusy, resource_lock  # noqa: E402
from infer_decomposed import predict_levels_cnn  # noqa: E402


def main() -> int:
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--level-ckpt", default="runtime/ckpt/EXP-013-level-cnn.pt")
    ap.add_argument("--npy", default="real_sem.npy", help="캐시 안의 소스 배열")
    ap.add_argument("--out", required=True, help="사후확률 .npy 출력 경로 (N,4) float32")
    args = ap.parse_args()

    try:
        with resource_lock(GPU_LOCK):
            return _run_locked(args)
    except ResourceBusy as e:
        ap.error(f"{GPU_LOCK} 사용 중: {e}")


def _run_locked(args) -> int:
    """추론과 산출물 저장은 같은 배타 구간이다."""
    _, proba = predict_levels_cnn(Path(args.cache_dir), args.level_ckpt,
                                  return_proba=True, npy=args.npy)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, proba.astype(np.float32))

    print(json.dumps({
        "x_domain": "real", "y_source": "real_group_label",
        "source_npy": args.npy, "level_ckpt": args.level_ckpt,
        "out": str(out), "shape": list(proba.shape),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
