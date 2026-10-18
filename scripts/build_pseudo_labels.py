"""H5 1단계 — teacher 구조 회귀기로 real train SEM에 soft pseudo-label을 만든다.

    X = real train SEM 60,664장 (`real_sem.npy`, `--source`로 위치만 바꿀 수 있고 test는 이름부터
        거부된다) / y = teacher(EXP-005 계열 PlainMLP, `--adabn real --adabn-shuffle 42` 고정)가
        내는 연속값 soft pseudo-label ŝ ∈ [0,1] — **실제 real depth GT가 아니다.**

teacher 추론은 학습이 아니라 결정적 순전파(+AdaBN 통계 재계산)라 sim 홀드아웃으로 선택할
체크포인트가 없다 — 이 스크립트는 어떤 단계에서도 sim 홀드아웃으로 아무것도 고르지 않는다.

로직은 전부 `ai_co_scientist.self_training`에 있다(계약 위반은 전부 `ContractError`). 여기는
그것을 엮어 한 번 돌리는 얇은 CLI다 — torch는 계약 검증을 모두 통과한 뒤 `build` 서브커맨드
안에서만 import한다(이 워크트리는 dev 그룹만 sync되어 torch가 없다).

  생성:  python scripts/build_pseudo_labels.py build --teacher-ckpt runtime/ckpt/EXP-005-structure.pt \
           --cache-dir runtime/cache --out runtime/pseudo/H5-real-pseudo-s.npy \
           --manifest runtime/pseudo/H5-real-pseudo-s.manifest.json
         깨끗한 커밋에서만 돈다(미추적 파일 포함). 같은 명령을 다른 경로로 한 번 더 돌려
         `compare`가 빈 diff여야 학습 입력으로 쓴다 — 두 빌드는 같은 커밋·장치여야 한다.
  검증:  python scripts/build_pseudo_labels.py verify --manifest PATH.json
  비교:  python scripts/build_pseudo_labels.py compare A.json B.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.locks import ResourceBusy, resource_lock
from ai_co_scientist.self_training import (
    EXPECTED_REAL_N,
    GPU_LOCK,
    REAL_TRAIN_NPY,
    TEACHER_ADABN_BATCH,
    TEACHER_ADABN_DROP_LAST,
    TEACHER_ADABN_SHUFFLE,
    TEACHER_ADABN_SOURCE,
    ContractError,
    build_pseudo_manifest,
    git_head,
    load_manifest,
    reject_aliases,
    require_clean_commit,
    require_real_train_source,
    require_reproduced,
    verify_pseudo_manifest,
    write_manifest,
)

SCRIPTS_DIR = Path(__file__).resolve().parent


def _cmd_build(args, ap: argparse.ArgumentParser) -> None:
    source = args.source or str(Path(args.cache_dir) / REAL_TRAIN_NPY)
    try:
        resolved_source = require_real_train_source(source, Path(args.cache_dir), args.expected_n)
        reject_aliases(teacher_ckpt=args.teacher_ckpt, out=args.out, manifest=args.manifest,
                       source=str(resolved_source))
    except ContractError as e:
        ap.error(str(e))
        return

    out_path, manifest_path = Path(args.out), Path(args.manifest)
    if out_path.exists():
        ap.error(f"--out가 이미 존재한다 (덮어쓰지 않는다): {out_path}")
    if manifest_path.exists():
        ap.error(f"--manifest가 이미 존재한다 (덮어쓰지 않는다): {manifest_path}")

    # AdaBN(`--adabn real`)은 cache/real_sem.npy를 직접 다시 읽는다 — 라벨을 만드는 SOURCE
    # 배열과 다른 파일이면 두 소비자가 같은 real 분포를 보지 못한다.
    cache = Path(args.cache_dir)
    expected_source = (cache / REAL_TRAIN_NPY).resolve()
    if resolved_source.resolve() != expected_source:
        ap.error(f"--source가 --cache-dir의 {REAL_TRAIN_NPY}와 같은 파일이어야 AdaBN과 "
                 f"라벨이 같은 배열을 본다: {resolved_source} != {expected_source}")
    try:
        commit = require_clean_commit(git_head(cwd=SCRIPTS_DIR))
    except ContractError as e:
        ap.error(str(e))

    # GPU 구간만 락 안에서 돈다 — 검증·--help는 락 없이 CPU로 끝난다. 모든 워크트리가 같은
    # 기계 단위 락을 다투므로 다른 학습/추론이 GPU를 쓰는 동안에는 즉시 거부된다.
    try:
        with resource_lock(GPU_LOCK):
            runtime = _label_on_gpu(args, ap, cache, resolved_source, out_path)
    except ResourceBusy as e:
        ap.error(f"{GPU_LOCK}를 다른 실행이 잡고 있다 (GPU는 한 번에 하나): {e}")

    try:
        manifest = build_pseudo_manifest(
            labels_path=str(out_path), source_path=str(resolved_source),
            teacher_ckpt=str(args.teacher_ckpt), expected_n=args.expected_n,
            batch_size=args.batch_size, source_commit=commit, runtime=runtime)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest_path, manifest)
    except ContractError as e:
        ap.error(str(e))
    print(f"manifest → {manifest_path}", flush=True)
    print(json.dumps(manifest, ensure_ascii=False))


def _label_on_gpu(args, ap: argparse.ArgumentParser, cache: Path, source: Path,
                  out_path: Path) -> dict:
    """teacher 추론 — 이 스크립트의 유일한 GPU 구간. `GPU_LOCK` 안에서만 부른다.

    반환: 라벨 바이트를 정하는 환경(device·torch·CUDA) — 재현 비교에서 해시가 다르면 원인을
    말해 준다. torch는 여기서만 import한다(계약 검증을 모두 통과한 뒤).
    """
    import torch
    sys.path.insert(0, str(SCRIPTS_DIR))
    from infer_decomposed import adapt_bn  # noqa: E402
    from train_structure import DEVICE, load_model  # noqa: E402

    model, arch = load_model(args.teacher_ckpt, "mlp")
    if arch != "mlp":  # H5 teacher는 EXP-005 계열 PlainMLP다 — 다른 백본이면 다른 실험이다
        ap.error(f"teacher arch가 mlp가 아니다: {arch}")
    # batch 512·drop_last=False·shuffle_seed=42 — EXP-019가 고정한 그 레시피. manifest가
    # 기록하는 상수를 그대로 넘겨 기록과 실행이 갈라질 수 없게 한다. --batch-size는 아래
    # 라벨 예측 배치 크기에만 쓴다.
    n_bn = adapt_bn(model, cache, TEACHER_ADABN_SOURCE, None, batch=TEACHER_ADABN_BATCH,
                    drop_last=TEACHER_ADABN_DROP_LAST, shuffle_seed=TEACHER_ADABN_SHUFFLE)
    model.eval()
    print(f"teacher AdaBN({TEACHER_ADABN_SOURCE}, shuffle={TEACHER_ADABN_SHUFFLE}): "
          f"{n_bn}장으로 BN 통계 재계산", flush=True)

    sem = np.load(source, mmap_mode="r")
    n = len(sem)
    h, w = sem.shape[1], sem.shape[2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, h, w))
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            a = np.asarray(sem[s:s + args.batch_size]).astype(np.float32)[:, None] / 255.0
            pred = model(torch.from_numpy(a).to(DEVICE))
            labels[s:s + a.shape[0]] = pred.reshape(-1, h, w).cpu().numpy()
    labels.flush()
    del labels
    print(f"pseudo-label {n}장 → {out_path}", flush=True)
    return {"device": str(DEVICE), "torch": torch.__version__, "cuda": torch.version.cuda}


def _cmd_verify(args, ap: argparse.ArgumentParser) -> None:
    try:
        result = verify_pseudo_manifest(args.manifest, teacher_ckpt=args.teacher_ckpt)
    except ContractError as e:
        ap.error(str(e))
        return
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))


def _cmd_compare(args, ap: argparse.ArgumentParser) -> None:
    try:
        a = load_manifest(args.a)
        b = load_manifest(args.b)
        require_reproduced(a, b)
    except ContractError as e:
        ap.error(str(e))
        return
    print(json.dumps({"ok": True, "labels_sha256": a["labels"]["sha256"]}, ensure_ascii=False))


def main(argv=None) -> None:
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser(description="H5 teacher pseudo-label 생성/검증/비교 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="teacher 체크포인트로 real train pseudo-label 생성")
    build.add_argument("--teacher-ckpt", required=True, help="EXP-005 계열 구조 회귀기 ckpt")
    build.add_argument("--cache-dir", required=True)
    build.add_argument("--out", required=True, help="생성할 pseudo-label .npy 경로")
    build.add_argument("--manifest", required=True, help="생성할 manifest .json 경로")
    build.add_argument("--source", default="", help="기본: <cache-dir>/real_sem.npy")
    build.add_argument("--expected-n", type=int, default=EXPECTED_REAL_N)
    build.add_argument("--batch-size", type=int, default=512, help="라벨 예측 배치 크기")

    verify = sub.add_parser("verify", help="pseudo-label manifest를 파일 재해시로 재검증")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--teacher-ckpt", default=None,
                        help="주어지면 그 ckpt의 sha256이 manifest와 일치하는지도 확인")

    compare = sub.add_parser("compare", help="두 manifest가 같은 명령의 재현인지 비교")
    compare.add_argument("a", metavar="A.json")
    compare.add_argument("b", metavar="B.json")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        _cmd_build(args, ap)
    elif args.cmd == "verify":
        _cmd_verify(args, ap)
    elif args.cmd == "compare":
        _cmd_compare(args, ap)


if __name__ == "__main__":
    main()
