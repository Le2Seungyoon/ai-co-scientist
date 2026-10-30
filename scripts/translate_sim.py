"""H6 — 학습된 CycleGAN으로 sim SEM을 real 외관으로 한 번 변환해 고정한다.

기하 위생 gate(`scripts/train_cyclegan.py gate`)를 통과한 체크포인트만 받는다 —
`ai_co_scientist.cyclegan.require_gate_passed(gate_json, ckpt, report_id=, sim_case_path=,
sim_sem_path=)`가
gate JSON을 읽어 확인하고, 실패하면 **torch를 import하기도 전에** 거부한다(이 워크트리엔 torch가
없으므로 이 순서가 곧 "torch 없이 거부 테스트 가능"의 근거다). 이 한 번의 호출이 체크포인트
파일명 규약·gate의 report_id·저장된 epoch(=사전등록 총 에폭)·config·임계값·표본 크기·원본
shifts/roundtrip_mae 재평가(저장된 `passed`를 신뢰하지 않고 다시 계산)·인덱스 지문·ckpt
sha256·sim_case 분할 provenance·sim_sem.npy sha256까지 전부 확인하는 단일 하드 스톱이다.
그 뒤로도 경로
위생·출력 디렉터리 안전·누수 가드까지 전부 순수 경로/해시 비교로 끝내고, 실제 배열을 읽고
번역기를 돌리는 지점에서만 torch를 불러온다.

쓰기는 **원자적**이다: 모든 산출물(번역된 `sim_sem.npy` + `sim_depth.npy`/`sim_case.npy` 원본
그대로 + `manifest.json`)을 `<out-cache-dir>.partial/`에 먼저 쓰고, manifest 작성과
`verify_manifest` 통과까지 확인한 뒤에만 `os.rename`으로 최종 이름 `<out-cache-dir>`로 승격한다.
크래시나 gate 재검증 실패로 중간에 멈추면 `.partial`만 남고 `train_structure.py --cache-dir
<translated dir>`가 읽을 최종 디렉터리는 아예 생기지 않는다.

**쓰기 후 재검증**: 디스크에 실제로 저장된 `sim_sem.npy`를 다시 읽어, gate가 쓴 것과 같은 2,048
고정 표본 인덱스로 전역 phase-correlation shift를 다시 재고(국소 블록 매칭은 진단 기록만)
(`measure_geometry` → `evaluate_gate`, gate가 기록한 `roundtrip_mae`와 함께) 판정한다.
round-trip 변환은 다시 돌리지 않는다 —
round-trip MAE는 쓰기 전/후로 달라질 이유가 없는 값이라, `G_real2sim`을 다시 호출하는 비용을
피한다. 이 재검증이 실패하면 `.partial`을 지우지 않고 남긴 채(조사용) 거부한다.

순서: 경로 위생(인자로 받은 경로를 열기 전) → gate 하드 스톱(gate JSON·ckpt·sim_sem의 **해시
읽기** — 락 밖이다) → 출력 안전 → 누수 가드(cache의 `test_sem.npy`가 있으면 **해시만** 읽어
real_sem과 비교 — 배열로 로드하지 않는다) → `resource_lock(GPU_LOCK)`(= `gpu-0`, 대기 없음 —
이미 잡혀 있으면 코드 4) 안에서만 배열 로드·torch·번역·쓰기. 해시는 CPU 순차 읽기라 GPU 경합이
아니므로 락 밖에 둔다. 덮어쓰기 정책: 최종·`.partial` 디렉터리 둘 중 하나라도 있으면 거부하고,
승격 직전에 최종 디렉터리를 다시 확인한다(락 안에서 확인 → rename; Windows `os.rename`은 기존
대상을 거부하고, POSIX에서는 그 사이 누가 만든 **빈** 디렉터리만 대체될 수 있다).
manifest 작성·검증 실패는 코드 3으로 거부하고 `.partial`을 조사용으로 남긴다.

ckpt 로드(`load_generators`)와 배치 번역(`translate_u8`)은 `train_cyclegan.py gate`와 같은
`ai_co_scientist.cyclegan` 함수다 — gate가 본 이미지와 downstream이 받는 이미지가 같은 경로로
만들어진다.

미래 실행 레시피(`uv run --group baseline` 필요 — 이 워크트리에서는 절대 실행하지 않는다):

    uv run --group baseline python scripts/translate_sim.py --report-id EXP-0NN \\
        --ckpt runtime/ckpt/EXP-0NN-cyclegan.pt \\
        --gate-json runtime/ckpt/EXP-0NN-cyclegan-gate.json \\
        --cache-dir runtime/cache --out-cache-dir runtime/cache/EXP-0NN-translated \\
        --git-commit "$(git rev-parse HEAD)"
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.cyclegan import (
    GPU_LOCK,
    GateFailedError,
    build_manifest,
    check_ckpt_name,
    evaluate_gate,
    load_sim_cache_split,
    load_generators,
    measure_geometry,
    reject_test_paths,
    require_gate_passed,
    require_current_git_commit,
    require_source_name,
    sha256_file,
    translate_u8,
    validation_gate_indices,
    verify_manifest,
    write_once_json,
)
from ai_co_scientist.locks import ResourceBusy, resource_lock

DEFAULT_CACHE_DIR = "runtime/cache"  # config.yaml paths.cache_dir과 손으로 맞춤


def _refuse(reason: str, **extra) -> int:
    print(json.dumps({"status": "refused", "reason": reason, **extra}, ensure_ascii=False))
    return 3


def main() -> int:
    ensure_utf8_console()  # argparse가 help를 찍기 전에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-id", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gate-json", required=True)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out-cache-dir", required=True,
                    help="번역된 sim_sem.npy + 원본 그대로의 sim_depth/sim_case.npy + manifest.json"
                         "을 쓸 디렉터리. 기본값 없음 — 워크트리에서 상대 기본값이 빈 runtime/으로 "
                         "조용히 풀리는 것을 피한다")
    ap.add_argument("--batch-size", type=int, default=256, help="번역 배치 크기 (미사전등록)")
    ap.add_argument("--git-commit", default="",
                    help="manifest에 남길 코드 커밋 — 실행한 워크트리의 HEAD "
                         "(exp.py --source-commit과 같은 관례: 호출자가 명시한다)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    gate_json_path = Path(args.gate_json)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_cache_dir)
    partial_dir = out_dir.parent / (out_dir.name + ".partial")
    sim_path = cache_dir / "sim_sem.npy"
    depth_path = cache_dir / "sim_depth.npy"
    case_path = cache_dir / "sim_case.npy"
    real_path = cache_dir / "real_sem.npy"
    test_path = cache_dir / "test_sem.npy"

    # 0) 경로 위생 -- 인자로 받은 어떤 파일도 읽기(해시 포함) 전에.
    try:
        reject_test_paths([sim_path, depth_path, case_path, real_path, ckpt_path, gate_json_path,
                           out_dir, partial_dir])
        require_source_name(sim_path, "sim_sem.npy")
        check_ckpt_name(ckpt_path, args.report_id)
    except (ValueError, GateFailedError) as e:
        return _refuse(str(e))

    repo_root = Path(__file__).resolve().parents[1]
    try:
        require_current_git_commit(args.git_commit, repo_root)
    except ValueError as e:
        return _refuse(str(e))

    # 3) 출력 디렉터리 안전 -- 최종본과 임시(.partial)본 둘 다 없어야 한다
    if out_dir.exists():
        return _refuse(f"출력 디렉터리가 이미 있다 -> {out_dir}")
    if partial_dir.exists():
        return _refuse(f"임시 출력 디렉터리가 이미 있다(이전 실행 잔재?) -> {partial_dir}")
    if out_dir.resolve() == cache_dir.resolve():
        return _refuse("출력 디렉터리가 입력 캐시와 같다 — runtime/cache/sim_sem.npy를 "
                       "덮어쓸 수 없다")

    if not sim_path.exists() or not depth_path.exists() or not case_path.exists():
        return _refuse(f"sim 캐시가 불완전하다(sim_sem/sim_depth/sim_case.npy) -> {cache_dir}")

    # 4) 누수 가드 -- test_sem.npy가 real_sem.npy와 바이트 단위로 같으면(이름만 바꾼 사본일
    #    가능성) 거부한다. 순수 파일 해시 비교라 torch 이전에 할 수 있다.
    if test_path.exists() and real_path.exists():
        if sha256_file(test_path) == sha256_file(real_path):
            return _refuse("cache/test_sem.npy와 real_sem.npy의 sha256이 같다 -- 이름만 바꾼 "
                           "test 사본이 real 자리에 들어왔을 가능성이 있다(누수 가드)")

    try:
        with resource_lock(GPU_LOCK):  # 여기부터 real 데이터·GPU·쓰기 (모듈 docstring 순서)
            # sim_case split을 읽는 gate provenance 검사도 lock 안에서, torch import 전 수행한다.
            try:
                gate = require_gate_passed(
                    gate_json_path, ckpt_path, report_id=args.report_id,
                    sim_sem_path=sim_path, sim_case_path=case_path, real_sem_path=real_path)
            except GateFailedError as e:
                return _refuse(f"gate 실패: {e}")
            except OSError as e:
                return _refuse(f"gate/ckpt/sim 파일 접근 실패: {e}")
            return _translate_locked(args, gate, sim_path, depth_path, case_path, real_path,
                                     ckpt_path, out_dir, partial_dir)
    except ResourceBusy as e:
        print(json.dumps({"status": "busy", "reason": f"{GPU_LOCK} 사용 중 — {e}"},
                         ensure_ascii=False))
        return 4


def _translate_locked(args, gate, sim_path, depth_path, case_path, real_path, ckpt_path,
                      out_dir, partial_dir) -> int:
    sim, case, _train_idx, _val_idx = load_sim_cache_split(sim_path, case_path)

    # ── 여기서부터만 torch -- 위 모든 거부 경로는 torch 없이 통과해야 한다 ──
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        cfg, ckpt_epoch, g_sim2real, _g_real2sim, _training_data = load_generators(
            ckpt_path, device)
    except GateFailedError as e:
        return _refuse(str(e))

    sim_full = np.ascontiguousarray(sim)
    translated = translate_u8(g_sim2real, sim_full, args.batch_size, device)

    partial_dir.mkdir(parents=True, exist_ok=False)
    out_sim_path = partial_dir / "sim_sem.npy"
    np.save(out_sim_path, translated)
    shutil.copyfile(depth_path, partial_dir / "sim_depth.npy")
    shutil.copyfile(case_path, partial_dir / "sim_case.npy")

    # 5) 쓰기 후 재검증 -- 디스크에 실제로 쓰인 배열로, gate와 같은 표본 인덱스에 대해 다시 잰다.
    written = np.load(out_sim_path, mmap_mode="r")
    idx = validation_gate_indices(case)
    geo = measure_geometry(np.ascontiguousarray(sim[idx]), np.ascontiguousarray(written[idx]))
    del written  # Windows에서 열린 memmap이 .partial 디렉터리 rename을 막지 않게 한다.
    post_write_gate = evaluate_gate(geo["shifts"], float(gate["roundtrip_mae"]), local=geo["local"])
    if not post_write_gate["passed"]:
        print(json.dumps({
            "status": "refused",
            "reason": "쓰기 후 재검증 gate 실패 -- .partial을 지우지 않고 조사용으로 남긴다",
            "post_write_gate": post_write_gate, "partial_dir": str(partial_dir),
        }, ensure_ascii=False))
        return 3

    manifest_path = partial_dir / "manifest.json"
    try:
        manifest = build_manifest(
            report_id=args.report_id, config=cfg.to_dict(),
            gate={**gate, "post_write_gate": post_write_gate}, ckpt_path=ckpt_path,
            source_files={"sim_sem": sim_path, "sim_depth": depth_path, "sim_case": case_path,
                          "real_sem": real_path},
            output_files={"sim_sem": out_sim_path, "sim_depth": partial_dir / "sim_depth.npy",
                          "sim_case": partial_dir / "sim_case.npy"},
            git_commit=args.git_commit)
        write_once_json(manifest_path, manifest)
        verify_manifest(manifest_path)  # 여기까지 통과해야만 아래에서 최종 이름으로 승격한다
    except (ValueError, OSError) as e:  # FileExistsError는 OSError — .partial은 조사용으로 남긴다
        return _refuse(f"manifest 작성/검증 실패: {e}", partial_dir=str(partial_dir))

    if out_dir.exists():  # 번역 도중 누가 만들었다 -- 덮지 않고 .partial을 조사용으로 남긴다
        return _refuse(f"승격 직전 출력 디렉터리가 생겼다 -> {out_dir}", partial_dir=str(partial_dir))
    os.rename(partial_dir, out_dir)  # 매니페스트 검증 이후에만 최종 이름이 된다

    print(json.dumps({
        "status": "ok", "report_id": args.report_id, "ckpt_epoch": ckpt_epoch,
        "x_domain": "sim_translated_to_real_appearance", "y_source": "sim_depth_gt",
        "out_cache_dir": str(out_dir), "manifest": str(out_dir / "manifest.json"),
        "n": int(len(translated)), "post_write_gate": post_write_gate,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
