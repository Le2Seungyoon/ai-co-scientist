"""H8 — 2-head 구조 회귀 추론. arm A(single)/arm B(two_head) 공용, EXP-019 경로를 그대로 쓴다.

    d̂ = L̂ · (1 − ŝ)      [scripts/infer_decomposed.py와 동일 재구성]

arm 분기는 `train_two_head.compose_module`이 흡수한다 — 이 스크립트가 보는 모델은 항상
`model(x) → (B,1,H,W)` 한 종류다. AdaBN·레벨 분류·조립은 `infer_decomposed`의 함수를 **그대로**
재사용한다(중복 조립 금지, coding-patterns.md). 이 모듈을 import해도 torch/cv2는 로드되지
않는다 — 전부 `main()` 안에서 지연 import한다.

GPU 구간 전체를 `main()` 하나에 몰아둔 이유: `submit_path.exists()`가 첫 torch 사용보다,
`submission_gate(...)`가 `reconstruct_and_zip(...)`보다 **소스 텍스트상으로** 먼저 나와야 한다는
계약(tests/test_two_head.py T7)이 있다 — 별도 헬퍼로 쪼개면 그 헬퍼가 파일 앞쪽에 정의되어
`main()`의 소스 슬라이스 안에서 두 호출이 보이지 않게 된다.

reviewer Phase 1 수정 반영:
  - `--peer-manifest`(사이드카 JSON) 대신 `--peer-ckpt`를 받아 ckpt 안의 manifest를 직접
    `resolve_ckpt`로 읽는다 — 오래된 사이드카 파일을 참조하는 사고를 막는다.
  - mask-rate 패스는 BN 통계를 건드리면 안 된다(H2) — eval 모드 확인 + BN 버퍼 해시 불변 검사.
  - 게이트는 `reconstruct_and_zip`에 넘기는 바로 그 `structure` 배열에 `expected_shape`까지
    걸어 검사한다. zip을 쓴 뒤 `verify_submission`으로 다시 열어 확인하고, 실패하면 그 zip을
    지우고 exit 3으로 끝낸다 — "만들어지긴 했지만 검증되지 않은 zip"을 절대 남기지 않는다.

reviewer Phase 2 수정 반영:
  - GPU 구간 전체(모델→DEVICE, AdaBN, mask 패스, 레벨, predict_structure, 게이트, zip, 검증)를
    `two_head.GPU_LOCK` 배타 락으로 두른다. `--help`/인자 검증/`--submit` 존재 검사/ckpt CPU
    로드/사전 게이트는 락 없이 돈다 — `resource_lock`을 **모듈 레벨 이름**으로 불러 테스트가
    `infer_two_head.resource_lock`을 monkeypatch할 수 있게 한다. `from infer_decomposed import
    ...`(CUDA를 건드린다)는 락 진입 **이후**에야 실행된다.
  - real-test mask rate는 여전히 로깅 전용 진단이다 — 실제 게이트는 학습 시점에 고정된
    `manifest["sim_mask_gate"]`가 `submission_gate` 안에서 담당한다(two_head arm).
  - `--level-smooth` 적용 **후**에 `infer_decomposed._diag`로 `level_diag`를 다시 계산한다 —
    이전에는 평활 전 클래스 분포가 그대로 남아 있었다.
  - mask-rate 패스는 `assert` 대신 `RuntimeError`를 던진다(-O 최적화로 assert가 사라져도
    H2 불변식이 계속 지켜지도록).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.locks import ResourceBusy, resource_lock  # stdlib-only, torch-free
from ai_co_scientist import two_head

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 형제 스크립트 지연 import용


def _bn_buffer_hash(model) -> str:
    """BatchNorm running_mean/running_var 버퍼 전체를 이어붙여 해시한다.

    mask-rate 패스(H2) 전후로 이 값이 같아야 한다 — 다르면 `no_grad`/`eval` 밖에서 통계가
    갱신됐다는 뜻이고, 그러면 이후 predict_structure가 쓰는 BN 통계가 AdaBN 결과와 어긋난다.
    """
    import torch.nn as nn

    h = hashlib.sha256()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            h.update(m.running_mean.detach().cpu().numpy().tobytes())
            h.update(m.running_var.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def compute_mask_rate(model, cache: Path, h: int, w: int, batch: int = 512):
    """real test SEM 전체에서 mask 양성 확률(sigmoid(mask_logit))을 뽑는다.

    AdaBN **이후**, 구조 예측과는 별도인 no_grad pass. `model`은 raw 2-head 모델(합성 전)이어야
    mask_logit에 접근할 수 있다 — `compose_module`로 감싼 모델을 넘기면 안 된다.
    반환: (mask_rate: float, mask_prob: (N,H,W) float32). **로깅 전용 진단이다** — 제출 게이트는
    학습 시점에 고정된 `manifest["sim_mask_gate"]`가 담당하고, 이 값은 real-test에서 그 판단이
    유지되는지 보는 참고 지표일 뿐 게이트를 통과/차단하지 않는다.
    """
    import torch

    device = next(model.parameters()).device
    if model.training:
        # AdaBN(`adapt_bn`)은 항상 `model.eval()`로 끝난다 — 여기서 train 모드면 AdaBN이
        # 통계를 eval로 되돌리지 못했다는 뜻이고, 그러면 이 패스의 BN 통계가 신뢰할 수 없다.
        raise RuntimeError(
            "mask-rate 패스 진입 시 모델이 train 모드다 — AdaBN이 eval 모드로 되돌리지 않았다 "
            "(H2 위반)")
    model.eval()
    before = _bn_buffer_hash(model)

    sem = np.load(cache / "test_sem.npy", mmap_mode="r")
    prob = np.empty((len(sem), h, w), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(sem), batch):
            a = np.asarray(sem[s:s + batch]).astype(np.float32)[:, None] / 255.0
            mask_logit, _ = model(torch.from_numpy(a).to(device))
            prob[s:s + len(a)] = torch.sigmoid(mask_logit).reshape(-1, h, w).cpu().numpy()

    after = _bn_buffer_hash(model)
    if any(m.training for m in model.modules()):
        raise RuntimeError("mask-rate pass가 모델을 train 모드로 바꿨다 (H2 위반)")
    if before != after:
        raise RuntimeError(
            "mask-rate pass가 BN 버퍼(running_mean/var)를 바꿨다 — AdaBN 통계가 오염됐다 (H2 위반)")

    mask_rate = float((prob > two_head.MASK_THRESHOLD).mean())
    return mask_rate, prob


def _output_pos_rate(structure: np.ndarray, levels: np.ndarray) -> float:
    """round(L·(1−ŝ)) < L 인 픽셀 비율 — 배경(=L 그대로)이 아닌 픽셀 비율. 로그 전용 진단."""
    lv = np.asarray(levels, dtype=np.float32).reshape(-1, 1, 1)
    rec = np.round(lv * (1.0 - structure))
    return float((rec < lv).mean())


def build_parser() -> argparse.ArgumentParser:
    """`--help`는 torch 없이도 동작해야 한다 — 추론 기본값은 EXP-019 경로
    (`two_head.INFERENCE_PREREGISTERED`)와 항상 일치한다."""
    ap = argparse.ArgumentParser(
        description="H8 2-head 구조 회귀 추론 — EXP-019 경로, arm A/B 공용")
    ap.add_argument("--ckpt", required=True, help="train_two_head.py가 저장한 체크포인트(.pt)")
    ap.add_argument("--peer-ckpt", required=True,
                    help="반대 arm의 ckpt(.pt) — arm parity 게이트에 필요, manifest는 여기서 직접 읽는다")
    ap.add_argument("--submit", required=True, help="생성할 제출 zip 경로 (이미 있으면 시작을 거부한다)")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--level-source", default=two_head.INFERENCE_PREREGISTERED["level_source"],
                    choices=("qda", "mean_only", "cnn"))
    ap.add_argument("--level-ckpt", default=two_head.INFERENCE_PREREGISTERED["level_ckpt"])
    ap.add_argument("--adabn", default=two_head.INFERENCE_PREREGISTERED["adabn"],
                    choices=("none", "test", "real", "realtest"))
    ap.add_argument("--adabn-shuffle", type=int,
                    default=two_head.INFERENCE_PREREGISTERED["adabn_shuffle"])
    ap.add_argument("--tau", type=float, default=two_head.INFERENCE_PREREGISTERED["tau"])
    ap.add_argument("--level-smooth", type=int,
                    default=two_head.INFERENCE_PREREGISTERED["level_smooth"])
    ap.add_argument("--dump-structure", default="")
    return ap


def main():
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    args = build_parser().parse_args()

    submit_path = Path(args.submit)
    if submit_path.exists():
        # 이미 있는 zip을 덮어쓰면 "무엇의 점수인지 알 수 없는" 상태가 된다 — GPU 작업 전에 거부.
        print(json.dumps({"gates": [f"--submit 경로가 이미 존재한다: {submit_path}"], "zip": None},
                         ensure_ascii=False))
        raise SystemExit(2)
    # np.save는 .npy가 없으면 붙여서 쓴다 — 존재 검사도 실제로 쓰일 경로에 해야 한다
    dump_path = None
    if args.dump_structure:
        dump_path = Path(args.dump_structure)
        if dump_path.suffix != ".npy":
            dump_path = dump_path.with_name(dump_path.name + ".npy")
        if dump_path.exists():
            print(json.dumps({"gates": [f"--dump-structure 경로가 이미 존재한다: {dump_path}"],
                              "zip": None}, ensure_ascii=False))
            raise SystemExit(2)

    import torch

    # ── 게이트 1: resolve_ckpt + validate_arm_pair + validate_inference_config — CPU, 락 전 ──
    ckpt_obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    peer_obj = torch.load(args.peer_ckpt, map_location="cpu", weights_only=False)
    try:
        arm, manifest = two_head.resolve_ckpt(ckpt_obj)
        _peer_arm, peer_manifest = two_head.resolve_ckpt(peer_obj)
    except ValueError as exc:
        print(json.dumps({"gates": [str(exc)], "zip": None}, ensure_ascii=False))
        raise SystemExit(2) from exc

    cfg = {
        "level_source": args.level_source, "adabn": args.adabn,
        "adabn_shuffle": args.adabn_shuffle, "tau": args.tau, "level_smooth": args.level_smooth,
        "level_ckpt": args.level_ckpt,
        # 아래 셋은 CLI 플래그가 없다 — EXP-019 경로에 고정된 값이라 노출하지 않는다 (reviewer 지적)
        "adabn_stats": "batch", "histmatch": False, "adabn_drop_last": False, "level_hmm": False,
    }
    early_errors = (two_head.validate_inference_config(cfg)
                    + two_head.validate_arm_pair(manifest, peer_manifest))
    if early_errors:
        print(json.dumps({"gates": early_errors, "zip": None}, ensure_ascii=False))
        raise SystemExit(2)

    try:
        with resource_lock(two_head.GPU_LOCK, timeout=0):
            # ── 여기서부터 GPU 구간. infer_decomposed import 자체가 CUDA를 건드리므로 락
            # 진입 이후에야 불러온다 (reviewer P2-5). ──
            from ai_co_scientist.sem import LEVELS, smooth_levels
            from ai_co_scientist.submission import verify_submission
            from infer_decomposed import (  # noqa: E402
                DEVICE, H, W, _diag, adapt_bn, fit_predict_levels, predict_structure,
                reconstruct_and_zip,
            )
            from train_two_head import compose_module, make_arm_model  # noqa: E402

            model = make_arm_model(arm).to(DEVICE)
            model.load_state_dict(ckpt_obj["state_dict"])
            composed = compose_module(model, arm).to(DEVICE)

            cache = Path(args.cache_dir)
            n_bn = adapt_bn(composed, cache, args.adabn, lut=None,
                            shuffle_seed=args.adabn_shuffle)
            print(f"AdaBN: {args.adabn} {n_bn}장으로 BN 통계 재계산 "
                  f"(shuffle_seed={args.adabn_shuffle})", flush=True)

            mask_rate, mask_diag = None, None
            if arm == "two_head":
                # 로깅 전용 — 게이트는 manifest["sim_mask_gate"](학습 시점에 고정)가 담당한다.
                mask_rate, mask_prob = compute_mask_rate(model, cache, H, W)
                mask_diag = two_head.mask_diagnostics(mask_prob)
                print(f"mask_rate={mask_rate:.4f} "
                      f"(sim_gt_pos_rate={manifest['sim_gt_pos_rate']:.4f}, "
                      "로깅 전용 — 게이트는 sim_mask_gate가 담당)", flush=True)

            cls, diag = fit_predict_levels(Path(args.data_dir), cache, args.level_source,
                                           args.level_ckpt)
            if args.level_smooth > 1:
                before = cls.copy()
                cls = smooth_levels(cls, args.level_smooth)
                changed = int((before != cls).sum())
                diag = {**_diag(args.level_source, cls), "smoothed": args.level_smooth,
                        "changed": changed, "changed_frac": round(changed / len(cls), 4)}
            cls_sha256 = two_head.sha256_array(cls)

            structure = predict_structure(composed, cache)

            names = json.loads((cache / "test_names.json").read_text(encoding="utf-8"))
            expected_shape = (len(names), H, W)
            # 실제 배정된 레벨 값 (LEVELS[cls]) — output_pos_rate 계산용
            levels_assigned = np.array(LEVELS, dtype=np.float32)[cls]
            out_pos_rate = _output_pos_rate(structure, levels_assigned)

            gate_errors = two_head.submission_gate(structure, arm, mask_rate, manifest,
                                                   peer_manifest, cfg,
                                                   expected_shape=expected_shape)
            if gate_errors:
                print(json.dumps({"gates": gate_errors, "zip": None}, ensure_ascii=False))
                raise SystemExit(2)
            if dump_path is not None:  # 게이트를 통과한 structure만 남긴다
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(dump_path, structure)
                print(f"구조 성분 덤프 → {dump_path} {structure.shape}", flush=True)

            try:
                n = reconstruct_and_zip(structure, cache, cls, args.tau, submit_path)
                print(f"제출본 {n}장 → {submit_path}", flush=True)
                check = verify_submission(submit_path)
            except BaseException:
                # 조립/검증 중 무엇이 터지든 반쯤 쓰인 zip을 남기지 않는다.
                if submit_path.exists():
                    submit_path.unlink(missing_ok=True)
                raise

            if not check.ok:
                # 검증되지 않은 zip은 남기지 않는다 — "제출본이 있지만 점수를 보장 못 함"이 최악이다.
                submit_path.unlink(missing_ok=True)
                print(json.dumps({"gates": ["submission_verify_failed"],
                                  "verify": check.to_dict(), "zip": None}, ensure_ascii=False))
                raise SystemExit(3)

            # 해시·최종 보고까지 락 안에서 — 모델이 아직 GPU에 있는 동안 락을 놓지 않는다
            ckpt_sha256 = hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()
            peer_ckpt_sha256 = hashlib.sha256(Path(args.peer_ckpt).read_bytes()).hexdigest()
            level_ckpt_sha256 = hashlib.sha256(Path(args.level_ckpt).read_bytes()).hexdigest()

            print(json.dumps({
                "x_domain": "real", "y_source": "real_depth_gt",
                "metric": {"name": "leaderboard_rmse", "x_domain": "real",
                           "y_source": "real_depth_gt"},
                "arm": arm, "ckpt": args.ckpt, "ckpt_sha256": ckpt_sha256,
                "peer_ckpt": args.peer_ckpt, "peer_ckpt_sha256": peer_ckpt_sha256,
                "level_ckpt_sha256": level_ckpt_sha256,
                "config": cfg, "mask_rate": mask_rate, "mask_diagnostics": mask_diag,
                "output_pos_rate": out_pos_rate, "cls_sha256": cls_sha256,
                "sim_gt_pos_rate": manifest.get("sim_gt_pos_rate"),
                "sim_mask_gate": manifest.get("sim_mask_gate"),
                "gates": [], "zip": str(submit_path), "n": n, "level_diag": diag,
                "verify": check.to_dict(),
            }, ensure_ascii=False))
    except ResourceBusy as exc:
        print(json.dumps({"gates": [f"gpu-0 busy: {exc}"], "zip": None}, ensure_ascii=False))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
