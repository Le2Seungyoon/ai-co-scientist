"""H5 2단계 — student 구조 회귀기를 arm별로 학습한다 (sim-only 대조군 vs sim+real pseudo-label).

    X = sim train SEM 138,648장(EXP-005와 같은 `map_level_split(case, 0.2, 42)`의 train, arm
        seed와 무관하게 고정) 전 arm 공통 + arm1(`sim_pseudo`)만 real train SEM 60,664장을 epoch
        마다 추가한다(`RealSamplerSpec`, epoch당 real ≤ sim). / y = sim은 `s=(L-d)/L` GT, arm1의
        real 부분은 `build_pseudo_labels.py`가 만든 teacher soft pseudo-label이다 — **실제 real
        depth GT는 어디서도 쓰지 않는다.**

세 arm(arm0/arm0b/arm1)은 `STUDENT_HPARAMS`로 고정된 동일 하이퍼파라미터로 scratch에서
학습하고, **항상 마지막 에폭을 저장한다** — sim 홀드아웃으로 체크포인트를 고르지 않는다
(sim 홀드아웃은 X를 정의할 뿐 어떤 선택에도 쓰지 않는다). 판정은 리더보드에서만 난다:
학생 manifest의 `metric`은 항상 null이다.

순서가 계약이다 — 설정 차이는 **학습 전에** 막는다(사전보고 "실행 전 중단"):
  1. plan   각 arm 설정을 torch 없이 확정해 파일로 쓴다
  2. parity 세 plan이 seed·데이터 구성 외에 같은지 확인한다
  3. train  같은 인자 + `--plan`으로 학습한다 — 지문이 plan과 다르면 거부
  4. parity 학습이 쓴 student manifest로 한 번 더 (학습 환경 `runtime`까지 비교)

로직은 전부 `ai_co_scientist.self_training`에 있다. 여기는 계약을 검증하고 나서만 torch를
부르는 얇은 CLI다 — 이 워크트리는 dev 그룹만 sync되어 torch가 없다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.locks import ResourceBusy, resource_lock
from ai_co_scientist.self_training import (
    ARMS,
    GPU_LOCK,
    REAL_TRAIN_NPY,
    STUDENT_HPARAMS,
    ContractError,
    RealSamplerSpec,
    arm_spec,
    check_arm_parity,
    git_head,
    load_manifest,
    reject_aliases,
    reject_teacher_copy,
    require_clean_commit,
    sha256_file,
    sim_subset_record,
    sim_train_indices,
    student_config,
    student_manifest_path,
    student_resume_path,
    verify_pseudo_manifest,
    write_manifest,
)

SCRIPTS_DIR = Path(__file__).resolve().parent


class PseudoRealDataset:
    """real train SEM + teacher pseudo-label ŝ. H5 arm1 전용 — real depth GT는 없다.

    모듈 최상위에 둔다: Windows DataLoader 워커는 데이터셋을 pickle하므로 함수 안에서 정의한
    클래스는 `--num-workers > 0`에서 죽는다. torch는 `__getitem__`에서만 부른다(지연 import).
    """

    def __init__(self, real_sem, pseudo):
        self.real_sem, self.pseudo = real_sem, pseudo

    def __len__(self) -> int:
        return len(self.real_sem)

    def __getitem__(self, i):
        import torch
        x = np.ascontiguousarray(self.real_sem[i]).astype(np.float32)[None] / 255.0
        y = np.ascontiguousarray(self.pseudo[i]).astype(np.float32)[None]
        return torch.from_numpy(x), torch.from_numpy(y)


def _prepare(args, ap: argparse.ArgumentParser) -> dict:
    """plan과 train이 공유하는 torch 없는 계약 검증. 반환: cfg와 학습에 필요한 값들."""
    is_arm1 = args.arm == "arm1"
    if is_arm1 and not args.pseudo_manifest:
        ap.error("--arm arm1은 --pseudo-manifest가 필요하다")
    if not is_arm1 and args.pseudo_manifest:
        ap.error(f"--arm {args.arm}은 sim-only 대조군이라 --pseudo-manifest를 받지 않는다")
    if not Path(args.teacher_ckpt).is_file():
        ap.error(f"--teacher-ckpt 파일이 없다: {args.teacher_ckpt}")

    out = Path(args.out)
    resume_path = student_resume_path(out)
    manifest_path = student_manifest_path(out)
    cache = Path(args.cache_dir)
    plan_path = getattr(args, "plan", None) or getattr(args, "write", None)

    pm = None
    try:
        alias_paths = {"teacher_ckpt": args.teacher_ckpt, "out": str(out),
                       "resume": str(resume_path), "student_manifest": str(manifest_path),
                       "plan": plan_path}
        if is_arm1:
            pm = load_manifest(args.pseudo_manifest)
            alias_paths["pseudo_manifest"] = args.pseudo_manifest
            alias_paths["pseudo_labels"] = pm["labels"]["path"]
        reject_aliases(**alias_paths)

        teacher_sha = sha256_file(args.teacher_ckpt)
        reject_teacher_copy(out, teacher_sha)
        reject_teacher_copy(resume_path, teacher_sha)

        if is_arm1:
            pm = verify_pseudo_manifest(args.pseudo_manifest, teacher_ckpt=args.teacher_ckpt)
            cache_real = (cache / REAL_TRAIN_NPY).resolve()
            if Path(pm["source"]["path"]).resolve() != cache_real:
                raise ContractError(
                    f"pseudo-label manifest의 source가 --cache-dir의 {REAL_TRAIN_NPY}가 "
                    f"아니다: {pm['source']['path']} != {cache_real}")

        commit = require_clean_commit(git_head(cwd=SCRIPTS_DIR))
        sim_idx = sim_train_indices(np.load(cache / "sim_case.npy"))
        spec = arm_spec(args.arm)
        n_real = pm["labels"]["n"] if is_arm1 else 0
        # real_per_epoch은 CLI로 열지 않는다 — 사전보고가 고정하지 않은 손잡이를 arm1만
        # 돌릴 수 있으면 그 자체가 arm 차이다. 기본 min(n_real, n_sim) = real 전량/epoch.
        sampler = RealSamplerSpec.build(len(sim_idx), n_real, seed=spec["seed"])
        pseudo_sha = sha256_file(args.pseudo_manifest) if is_arm1 else None
        cfg = student_config(args.arm, sampler, pseudo_manifest_sha256=pseudo_sha,
                             source_commit=commit, sim_subset=sim_subset_record(sim_idx))
    except ContractError as e:
        ap.error(str(e))
    except (OSError, KeyError, json.JSONDecodeError) as e:
        ap.error(f"입력 파일을 읽을 수 없다: {e}")

    return {"cfg": cfg, "pm": pm, "sim_idx": sim_idx, "sampler": sampler, "spec": spec,
            "out": out, "resume_path": resume_path, "manifest_path": manifest_path,
            "cache": cache, "is_arm1": is_arm1}


def _cmd_plan(args, ap: argparse.ArgumentParser) -> None:
    """학습 **전에** arm 설정을 확정한다 — `parity`를 이 파일들에 돌려 통과해야 train이 돈다."""
    prep = _prepare(args, ap)
    try:
        write_manifest(args.write, prep["cfg"])
    except ContractError as e:
        ap.error(str(e))
    print(json.dumps(prep["cfg"], ensure_ascii=False))


def _cmd_train(args, ap: argparse.ArgumentParser) -> None:
    prep = _prepare(args, ap)
    cfg, out = prep["cfg"], prep["out"]
    resume_path, manifest_path = prep["resume_path"], prep["manifest_path"]

    try:
        plan = load_manifest(args.plan)
    except (OSError, json.JSONDecodeError) as e:
        ap.error(f"--plan을 읽을 수 없다: {e}")
    if plan.get("config_fingerprint") != cfg["config_fingerprint"]:
        ap.error(f"--plan과 지금 설정이 다르다 — parity를 통과한 설정으로만 학습한다 "
                 f"({plan.get('config_fingerprint')} != {cfg['config_fingerprint']})")
    if manifest_path.exists():
        ap.error(f"학생 manifest가 이미 존재한다 (덮어쓰지 않는다): {manifest_path}")
    if out.exists() and not args.resume:
        ap.error(f"--out이 이미 존재한다 (덮어쓰지 않는다, 재개는 --resume): {out}")
    resume_ok = bool(args.resume and resume_path.exists())

    # GPU 구간만 락 안에서 돈다 — plan·parity·검증은 락 없이 CPU로 끝난다. 모든 워크트리가
    # 같은 기계 단위 락을 다투므로 다른 학습/추론이 GPU를 쓰는 동안에는 즉시 거부된다.
    try:
        with resource_lock(GPU_LOCK):
            _train_on_gpu(args, ap, prep, resume_ok)
    except ResourceBusy as e:
        ap.error(f"{GPU_LOCK}를 다른 실행이 잡고 있다 (GPU는 한 번에 하나): {e}")


def _train_on_gpu(args, ap: argparse.ArgumentParser, prep: dict, resume_ok: bool) -> None:
    """student 학습 — 이 스크립트의 유일한 GPU 구간. `GPU_LOCK` 안에서만 부른다."""
    cfg, pm, sampler, spec = prep["cfg"], prep["pm"], prep["sampler"], prep["spec"]
    out, resume_path, manifest_path = prep["out"], prep["resume_path"], prep["manifest_path"]
    cache, is_arm1 = prep["cache"], prep["is_arm1"]

    # 계약 검증을 모두 통과한 뒤에만 torch를 부른다.
    import torch
    import torch.nn as nn
    from torch.utils.data import ConcatDataset, DataLoader, Subset
    sys.path.insert(0, str(SCRIPTS_DIR))
    from train_structure import DEVICE, StructureDataset, make_model, seed_everything  # noqa: E402

    seed_everything(spec["seed"])

    sim_sem = np.load(cache / "sim_sem.npy", mmap_mode="r")
    sim_depth = np.load(cache / "sim_depth.npy", mmap_mode="r")
    sim_case = np.load(cache / "sim_case.npy")
    sim_full = StructureDataset(sim_sem, sim_depth, sim_case,
                                blur_sigma=STUDENT_HPARAMS["blur_sigma"])
    sim_ds = Subset(sim_full, prep["sim_idx"].tolist())  # EXP-005 train 분할 138,648장
    if is_arm1:
        real_sem = np.load(pm["source"]["path"], mmap_mode="r")
        pseudo = np.load(pm["labels"]["path"], mmap_mode="r")
        ds = ConcatDataset([sim_ds, PseudoRealDataset(real_sem, pseudo)])
    else:
        ds = sim_ds

    model = make_model(STUDENT_HPARAMS["arch"], STUDENT_HPARAMS["width"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=STUDENT_HPARAMS["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STUDENT_HPARAMS["epochs"])
    crit = nn.L1Loss().to(DEVICE)

    start_ep = 1
    if resume_ok:
        ck = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        if ck.get("config_fingerprint") != cfg["config_fingerprint"]:
            ap.error(f"--resume 거부: {resume_path}는 다른 설정에서 만들어졌다 "
                     f"({ck.get('config_fingerprint')} != {cfg['config_fingerprint']})")
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        print(f"resume: {resume_path} → epoch {start_ep}부터", flush=True)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | arm={args.arm} params={n_par:,} | "
          f"sim {sampler.n_sim} + real/epoch {sampler.real_per_epoch}", flush=True)

    for ep in range(start_ep, STUDENT_HPARAMS["epochs"] + 1):
        model.train()
        idx = sampler.epoch_indices(ep).tolist()
        loader = DataLoader(ds, batch_size=STUDENT_HPARAMS["batch_size"], sampler=idx,
                            num_workers=args.num_workers)
        losses = []
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        print(f"epoch {ep}: train_l1={np.mean(losses):.5f}", flush=True)

        resume_path.parent.mkdir(parents=True, exist_ok=True)
        # 에폭별 재개점 — best/holdout 선택 없이 진행 상황만 담는다. out(추론 계약)과는
        # 별도 파일이다 (coding-patterns.md). 샘플러 순서는 (seed, epoch)만의 함수라 재개해도
        # 끊기지 않은 실행과 배치 순서가 같다.
        torch.save({"arch": STUDENT_HPARAMS["arch"], "width": STUDENT_HPARAMS["width"],
                    "epoch": ep, "config_fingerprint": cfg["config_fingerprint"],
                    "state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, resume_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": STUDENT_HPARAMS["arch"], "width": STUDENT_HPARAMS["width"],
                "blur_sigma": STUDENT_HPARAMS["blur_sigma"],
                "state_dict": model.state_dict()}, out)
    print(f"student(마지막 에폭, 홀드아웃 선택 없음) → {out}", flush=True)

    manifest = {**cfg, "out": str(out),
                "x_domain": "sim+real" if is_arm1 else "sim",
                "y_source": "sim_depth_gt+pseudo_label" if is_arm1 else "sim_depth_gt",
                "metric": None, "note": "judged by leaderboard only",
                # 학습 환경 — parity가 비교한다 (arm0은 CPU, arm1은 GPU 같은 차이를 잡는다)
                "runtime": {"device": str(DEVICE), "torch": torch.__version__,
                            "cuda": torch.version.cuda, "num_workers": args.num_workers}}
    try:
        write_manifest(manifest_path, manifest)
    except ContractError as e:
        ap.error(str(e))
    print(json.dumps(manifest, ensure_ascii=False))


def _cmd_parity(args, ap: argparse.ArgumentParser) -> None:
    if len(args.manifests) < 2:
        ap.error("parity는 manifest가 최소 2개 필요하다")
    try:
        manifests = [load_manifest(p) for p in args.manifests]
        check_arm_parity(manifests)
    except ContractError as e:
        ap.error(str(e))
    except (OSError, json.JSONDecodeError) as e:
        ap.error(f"manifest를 읽을 수 없다: {e}")
    print(json.dumps({"ok": True, "arms": [m.get("arm") for m in manifests]}, ensure_ascii=False))


def main(argv=None) -> None:
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser(description="H5 self-training student plan/학습/parity CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _arm_args(sp):
        sp.add_argument("--arm", required=True, choices=tuple(ARMS))
        sp.add_argument("--cache-dir", required=True)
        sp.add_argument("--out", required=True, help="student ckpt 저장 경로 (load_model 호환)")
        sp.add_argument("--teacher-ckpt", required=True,
                        help="student가 teacher 가중치의 사본이 아님을 검증하는 기준")
        sp.add_argument("--pseudo-manifest", default="", help="arm1 필수, arm0/arm0b는 금지")

    plan = sub.add_parser("plan", help="학습 없이 arm 설정을 확정해 파일로 쓴다 (torch 불필요)")
    _arm_args(plan)
    plan.add_argument("--write", required=True, help="plan JSON 경로 (덮어쓰지 않는다)")

    train = sub.add_parser("train", help="parity를 통과한 plan대로 arm 하나를 scratch 학습")
    _arm_args(train)
    train.add_argument("--plan", required=True, help="같은 인자로 만든 plan JSON — 지문 일치 필수")
    train.add_argument("--num-workers", type=int, default=0, help="Windows는 0 권장")
    train.add_argument("--resume", action="store_true",
                       help="<out>.resume.pt가 있고 설정 지문이 같으면 이어서 학습")

    parity = sub.add_parser("parity", help="여러 arm manifest가 seed/data 외에는 동일한지 확인")
    parity.add_argument("manifests", nargs="+", metavar="MANIFEST.json")

    args = ap.parse_args(argv)
    if args.cmd in ("plan", "train"):
        args.pseudo_manifest = args.pseudo_manifest or None
        (_cmd_plan if args.cmd == "plan" else _cmd_train)(args, ap)
    elif args.cmd == "parity":
        _cmd_parity(args, ap)


if __name__ == "__main__":
    main()
