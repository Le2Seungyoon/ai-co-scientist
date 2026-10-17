"""H7 — DANN 특징 도메인 정렬. sim 구조 회귀 + sim/real 도메인 판별을 한 번에 학습하는 얇은 CLI.

    d̂ = L̂ · (1 − ŝ)                                          [docs/data-facts.md §2]
    구조 손실  L1(ŝ, s), s = depth_to_s(d, L)                   (sim만, real depth GT는 쓰지 않는다)
    도메인 손실 BCE(disc(GRL(f)), domain), domain: sim=0 / real_train=1

arm은 `lambda_max` 하나만 다르다(A=0.0, B=1.0) — GRL 계수 최대값 외 모든 것(seed·step 순서·
optimizer state 모양)이 같아야 비교가 성립한다 (`docs/experiment/H7-dann-feature-alignment.md`).
`--lambda-max`는 그 확인용 CLI 플래그다 — 값 자체는 `DannConfig.for_arm`이 `ARMS[arm]`에서
고정하고, `validate_arm`이 둘이 어긋나면(예: `--arm A --lambda-max 1.0`) 즉시 막는다.

로직은 전부 `ai_co_scientist.dann`에 있다 — 이 파일은 데이터 로드·루프·체크포인트 저장만 한다.
**top-level에서 torch를 import하지 않는다**: `torch` / `train_structure` / `tqdm`은 `train()`
안에서만 import해, 이 worktree(dev deps뿐, torch 없음)에서도 `train_dann` 모듈 자체는 import되고
`--help`/`--dry-plan`/`check-parity`는 동작한다. 학습 하이퍼파라미터(lambda·epochs·lr·batch·
seed)는 전부 `DannConfig.for_arm`이 사전등록값으로 고정한다 — CLI 플래그로 바꿀 수 없다.

실행은 두 단계다. `prepare()`는 CPU만 쓴다(산출물 정책, 코드 fingerprint, 데이터 해시, 분할,
step plan) — GPU를 잡기 전에 끝낸다. `train()`(모델 초기화부터 마지막 체크포인트 저장까지)만
`resource_lock("gpu-0")` 안에서 돈다 — 8GB 카드 하나를 워크트리 여럿이 다툴 수 있으므로
(`architecture.md` → Parallel execution contract), 기본 timeout=0으로 다른 실행이 GPU를 쥐고
있으면 큐잉하지 않고 즉시 실패한다.

두 arm의 parity 확인:
    train_dann.py --arm B ... --parity-with <arm A manifest>   # arm B 학습 전에 중단 조건 검사
    train_dann.py check-parity <manifest A> <manifest B>        # 사후 대조 (CPU, 파일만 읽는다)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ai_co_scientist import dann
from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.locks import resource_lock

REPO_ROOT = Path(__file__).resolve().parents[1]

# 캐시에서 실제로 여는 파일 — manifest의 sources가 이 목록에서 나온다.
CACHE_FILES = ("sim_sem.npy", "sim_depth.npy", "sim_case.npy", "real_sem.npy")
DOMAIN_FILES = ("sim_sem.npy", "real_sem.npy")  # (sim, real) — domain 판별이 보는 입력 전부


def default_out(report_id: str, arm: str) -> str:
    """--out 기본값. report_id·arm이 파일명에 박혀 있어야 두 arm의 산출물이 서로를 덮어쓰지 않는다."""
    return f"runtime/ckpt/{report_id}-dann-{arm}.pt"


def build_parser() -> argparse.ArgumentParser:
    ensure_utf8_console()  # argparse가 --help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser(
        description="H7 DANN — sim 구조 회귀 + sim/real 도메인 판별 동시 학습 "
                    "(사후 parity 대조는 `check-parity <manifest A> <manifest B>`)")
    ap.add_argument("--arm", choices=sorted(dann.ARMS), required=True,
                    help="A=lambda_max 0.0(대조군) / B=lambda_max 1.0")
    ap.add_argument("--lambda-max", type=float, required=True,
                    help="arm 확인용(validate_arm). DannConfig.for_arm이 실제 값은 ARMS[arm]에서 "
                         "고정하므로, 여기 값이 그것과 다르면(예: --arm A --lambda-max 1.0) 즉시 "
                         "실패한다 — 두 arm이 lambda_max 하나만 달라야 한다는 사전등록을 CLI에서도 "
                         "명시적으로 확인시킨다")
    ap.add_argument("--report-id", required=True, help="사전등록 report_id (EXP-0NN)")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="",
                    help="기본값: runtime/ckpt/<report-id>-dann-<arm>.pt")
    ap.add_argument("--amp", action="store_true", help="혼합정밀 (기본 off)")
    ap.add_argument("--resume", action="store_true",
                    help="manifest가 있고 완료 표식 result_json이 없을 때만 이어서 학습한다 "
                         "(check_output_policy). manifest의 parity_key·arm·report_id가 재계산과 "
                         "같을 때만 실제로 이어붙인다")
    ap.add_argument("--parity-with", default="",
                    help="다른 arm의 manifest. 주면 이 arm의 manifest를 쓰기 전에(첫 optimizer "
                         "step 전에) check_arm_parity로 대조하고, 어긋나면 학습 없이 중단한다")
    ap.add_argument("--dry-plan", action="store_true",
                    help="데이터를 읽지 않고 config + 실행 계약 + 산출물 경로 5종 + 코드 "
                         "fingerprint JSON만 찍는다 (파일 생성 없음, 다만 --out이 이미 쓰였는지 "
                         "존재 검사는 한다)")
    return ap


def build_check_parity_parser() -> argparse.ArgumentParser:
    ensure_utf8_console()
    ap = argparse.ArgumentParser(
        prog="train_dann.py check-parity",
        description="두 arm의 manifest가 ARM_FIELDS 밖에서 같은지 대조한다 (CPU, 파일만 읽는다)")
    ap.add_argument("manifest_a")
    ap.add_argument("manifest_b")
    return ap


def _resolve_paths(args: argparse.Namespace) -> dict:
    """--out → output_paths(5종) → check_output_policy. 데이터를 읽기 전에, --dry-plan에서도 부른다.

    check_output_policy는 `exists()` 검사만 하고 아무것도 만들지 않으므로 --dry-plan에서 불러도
    "파일을 생성하지 않는다"는 계약을 깨지 않는다 — 오히려 --dry-plan이 사고(이미 있는 out을
    덮어쓰려는 실행)를 학습 시작 전에 잡아주는 자리가 된다.
    """
    out = Path(args.out or default_out(args.report_id, args.arm))
    paths = dann.output_paths(out)
    dann.check_output_policy(paths, args.resume)
    return paths


def _dry_plan(args: argparse.Namespace) -> dict:
    """`--dry-plan` 전용 미리보기. 데이터도 torch도 건드리지 않는다. 코드 fingerprint는
    보여주기만 하고 dirty여도 거부하지 않는다 — 거부는 실제 학습(`prepare`)의 몫이다."""
    cfg = dann.DannConfig.for_arm(args.arm, amp=args.amp)
    paths = _resolve_paths(args)
    return {
        "dry_plan": True,
        "report_id": args.report_id,
        "arm": args.arm,
        "config": cfg.to_dict(),
        "contract": dann.run_contract(cfg),
        "code": dann.code_fingerprint(REPO_ROOT),
        "out": str(paths["ckpt"]),
        "paths": {k: str(v) for k, v in paths.items()},
    }


def prepare(args: argparse.Namespace) -> dict:
    """GPU 락 **밖**에서 끝내는 CPU 단계 — 산출물 정책, 코드 fingerprint, 데이터 해시·개수,
    분할, step plan, lambda 배열. 여기서 실패하면 GPU를 잡지도 않는다."""
    from ai_co_scientist.sem import load_labels

    cfg = dann.DannConfig.for_arm(args.arm, amp=args.amp)
    paths = _resolve_paths(args)  # 데이터 로드 전에 산출물 5종 정책을 확인해 빨리 실패한다
    code = dann.code_fingerprint(REPO_ROOT)
    dann.assert_clean_code(code)  # registry의 source_commit이 실제 실행 코드여야 한다

    cache, data_dir = Path(args.cache_dir), Path(args.data_dir)
    # domain 분기에 들어가는 배열은 이 두 파일뿐이다 — 이름을 여기서 한 번 정하고 그대로 연다.
    sim_sem_file, real_sem_file = DOMAIN_FILES
    dann.assert_domain_sources(DOMAIN_FILES)

    # ── 데이터 ──────────────────────────────────────────────
    sim_sem = np.load(cache / sim_sem_file, mmap_mode="r")
    sim_depth = np.load(cache / "sim_depth.npy", mmap_mode="r")
    sim_case = np.load(cache / "sim_case.npy")
    real_sem = np.load(cache / real_sem_file, mmap_mode="r")

    n_sim_total, n_real = len(sim_sem), len(real_sem)
    train_mask = dann.sim_structure_split(sim_case)  # EXP-005 파티션(seed 42)의 train 쪽만 구조 학습
    n_sim_train = int(train_mask.sum())
    dann.check_counts(n_sim_total, n_sim_train, n_real)

    y, site = load_labels(data_dir, n_real)
    real_probe_mask = dann.real_domain_split(site, y, cfg.probe_frac, cfg.split_seed)
    real_domain_idx = np.where(~real_probe_mask)[0]  # 도메인 학습 80%
    real_probe_idx = np.where(real_probe_mask)[0]    # 고정 도메인 probe 20%
    sim_train_idx = np.where(train_mask)[0]
    # sim probe는 구조 분할의 holdout에서, real probe와 그룹(=level)별 개수를 맞춰 뽑는다 —
    # sim 도메인 학습 표본은 sim_train_idx뿐이라 두 probe 모두 학습과 서로소다(대칭).
    sim_probe_idx = dann.sim_probe_indices(
        ~train_mask, sim_case, y[real_probe_idx], cfg.split_seed)
    sim_probe_order, real_probe_order = dann.probe_order(sim_probe_idx, real_probe_idx, cfg.seed)

    dann.assert_disjoint(sim_probe_idx, sim_train_idx, "sim probe/train")
    dann.assert_disjoint(real_probe_idx, real_domain_idx, "real probe/domain")

    sources = [dann.file_fingerprint(cache / name) for name in CACHE_FILES]

    # 에폭별 step plan을 미리 전부 뽑아 digest·steps_per_epoch를 만들고, 학습 루프도 이걸 그대로
    # 재사용한다 — step_plan은 (seed, stream, epoch)에 대해 stateless라 재호출과 재사용이 같다.
    plan_by_epoch = [dann.step_plan(sim_train_idx, real_domain_idx, cfg, ep)
                     for ep in range(1, cfg.epochs + 1)]
    steps_per_epoch = plan_by_epoch[0][0].shape[0]
    total_steps = steps_per_epoch * cfg.epochs
    lambda_arr = dann.lambda_schedule(total_steps, cfg.lambda_max, cfg.gamma)

    return {
        "cfg": cfg, "paths": paths, "code": code,
        "sim_sem": sim_sem, "sim_depth": sim_depth, "sim_case": sim_case, "real_sem": real_sem,
        "sim_probe_idx": sim_probe_order, "real_probe_idx": real_probe_order,
        "plan_by_epoch": plan_by_epoch, "steps_per_epoch": steps_per_epoch,
        "lambda_arr": lambda_arr,
        "manifest_inputs": {
            "sources": sources,
            "n_sim_train": n_sim_train, "n_real_domain": len(real_domain_idx),
            "n_real_probe": len(real_probe_idx), "n_sim_probe": len(sim_probe_idx),
            "split_digest": dann.array_digest(train_mask, real_probe_mask, sim_probe_idx),
            "plan_digest": dann.array_digest(*[arr for pair in plan_by_epoch for arr in pair]),
            "lambda_digest": dann.array_digest(lambda_arr),
            "steps_per_epoch": steps_per_epoch,
            "outputs": {k: str(v) for k, v in paths.items()},
        },
    }


def _open_manifest(args: argparse.Namespace, paths: dict, manifest: dict) -> dict:
    """manifest를 한 번만 쓴다. 이미 있으면 --resume이어야 하고 저장본이 재계산과 같은 실행
    (parity_key·arm·report_id)일 때만 그 저장본으로 이어간다 — 아니면 예외로 abort한다."""
    try:
        dann.write_manifest(paths["manifest"], manifest)
        return manifest
    except FileExistsError:
        if not args.resume:
            raise
    existing = dann.read_manifest(paths["manifest"])
    for key in ("parity_key", "arm", "report_id", "lambda_digest"):
        if existing.get(key) != manifest.get(key):
            raise ValueError(f"resume manifest mismatch — {key}가 재계산과 다르다")
    return existing


def train(args: argparse.Namespace, prep: dict) -> dict:
    """모델 초기화부터 마지막 체크포인트 저장까지 — 호출부가 `gpu-0` 락을 쥐고 부른다."""
    import torch
    from tqdm.auto import tqdm

    from ai_co_scientist.sem import CASE_LEVEL, depth_to_s
    from train_structure import DEVICE, make_model, seed_everything

    cfg, paths = prep["cfg"], prep["paths"]
    sim_sem, sim_depth, sim_case, real_sem = (
        prep["sim_sem"], prep["sim_depth"], prep["sim_case"], prep["real_sem"])
    lambda_arr, steps_per_epoch = prep["lambda_arr"], prep["steps_per_epoch"]

    # sampler RNG(epoch_rng → plan_by_epoch)는 prepare에서 이미 다 뽑았다. 여기서부터는 모델
    # 가중치 초기화만 torch RNG를 소비한다 — 두 arm이 같은 순서로 같은 모듈을 만든다.
    seed_everything(cfg.seed)
    model = make_model(cfg.arch, cfg.width).to(DEVICE)
    disc = dann.make_discriminator(cfg.feature_dim, cfg.disc_hidden).to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(disc.parameters()), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    optimizer_signature = {
        "param_groups": len(opt.param_groups),
        "params": sorted(
            [(f"model.{n}", list(p.shape)) for n, p in model.named_parameters()]
            + [(f"disc.{n}", list(p.shape)) for n, p in disc.named_parameters()]
        ),
        "lr": cfg.lr,
        "weight_decay": opt.defaults["weight_decay"],
        "betas": list(opt.defaults["betas"]),
    }
    environment = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(DEVICE),
        "code_sha256": prep["code"]["files_sha256"],
    }
    source = {k: v for k, v in prep["code"].items() if k != "files_sha256"}
    manifest = dann.build_manifest(
        cfg, report_id=args.report_id, optimizer_signature=optimizer_signature,
        environment=environment, source=source, **prep["manifest_inputs"])

    # 실행 전 중단 조건: 다른 arm과 manifest가 ARM_FIELDS 밖에서 다르면 쓰기도 전에 abort.
    if args.parity_with:
        dann.check_arm_parity(dann.read_manifest(args.parity_with), manifest)
    manifest = _open_manifest(args, paths, manifest)

    start_ep, epoch_logs = 1, []
    if args.resume and paths["resume_ckpt"].exists():
        ck = torch.load(paths["resume_ckpt"], map_location=DEVICE, weights_only=False)
        if ck["parity_key"] != manifest["parity_key"]:
            raise ValueError("resume_ckpt가 이 manifest의 실행이 아니다 — parity_key 불일치")
        model.load_state_dict(ck["state_dict"])
        disc.load_state_dict(ck["disc_state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["torch_rng"])
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        epoch_logs = list(ck["epoch_logs"])  # 크래시 전 epoch 로그를 결과에 보존한다
        start_ep = ck["epoch"] + 1
        print(f"resume: {paths['resume_ckpt']} → epoch {start_ep}부터", flush=True)

    def to_tensor(arr, idx):
        return torch.from_numpy(
            np.ascontiguousarray(arr[idx]).astype(np.float32)[:, None] / 255.0).to(DEVICE)

    probe_sources = [(sim_sem, prep["sim_probe_idx"], 0.0),
                     (real_sem, prep["real_probe_idx"], 1.0)]

    global_step = (start_ep - 1) * steps_per_epoch
    for ep in range(start_ep, cfg.epochs + 1):
        model.train()
        disc.train()
        sim_steps, real_steps = prep["plan_by_epoch"][ep - 1]
        l1_losses, domain_losses, lambdas = [], [], []
        for sb, rb in tqdm(list(zip(sim_steps, real_steps)), desc=f"epoch {ep}", leave=False):
            lam = float(lambda_arr[global_step])
            x_sim = to_tensor(sim_sem, sb)
            d_sim = np.ascontiguousarray(sim_depth[sb]).astype(np.float32)[:, None]
            lv = np.array([CASE_LEVEL[int(c)] for c in sim_case[sb]],
                          dtype=np.float32)[:, None, None, None]
            s_sim = torch.from_numpy(depth_to_s(d_sim, lv)).to(DEVICE)
            x_real = to_tensor(real_sem, rb)

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                losses = dann.dann_step_losses(model, disc, x_sim, s_sim, x_real, lam)
            scaler.scale(losses["total"]).backward()
            scaler.step(opt)
            scaler.update()

            l1_losses.append(losses["l1"].item())
            domain_losses.append(losses["domain"].item())
            lambdas.append(lam)
            global_step += 1

        sched.step()
        auc = dann.domain_probe_auc(model, disc, probe_sources, cfg.batch_size, to_tensor)
        log = {"epoch": ep, "train_l1": float(np.mean(l1_losses)),
               "domain_loss": float(np.mean(domain_losses)),
               "probe_auc": auc, "lambda": float(np.mean(lambdas)),
               "lr": float(sched.get_last_lr()[0]), "global_step": global_step}
        epoch_logs.append(log)
        print(f"epoch {ep}: train_l1={log['train_l1']:.5f} domain_loss={log['domain_loss']:.5f} "
              f"probe_auc={log['probe_auc']:.4f} lambda={log['lambda']:.4f}", flush=True)

        # 에폭별 재개점 — 크래시 손실을 1에폭으로 묶는다. 임시 파일에 다 쓴 뒤 교체하므로 저장
        # 도중 죽어도 직전 epoch의 완전한 재개점이 남는다. 로그·RNG 상태도 함께 들고 간다.
        state = {"arch": cfg.arch, "width": cfg.width, "epoch": ep,
                 "parity_key": manifest["parity_key"], "epoch_logs": epoch_logs,
                 "state_dict": model.state_dict(), "disc_state_dict": disc.state_dict(),
                 "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": (torch.cuda.get_rng_state_all()
                              if torch.cuda.is_available() else None)}
        dann.replace_atomic(paths["resume_ckpt"], lambda tmp, st=state: torch.save(st, tmp))

    # 마지막 에폭만 저장한다(사전등록: checkpoint_selection = final_epoch, holdout 선택 없음).
    # ckpt는 추론 계약(arch+width+state_dict — train_structure.load_model이 weights_only 기본값
    # 으로 읽는다)이라 discriminator·optimizer를 섞지 않는다. 둘 다 한 번만 원자적으로 게시한다.
    # result_json이 완료 표식이라 맨 마지막이다. 게시 도중 죽었다면 --resume이 마지막 epoch
    # 재개점(= 이미 게시된 파일과 같은 최종 상태)에서 돌아와 남은 것만 게시한다.
    finals = {
        "ckpt": {"arch": cfg.arch, "width": cfg.width, "state_dict": model.state_dict()},
        "disc_ckpt": {"disc_hidden": cfg.disc_hidden, "feature_dim": cfg.feature_dim,
                      "state_dict": disc.state_dict()},
    }
    for key, obj in finals.items():
        if paths[key].exists():
            if not (args.resume and start_ep > cfg.epochs):
                raise FileExistsError(f"마지막 epoch 재개가 아닌데 {paths[key]}가 이미 있다")
            continue
        dann.publish_once(paths[key], lambda tmp, o=obj: torch.save(o, tmp))

    result = {
        "manifest": str(paths["manifest"]), "parity_key": manifest["parity_key"],
        "report_id": args.report_id, "arm": args.arm, "lambda_max": cfg.lambda_max,
        "x_domain": manifest["x_domain"], "y_source": manifest["y_source"],
        "epochs": epoch_logs, "ckpt": str(paths["ckpt"]), "disc_ckpt": str(paths["disc_ckpt"]),
    }
    dann.write_json_exclusive(paths["result_json"], result)
    return result


def run(args: argparse.Namespace) -> None:
    """CPU 준비 → `gpu-0` 락 → 학습. 락은 준비가 끝난 뒤에만, 학습 동안만 쥔다."""
    prep = prepare(args)
    with resource_lock("gpu-0"):
        result = train(args, prep)
    print(json.dumps(result, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check-parity"]:
        ns = build_check_parity_parser().parse_args(argv[1:])
        dann.check_manifest_files(ns.manifest_a, ns.manifest_b)
        print(json.dumps({"parity": "ok", "manifests": [ns.manifest_a, ns.manifest_b]}))
        return
    args = build_parser().parse_args(argv)
    dann.validate_arm(args.arm, args.lambda_max)  # --dry-plan에서도 적용된다 (분기 이전)
    if args.dry_plan:
        print(json.dumps(_dry_plan(args), ensure_ascii=False))
        return
    run(args)


if __name__ == "__main__":
    main()
