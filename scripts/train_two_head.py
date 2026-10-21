"""H8 — 2-head 구조 회귀 학습. arm A(single)/arm B(two_head) 공통 스크립트.

    arm A: 기존 단일 출력 s 회귀 (train_structure.PlainMLP, L1 loss)
    arm B: mask_logit + s_pos_hat 2-head, BCEWithLogits + L1(m=1) 1:1 합산

두 arm이 backbone·optimizer·schedule·seed·split을 전부 공유해야 비교가 성립한다(H8 pre-report
"공통 backbone"). 이 스크립트는 그 공유 경로 하나이고, arm별 분기는 모델 생성·손실 두 곳뿐이다.

**이 모듈을 import해도 torch/cv2가 로드되지 않는다** — 순수 로직은 `ai_co_scientist.two_head`
(numpy)에 있고, torch가 필요한 모든 것은 함수 안에서 지연 import한다(coding-patterns.md).
체크포인트 선택은 sim 홀드아웃 순위와 무관하다: **항상 마지막 에폭**을 저장한다(H8 pre-report
"ckpt_selection: final epoch") — arm B가 유리한 에폭만 골라 채택 편향을 만들지 않기 위해서다.

reviewer Phase 1 수정 반영: 배치 순서 결정 RNG를 전역이 아닌 명시적 `torch.Generator`로 분리하고
(B1, arm B의 추가 head 초기화가 매 에폭 셔플을 밀어내지 않게), `--resume`으로 순서까지 이어받는다.
2-head 백본은 arm A의 PlainMLP 출력층 객체를 그대로 재사용해 초기화값을 공유한다(B2).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.locks import ResourceBusy, resource_lock  # stdlib-only, torch-free
from ai_co_scientist.sem import CASE_LEVEL, map_level_split
from ai_co_scientist import two_head

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 형제 스크립트(train_structure) 지연 import용


def make_arm_model(arm: str):
    """arm에 맞는 backbone을 만든다 (torch 지연 import).

    두 arm 모두 `train_structure.PlainMLP()`를 **먼저** 만든다 — encoder/decoder 초기화가
    같은 시드 아래서 arm A와 bit-identical해야 backbone 폭 변경이 confound가 아니게 된다
    (H8 pre-report). arm B는 encoder + decoder[:-1]을 trunk로 재사용하고, **decoder의 마지막
    Linear(=arm A의 출력층)를 그대로 depth_head로 재사용**해(B2) 같은 초기화값을 공유한다.
    mask_head는 그 다음에 새로 만든다(RNG 소비 순서). PlainMLP 객체 자체는 attribute로
    보관하지 않는다 — 보관하면 미사용 파라미터가 두 번 등록될 위험이 있다(n_params 계약).
    """
    import torch.nn as nn
    from train_structure import H, W, PlainMLP  # noqa: E402  (지연 import)

    base = PlainMLP()
    if arm == "single":
        return base
    if arm != "two_head":
        raise ValueError(f"알 수 없는 arm: {arm}")

    class TwoHeadMLP(nn.Module):
        """forward는 항상 (mask_logit, s_pos_hat) — 둘 다 (B,1,H,W), 활성화 없는 raw 값."""

        def __init__(self, base_model: nn.Module):
            super().__init__()
            layers = list(base_model.decoder.children())
            self.encoder = base_model.encoder
            self.backbone = nn.Sequential(*layers[:-1])
            self.depth_head = layers[-1]  # arm A 출력층과 동일 초기화값 (재사용, 재생성 아님)
            self.mask_head = nn.Linear(1024, H * W)  # depth_head 이후 생성 — RNG 소비 순서 유지

        def forward(self, x):
            b = x.shape[0]
            z = self.backbone(self.encoder(x.view(b, -1)))
            mask_logit = self.mask_head(z).view(b, 1, H, W)
            s_pos_hat = self.depth_head(z).view(b, 1, H, W)
            return mask_logit, s_pos_hat

    return TwoHeadMLP(base)


def compose_module(model, arm: str):
    """단일 텐서 (B,1,H,W)를 내는 wrapper. `infer_decomposed.adapt_bn`/`predict_structure`가
    arm과 무관하게 `model(x)` 한 번으로 동작하도록 arm B의 2-head 출력을 합성한다."""
    if arm == "single":
        return model
    if arm != "two_head":
        raise ValueError(f"알 수 없는 arm: {arm}")

    import torch.nn as nn

    class ComposedTwoHead(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            mask_logit, s_pos_hat = self.inner(x)
            return two_head.compose_output_torch(mask_logit, s_pos_hat)

    return ComposedTwoHead(model)


def evaluate_epoch(model, arm: str, loader, levels: np.ndarray) -> dict:
    """sim 홀드아웃 지표 (`two_head.StructureMetrics`) — **위생/기전 지표이며 채택 지표가
    아니다**(H8 pre-report). ckpt 저장 여부와 무관하게 매 에폭 로그만 한다."""
    import torch

    model.eval()
    metrics = two_head.StructureMetrics()
    device = next(model.parameters()).device
    off = 0
    with torch.no_grad():
        for x, s in loader:
            x, s = x.to(device), s.to(device)
            lv = levels[off:off + len(x)]
            off += len(x)
            if arm == "single":
                s_hat = model(x)
                mask_prob = None
            else:
                mask_logit, s_pos_hat = model(x)
                s_hat = two_head.compose_output_torch(mask_logit, s_pos_hat)
                mask_prob = torch.sigmoid(mask_logit).cpu().numpy()
            metrics.update(s_hat.cpu().numpy(), s.cpu().numpy(), lv, mask_prob=mask_prob)
    return metrics.value()


def build_parser() -> argparse.ArgumentParser:
    """`--help`는 torch 없이도 동작해야 한다 — 기본값은 전부 `two_head.PREREGISTERED`에서
    가져와 기본 실행이 사전등록 하이퍼파라미터와 항상 일치하게 한다."""
    ap = argparse.ArgumentParser(
        description="H8 2-head 구조 회귀 학습 — arm A(single)/arm B(two_head) 공통 스크립트")
    ap.add_argument("--arm", required=True, choices=two_head.ARMS, help="single | two_head")
    ap.add_argument("--out", required=True, help="체크포인트 저장 경로(.pt)")
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--epochs", type=int, default=two_head.PREREGISTERED["epochs"])
    ap.add_argument("--batch-size", type=int, default=two_head.PREREGISTERED["batch_size"])
    ap.add_argument("--lr", type=float, default=two_head.PREREGISTERED["lr"])
    ap.add_argument("--seed", type=int, default=two_head.PREREGISTERED["seed"])
    ap.add_argument("--split-seed", type=int, default=two_head.PREREGISTERED["split_seed"])
    ap.add_argument("--val-frac", type=float, default=two_head.PREREGISTERED["val_frac"])
    ap.add_argument("--num-workers", type=int, default=0, help="Windows는 0 권장")
    ap.add_argument("--resume", action="store_true",
                    help="<out>.resume.pt가 있으면 그 다음 에폭부터, 같은 배치 순서로 이어서 학습")
    return ap


def _git(*args: str):
    """읽기 전용 git 조회. 실패(exit≠0·git 없음)면 None — 빈 출력과 구별해야 `git status`
    실패를 깨끗한 트리로 오인하지 않는다(`two_head.git_preflight_errors`가 None을 거부)."""
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             check=False, cwd=Path(__file__).resolve().parents[1])
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _atomic_torch_save(obj, path: Path) -> None:
    """임시 파일에 쓴 뒤 os.replace — 저장 도중 죽어도 이전 파일이 반쯤 쓰인 채 남지 않는다."""
    import torch

    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _run(args, start_commit: str) -> None:
    """실제 학습 본체. `two_head.GPU_LOCK`을 쥔 채로만 호출된다(`main`이 락을 두른다) —
    torch import를 포함한 모든 GPU/장시간 작업이 여기 있다. git·하이퍼파라미터·산출물 경로
    preflight는 `main`이 락 전에 끝냈고, `start_commit`은 그때 읽은 HEAD다."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
    from tqdm.auto import tqdm
    from train_structure import DEVICE, StructureDataset, seed_everything  # noqa: E402

    seed_everything(args.seed)
    cache = Path(args.cache_dir)
    sem_arr = np.load(cache / "sim_sem.npy", mmap_mode="r")
    depth = np.load(cache / "sim_depth.npy", mmap_mode="r")
    case = np.load(cache / "sim_case.npy")

    model = make_arm_model(args.arm).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE} | arm={args.arm} params={n_par:,} | sim {len(sem_arr)}장", flush=True)

    va_mask = map_level_split(case, args.val_frac, args.split_seed)
    tr_mask = ~va_mask
    train_idx = np.where(tr_mask)[0]
    va_idx = np.where(va_mask)[0]
    print(f"split(depth-map 단위): train {tr_mask.sum()} / val {va_mask.sum()}", flush=True)
    if two_head.batch_errors(len(train_idx), args.batch_size):
        print(json.dumps({"errors": two_head.batch_errors(len(train_idx), args.batch_size),
                          "ckpt": None}, ensure_ascii=False))
        raise SystemExit(2)

    # 데이터 identity — manifest와 --resume 검증이 같은 값을 쓴다 (학습 전에 계산)
    fingerprint = {
        "n_sim": int(len(sem_arr)), "sem_shape": list(sem_arr.shape),
        "depth_shape": list(depth.shape),
        "case_counts": {int(k): int(v) for k, v in zip(*np.unique(case, return_counts=True))},
        "case_sha256": two_head.sha256_array(case),
        "val_idx_sha256": two_head.sha256_array(va_idx.astype(np.int64)),
        # 결정적 부분표본 — depth 전체(수GB)를 해시하지 않고도 GT 내용이 같은지 확인한다
        "depth_sample_sha256": two_head.sha256_array(np.ascontiguousarray(depth[::997])),
    }

    ds = StructureDataset(sem_arr, depth, case, blur_sigma=0.0)
    # B1: 배치 순서 RNG를 전역과 분리한다 — arm B는 추가 head 생성으로 전역 RNG를 arm A보다
    # 더 소비하므로, 전역에 얹으면 두 arm의 배치 순서가 갈라져 optimizer 궤적이 달라진다.
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    sampler = SubsetRandomSampler(train_idx.tolist(), generator=gen)
    tl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)
    vl = DataLoader(Subset(ds, va_idx.tolist()), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers)
    va_levels = np.array([CASE_LEVEL[int(c)] for c in case[va_idx]], dtype=np.float32)

    # 두 arm 모두 학습 전에 결정적으로 계산 — 같은 depth/case/split이면 arm 무관하게 일치한다.
    sim_gt_pos_rate = two_head.sim_gt_pos_rate_from_depth(depth, case, train_idx)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    l1 = nn.L1Loss().to(DEVICE)

    # 에폭별 재개점 — best ckpt(`out`)와 **별도 파일**(coding-patterns.md 규약과 동일 이유).
    paths = two_head.train_artifact_paths(args.out)
    resume_path = paths["resume"]
    start_ep = 1
    resumed_from_epoch = None
    epoch_metrics = []  # sim 홀드아웃 — 위생/기전 지표. 선택에 쓰지 않는다 (최종 ckpt는 마지막 에폭)
    if args.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        # P2-9: 재개 파일의 하이퍼파라미터·git_commit·data_fingerprint가 이번 실행과 어긋나면
        # 거부한다 — 안 그러면 ckpt가 다른 설정/코드/데이터의 혼합물이 된다.
        current = {"arm": args.arm, "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                   "split_seed": args.split_seed, "val_frac": args.val_frac,
                   "batch_size": args.batch_size, "git_commit": start_commit,
                   "data_fingerprint": fingerprint}
        mismatches = two_head.resume_identity_errors(ck, current)
        if mismatches:
            print(json.dumps({"errors": mismatches, "ckpt": None}, ensure_ascii=False))
            raise SystemExit(2)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        # P2-1: map_location=DEVICE가 sampler_state(ByteTensor)까지 GPU로 옮긴다 —
        # Generator.set_state는 CPU ByteTensor만 받으므로 되돌린다.
        gen.set_state(ck["sampler_state"].cpu())  # 배치 순서까지 이어받는다 (B1)
        epoch_metrics = list(ck["epoch_metrics"])  # 재개 전 에폭 로그도 최종 JSON에 남긴다
        resumed_from_epoch = ck["epoch"]
        start_ep = ck["epoch"] + 1
        print(f"resume: {resume_path} → epoch {start_ep}부터 (배치 순서 복원됨)", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        losses = []
        for x, s in tqdm(tl, desc=f"epoch {ep}", leave=False):
            x, s = x.to(DEVICE), s.to(DEVICE)
            opt.zero_grad()
            if args.arm == "single":
                loss = l1(model(x), s)
            else:
                mask_logit, s_pos_hat = model(x)
                loss, _bce, _l1_pos = two_head.two_head_loss(mask_logit, s_pos_hat, s)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        m = evaluate_epoch(model, args.arm, vl, va_levels)
        epoch_metrics.append({"epoch": ep, **m})
        print(f"epoch {ep}: train_loss={np.mean(losses):.5f} s_rmse={m['s_rmse']:.5f} "
              f"depth_rmse={m['depth_rmse']:.4f} mask_auroc={m.get('mask_auroc')}", flush=True)

        resume_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save({"arm": args.arm, "epoch": ep, "epochs": args.epochs, "lr": args.lr,
                            "seed": args.seed, "split_seed": args.split_seed,
                            "val_frac": args.val_frac,
                            "batch_size": args.batch_size,  # P2-9: 다음 --resume이 검증
                            "git_commit": start_commit, "data_fingerprint": fingerprint,
                            "state_dict": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "sampler_state": gen.get_state(),
                            "epoch_metrics": epoch_metrics}, resume_path)

    if not epoch_metrics:
        # --resume이 이미 마지막 에폭까지 끝난 ckpt를 다시 불렀다 — 루프가 한 번도 안 돌았다.
        # sim_mask_gate/최종 JSON은 마지막 에폭 지표를 요구하므로 로드된 모델로 1회만 평가한다.
        m = evaluate_epoch(model, args.arm, vl, va_levels)
        epoch_metrics.append({"epoch": args.epochs, **m})

    last_epoch = epoch_metrics[-1]["epoch"]
    sim_gate = two_head.sim_mask_gate(epoch_metrics[-1], args.arm, last_epoch)

    # P2-2: HEAD가 학습 도중 바뀌지 않았는지 재확인 — 바뀌었으면 manifest의 git_commit이 실제로
    # 돌아간 코드를 더 이상 대표하지 못하므로 ckpt를 저장하지 않는다.
    end_commit = _git("rev-parse", "HEAD")
    if end_commit != start_commit:
        print(json.dumps({"errors": [f"git HEAD가 학습 중 바뀌었다: {start_commit} → {end_commit}"],
                          "ckpt": None}, ensure_ascii=False))
        raise SystemExit(2)

    # holdout 순위와 무관하게 **항상 마지막 에폭**을 저장한다 (H8 pre-report ckpt_selection).
    manifest = two_head.build_manifest(
        arm=args.arm, n_train=int(tr_mask.sum()), n_val=int(va_mask.sum()), n_params=n_par,
        data_fingerprint=fingerprint, git_commit=start_commit, git_dirty=False,
        sim_gt_pos_rate=float(sim_gt_pos_rate), resumed_from_epoch=resumed_from_epoch,
        sim_mask_gate=sim_gate,
        arch=two_head.PREREGISTERED["arch"], batch_size=args.batch_size, lr=args.lr,
        optimizer=two_head.PREREGISTERED["optimizer"], schedule=two_head.PREREGISTERED["schedule"],
        epochs=args.epochs, seed=args.seed, split_seed=args.split_seed, val_frac=args.val_frac)

    errors = two_head.validate_manifest(manifest)
    if errors:
        # 사전등록 하이퍼파라미터에서 벗어난 실행 — ckpt를 만들지 않는다 (arm parity 전제 위반).
        print(json.dumps({"errors": errors, "ckpt": None}, ensure_ascii=False))
        raise SystemExit(2)

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    manifest_path = paths["manifest"]
    # manifest를 ckpt보다 먼저 쓴다 — ckpt가 존재하면 manifest도 반드시 있다(역은 성립 안 해도
    # 다음 실행의 train_output_errors가 거부하므로 조용한 덮어쓰기는 없다).
    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    _atomic_torch_save(two_head.ckpt_payload(args.arm, manifest, state_dict), paths["ckpt"])
    ckpt_sha256 = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(f"ckpt → {args.out} (sha256={ckpt_sha256[:12]}...) · manifest → {manifest_path}",
          flush=True)

    print(json.dumps({
        "x_domain": "sim", "y_source": "sim_depth_gt",
        "metric": {"name": f"sim_holdout_{args.arm}", "x_domain": "sim",
                   "y_source": "sim_depth_gt", "hygiene_only": True},
        "arm": args.arm, "ckpt": args.out, "ckpt_sha256": ckpt_sha256,
        "manifest": str(manifest_path),
        "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()), "params": n_par,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "split_seed": args.split_seed, "val_frac": args.val_frac,
        "resumed_from_epoch": resumed_from_epoch,
        "sim_gt_pos_rate": manifest["sim_gt_pos_rate"],
        "sim_mask_gate": sim_gate,
        "sim_mask_gate_label": "stop gate, not a metric",
        "sim_holdout": epoch_metrics,
        "note": "sim_holdout은 에폭별 위생/기전 지표다 — 채택에도 ckpt 선택에도 쓰지 않는다 "
                "(H8 pre-report: 항상 마지막 에폭 저장). sim_mask_gate는 지표가 아니라 "
                "stop gate다 — 실패해도 ckpt는 저장되지만(진단용), 그 ckpt로 추론하면 "
                "infer_two_head.py가 제출을 거부한다.",
    }, ensure_ascii=False))


def main():
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    args = build_parser().parse_args()

    # P2-3: 락/torch import 전에 하이퍼파라미터부터 사전등록과 맞춰본다 — GPU를 잡기 전에
    # 드리프트를 잡아야 락을 쥔 채로 실패하는 낭비가 없다.
    hp = {"arch": two_head.PREREGISTERED["arch"], "batch_size": args.batch_size, "lr": args.lr,
         "optimizer": two_head.PREREGISTERED["optimizer"],
         "schedule": two_head.PREREGISTERED["schedule"], "epochs": args.epochs,
         "seed": args.seed, "split_seed": args.split_seed, "val_frac": args.val_frac}
    # P2-2: git preflight도 락 전에 — src/scripts/pyproject.toml/uv.lock이 untracked까지 깨끗해야
    # manifest의 git_commit이 실제로 돌아간 코드를 대표한다. 종료 시점 HEAD 재검사는 _run이 한다.
    start_commit = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain", "--untracked-files=all", "--",
                     *two_head.GIT_PREFLIGHT_PATHS)
    errors = (two_head.validate_hparams(hp)
              + two_head.git_preflight_errors(start_commit, porcelain)
              + two_head.train_output_errors(args.out, args.resume))
    if errors:
        print(json.dumps({"errors": errors, "ckpt": None}, ensure_ascii=False))
        raise SystemExit(2)

    try:
        with resource_lock(two_head.GPU_LOCK, timeout=0):
            _run(args, start_commit)
    except ResourceBusy as exc:
        print(json.dumps({"errors": [f"gpu-0 busy: {exc}"], "ckpt": None}, ensure_ascii=False))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
