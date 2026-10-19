"""H6 — CycleGAN sim↔real 학습/기하 위생 gate 진입점. `ai_co_scientist.cyclegan`이 로직을,
여기는 argparse·IO·학습 루프만 갖는다 (조립은 `scripts/exp.py`와 같은 관례).

**최상단에서 torch/cv2를 import하지 않는다.** 이 워크트리는 dev 그룹만 sync하므로 둘 다 없다.
`plan`/`gate`의 거부 경로(경로 위생·설정 편차·체크포인트 이름/존재)는 전부 numpy만으로 끝나고,
`train`/`gate`가 실제로 학습·번역을 돌리는 지점에서만 함수 안에서 `import torch`한다.

에폭의 정의: 1 에폭 = sim 전체(138,648장, batch_size=8)를 한 번 통과하는 것이다(`DataLoader`
`shuffle=True`, `drop_last=True` — sim이 `--num-workers` 순회의 기준 로더다). real은 작은
도메인(60,664장)이라 매 스텝 `torch.Generator(seed=42)`로 **복원추출**해 sim 배치와 짝짓는다.
**이 표본추출 방식(에폭=sim 기준·real 복원추출·시드 42)은 사전등록(H6 pre-report)에 없다** —
사전등록은 배치 크기·손실·아키텍처·학습률 스케줄만 고정했고 unpaired 배치 구성 방법은 명시하지
않았으므로, 재현 가능하도록 이 스크립트가 고정한 값이다.

체크포인트 이름 규약: 학습이 끝난(= `epoch == cfg.total_epochs`) 최종 체크포인트만
`<out-dir>/<report-id>-cyclegan.pt`로 저장된다(`expected_ckpt_name`). 에폭별 재개점은
`<out-dir>/<report-id>-cyclegan.resume.pt`로 **별도 파일**이라 이름 규약에 맞지 않고, 따라서
`gate`/`translate_sim.py`는 재개점을 절대 받아들이지 않는다(이름이 다르면 torch를 불러오기
전에 거부한다).

덮어쓰기 정책(결정론적 — 어느 경우에도 "상황에 따라 조용히"가 없다):

- 최종 ckpt가 이미 있으면 거부한다. 저장은 `.tmp`에 끝까지 쓴 뒤 `os.link`로 최종 이름을
  배타적으로 붙인다 — 경합에서도 안 덮고, 저장 중 크래시가 최종 이름에 찢긴 파일을 남기지 않는다.
- 재개점이 있는데 `--resume`이 없으면 거부한다(조용히 처음부터 돌며 재개점을 덮지 않는다).
  `--resume`인데 재개점이 없으면 거부한다(조용히 처음부터 시작하지 않는다).
- 재개점만 매 에폭 **원자적으로** 덮어쓴다(`.tmp`에 쓰고 `os.replace`).
- gate JSON은 write-once다(`write_once_json`).

GPU 락: `train`/`gate`는 모든 거부 검사를 끝낸 **뒤**, 실제 데이터를 읽기 **전**에
`resource_lock(GPU_LOCK)`(= `gpu-0`, 대기 없음)을 잡고 runtime 전체를 그 안에서 돈다. 이미
잡혀 있으면 코드 4로 거부한다. `plan`과 `--help`는 락을 잡지 않는다(CPU-safe).

미래 실행 레시피(둘 다 `uv run --group baseline` 필요 — 이 워크트리에서는 절대 실행하지 않는다):

    uv run --group baseline python scripts/train_cyclegan.py plan --report-id EXP-0NN

    uv run --group baseline python scripts/train_cyclegan.py train --report-id EXP-0NN \\
        --cache-dir runtime/cache --out-dir runtime/ckpt --num-workers 0

    uv run --group baseline python scripts/train_cyclegan.py gate --report-id EXP-0NN \\
        --ckpt runtime/ckpt/EXP-0NN-cyclegan.pt --cache-dir runtime/cache \\
        --out-json runtime/ckpt/EXP-0NN-cyclegan-gate.json
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.cyclegan import (
    GATE_SAMPLE_N,
    GPU_LOCK,
    PREREGISTERED,
    REAL_TRAIN_N,
    SIM_TRAIN_N,
    CycleGANConfig,
    GateFailedError,
    build_discriminator,
    build_generator,
    check_ckpt_name,
    check_sem_array,
    evaluate_gate,
    expected_ckpt_name,
    gate_sample_indices,
    indices_sha256,
    load_generators,
    lr_multiplier,
    measure_geometry,
    reject_test_paths,
    require_source_name,
    roundtrip_mae,
    sha256_file,
    to_signed,
    translate_u8,
    validate_config,
    write_once_json,
)
from ai_co_scientist.locks import ResourceBusy, resource_lock

DEFAULT_CACHE_DIR = "runtime/cache"  # config.yaml paths.cache_dir과 손으로 맞춤
DEFAULT_CKPT_DIR = "runtime/ckpt"  # config.yaml paths.ckpt_dir과 손으로 맞춤
REAL_SAMPLE_SEED = 42  # real 도메인 복원추출 전용 generator 시드 — 미사전등록 (모듈 docstring 참고)


def _resolve_config(config_json: str) -> CycleGANConfig:
    """`--config-json`이 없으면 사전등록 기본값, 있으면 그걸로 덮어써 `validate_config`에 넘긴다.

    사전등록은 하이퍼파라미터를 전부 고정했으므로 `--config-json`으로 뭔가를 바꾸면 거의 항상
    `validate_config`가 거부한다 — 이 흐름 자체가 "설정 편차" 거부 경로이고, torch 없이도
    (numpy/json만으로) 테스트 가능하다. 계약 문서(h6_contract.md)에 리터럴로 나열된 CLI 플래그는
    아니지만, 편차 거부 경로를 torch 없이 실행/테스트하려면 덮어쓸 입구가 있어야 해서 추가했다 —
    사전등록 값 자체를 바꾸는 기능이 아니라 "다르면 거부한다"를 증명하는 기능이다.
    """
    if not config_json:
        cfg = CycleGANConfig()
    else:
        overrides = json.loads(Path(config_json).read_text(encoding="utf-8"))
        cfg = CycleGANConfig.from_dict({**PREREGISTERED, **overrides})
    validate_config(cfg)
    return cfg


def _seed_everything(seed: int) -> None:
    """`scripts/train_structure.py`의 `seed_everything`을 **그대로 복사**한 것이다.

    `cudnn.benchmark=True`를 빼면 학습 중 cuDNN이 워크스페이스를 가변 요청해 PyTorch 할당자
    밖에서 OOM이 난다(`.agents/rules/coding-patterns.md` → Gotchas). 빼는 게 버그다.
    """
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ── plan ────────────────────────────────────────────────────

def plan(args) -> int:
    cfg = _resolve_config(args.config_json)
    cache_dir = Path(args.cache_dir)
    sim_path = cache_dir / "sim_sem.npy"
    real_path = cache_dir / "real_sem.npy"
    reject_test_paths([sim_path, real_path])
    require_source_name(sim_path, "sim_sem.npy")
    require_source_name(real_path, "real_sem.npy")
    print(json.dumps({
        "report_id": args.report_id,
        "config": cfg.to_dict(),
        "cache_dir": str(cache_dir),
        "inputs": {"sim_sem": str(sim_path), "real_sem": str(real_path)},
        "expected_ckpt_name": expected_ckpt_name(args.report_id),
        "x_domain": "sim+real_train_sem_unpaired", "y_source": "none",
    }, ensure_ascii=False))
    return 0


# ── train ───────────────────────────────────────────────────

def train(args) -> int:
    cfg = _resolve_config(args.config_json)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    sim_path = cache_dir / "sim_sem.npy"
    real_path = cache_dir / "real_sem.npy"
    ckpt_path = out_dir / expected_ckpt_name(args.report_id)
    resume_path = out_dir / f"{args.report_id}-cyclegan.resume.pt"

    reject_test_paths([sim_path, real_path, out_dir, ckpt_path])
    require_source_name(sim_path, "sim_sem.npy")
    require_source_name(real_path, "real_sem.npy")

    # 출력 존재 검사 — torch/배열 어느 것도 건드리기 전에 (덮어쓰기 정책: 모듈 docstring)
    if ckpt_path.exists():
        raise SystemExit(f"거부: 출력 체크포인트가 이미 있다(덮어쓰지 않음) -> {ckpt_path}")
    if resume_path.exists() and not args.resume:
        raise SystemExit(f"거부: 재개점이 이미 있다 — 이어가려면 --resume, 아니면 사람이 먼저 "
                         f"치울 것(조용히 덮지 않음) -> {resume_path}")
    if args.resume and not resume_path.exists():
        raise SystemExit(f"거부: --resume인데 재개점이 없다(조용히 처음부터 시작하지 않음) "
                         f"-> {resume_path}")
    if not sim_path.exists():
        raise SystemExit(f"거부: sim SEM 캐시가 없다 -> {sim_path}")
    if not real_path.exists():
        raise SystemExit(f"거부: real SEM 캐시가 없다 -> {real_path}")

    with resource_lock(GPU_LOCK):  # 여기부터 real 데이터·GPU — 모듈 docstring의 GPU 락
        return _train_locked(args, cfg, sim_path, real_path, out_dir, ckpt_path, resume_path)


def _train_locked(args, cfg, sim_path, real_path, out_dir, ckpt_path, resume_path) -> int:
    sim = np.load(sim_path, mmap_mode="r")
    real = np.load(real_path, mmap_mode="r")
    check_sem_array(sim, "sim_sem", SIM_TRAIN_N)
    check_sem_array(real, "real_sem", REAL_TRAIN_N)

    # ── 여기서부터만 torch를 불러온다 — 위 거부 경로는 전부 torch 없이 통과해야 한다 ──
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    _seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _SimDataset(Dataset):
        """sim SEM(도메인 A) — 이 로더의 한 바퀴가 "1 에폭"의 정의다(모듈 docstring)."""

        def __init__(self, arr):
            self.arr = arr

        def __len__(self):
            return len(self.arr)

        def __getitem__(self, i):
            x = to_signed(np.ascontiguousarray(self.arr[i]))[None]
            return torch.from_numpy(x)

    loader_a = DataLoader(_SimDataset(sim), batch_size=cfg.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)

    real_gen = torch.Generator().manual_seed(REAL_SAMPLE_SEED)  # 미사전등록 (모듈 docstring)

    def _sample_real_batch(bs: int) -> "torch.Tensor":
        """real(도메인 B, 60,664장)에서 복원추출로 `bs`장을 뽑아 (-1,1) 텐서로 만든다."""
        idx = torch.randint(0, REAL_TRAIN_N, (bs,), generator=real_gen).numpy()
        chunk = np.ascontiguousarray(real[idx])
        x = to_signed(chunk)[:, None, :, :]
        return torch.from_numpy(x)

    g_sim2real = build_generator(cfg).to(device)
    g_real2sim = build_generator(cfg).to(device)
    d_sim = build_discriminator(cfg).to(device)
    d_real = build_discriminator(cfg).to(device)

    opt_g = torch.optim.Adam(
        list(g_sim2real.parameters()) + list(g_real2sim.parameters()),
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(
        list(d_sim.parameters()) + list(d_real.parameters()),
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lambda ep: lr_multiplier(ep, cfg))
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lambda ep: lr_multiplier(ep, cfg))

    gan_loss = nn.MSELoss().to(device)  # LSGAN
    l1 = nn.L1Loss().to(device)
    scaler_g = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=args.amp)

    total_epochs = cfg.total_epochs
    start_ep = 0
    if args.resume:  # 존재는 train()이 락 전에 확인했다
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        g_sim2real.load_state_dict(ck["state_dict"]["G_sim2real"])
        g_real2sim.load_state_dict(ck["state_dict"]["G_real2sim"])
        d_sim.load_state_dict(ck["state_dict"]["D_sim"])
        d_real.load_state_dict(ck["state_dict"]["D_real"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        sched_g.load_state_dict(ck["sched_g"])
        sched_d.load_state_dict(ck["sched_d"])
        start_ep = ck["epoch"] + 1
        # real_gen/loader_a 셔플 순서는 재개점에 없다 — train_structure.py --resume과 같은
        # 이유(샘플러 RNG 비보존)로, 재현이 필요한 실행은 처음부터 돌릴 것.
        print(f"resume: {resume_path} → epoch {start_ep}부터", flush=True)

    for ep in range(start_ep, total_epochs):
        g_sim2real.train()
        g_real2sim.train()
        d_sim.train()
        d_real.train()
        g_losses, d_losses = [], []
        for real_a in loader_a:
            real_a = real_a.to(device)
            real_b = _sample_real_batch(real_a.shape[0]).to(device)

            opt_g.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake_b = g_sim2real(real_a)
                fake_a = g_real2sim(real_b)
                pred_fake_b = d_real(fake_b)
                pred_fake_a = d_sim(fake_a)
                loss_gan = (gan_loss(pred_fake_b, torch.ones_like(pred_fake_b))
                           + gan_loss(pred_fake_a, torch.ones_like(pred_fake_a)))

                rec_a = g_real2sim(fake_b)
                rec_b = g_sim2real(fake_a)
                loss_cycle = (l1(rec_a, real_a) + l1(rec_b, real_b)) * cfg.lambda_cycle

                idt_b = g_sim2real(real_b)
                idt_a = g_real2sim(real_a)
                loss_idt = (l1(idt_b, real_b) + l1(idt_a, real_a)) * cfg.lambda_identity

                loss_g = loss_gan + loss_cycle + loss_idt
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            opt_d.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp):
                pred_real_a = d_sim(real_a)
                pred_fake_a_det = d_sim(fake_a.detach())
                loss_d_sim = 0.5 * (
                    gan_loss(pred_real_a, torch.ones_like(pred_real_a))
                    + gan_loss(pred_fake_a_det, torch.zeros_like(pred_fake_a_det)))

                pred_real_b = d_real(real_b)
                pred_fake_b_det = d_real(fake_b.detach())
                loss_d_real = 0.5 * (
                    gan_loss(pred_real_b, torch.ones_like(pred_real_b))
                    + gan_loss(pred_fake_b_det, torch.zeros_like(pred_fake_b_det)))

                loss_d = loss_d_sim + loss_d_real
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

        sched_g.step()
        sched_d.step()
        print(f"epoch {ep + 1}/{total_epochs}: loss_G={np.mean(g_losses):.4f} "
              f"loss_D={np.mean(d_losses):.4f}", flush=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        resume_tmp = resume_path.with_name(resume_path.name + ".tmp")
        torch.save({
            "epoch": ep,
            "state_dict": {
                "G_sim2real": g_sim2real.state_dict(), "G_real2sim": g_real2sim.state_dict(),
                "D_sim": d_sim.state_dict(), "D_real": d_real.state_dict(),
            },
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
            "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
            "config": cfg.to_dict(),
        }, resume_tmp)
        os.replace(resume_tmp, resume_path)  # 매 에폭 원자적으로 덮는 별도 파일 — 찢긴 재개점 없음

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # 임시 파일에 끝까지 쓴 뒤 하드링크로 최종 이름을 **배타적으로** 붙인다 — os.link는 대상이
    # 있으면 FileExistsError(덮지 않음)이고, 저장 도중 크래시는 최종 이름에 찢긴 파일을 남기지 않는다
    final_tmp = ckpt_path.with_name(ckpt_path.name + ".tmp")
    torch.save({
        "config": cfg.to_dict(),
        "epoch": total_epochs,  # gate/translate가 "완주"를 확인하는 값 — 재개점의 0-idx ep와 다르다
        "state_dict": {
            "G_sim2real": g_sim2real.state_dict(), "G_real2sim": g_real2sim.state_dict(),
            "D_sim": d_sim.state_dict(), "D_real": d_real.state_dict(),
        },
        # 미사전등록 실행 옵션 — 판정엔 안 쓰지만 재현을 위해 기록한다(모듈 docstring)
        "runtime_options": {"amp": bool(args.amp), "num_workers": int(args.num_workers),
                            "real_sample_seed": REAL_SAMPLE_SEED, "resumed": bool(args.resume)},
    }, final_tmp)
    os.link(final_tmp, ckpt_path)
    final_tmp.unlink()

    print(json.dumps({
        "report_id": args.report_id,
        "x_domain": "sim+real_train_sem_unpaired", "y_source": "none",
        "config": cfg.to_dict(),
        "ckpt": str(ckpt_path), "ckpt_sha256": sha256_file(ckpt_path),
        "epochs": total_epochs,
    }, ensure_ascii=False))
    return 0


# ── gate ────────────────────────────────────────────────────

def gate(args) -> int:
    cache_dir = Path(args.cache_dir)
    ckpt_path = Path(args.ckpt)
    out_json = Path(args.out_json)
    sim_path = cache_dir / "sim_sem.npy"

    reject_test_paths([sim_path, ckpt_path, out_json])
    require_source_name(sim_path, "sim_sem.npy")
    check_ckpt_name(ckpt_path, args.report_id)  # torch 이전 — 이름만 본다

    if out_json.exists():
        raise SystemExit(f"거부: gate 출력이 이미 있다(덮어쓰지 않음) -> {out_json}")
    if not ckpt_path.exists():
        raise SystemExit(f"거부: 체크포인트가 없다 -> {ckpt_path}")
    if not sim_path.exists():
        raise SystemExit(f"거부: sim SEM 캐시가 없다 -> {sim_path}")

    with resource_lock(GPU_LOCK):  # 여기부터 real 데이터·GPU — 모듈 docstring의 GPU 락
        return _gate_locked(args, sim_path, ckpt_path, out_json)


def _gate_locked(args, sim_path, ckpt_path, out_json) -> int:
    sim = np.load(sim_path, mmap_mode="r")
    check_sem_array(sim, "sim_sem", SIM_TRAIN_N)

    idx = gate_sample_indices(SIM_TRAIN_N, GATE_SAMPLE_N, seed=42)
    orig = np.ascontiguousarray(sim[idx])

    # ── 여기서부터만 torch — 위 거부 경로는 전부 torch 없이 통과해야 한다 ──
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, ckpt_epoch, g_sim2real, g_real2sim = load_generators(
        ckpt_path, device)  # epoch/config 검사는 torch.load 이후

    translated = translate_u8(g_sim2real, orig, args.batch_size, device)
    roundtrip = translate_u8(g_real2sim, translated, args.batch_size, device)

    geo = measure_geometry(orig, translated)  # 전역 phase correlation + 국소 블록 매칭
    mae = roundtrip_mae(orig, roundtrip)
    gate_result = evaluate_gate(geo["shifts"], mae, local=geo["local"],
                                signed=geo["signed"])  # signed= → diagnostics에 mean_dy/dx

    payload = {
        **gate_result,
        "report_id": args.report_id,
        "ckpt": str(ckpt_path), "ckpt_sha256": sha256_file(ckpt_path), "ckpt_epoch": ckpt_epoch,
        "config": cfg.to_dict(),
        "shifts": [float(m) for m in geo["shifts"]],  # require_gate_passed가 재평가할 원본값
        "local_shifts": [float(m) for m in geo["local"]],  # 국소 기준 원본 — 없으면 통과 불가
        "signed_shifts": geo["signed"].tolist(),  # 진단용 원본 (dy,dx) — 판정에는 안 쓰인다
        "roundtrip_mae": mae,
        "indices_sha256": indices_sha256(idx),
        "sim_sem_sha256": sha256_file(sim_path),
    }
    print(json.dumps(payload, ensure_ascii=False))  # 쓰기 전에 — 쓰기가 실패해도 결과는 남는다
    write_once_json(out_json, payload)
    return 0 if gate_result["passed"] else 3


def main() -> int:
    ensure_utf8_console()  # argparse가 help를 찍기 전에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="설정 해석만 하고 아무것도 학습하지 않는다 (torch 불필요)")
    pl.add_argument("--report-id", required=True)
    pl.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    pl.add_argument("--config-json", default="",
                    help="사전등록을 덮어쓸 JSON (편차 거부 경로 확인용 — 정상 실행에선 안 씀)")

    tr = sub.add_parser("train", help="CycleGAN sim<->real 학습")
    tr.add_argument("--report-id", required=True)
    tr.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    tr.add_argument("--out-dir", default=DEFAULT_CKPT_DIR)
    tr.add_argument("--num-workers", type=int, default=0,
                    help="Windows는 0 권장 (미사전등록 기본값)")
    tr.add_argument("--amp", action="store_true", help="혼합정밀 (기본 꺼짐, 미사전등록)")
    tr.add_argument("--resume", action="store_true",
                    help="<out-dir>/<report-id>-cyclegan.resume.pt가 있으면 이어서 학습")
    tr.add_argument("--config-json", default="", help="plan과 동일 — 사전등록 편차 거부용")

    ga = sub.add_parser("gate", help="기하 위생 gate: 전역 phase-correlation shift + 국소 블록 "
                                     "매칭 shift + round-trip MAE")
    ga.add_argument("--report-id", required=True)
    ga.add_argument("--ckpt", required=True)
    ga.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ga.add_argument("--out-json", required=True)
    ga.add_argument("--batch-size", type=int, default=256, help="번역 배치 크기 (미사전등록)")

    args = ap.parse_args()
    try:
        if args.cmd == "plan":
            return plan(args)
        if args.cmd == "train":
            return train(args)
        if args.cmd == "gate":
            return gate(args)
    except GateFailedError as e:  # ckpt 이름·epoch·config 불일치 — translate_sim과 같은 코드 3
        print(f"거부: {e}", file=sys.stderr)
        return 3
    except ResourceBusy as e:  # 다른 실행이 gpu-0을 쥐고 있다 — 기다리거나 다른 자원으로 옮기지 않는다
        print(f"거부: {GPU_LOCK} 사용 중 — {e}", file=sys.stderr)
        return 4
    except ValueError as e:
        print(f"거부: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
