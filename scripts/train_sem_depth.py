"""SEM -> Depth 개선 학습기 — baseline(MLP AE) 대비 Conv U-Net + 스윕 가능한 하이퍼파라미터.

Lightning Studio에서 단독 실행 가능해야 하므로 repo 패키지를 import하지 않는다 (standalone).
baseline_sem_depth.py와 동일한 데이터 규약(경로·짝맞춤·48x72·[0,1] 스케일·80/20 split)을
유지해 val_rmse가 직접 비교 가능하다. 최초 실행 시 PNG 35만 장을 uint8 npy 캐시로 굽고
(IO 병목 제거), 이후 실행은 캐시를 mmap으로 읽는다.

사용 예:
  python train_sem_depth.py --arch unet --width 32 --lr 1e-3 --loss l1 --epochs 30
"""

import argparse
import glob
import json
import os
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

H, W = 72, 48

# ── 전처리 (SEM 입력에만 적용; depth 타겟은 항상 /255) ────────────
# 도메인-중립 기법(standardize/clahe)은 train sim·val sim·test real에 동일 적용 → sim-val로 판정 가능.
# 도메인-갭 기법(histmatch: sim→real)은 sim에만 적용 → sim-val로 측정 불가, real 프록시/제출 필요.
# CLAHE는 지연 생성이라 워커(Windows spawn)에서 각자 만들어짐 → 전역 OK.
# 반면 histmatch 참조 CDF는 런타임 값이라 전역으로 두면 워커에 안 넘어감 →
# 반드시 ref_cdf 인자로 전달(데이터셋 속성으로 피클되게). (E2b 크래시 교훈)
_CLAHE = None


def apply_preproc(img_u8: np.ndarray, mode: str, ref_cdf: np.ndarray | None = None) -> np.ndarray:
    """uint8 (H,W) → float32 (H,W). 정규화까지 포함해 모델 입력으로 바로 쓸 수 있게 반환."""
    if mode == "none":
        return img_u8.astype(np.float32) / 255.0
    if mode == "standardize":
        f = img_u8.astype(np.float32)
        return (f - f.mean()) / (f.std() + 1e-6)
    if mode == "clahe":
        global _CLAHE
        if _CLAHE is None:
            _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return _CLAHE.apply(img_u8).astype(np.float32) / 255.0
    if mode == "histmatch":
        # 실측 참조 CDF로 히스토그램 매칭 (sim→real 도메인 갭 축소 가설)
        if ref_cdf is None:
            raise ValueError("histmatch에는 ref_cdf 필요 (데이터셋 속성으로 전달)")
        src = img_u8.ravel()
        hist = np.bincount(src, minlength=256).astype(np.float64)
        src_cdf = np.cumsum(hist) / src.size
        lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
        return lut[img_u8].astype(np.float32) / 255.0
    raise ValueError(f"unknown preproc: {mode}")


def build_histmatch_ref(cache_dir: Path) -> np.ndarray:
    """실측 test SEM의 평균 CDF — histmatch 참조(sim을 실측 밝기 분포로 맞춤)."""
    real = np.load(cache_dir / "test_sem.npy", mmap_mode="r")
    hist = np.bincount(np.asarray(real[::20]).ravel(), minlength=256).astype(np.float64)
    return np.cumsum(hist) / hist.sum()


def build_fda_ref(cache_dir: Path, n: int = 3000) -> np.ndarray:
    """FDA 참조 풀 — 실측 test SEM 일부(진폭 스펙트럼 공여자)."""
    real = np.load(cache_dir / "test_sem.npy", mmap_mode="r")
    idx = np.linspace(0, len(real) - 1, min(n, len(real))).astype(int)
    return np.ascontiguousarray(real[idx])


def fda_transform(src_u8: np.ndarray, ref_u8: np.ndarray, beta: float) -> np.ndarray:
    """Fourier Domain Adaptation: src의 저주파 진폭을 ref(real)로 교환, 위상(구조)은 보존.
    beta는 교환할 중심 대역 비율. FFT만 쓰는 학습-불필요 도메인 정합 (Yang et al. CVPR'20)."""
    src = src_u8.astype(np.float32)
    fs = np.fft.fft2(src)
    amp_s, pha_s = np.abs(fs), np.angle(fs)
    amp_r = np.abs(np.fft.fft2(ref_u8.astype(np.float32)))
    amp_s_sh = np.fft.fftshift(amp_s)
    amp_r_sh = np.fft.fftshift(amp_r)
    h, w = src.shape
    b = int(round(min(h, w) * beta))
    cy, cx = h // 2, w // 2
    amp_s_sh[cy - b:cy + b + 1, cx - b:cx + b + 1] = amp_r_sh[cy - b:cy + b + 1, cx - b:cx + b + 1]
    out = np.real(np.fft.ifft2(np.fft.ifftshift(amp_s_sh) * np.exp(1j * pha_s)))
    return np.clip(out, 0, 255).astype(np.uint8)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ── 데이터 캐시 ──────────────────────────────────────────────

def _load_png_stack(paths: list[str], desc: str) -> np.ndarray:
    arr = np.empty((len(paths), H, W), dtype=np.uint8)
    for i, p in enumerate(tqdm(paths, desc=desc)):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        arr[i] = img
    return arr


def build_cache(data_dir: Path, cache_dir: Path) -> None:
    """baseline과 동일한 정렬·짝맞춤(SEM itr0/itr1 x2 depth 복제) 후 npy로 저장."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sem = sorted(glob.glob(str(data_dir / "simulation_data" / "SEM" / "*" / "*" / "*.png")))
    dep = sorted(
        glob.glob(str(data_dir / "simulation_data" / "Depth" / "*" / "*" / "*.png")) * 2)
    assert len(sem) == len(dep), f"SEM/Depth 짝 불일치: {len(sem)} vs {len(dep)}"
    np.save(cache_dir / "sim_sem.npy", _load_png_stack(sem, "cache sim SEM"))
    np.save(cache_dir / "sim_depth.npy", _load_png_stack(dep, "cache sim Depth"))

    test = sorted(glob.glob(str(data_dir / "test" / "SEM" / "*.png")))
    np.save(cache_dir / "test_sem.npy", _load_png_stack(test, "cache test SEM"))
    (cache_dir / "test_names.json").write_text(
        json.dumps([os.path.basename(p) for p in test]), encoding="utf-8")
    print(f"cache built at {cache_dir}: sim={len(sem)}, test={len(test)}")


def build_sim_case_cache(data_dir: Path, cache_dir: Path) -> None:
    """sim_sem.npy와 동일 순서(sorted glob)로 case 라벨(1~4)을 파생 저장 — 이미지 로드 없이 경로만.

    case를 통째로 val로 격리하는 leave-one-case-out 검증용(train↔val 분포차 → sim→real 일반화 프록시).
    """
    sem = sorted(glob.glob(str(data_dir / "simulation_data" / "SEM" / "*" / "*" / "*.png")))
    case = np.array([int(Path(p).parts[-3].split("_")[1]) for p in sem], dtype=np.int8)
    np.save(cache_dir / "sim_case.npy", case)
    print(f"sim_case cache: {len(case)} imgs, cases {sorted(set(case.tolist()))}", flush=True)


def build_realproxy_cache(data_dir: Path, cache_dir: Path) -> None:
    """실측 train SEM + 사이트별 평균 depth를 캐시 (도메인 매칭 프록시용).

    경로 train/SEM/Depth_<d>/site_<n>/SEM_*.png → csv 키 'depth_<d>_site_<n>'로 평균 depth 조회.
    """
    import csv as _csv
    avg: dict[str, float] = {}
    with open(data_dir / "train" / "average_depth.csv", encoding="utf-8") as f:
        r = _csv.reader(f)
        next(r)
        for name, v in r:
            avg[name.strip().lower()] = float(v)
    paths = sorted(glob.glob(str(data_dir / "train" / "SEM" / "*" / "site_*" / "*.png")))
    site_idx = np.empty(len(paths), dtype=np.int32)
    key_to_idx: dict[str, int] = {}
    depths: list[float] = []
    for i, p in enumerate(paths):
        parts = Path(p).parts  # split('/') 금지 (Windows 역슬래시) — Path.parts 사용
        key = f"{parts[-3]}_{parts[-2]}".lower()  # depth_140_site_00233
        if key not in key_to_idx:
            if key not in avg:
                raise KeyError(f"average_depth.csv에 없는 사이트: {key}")
            key_to_idx[key] = len(depths)
            depths.append(avg[key])
        site_idx[i] = key_to_idx[key]
    np.save(cache_dir / "real_sem.npy", _load_png_stack(paths, "cache real train SEM"))
    np.save(cache_dir / "real_site_idx.npy", site_idx)
    np.save(cache_dir / "real_site_depth.npy", np.array(depths, dtype=np.float32))
    print(f"realproxy cache: {len(paths)} imgs, {len(depths)} sites", flush=True)


@torch.no_grad()
def eval_realproxy(model, cache_dir: Path, device, preproc: str, batch_size: int = 256) -> float:
    """실측 SEM으로 예측한 depth의 사이트별 평균 vs average_depth.csv → RMSE (도메인 매칭 지표).

    sim-val이 못 보는 sim↔real 도메인 갭에 민감. 절대 depth만 보고 공간 구조는 안 봄(약한 신호지만
    실측 도메인이라 도메인-갭 전처리(histmatch 등) 판정에 sim-val보다 신뢰도 높음).
    """
    sem = np.load(cache_dir / "real_sem.npy", mmap_mode="r")
    site_idx = np.load(cache_dir / "real_site_idx.npy")
    true_depth = np.load(cache_dir / "real_site_depth.npy")
    pp = "none" if preproc == "histmatch" else preproc  # histmatch는 실측에 적용 안 함
    sum_d = np.zeros(len(true_depth), dtype=np.float64)
    cnt = np.zeros(len(true_depth), dtype=np.float64)
    model.eval()
    for start in range(0, len(sem), batch_size):
        imgs = np.asarray(sem[start:start + batch_size])
        batch = torch.from_numpy(np.stack([apply_preproc(im, pp) for im in imgs])).unsqueeze(1)
        pred = (model(batch.to(device)) * 255.0).clamp(0, 255)
        per_img = pred.mean(dim=(1, 2, 3)).cpu().numpy()
        sites = site_idx[start:start + len(per_img)]
        np.add.at(sum_d, sites, per_img)
        np.add.at(cnt, sites, 1)
    pred_site = sum_d / np.maximum(cnt, 1)
    return float(np.sqrt(np.mean((pred_site - true_depth) ** 2)))


class AugmentDataset(Dataset):
    """uint8 스택 -> SEM은 preproc 적용, depth는 /255 + (옵션) SEM/Depth 동기 좌우반전."""

    def __init__(self, sem: np.ndarray, depth: np.ndarray, hflip: bool, preproc: str = "none",
                 histmatch_ref: np.ndarray | None = None, blur_aug: float = 0.0,
                 fda_ref: np.ndarray | None = None, fda_beta: float = 0.0):
        self.sem, self.depth, self.hflip, self.preproc = sem, depth, hflip, preproc
        self.histmatch_ref = histmatch_ref  # 데이터셋 속성 → 워커에 피클됨
        self.blur_aug = blur_aug  # >0이면 σ∈(0.1,blur_aug] 랜덤 블러(텍스처 강건성, 도메인 갭)
        self.fda_ref, self.fda_beta = fda_ref, fda_beta  # FDA: 실측 진폭 스펙트럼 공여 풀

    def __len__(self):
        return len(self.sem)

    def __getitem__(self, i):
        sem, depth = self.sem[i], self.depth[i]
        if self.hflip and random.random() < 0.5:
            sem, depth = sem[:, ::-1], depth[:, ::-1]
        sem = np.ascontiguousarray(sem)
        if self.fda_beta > 0 and self.fda_ref is not None:
            ref = self.fda_ref[random.randrange(len(self.fda_ref))]
            sem = fda_transform(sem, ref, self.fda_beta)
        if self.blur_aug > 0 and random.random() < 0.5:
            sem = cv2.GaussianBlur(sem, (0, 0), random.uniform(0.1, self.blur_aug))
        sem = apply_preproc(sem, self.preproc, self.histmatch_ref)
        return (torch.from_numpy(sem).unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0).float() / 255.0)


# ── 모델 ────────────────────────────────────────────────────

class BaselineMLP(nn.Module):
    """baseline_sem_depth.py의 BaseModel 동일 구조 (비교 대조군)."""

    def __init__(self):
        super().__init__()
        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]
        self.encoder = nn.Sequential(*block(H * W, 1024), *block(1024, 512),
                                     *block(512, 256), *block(256, 128))
        self.decoder = nn.Sequential(*block(128, 256), *block(256, 512),
                                     *block(512, 1024), nn.Linear(1024, H * W))

    def forward(self, x):
        b = x.shape[0]
        return self.decoder(self.encoder(x.view(b, -1))).view(b, 1, H, W)


def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNetSmall(nn.Module):
    """3단 U-Net — 72x48은 2회 다운샘플(18x12)까지 나누어떨어짐."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.enc3 = conv_block(1, w), conv_block(w, w * 2), conv_block(w * 2, w * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.dec2 = conv_block(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.dec1 = conv_block(w * 2, w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class SmpModel(nn.Module):
    """일반 segmentation_models_pytorch dense-prediction 모델.

    72×48을 인코더 요구 크기로 리사이즈 후 출력을 72×48로 원복. 1채널 SEM(in_channels=1,
    smp가 사전학습 1st-conv를 채널평균으로 초기화). arch: unet/segformer/dpt/fpn/deeplabv3plus/
    unetpp/manet. DPT는 ViT 고정 크기(224) 필요, 나머지는 96×64(32 배수).
    """

    def __init__(self, arch: str, encoder: str, in_size: tuple[int, int]):
        super().__init__()
        import segmentation_models_pytorch as smp
        archs = {"unet": smp.Unet, "segformer": smp.Segformer, "dpt": smp.DPT,
                 "fpn": smp.FPN, "deeplabv3plus": smp.DeepLabV3Plus,
                 "unetpp": smp.UnetPlusPlus, "manet": smp.MAnet}
        self.net = archs[arch](encoder_name=encoder, encoder_weights="imagenet",
                               in_channels=1, classes=1)
        self._in = in_size

    def forward(self, x):
        h, w = x.shape[-2:]
        x = nn.functional.interpolate(x, size=self._in, mode="bilinear", align_corners=False)
        x = self.net(x)
        return nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


def predict_tta(model, batch, tta: bool):
    """TTA: 원본 + hflip + vflip 예측을 평균 (되돌려 정렬). tta=False면 원본만."""
    pred = model(batch)
    if tta:
        pred = pred + torch.flip(model(torch.flip(batch, [-1])), [-1])
        pred = pred + torch.flip(model(torch.flip(batch, [-2])), [-2])
        pred = pred / 3.0
    return pred


def make_model(arch: str, width: int) -> nn.Module:
    if arch == "mlp":
        return BaselineMLP()
    if arch == "unet":
        return UNetSmall(width)
    if arch.startswith("pretrained:"):  # 하위호환: pretrained:resnet18 → smp Unet
        return SmpModel("unet", arch.split(":", 1)[1], (96, 64))
    if arch.startswith("smp:"):  # smp:<arch>:<encoder> (예: smp:segformer:mit_b0)
        _, a, enc = arch.split(":", 2)
        return SmpModel(a, enc, (224, 224) if a == "dpt" else (96, 64))
    raise ValueError(arch)


# ── 학습/평가 ────────────────────────────────────────────────

@torch.no_grad()
def validate(model, criterion, loader, device):
    model.eval()
    mse = nn.MSELoss().to(device)
    losses, rmses = [], []
    for sem, depth in loader:
        sem, depth = sem.to(device), depth.to(device)
        pred = model(sem)
        losses.append(criterion(pred, depth).item())
        p255 = (pred * 255.0).round().clamp(0, 255)
        t255 = (depth * 255.0).round().clamp(0, 255)
        rmses.append(torch.sqrt(mse(p255, t255)).item())
    return float(np.mean(losses)), float(np.mean(rmses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--arch", default="unet",
                    help="mlp | unet | pretrained:<encoder> (예: pretrained:resnet18)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["l1", "l2", "huber"], default="l1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=0, help="선형 warmup (트랜스포머 안정화)")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="grad norm 클리핑 (0=off, 트랜스포머 권장 1.0)")
    ap.add_argument("--no-amp", action="store_true", help="AMP 끄기 (트랜스포머 fp16 발산 회피)")
    ap.add_argument("--blur-aug", type=float, default=0.0, help="랜덤 블러 최대 σ (텍스처 강건성, 0=off)")
    ap.add_argument("--fda-beta", type=float, default=0.0, help="FDA 진폭교환 대역 비율 (0=off, 예: 0.01/0.1)")
    ap.add_argument("--clamp-lo", type=float, default=0.0, help="추론 예측 clamp 하한 (EDA: 25)")
    ap.add_argument("--clamp-hi", type=float, default=255.0, help="추론 예측 clamp 상한 (EDA: 170)")
    ap.add_argument("--tta", action="store_true", help="추론 시 hflip/vflip TTA 평균")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-subsample", type=int, default=0,
                    help="학습 표본 상한 (0=전체). 저비용 스크리닝용 데이터 축소")
    ap.add_argument("--val-case", type=int, default=0,
                    help="N(1~4): Case_N을 통째로 val로 격리(leave-one-case-out). 0=기존 랜덤분할")
    ap.add_argument("--val-subsample", type=int, default=0,
                    help="검증 표본 상한 (0=전체). 스크리닝 시 검증도 빠르게")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--preproc", choices=["none", "standardize", "clahe", "histmatch"],
                    default="none", help="SEM 입력 전처리 (E2 스크리닝)")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--real-proxy", action="store_true",
                    help="학습 후 실측 avg-depth 프록시 RMSE 계산·로깅 (도메인-갭 전처리 판정용)")
    ap.add_argument("--build-cache-only", action="store_true")
    ap.add_argument("--skip-inference", action="store_true")
    args = ap.parse_args()

    data_dir, cache_dir = Path(args.data_dir), Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (cache_dir / "sim_sem.npy").exists():
        build_cache(data_dir, cache_dir)
    if args.real_proxy and not (cache_dir / "real_sem.npy").exists():
        build_realproxy_cache(data_dir, cache_dir)
    if args.build_cache_only:
        return

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    sem = np.load(cache_dir / "sim_sem.npy", mmap_mode="r")
    depth = np.load(cache_dir / "sim_depth.npy", mmap_mode="r")
    n_train = int(len(sem) * 0.8)

    def subsample(lo, hi, cap):
        """[lo,hi) 구간에서 최대 cap개를 고정 시드로 무작위 추출(정렬해 mmap 효율 유지). cap=0이면 전체."""
        if cap <= 0 or cap >= (hi - lo):
            return slice(lo, hi)
        rng = np.random.default_rng(args.seed)
        return np.sort(rng.choice(np.arange(lo, hi), size=cap, replace=False))

    def subsample_pool(pool, cap):
        """임의 인덱스 배열 pool에서 최대 cap개 무작위 추출(정렬)."""
        if cap <= 0 or cap >= len(pool):
            return np.sort(pool)
        rng = np.random.default_rng(args.seed)
        return np.sort(rng.choice(pool, size=cap, replace=False))

    hm_ref = build_histmatch_ref(cache_dir) if args.preproc == "histmatch" else None
    fda_ref = build_fda_ref(cache_dir) if args.fda_beta > 0 else None

    if args.val_case > 0:  # leave-one-case-out: Case_N 통째 격리 (train↔val 분포차 = 일반화 프록시)
        if not (cache_dir / "sim_case.npy").exists():
            build_sim_case_cache(data_dir, cache_dir)
        case = np.load(cache_dir / "sim_case.npy")
        tr_idx = subsample_pool(np.where(case != args.val_case)[0], args.train_subsample)
        va_idx = subsample_pool(np.where(case == args.val_case)[0], args.val_subsample)
        print(f"[val-case] Case_{args.val_case} 격리: train {len(tr_idx)} / val {len(va_idx)}", flush=True)
    else:
        tr_idx = subsample(0, n_train, args.train_subsample)
        va_idx = subsample(n_train, len(sem), args.val_subsample)
    # mmap fancy-index는 메모리로 로드됨 — 스크리닝의 작은 cap에서만 쓰고, 전체(slice)는 mmap 유지
    train_ds = AugmentDataset(sem[tr_idx], depth[tr_idx], hflip=args.hflip,
                              preproc=args.preproc, histmatch_ref=hm_ref, blur_aug=args.blur_aug,
                              fda_ref=fda_ref, fda_beta=args.fda_beta)
    # val은 clean sim (FDA/블러 미적용) — sim-val을 무-FDA 챔피언과 공정 비교하기 위함.
    # FDA를 '증강'으로 보고 clean sim-val을 낮추는 β를 찾는 스크리닝 (sim-val↔리더보드 상관 활용).
    val_ds = AugmentDataset(sem[va_idx], depth[va_idx], hflip=False,
                            preproc=args.preproc, histmatch_ref=hm_ref)
    print(f"train={len(train_ds)} val={len(val_ds)} "
          f"(subsample tr={args.train_subsample or 'full'} va={args.val_subsample or 'full'})", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    run_name = args.run_name or (
        f"{args.arch}-w{args.width}-lr{args.lr:g}-{args.loss}"
        f"{'-hflip' if args.hflip else ''}-e{args.epochs}")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ai-co-scientist"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_name,
        config={"arch": args.arch, "width": args.width, "lr": args.lr, "loss": args.loss,
                "epochs": args.epochs, "batch_size": args.batch_size, "hflip": args.hflip,
                "seed": args.seed, "train_subsample": args.train_subsample,
                "val_subsample": args.val_subsample, "preproc": args.preproc,
                "blur_aug": args.blur_aug, "warmup_epochs": args.warmup_epochs,
                "grad_clip": args.grad_clip, "fda_beta": args.fda_beta},
        mode="online" if os.environ.get("WANDB_API_KEY") else "disabled",
    )

    model = make_model(args.arch, args.width).to(device)
    criterion = {"l1": nn.L1Loss(), "l2": nn.MSELoss(),
                 "huber": nn.HuberLoss(delta=0.1)}[args.loss].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cosine_t = max(1, args.epochs - args.warmup_epochs)
    if args.warmup_epochs > 0:  # 선형 warmup → cosine (트랜스포머 발산 방지)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                               total_iters=args.warmup_epochs),
             torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t)],
            milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_rmse, best_path = float("inf"), output_dir / f"{run_name}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for sem_b, depth_b in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            sem_b = sem_b.to(device, non_blocking=True)
            depth_b = depth_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(sem_b), depth_b)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        val_loss, val_rmse = validate(model, criterion, val_loader, device)
        train_loss = float(np.mean(losses))
        print(f"epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
              f"val_rmse={val_rmse:.5f}", flush=True)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_rmse": val_rmse,
                   "lr": scheduler.get_last_lr()[0]}, step=epoch)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_path)

    wandb.summary["best_val_rmse"] = best_rmse
    print(f"best val_rmse: {best_rmse:.5f}")

    proxy_rmse = None
    if args.real_proxy:
        model.load_state_dict(torch.load(best_path, map_location=device))
        proxy_rmse = eval_realproxy(model, cache_dir, device, args.preproc, args.batch_size)
        wandb.summary["real_proxy_rmse"] = proxy_rmse
        print(f"real_proxy_rmse: {proxy_rmse:.5f}", flush=True)

    if not args.skip_inference:
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        test_sem = np.load(cache_dir / "test_sem.npy", mmap_mode="r")
        names = json.loads((cache_dir / "test_names.json").read_text(encoding="utf-8"))
        test_loader = DataLoader(
            TensorDataset(torch.arange(len(test_sem))), batch_size=args.batch_size)
        # histmatch는 sim→real 변환이므로 실측 test엔 적용 안 함(이미 real 도메인). 나머지는 동일 적용.
        test_preproc = "none" if args.preproc == "histmatch" else args.preproc
        sub_dir = output_dir / f"submission_{run_name}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / f"submission_{run_name}.zip"
        with torch.no_grad(), zipfile.ZipFile(zip_path, "w") as zf:
            for (idx_b,) in tqdm(test_loader, desc="infer"):
                imgs = np.asarray(test_sem[idx_b.numpy()])
                batch = torch.from_numpy(np.stack(
                    [apply_preproc(im, test_preproc) for im in imgs])).unsqueeze(1).to(device)
                pred = predict_tta(model, batch, args.tta) * 255.0
                pred = pred.round().clamp(args.clamp_lo, args.clamp_hi).cpu().numpy().astype(np.uint8)
                for j, i in enumerate(idx_b.tolist()):
                    img_path = sub_dir / names[i]
                    cv2.imwrite(str(img_path), pred[j, 0])
                    zf.write(img_path, arcname=names[i])
        print(f"submission written to {zip_path}", flush=True)

    wandb.finish()
    print(json.dumps({"best_val_rmse": best_rmse, "real_proxy_rmse": proxy_rmse,
                      "run_name": run_name}))


if __name__ == "__main__":
    main()
