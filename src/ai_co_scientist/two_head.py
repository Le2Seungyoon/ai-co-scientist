"""H8 마스크·양의 깊이 2-head 순수 로직 — 타깃 분해·조립·손실·지표·매니페스트·게이트.

**이 모듈은 torch를 import하지 않는다.** `sem.py`와 같은 경계: 패키지 본 의존성은 numpy까지고
torch는 `two_head_loss`/`compose_output_torch` 두 함수 안에서만 지연 import한다 — 학습 루프와
모델 정의는 `scripts/train_two_head.py`/`scripts/infer_two_head.py`에 남는다.

`docs/experiment/H8-two-head-mask-depth.md`의 사전등록을 코드로 고정한다:
    s = (L - d) / L,  m = 1[s>0],  s_pos = s|m
    arm A(single): ŝ = sigmoid(logit), 전체 픽셀 L1
    arm B(two_head): s_pos_hat = sigmoid(depth_logit)  (arm A와 같은 출력 비선형)
                     BCEWithLogits(mask_logit, m) + L1(s_pos_hat, s | m=1), 1:1 합산
    추론: ŝ = sigmoid(mask_logit) * s_pos_hat  (soft product, hard threshold 없음)
사전등록 값이 드리프트하면 arm A/B 비교가 confound를 갖게 되므로, 매니페스트/체크포인트/제출
게이트가 이 드리프트를 실행 전에 잡는다.
"""
import hashlib
from pathlib import Path

import numpy as np

from ai_co_scientist.sem import CASE_LEVEL

# ── 상수 ────────────────────────────────────────────────────

ARMS = ("single", "two_head")
CKPT_FORMAT = "h8-two-head/v2"  # v1 = raw depth head + clamp — 키가 같아 조용히 로드되므로 거부

# 학습 사전등록 — arm A/B가 공유해야 하는 backbone·optimizer·split 하이퍼파라미터
PREREGISTERED = {"arch": "mlp", "batch_size": 128, "lr": 1e-3, "optimizer": "AdamW",
                 "schedule": "cosine", "epochs": 15, "seed": 42, "split_seed": 0,
                 "val_frac": 0.2}

# 추론 사전등록 — EXP-019 경로 (level 소스·AdaBN·tau·평활 등), 두 arm이 동일해야 한다
INFERENCE_PREREGISTERED = {
    "level_source": "cnn", "adabn": "real", "adabn_shuffle": 42, "adabn_stats": "batch",
    "tau": 0.0, "level_smooth": 9, "level_ckpt": "runtime/ckpt/EXP-013-level-cnn.pt",
    "histmatch": False, "adabn_drop_last": False, "level_hmm": False,
}

MASK_RATE_TOLERANCE = 0.10  # |pred_rate - sim_gt_rate| >= 0.10 -> 제출 중단 (사전등록: 10pp)
MASK_THRESHOLD = 0.5        # 붕괴 게이트/비율 진단 전용 — 출력 조립에는 쓰지 않는다 (soft product)

GPU_LOCK = "gpu-0"  # train_two_head.py·infer_two_head.py가 공유하는 자원 락 이름 (locks.resource_lock)

# 두 arm 체크포인트 매니페스트가 반드시 일치해야 하는 키 (arm 자체와 n_params·loss·output·
# ckpt_selection·git_dirty·resumed_from_epoch·sim_mask_gate는 의도적으로 제외 — arm마다
# 다르거나 재개 여부·에폭별 sim 홀드아웃 값으로 자연히 갈릴 수 있는 필드다)
PARITY_KEYS = ("format", "arch", "batch_size", "lr", "optimizer", "schedule", "epochs", "seed",
              "split_seed", "val_frac", "n_train", "n_val", "data_fingerprint", "git_commit",
              "sim_gt_pos_rate", "x_domain", "y_source", "target")

_DATA_FINGERPRINT_KEYS = ("n_sim", "sem_shape", "depth_shape", "case_counts", "case_sha256",
                          "val_idx_sha256", "depth_sample_sha256")

_SIM_MASK_GATE_KEYS = ("applies", "threshold", "tolerance", "split", "epoch", "pred_pos_rate",
                      "gt_pos_rate", "errors")

_MANIFEST_REQUIRED_KEYS = (set(PARITY_KEYS)
                          | {"arm", "n_params", "loss", "output", "ckpt_selection", "git_dirty",
                             "resumed_from_epoch", "sim_mask_gate"})


# ── 타깃 분해 / 출력 조립 (numpy) ─────────────────────────────

def split_target(s) -> tuple:
    """s → (m, s_pos). m = 1[s>0], s_pos = s (m=1인 곳만, 그 외 0). s는 [0,1] 유한값이어야 한다."""
    s = np.asarray(s, dtype=np.float32)
    if not np.all(np.isfinite(s)):
        raise ValueError("s에 NaN/Inf가 있다")
    if s.size and (float(s.min()) < 0.0 or float(s.max()) > 1.0):
        raise ValueError(f"s는 [0,1] 범위여야 한다 — min={s.min()}, max={s.max()}")
    m = (s > 0).astype(np.float32)
    s_pos = np.where(m > 0, s, np.float32(0.0)).astype(np.float32)
    return m, s_pos


def compose_output(mask_logit, s_pos_hat) -> np.ndarray:
    """ŝ = sigmoid(mask_logit) * s_pos_hat. s_pos_hat은 모델이 이미 sigmoid로 낸 값이라 clip은
    범위 가드일 뿐 no-op이다. 큰 |logit|에서도 overflow 경고 없이 수치적으로 안정적인 조각별
    sigmoid(음수 인자에만 exp를 건다)를 쓴다."""
    logit = np.asarray(mask_logit, dtype=np.float64)
    pos = np.asarray(s_pos_hat, dtype=np.float64)
    # np.where는 두 branch를 전부(배열 전체로) 미리 계산하므로 exp(-logit)를 선택 안 해도
    # logit=-1e4에서 exp(1e4) overflow 경고가 난다 — 마스크로 골라낸 부분에만 exp를 건다
    sig = np.empty_like(logit)
    nonneg = logit >= 0
    sig[nonneg] = 1.0 / (1.0 + np.exp(-logit[nonneg]))
    neg = ~nonneg
    exp_l = np.exp(logit[neg])
    sig[neg] = exp_l / (1.0 + exp_l)
    return (sig * np.clip(pos, 0.0, 1.0)).astype(np.float32)


def two_head_loss_reference(mask_logit, s_pos_hat, s) -> dict:
    """numpy 기준 손실 — torch 버전(two_head_loss)이 반드시 이 값과 같아야 한다.

    bce: 전체 픽셀 평균 BCEWithLogits(안정형 log-sum-exp).
    l1_pos: m=1 픽셀만, 모델 출력 s_pos_hat(=sigmoid(depth_logit))을 추가 clamp 없이 쓴다.
        배치 안 m=1 픽셀 전체의 평균이다(이미지별 평균 아님). n_pos==0이면 0.0.
    m=0 픽셀은 depth head에 gradient를 주지 않는다. 두 head 사이 stop-gradient는 없다 —
    두 손실 모두 공유 trunk로 역전파된다.
    total: bce + l1_pos (1:1).
    """
    logit = np.asarray(mask_logit, dtype=np.float64)
    pos_hat = np.asarray(s_pos_hat, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    m = (s > 0).astype(np.float64)
    bce_elem = np.maximum(logit, 0.0) - logit * m + np.log1p(np.exp(-np.abs(logit)))
    bce = float(bce_elem.mean())
    pos_mask = m > 0
    n_pos = int(pos_mask.sum())
    l1_pos = float(np.abs(pos_hat[pos_mask] - s[pos_mask]).mean()) if n_pos else 0.0
    return {"bce": bce, "l1_pos": l1_pos, "total": bce + l1_pos, "n_pos": n_pos}


def two_head_loss(mask_logit, s_pos_hat, s):
    """torch 버전. (total, bce, l1_pos) 텐서를 반환하며 값은 위 numpy 기준과 같아야 한다.

    n_pos==0이어도 l1_pos는 그래프를 유지한 채 0이어야 한다(옵티마이저가 s_pos_hat을 계속
    스텝할 수 있도록) — 그래서 `torch.zeros(...)`가 아니라 `s_pos_hat.sum() * 0`을 쓴다.
    """
    import torch.nn.functional as F

    m = (s > 0).to(s_pos_hat.dtype)
    bce = F.binary_cross_entropy_with_logits(mask_logit, m.to(mask_logit.dtype))
    pos_mask = m > 0
    n_pos = int(pos_mask.sum().item())
    if n_pos:
        l1_pos = (s_pos_hat[pos_mask] - s[pos_mask]).abs().mean()
    else:
        l1_pos = s_pos_hat.sum() * 0.0  # 그래프를 유지하는 0
    total = bce + l1_pos
    return total, bce, l1_pos


def compose_output_torch(mask_logit, s_pos_hat):
    """torch 버전 compose_output. 값은 numpy 버전과 같아야 한다."""
    import torch

    sig = torch.sigmoid(mask_logit)
    pos = torch.clamp(s_pos_hat, 0.0, 1.0)
    return (sig * pos).float()


# ── 지표 (numpy, streaming) ───────────────────────────────────

class BinnedAUROC:
    """히스토그램 기반 streaming AUROC. `bins`개 구간으로 점수를 양자화하고, 같은 구간 안의
    양성-음성 쌍은 동점(tie)으로 0.5씩 인정하는 Mann-Whitney U 공식을 쓴다."""

    def __init__(self, bins: int = 4096):
        self.bins = bins
        self.pos_counts = np.zeros(bins, dtype=np.int64)
        self.neg_counts = np.zeros(bins, dtype=np.int64)

    def update(self, scores, labels) -> None:
        scores = np.clip(np.asarray(scores, dtype=np.float64).reshape(-1), 0.0, 1.0)
        labels = np.asarray(labels).reshape(-1)
        idx = np.minimum((scores * self.bins).astype(np.int64), self.bins - 1)
        pos = labels == 1
        neg = labels == 0
        self.pos_counts += np.bincount(idx[pos], minlength=self.bins)
        self.neg_counts += np.bincount(idx[neg], minlength=self.bins)

    def value(self) -> float:
        n_pos = int(self.pos_counts.sum())
        n_neg = int(self.neg_counts.sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"AUROC: 한 클래스가 비어 있다 (n_pos={n_pos}, n_neg={n_neg})")
        cum_neg_before = np.cumsum(self.neg_counts) - self.neg_counts  # 구간 배타적 누적
        auc_sum = float(np.sum(self.pos_counts * (cum_neg_before + 0.5 * self.neg_counts)))
        return auc_sum / (n_pos * n_neg)


class StructureMetrics:
    """sim 홀드아웃 위생 지표 — **채택 지표가 아니다**, 로깅 전용.

    update(s_hat, s, levels, mask_prob=None). s_hat/s는 (B,H,W) 또는 (B,1,H,W).
    depth_rmse는 `train_structure.evaluate`의 tau=0 경로와 동일한 재구성
    (round → clip(0,255))을 쓴다.
    """

    def __init__(self, bins: int = 4096):
        self._se_s = 0.0
        self._se_d = 0.0
        self._se_pos = 0.0
        self._n_total = 0
        self._n_pos = 0
        self._gt_pos = 0
        self._auroc = BinnedAUROC(bins)
        self._mask_prob_seen = False
        self._pred_pos_count = 0
        self._pred_pos_total = 0

    @staticmethod
    def _squeeze(a: np.ndarray) -> np.ndarray:
        if a.ndim == 4 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim != 3:
            raise ValueError(f"(B,H,W) 또는 (B,1,H,W)여야 한다 — 받은 형태 {a.shape}")
        return a

    def update(self, s_hat, s, levels, mask_prob=None) -> None:
        s_hat = self._squeeze(np.asarray(s_hat, dtype=np.float64))
        s = self._squeeze(np.asarray(s, dtype=np.float64))
        levels = np.asarray(levels, dtype=np.float64)
        if levels.ndim != 1 or len(levels) != len(s):
            raise ValueError(f"levels는 (B,)이어야 한다 — s {len(s)}장, levels {levels.shape}")
        lv = levels.reshape(-1, 1, 1)
        m = (s > 0).astype(np.float64)

        self._se_s += float(((s_hat - s) ** 2).sum())
        self._n_total += s.size

        rec = np.clip(np.round(lv * (1.0 - s_hat)), 0, 255)
        true_d = lv * (1.0 - s)
        self._se_d += float(((rec - true_d) ** 2).sum())

        pos_mask = m > 0
        self._se_pos += float(((s_hat[pos_mask] - s[pos_mask]) ** 2).sum())
        self._n_pos += int(pos_mask.sum())

        if mask_prob is not None:
            mp = self._squeeze(np.asarray(mask_prob, dtype=np.float64))
            score = mp
            self._mask_prob_seen = True
            self._pred_pos_count += int((mp > MASK_THRESHOLD).sum())
            self._pred_pos_total += mp.size
        else:
            score = s_hat
        self._auroc.update(score.reshape(-1), m.reshape(-1))
        self._gt_pos += int(m.sum())

    def value(self) -> dict:
        return {
            "s_rmse": (self._se_s / self._n_total) ** 0.5,
            "depth_rmse": (self._se_d / self._n_total) ** 0.5,
            "pos_rmse": (self._se_pos / self._n_pos) ** 0.5 if self._n_pos else 0.0,
            "mask_auroc": self._auroc.value(),
            # 두 arm의 AUROC는 서로 비교 불가(single은 s_hat, two_head는 mask_prob이 자연스러운
            # 점수원) — 어느 쪽을 썼는지 명시한다
            "mask_auroc_score": "mask_prob" if self._mask_prob_seen else "s_hat",
            "pred_pos_rate": (self._pred_pos_count / self._pred_pos_total
                             if self._mask_prob_seen else None),
            "gt_pos_rate": self._gt_pos / self._n_total,
            "n_pixels": self._n_total,
        }


class RateCounter:
    """m(양성 마스크)의 스트리밍 양성률. sim train split epoch 1 위에서 `sim_gt_pos_rate`를
    구할 때 쓴다 (지금은 `sim_gt_pos_rate_from_depth`가 학습 전에 결정적으로 대체하지만,
    호환을 위해 남겨둔다)."""

    def __init__(self):
        self._pos = 0
        self._total = 0

    def update(self, m) -> None:
        m = np.asarray(m)
        self._pos += int((m > 0).sum())
        self._total += m.size

    def value(self) -> float:
        return self._pos / self._total if self._total else 0.0


def sim_gt_pos_rate_from_depth(depth, case, idx, chunk: int = 4096) -> float:
    """`idx`가 가리키는 이미지들에서 `depth < L(case)`(= s>0)인 픽셀 비율. 학습 시작 **전**에
    결정적으로 계산해 `sim_gt_pos_rate`를 고정한다 — epoch 1 RateCounter보다 먼저 도는 값이라
    학습 도중 배치 순서에 의존하지 않는다. `depth`/`case`는 배열 또는 memmap일 수 있다."""
    idx = np.asarray(idx)
    pos = 0
    total = 0
    for start in range(0, len(idx), chunk):
        batch_idx = idx[start:start + chunk]
        d = np.asarray(depth[batch_idx], dtype=np.float64)
        lv = np.array([CASE_LEVEL[int(c)] for c in case[batch_idx]], dtype=np.float64)
        lv = lv.reshape((-1,) + (1,) * (d.ndim - 1))
        pos += int((d < lv).sum())
        total += d.size
    return pos / total if total else 0.0


def sha256_array(a) -> str:
    """배열의 연속 바이트 + dtype/shape 프리픽스의 sha256 hex. 프리픽스가 없으면 같은 바이트가
    다른 shape/dtype에서 같은 해시가 되어 fingerprint가 구조 변화를 놓친다."""
    arr = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{arr.dtype}|{arr.shape}".encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def mask_diagnostics(mask_prob) -> dict:
    """로깅 전용 진단(게이트 아님): 평균 확률 · 예측 양성률 · '확신' 비율(<0.1 또는 >0.9)."""
    mp = np.asarray(mask_prob, dtype=np.float64)
    return {
        "mean_prob": float(mp.mean()),
        "pred_pos_rate": float((mp > MASK_THRESHOLD).mean()),
        "confident_frac": float(((mp < 0.1) | (mp > 0.9)).mean()),
    }


def sim_mask_gate(holdout: dict, arm: str, epoch: int) -> dict:
    """마스크 붕괴 중단 게이트 — **sim 홀드아웃(고정 split_seed/val_frac) 위에서만** 판정한다.

    real test GT는 감춰져 있어 real mask rate를 sim GT 비율과 비교할 근거가 없다(reviewer H1)
    — 그래서 이 게이트는 real이 아니라 마지막 에폭의 `StructureMetrics.value()`(sim 홀드아웃)를
    입력으로 받는다. `holdout["pred_pos_rate"]`/`holdout["gt_pos_rate"]`를
    `MASK_THRESHOLD`/`MASK_RATE_TOLERANCE`로 비교해 `errors`를 만든다. single arm은 아예
    적용되지 않는다(`applies=False`, `errors=[]`). two_head인데 `pred_pos_rate`가 None이면
    (StructureMetrics.update에 mask_prob을 안 넘겼다는 뜻) 그 자체가 에러다.
    """
    applies = arm == "two_head"
    pred = holdout.get("pred_pos_rate")
    gt = holdout.get("gt_pos_rate")
    if not applies:
        errors = []
    elif pred is None:
        errors = ["sim_mask_gate: two_head인데 pred_pos_rate가 None이다 "
                 "(StructureMetrics.update에 mask_prob을 전달했는지 확인할 것)"]
    else:
        errors = check_mask_rate(pred, gt)
    return {
        "applies": applies,
        "threshold": MASK_THRESHOLD,
        "tolerance": MASK_RATE_TOLERANCE,
        "split": "sim holdout, depth-map unit, split_seed/val_frac from manifest",
        "epoch": epoch,
        "pred_pos_rate": pred,
        "gt_pos_rate": gt,
        "errors": errors,
    }


# ── 매니페스트 ─────────────────────────────────────────────────

_LOSS_BY_ARM = {"single": "L1(all pixels)",
                "two_head": ("BCEWithLogits(mask_logit, m) "
                             "+ L1(sigmoid(depth_logit), s | m=1), 1:1")}
_OUTPUT_BY_ARM = {"single": "sigmoid(logit)",
                  "two_head": "sigmoid(mask_logit) * sigmoid(depth_logit)"}


def validate_hparams(hp: dict) -> list:
    """PREREGISTERED의 9개 키가 `hp`에 전부 있고 값이 정확히 같아야 한다 — 없으면 "누락",
    다르면 "불일치"로 키 이름을 대며 실패 이유를 남긴다. `validate_manifest`도 이 함수를
    재사용한다(사전등록 값 비교 로직을 두 벌 두지 않는다) — B는 학습 시작 전에 CLI 인자를
    조립한 hparams 딕셔너리를 이걸로 먼저 검사한다."""
    errs = []
    for k, v in PREREGISTERED.items():
        if k not in hp:
            errs.append(f"필수 hparams 키 누락: {k}")
        elif hp[k] != v:
            errs.append(f"사전등록 값 불일치: {k}={hp[k]!r} (기대 {v!r})")
    return errs


def build_manifest(*, arm: str, n_train: int, n_val: int, n_params: int, data_fingerprint: dict,
                   git_commit: str, git_dirty: bool, sim_gt_pos_rate: float,
                   sim_mask_gate: dict, resumed_from_epoch=None, **hparams) -> dict:
    """arm별 체크포인트 매니페스트를 만든다. **hparams는 PREREGISTERED의 9개 키를 모두
    담고 있어야 한다 — 하나라도 없으면 어떤 값이 빠졌는지 이름을 대며 실패한다.

    `sim_mask_gate`는 `sim_mask_gate()` 함수가 반환한 dict를 그대로 받는다(필수, 기본값 없음)
    — 마지막 에폭 sim 홀드아웃에서 계산한 마스크 붕괴 판정을 매니페스트에 실어, 제출 게이트가
    real GT 없이도(=신뢰할 수 없는 real 비교 없이도) 참조할 수 있게 한다.
    """
    if arm not in ARMS:
        raise ValueError(f"알 수 없는 arm: {arm!r} (허용: {ARMS})")
    missing = [k for k in PREREGISTERED if k not in hparams]
    if missing:
        raise ValueError(f"build_manifest: 필수 hparams 키 누락: {', '.join(missing)}")
    manifest = {
        "format": CKPT_FORMAT,
        "arm": arm,
        "x_domain": "sim",
        "y_source": "sim_depth_gt",
        "target": "s = (L - d) / L",
        "loss": _LOSS_BY_ARM[arm],
        "output": _OUTPUT_BY_ARM[arm],
        "ckpt_selection": "final epoch (no sim-holdout selection)",
        "n_train": n_train,
        "n_val": n_val,
        "n_params": n_params,
        "data_fingerprint": data_fingerprint,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "sim_gt_pos_rate": sim_gt_pos_rate,
        "sim_mask_gate": sim_mask_gate,
        "resumed_from_epoch": resumed_from_epoch,
    }
    for k in PREREGISTERED:
        manifest[k] = hparams[k]
    return manifest


def validate_manifest(m: dict) -> list:
    """errors (빈 리스트 = ok): format·arm·필수 키·사전등록 값 일치(validate_hparams 재사용)·
    도메인·양성률 범위·data_fingerprint 내용·sim_mask_gate 내용."""
    if not isinstance(m, dict):
        return ["manifest는 dict여야 한다"]
    errs = []
    if m.get("format") != CKPT_FORMAT:
        errs.append(f"format 불일치: {m.get('format')!r} (기대 {CKPT_FORMAT!r})")
    if m.get("arm") not in ARMS:
        errs.append(f"알 수 없는 arm: {m.get('arm')!r} (허용: {ARMS})")
    for k in sorted(_MANIFEST_REQUIRED_KEYS):
        if k not in m:
            errs.append(f"필수 키 누락: {k}")
    errs += validate_hparams(m)
    if "x_domain" in m and m["x_domain"] != "sim":
        errs.append(f"x_domain은 sim이어야 한다: {m['x_domain']!r}")
    if "y_source" in m and m["y_source"] != "sim_depth_gt":
        errs.append(f"y_source는 sim_depth_gt여야 한다: {m['y_source']!r}")
    if "sim_gt_pos_rate" in m:
        rate = m["sim_gt_pos_rate"]
        if not (isinstance(rate, (int, float)) and np.isfinite(rate) and 0.0 <= rate <= 1.0):
            errs.append(f"sim_gt_pos_rate는 [0,1] 범위의 유한값이어야 한다: {rate!r}")
    if "data_fingerprint" in m:
        fp = m["data_fingerprint"]
        if not isinstance(fp, dict):
            errs.append("data_fingerprint는 dict여야 한다")
        else:
            for k in _DATA_FINGERPRINT_KEYS:
                if k not in fp:
                    errs.append(f"data_fingerprint 필수 키 누락: {k}")
    if "sim_mask_gate" in m:
        smg = m["sim_mask_gate"]
        if not isinstance(smg, dict):
            errs.append("sim_mask_gate는 dict여야 한다")
        else:
            for k in _SIM_MASK_GATE_KEYS:
                if k not in smg:
                    errs.append(f"sim_mask_gate 필수 키 누락: {k}")
            if "applies" in smg and "arm" in m and smg["applies"] != (m["arm"] == "two_head"):
                errs.append(f"sim_mask_gate.applies가 arm과 불일치: {smg['applies']!r} "
                            f"(arm={m['arm']!r})")
    return errs


def validate_arm_pair(a: dict, b: dict) -> list:
    """errors: 둘 다 유효 · arm 쌍이 정확히 {single, two_head} · PARITY_KEYS 전부 일치
    (어긋난 키 이름을 명시) · git_dirty 둘 다 False · git_commit 둘 다 비어있지 않음."""
    errs = []
    errs += [f"a: {e}" for e in validate_manifest(a)]
    errs += [f"b: {e}" for e in validate_manifest(b)]
    arms = {a.get("arm"), b.get("arm")}
    if arms != set(ARMS):
        errs.append(f"arm 쌍이 {set(ARMS)}가 아니다: {arms}")
    for k in PARITY_KEYS:
        if a.get(k) != b.get(k):
            errs.append(f"parity 불일치: {k} a={a.get(k)!r} b={b.get(k)!r}")
    if a.get("git_dirty") or b.get("git_dirty"):
        errs.append("git_dirty: 작업 트리가 깨끗해야 한다")
    if not a.get("git_commit") or not b.get("git_commit"):
        errs.append("git_commit이 비어 있다")
    return errs


# ── 실행 전 점검 (git · 산출물 경로) ─────────────────────────

GIT_PREFLIGHT_PATHS = ("src", "scripts", "pyproject.toml", "uv.lock")


def git_preflight_errors(head, porcelain) -> list:
    """errors: HEAD를 읽지 못했거나(None/빈 문자열), `git status`가 실패했거나(None — 빈 출력을
    깨끗한 트리로 오인하지 않는다), `git status --porcelain`(untracked 포함) 출력이 비어 있지
    않으면 학습을 시작하지 않는다 — manifest의 git_commit이 실제로 돌아간 코드를 대표해야
    한다. untracked 파일도 잡는다: 커밋 안 된 새 모듈은 git_commit에 없는 코드다."""
    errs = []
    if not head:
        errs.append("git HEAD를 읽지 못했다 — git_commit 없이 학습하지 않는다")
    if porcelain is None:
        errs.append("git status가 실패했다 — 작업 트리가 깨끗한지 확인할 수 없다")
    elif porcelain.strip():
        errs.append(f"작업 트리가 깨끗하지 않다({'/'.join(GIT_PREFLIGHT_PATHS)} 대상): "
                    f"{porcelain.strip()}")
    return errs


def train_artifact_paths(out) -> dict:
    """train_two_head.py가 쓰는 산출물 경로 — 최종 ckpt · manifest 사이드카 · 에폭별 재개 파일."""
    out = Path(out)
    return {"ckpt": out, "manifest": out.with_suffix(".manifest.json"),
            "resume": out.with_suffix(".resume.pt")}


def train_output_errors(out, resume: bool) -> list:
    """결정적 덮어쓰기 정책 — errors가 비어야 학습을 시작한다.

    - 최종 ckpt나 manifest가 이미 있으면 **항상** 거부한다(`--resume`이어도). 끝난 arm의
      산출물을 다시 쓰면 제출된 zip이 어느 ckpt의 것인지 알 수 없게 된다.
    - 재개 파일이 있는데 `--resume`이 없으면 거부한다 — 조용히 처음부터 다시 돌며 덮어쓰지
      않는다. `--resume`인데 재개 파일이 없으면 처음부터 시작한다(에러 아님).
    """
    paths = train_artifact_paths(out)
    errs = [f"{k} 경로가 이미 존재한다(덮어쓰지 않는다): {paths[k]}"
            for k in ("ckpt", "manifest") if paths[k].exists()]
    if paths["resume"].exists() and not resume:
        errs.append(f"재개 파일이 이미 있다: {paths['resume']} — 이어서 돌리려면 --resume, "
                    "버리려면 직접 지운 뒤 다시 실행")
    return errs


RESUME_IDENTITY_KEYS = ("arm", "epochs", "lr", "seed", "split_seed", "val_frac", "batch_size",
                        "git_commit", "data_fingerprint")


def resume_identity_errors(saved: dict, current: dict) -> list:
    """errors: 재개 파일에 박힌 하이퍼파라미터·git_commit·data_fingerprint가 이번 실행과 하나라도
    다르면 거부한다. 다른 커밋이나 다시 만든 cache 위에서 이어 돌리면 ckpt가 두 코드/데이터의
    혼합물이 되는데, manifest는 마지막 커밋 하나만 적어 arm parity 게이트가 그걸 못 잡는다."""
    return [f"--resume 불일치: {k}: resume파일={saved.get(k)!r} != 이번 실행={current.get(k)!r}"
            for k in RESUME_IDENTITY_KEYS if saved.get(k) != current.get(k)]


def batch_errors(n_train: int, batch_size: int) -> list:
    """errors: 학습 로더는 마지막 부분 배치를 버리지 않으므로 n_train % batch_size == 1이면
    크기 1 배치가 BatchNorm1d(train 모드)에서 죽는다 — GPU 시간을 쓰기 전에 거부한다."""
    if n_train % batch_size == 1:
        return [f"n_train={n_train} % batch_size={batch_size} == 1 — 크기 1 배치가 "
                "BatchNorm1d에서 실패한다"]
    return []


# ── 체크포인트 ───────────────────────────────────────────────

def ckpt_payload(arm: str, manifest: dict, state_dict) -> dict:
    return {"format": CKPT_FORMAT, "arm": arm, "arch": manifest.get("arch"),
           "manifest": manifest, "state_dict": state_dict}


def resolve_ckpt(obj) -> tuple:
    """(arm, manifest)를 돌려준다. bare state_dict나 train_structure 스타일({"arch",
    "state_dict"}, format 키 없음) ckpt는 h8 체크포인트가 아니므로 infer_decomposed.py를
    쓰라고 안내하며 실패한다."""
    if not isinstance(obj, dict) or "format" not in obj:
        raise ValueError("ckpt에 format 키가 없다 — bare state_dict 또는 train_structure 형식으로 "
                         "보인다. h8 체크포인트가 아니면 infer_decomposed.py를 쓸 것")
    if obj["format"] != CKPT_FORMAT:
        raise ValueError(f"알 수 없는 ckpt format: {obj['format']!r} (기대 {CKPT_FORMAT!r})")
    if obj.get("arm") not in ARMS:
        raise ValueError(f"알 수 없는 arm: {obj.get('arm')!r} (허용: {ARMS})")
    manifest = obj.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("ckpt에 manifest가 없다")
    if manifest.get("arm") != obj["arm"]:
        raise ValueError(f"manifest arm {manifest.get('arm')!r} != ckpt arm {obj['arm']!r}")
    errs = validate_manifest(manifest)
    if errs:
        raise ValueError("; ".join(errs))
    return obj["arm"], manifest


# ── 제출 전 게이트 ────────────────────────────────────────────

def check_output(structure, expected_shape=None) -> list:
    """errors: ndim==3, dtype==float32, (있으면) shape==expected_shape, 유한값, [0,1] 범위."""
    s = np.asarray(structure)
    errs = []
    if s.ndim != 3:
        errs.append(f"structure는 (N,H,W)이어야 한다 — 받은 형태 {s.shape}")
    if s.dtype != np.float32:
        errs.append(f"structure dtype은 float32여야 한다 — 받은 dtype {s.dtype}")
    if expected_shape is not None and tuple(s.shape) != tuple(expected_shape):
        errs.append(f"structure shape 불일치: {s.shape} (기대 {tuple(expected_shape)})")
    if not np.all(np.isfinite(s)):
        errs.append("structure에 NaN/Inf가 있다")
        return errs  # 유한하지 않으면 min/max 비교가 의미 없다
    mn, mx = float(s.min()), float(s.max())
    if mn < 0:
        errs.append(f"structure min이 0 미만이다: {mn}")
    if mx > 1:
        errs.append(f"structure max가 1 초과이다: {mx}")
    return errs


def check_mask_rate(pred_rate, sim_gt_rate, tol: float = MASK_RATE_TOLERANCE) -> list:
    """errors: |pred_rate - sim_gt_rate| >= tol 이면 중단. NaN/None도 중단.

    사전등록된 게이트 판정 자체는 이 함수다 — 다만 이제 `submission_gate`가 이걸 real test
    mask_rate에 직접 걸지 않는다(real GT가 없어 비교 근거가 없다, reviewer H1). 대신
    `sim_mask_gate()`가 **sim 홀드아웃**의 pred/gt 비율에 이 함수를 걸어 매니페스트에 실어두고,
    `submission_gate`는 그 결과(`manifest["sim_mask_gate"]["errors"]`)를 읽기만 한다.
    """
    if (pred_rate is None or sim_gt_rate is None
            or not np.isfinite(pred_rate) or not np.isfinite(sim_gt_rate)):
        return [f"mask rate가 NaN/None이다: pred={pred_rate!r} sim_gt={sim_gt_rate!r}"]
    diff = abs(pred_rate - sim_gt_rate)
    if diff >= tol:
        return [f"mask 양성률 이탈: |{pred_rate:.4f}-{sim_gt_rate:.4f}|={diff:.4f} >= {tol}"]
    return []


def validate_inference_config(cfg: dict) -> list:
    """errors: INFERENCE_PREREGISTERED의 각 키가 cfg에 있고 값이 같아야 한다. `level_ckpt`는
    `Path(...).as_posix()`로 정규화해 비교한다(윈도 역슬래시/슬래시 표기 차이를 무시)."""
    errs = []
    for k, v in INFERENCE_PREREGISTERED.items():
        if k not in cfg:
            errs.append(f"추론 설정 키 누락: {k}")
            continue
        cv = cfg[k]
        if k == "level_ckpt":
            ok = isinstance(cv, str) and Path(cv).as_posix() == Path(v).as_posix()
        else:
            ok = cv == v
        if not ok:
            errs.append(f"추론 설정 불일치: {k}={cv!r} (기대 {v!r})")
    return errs


def submission_gate(structure, arm, mask_rate, manifest, peer_manifest, cfg,
                    expected_shape=None) -> list:
    """제출 직전 종합 게이트: arm 쌍 parity + 추론 설정 사전등록 + 출력 범위/dtype/shape +
    (two_head만) `manifest["sim_mask_gate"]["errors"]`. 하나라도 에러가 있으면 호출자는 zip을
    만들지 않는다.

    `mask_rate`(real test에서 잰 마스크 양성률)는 **게이트가 아니라 로그용 진단**이다 — real
    depth GT가 감춰져 있어 sim GT 비율과 비교할 근거가 없다(reviewer H1). 실제 마스크 붕괴
    판정은 학습 시점에 sim 홀드아웃에서 고정된 `sim_mask_gate()`가 이미 매니페스트에 실어뒀고,
    여기서는 그 결과를 그대로 읽어 에러 문자열 앞에 출처를 표시해 붙인다. single arm은 여전히
    `mask_rate=None`이어야 하고(2-head 전용 진단을 잘못 실어 나르지 않도록), two_head arm은
    `mask_rate`가 float이든 None이든 둘 다 허용한다(어느 쪽이든 게이트에 영향이 없다).
    """
    errs = []
    errs += validate_arm_pair(manifest, peer_manifest)
    errs += validate_inference_config(cfg)
    errs += check_output(structure, expected_shape=expected_shape)
    if arm == "two_head":
        sim_gate = manifest.get("sim_mask_gate") or {}
        errs += [f"sim_mask_gate: {e}" for e in sim_gate.get("errors", [])]
    elif arm == "single":
        if mask_rate is not None:
            errs.append("single arm은 mask_rate가 None이어야 한다")
    else:
        errs.append(f"알 수 없는 arm: {arm!r} (허용: {ARMS})")
    return errs


# ── 파라미터 수 ───────────────────────────────────────────────

def mlp_param_counts(hw: int = 72 * 48) -> dict:
    """PlainMLP(EXP-005) 파라미터 8,468,736개는 hw=72*48에 대해 실측한 상수다 — decoder의
    마지막 Linear(1024, hw)만 두 개로 늘어나는 게 two_head arm의 구조적 증가분이다.

    single 값 자체는 hw로부터 재계산되지 않는다(첫 encoder층도 hw에 의존하지만 그 변화는
    이 함수가 추적하지 않는다) — 기본 hw=72*48에서만 self-consistent하다.
    """
    single = 8_468_736
    extra = 1024 * hw + hw  # 두 번째 Linear(1024, hw) head 하나만큼
    return {
        "single": single,
        "two_head": single + extra,
        "extra": extra,
        "extra_frac_total": extra / single,
        "extra_frac_output_layer": 1.0,  # 출력층 자체는 정확히 두 배가 된다
    }
