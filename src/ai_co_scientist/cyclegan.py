"""H6 — CycleGAN sim→real 외관 변환의 순수 로직.

**이 모듈은 import 시점에 numpy까지만 쓴다.** torch는 `build_generator`/`build_discriminator`
안에서만, 실제로 호출될 때 지연 import한다 — 이 워크트리(dev 그룹만 sync)에는 torch/cv2가 없고,
경로 검증·gate 판정·manifest 같은 순수 로직은 GPU 없이도 검증 가능해야 하기 때문이다
(`ai_co_scientist.sem`과 같은 경계 설계).

여기 있는 것: 설정(사전등록 값과의 정확한 일치 검사) · 경로 누수 방지(`test_*` 실제 테스트셋이
변환기 입력으로 새는 것을 막는다) · 형태 계약 · 기하 위생 gate(순수 numpy — 전역 phase
correlation 판정 + 진단 전용 국소 NCC probe) · manifest의 불변성(write-once + sha256·gate 재검증) · GPU 락
이름 · torch 아키텍처 팩토리(지연 import).
"""
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

from ai_co_scientist.sem import map_level_split

# ── 상수 ──────────────────────────────────────────────────────

IMG_H, IMG_W = 72, 48
SIM_TOTAL_N = 173_304
SIM_TRAIN_N = 138_648
SIM_VAL_N = 34_656
SIM_SPLIT_VAL_FRAC = 0.2
SIM_SPLIT_SEED = 42
REAL_TRAIN_N = 60_664
GATE_SAMPLE_N = 2048
SHIFT_MEDIAN_MAX = 0.5
SHIFT_P95_MAX = 1.0
ROUNDTRIP_MAE_MAX = 0.10

# 정지 규칙 = 사전등록 3기준뿐이다(`docs/experiment/H6-cyclegan-sim-to-real.md` §조건). gate 파일에
# 기록·대조되는 판정 설정 전부이고, 여기 없는 값은 `passed`에 관여하지 않는다.
GATE_THRESHOLDS = {
    "shift_median_max": SHIFT_MEDIAN_MAX,
    "shift_p95_max": SHIFT_P95_MAX,
    "roundtrip_mae_max": ROUNDTRIP_MAE_MAX,
}

# 국소 기하 probe (블록 매칭) — **진단 전용이며 미보정(uncalibrated)이다. 판정에 관여하지 않는다.**
# 전역 phase correlation은 대칭 팽창·국소 왜곡을 못 보고 비순환 subpixel 이동을 과소평가한다(테스트가
# 고정). 그 공백을 사람이 볼 수 있게 기록하지만, 합성 입력에서 알려진 한계 두 가지 때문에 정지
# 규칙에 넣을 수 없다: (1) 거짓 플래그 — 경사진 구멍 가장자리에 블러+감마(비선형 밝기)를 주면
# 등밝기 윤곽이 실제로 ~1 px 움직여, 강도 기반 매칭으로는 기하 이동과 원리적으로 구별되지 않는다.
# (2) 사각지대 — 한 타일 안에 중심이 있는 대칭 팽창은 타일 변위가 0이다. 실제 sim 2,048장에서
# 외관 전용 변환으로 보정하고 사전등록을 개정하기 전까지는 플래그만 남긴다.
LOCAL_TILE = 24  # 72x48 → 3x2 타일
LOCAL_SEARCH = 3  # 블록 매칭 탐색 반경(px). 이보다 큰 변위는 경계에 포화된다
LOCAL_MIN_STD = 2.0  # 원본 타일 std가 이보다 작으면(평탄) 변위가 정의되지 않아 NaN으로 뺀다
LOCAL_PROBE_ROLE = "diagnostic_only_uncalibrated"
# probe 기록에 함께 남는 측정 방법 — 방법이 바뀌면 옛 gate의 probe 요약이 재현되지 않아 거부된다.
# 플래그 문턱은 새로 만들지 않고 사전등록 px 문턱을 타일 단위로 재사용한다(보정된 값이 아니다).
LOCAL_PROBE_METHOD = {
    "tile": LOCAL_TILE,
    "search": LOCAL_SEARCH,
    "min_std": LOCAL_MIN_STD,
    "flag_median_over": SHIFT_MEDIAN_MAX,
    "flag_p95_over": SHIFT_P95_MAX,
}

GPU_LOCK = "gpu-0"  # 학습·gate·번역의 runtime 구간이 잡는 `locks.resource_lock` 이름 — 기계 단위 1장

ALLOWED_SOURCES = ("sim_sem.npy", "real_sem.npy")  # 변환기 입력으로 허용되는 파일명
FORBIDDEN_NAMES = ("test_sem.npy", "test_names.json")  # real test — 변환기 학습에 넣으면 안 된다


# ── 설정 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class CycleGANConfig:
    """H6 사전등록 하이퍼파라미터. 필드 하나하나가 `docs/experiment/H6-cyclegan-sim-to-real.md`의
    조건절에서 왔다 — 여기서 기본값을 바꾸는 것은 곧 사전등록을 바꾸는 것이라 `validate_config`가
    드리프트를 잡는다."""

    seed: int = 42
    batch_size: int = 8
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    epochs_fixed: int = 50
    epochs_decay: int = 50
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    n_res_blocks: int = 6
    ngf: int = 64
    ndf: int = 64
    n_disc_layers: int = 3
    in_channels: int = 1
    norm: str = "instance"
    gan_loss: str = "lsgan"
    gate_sample_n: int = 2048
    gate_seed: int = 42

    def to_dict(self) -> dict:
        """JSON 직렬화 가능한 평면 dict. 필드가 전부 int/float/str이라 `dataclasses.asdict`로
        충분하다 — 타입이 그대로 보존되어야 `validate_config`의 타입 엄격 비교가 의미를 가진다."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "CycleGANConfig":
        """알 수 없는 키나 누락된 키가 있으면 그 이름을 나열하며 거부한다 — 오타 하나가
        조용히 기본값으로 채워지는 것(대신 무엇을 기대했는지 말도 없이)을 막는다."""
        expected = {f.name for f in dc_fields(cls)}
        given = set(d)
        missing = expected - given
        unknown = given - expected
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"누락된 키: {sorted(missing)}")
            if unknown:
                parts.append(f"알 수 없는 키: {sorted(unknown)}")
            raise ValueError("; ".join(parts))
        return cls(**d)

    @property
    def total_epochs(self) -> int:
        return self.epochs_fixed + self.epochs_decay


PREREGISTERED: dict = CycleGANConfig().to_dict()


def validate_config(cfg: CycleGANConfig) -> None:
    """`cfg`가 `PREREGISTERED`와 정확히 같은지 검사한다 — 다른 필드를 **전부** 나열한다.

    타입 엄격: `8`(int)과 `8.0`(float)은 값이 같아도 다른 것으로 본다. `bool`은 `int`의
    서브클래스라 `type(x) is type(y)`로만 구분된다(`isinstance`는 `True == 1`을 통과시킨다).
    """
    got = cfg.to_dict()
    diffs = []
    for k, expected in PREREGISTERED.items():
        if k not in got:
            diffs.append(f"{k}: 필드 누락")
            continue
        v = got[k]
        if type(v) is not type(expected) or v != expected:
            diffs.append(f"{k}: 사전등록값 {expected!r}, 받은 값 {v!r}")
    extra = sorted(set(got) - set(PREREGISTERED))
    for k in extra:
        diffs.append(f"{k}: 사전등록에 없는 필드 (받은 값 {got[k]!r})")
    if diffs:
        raise ValueError("설정이 사전등록과 다르다 — " + "; ".join(diffs))


def lr_multiplier(epoch: int, cfg: CycleGANConfig) -> float:
    """`epoch`(0-indexed)에 곱할 learning rate 배율.

    `epochs_fixed` 동안은 1.0을 유지하고, 이후 `epochs_decay` 동안 선형으로 0을 향해 감쇠한다.
    `epoch=epochs_fixed`(감쇠 첫 epoch)에서 이미 1/(epochs_decay+1)만큼 깎이므로 마지막
    epoch(`total_epochs-1`)이 정확히 0이 되지 않고 `1/(epochs_decay+1)`에서 멈춘다 — CycleGAN
    원 논문 스케줄과 같은 관례다.
    """
    total = cfg.total_epochs
    if epoch < 0 or epoch >= total:
        raise ValueError(f"epoch은 [0, {total})이어야 한다 — 받은 값 {epoch}")
    if epoch < cfg.epochs_fixed:
        return 1.0
    return 1.0 - (epoch - cfg.epochs_fixed + 1) / (cfg.epochs_decay + 1)


# ── 경로 / 누수 방지 ────────────────────────────────────────────

def reject_test_paths(paths) -> None:
    """real test 파일이 변환기 입력·출력 경로 어디에도 섞이지 않았는지 확인한다.

    거부 조건 세 가지: (1) 파일명이 `FORBIDDEN_NAMES`에 있음, (2) 파일명이 `test_`로 시작하고
    `.npy`/`.json`로 끝남, (3) 경로의 어느 디렉터리 성분이든 대소문자 무시하고 정확히 `test`.
    `pathlib`의 `parts`만 쓴다 — `str.split('/')`은 Windows 백슬래시 경로에서 깨진다
    (`.agents/rules/coding-patterns.md`가 이미 한 번 잡은 결함).

    (3)은 **디렉터리 성분만** 본다 — `test_foo0` 같은 pytest `tmp_path` 디렉터리는 문자열이
    "test"가 아니라 걸리지 않는다.

    세 조건 모두 **대소문자를 무시**하고(Windows에서 `TEST_SEM.npy`는 같은 파일이다), 주어진
    경로와 `resolve()`한 경로 **둘 다**에 적용한다 — symlink·`..`로 test 파일을 가리키는 우회를 막는다.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in paths:
        for path in (Path(p), Path(p).resolve()):
            name = path.name.lower()
            if name in FORBIDDEN_NAMES:
                raise ValueError(f"금지된 파일이 입력/출력 경로에 있다: {p}")
            if name.startswith("test_") and (name.endswith(".npy") or name.endswith(".json")):
                raise ValueError(f"'test_' 접두 파일은 변환기 경로에 쓸 수 없다: {p}")
            for part in path.parts[:-1]:
                if part.lower() == "test":
                    raise ValueError(f"경로에 'test' 디렉터리 성분이 있다: {p}")


def require_source_name(path, expected: str) -> None:
    """`path`의 파일명이 정확히 `expected`인지 확인한다 — 도메인 A/B가 뒤바뀌는 것을 막는다."""
    if Path(path).name != expected:
        raise ValueError(f"파일명이 {expected!r}이어야 한다 — 받은 경로 {path}")


# ── 형태 계약 ────────────────────────────────────────────────

def check_sem_array(arr, name: str, expected_n: "int | None" = None) -> None:
    """SEM 캐시 배열의 형태 계약: `(N, 72, 48)` uint8, 필요하면 장수까지 확인."""
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"{name}: 3차원 배열이어야 한다 — 받은 shape {a.shape}")
    if a.shape[1:] != (IMG_H, IMG_W):
        raise ValueError(f"{name}: shape[1:]가 ({IMG_H}, {IMG_W})여야 한다 — 받은 shape {a.shape}")
    if a.dtype != np.uint8:
        raise ValueError(f"{name}: dtype이 uint8이어야 한다 — 받은 dtype {a.dtype}")
    if expected_n is not None and len(a) != expected_n:
        raise ValueError(f"{name}: 길이가 {expected_n}이어야 한다 — 받은 길이 {len(a)}")


def check_batch_shape(shape: tuple, channels: int = 1) -> None:
    """torch 배치 텐서의 형태 계약: `(B, channels, 72, 48)`, `B>=1`. torch 없이도 검사 가능하도록
    `shape`(tuple)만 받는다 — 텐서 자체를 요구하지 않는다."""
    if len(shape) != 4:
        raise ValueError(f"배치 shape은 4차원이어야 한다 — 받은 {shape}")
    b, c, h, w = shape
    if b < 1:
        raise ValueError(f"배치 크기는 1 이상이어야 한다 — 받은 {b}")
    if (c, h, w) != (channels, IMG_H, IMG_W):
        raise ValueError(f"shape이 (B, {channels}, {IMG_H}, {IMG_W})여야 한다 — 받은 {shape}")


def to_unit(u8) -> np.ndarray:
    """uint8 → float32, `[0, 1]` (`/255`)."""
    return (np.asarray(u8).astype(np.float64) / 255.0).astype(np.float32)


def to_signed(u8) -> np.ndarray:
    """uint8 → float32, `[-1, 1]` — GAN generator/discriminator의 입출력 범위(`tanh`)."""
    unit = np.asarray(u8).astype(np.float64) / 255.0
    return (unit * 2.0 - 1.0).astype(np.float32)


def signed_to_u8(x) -> np.ndarray:
    """`[-1, 1]` → uint8. 범위 밖 값은 클램프하고, `np.rint`(반올림 짝수 규칙)로 정수화한다.

    `to_signed`의 정확한 역함수라서 `signed_to_u8(to_signed(u8))`는 부동소수 잡음(1e-7 수준)이
    반올림 경계를 넘지 않는 한 원본을 픽셀 단위로 정확히 복원한다.
    """
    xf = np.clip(np.asarray(x).astype(np.float64), -1.0, 1.0)
    return np.rint((xf + 1.0) / 2.0 * 255.0).astype(np.uint8)


# ── 기하 위생 gate (순수 numpy) ──────────────────────────────

def gate_sample_indices(n_total: int, n_sample: int = GATE_SAMPLE_N, seed: int = 42) -> np.ndarray:
    """gate에 쓸 고정 sim 표본의 인덱스 — 정렬된 고유 int64, seed로 결정론적이다.

    `n_sample`이 `n_total`보다 크면 표본을 만들 수 없으므로 즉시 거부한다.
    """
    if n_sample > n_total:
        raise ValueError(f"n_sample({n_sample}) > n_total({n_total})")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_sample, replace=False)
    return np.sort(idx).astype(np.int64)


def sim_split_indices(case: np.ndarray, *, expected_total: "int | None" = None,
                      expected_train: "int | None" = None,
                      expected_val: "int | None" = None,
                      val_frac: "float | None" = None,
                      seed: "int | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """EXP-005와 같은 depth-map pair 단위 train/validation 전역 인덱스를 반환한다.

    작은 독립 fixture로 계약을 검증할 수 있도록 기대 개수와 split 인자는 주입 가능하지만,
    정상 실행은 H6 사전등록 상수를 사용한다. 반환값은 둘 다 정렬된 ``int64`` 전역 인덱스다.
    """
    total_n = SIM_TOTAL_N if expected_total is None else expected_total
    train_n = SIM_TRAIN_N if expected_train is None else expected_train
    val_n = SIM_VAL_N if expected_val is None else expected_val
    split_frac = SIM_SPLIT_VAL_FRAC if val_frac is None else val_frac
    split_seed = SIM_SPLIT_SEED if seed is None else seed

    a = np.asarray(case)
    if a.ndim != 1:
        raise ValueError(f"sim_case: 1차원 배열이어야 한다 — 받은 shape {a.shape}")
    if len(a) != total_n:
        raise ValueError(f"sim_case: 길이가 {total_n}이어야 한다 — 받은 길이 {len(a)}")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"sim_case: 정수 dtype이어야 한다 — 받은 dtype {a.dtype}")
    if len(a) % 2 or not np.array_equal(a[::2], a[1::2]):
        raise ValueError("sim_case: 연속한 두 SEM iteration의 case가 같은 depth-map pair여야 한다")

    val_mask = map_level_split(a, split_frac, split_seed)
    train_idx = np.flatnonzero(~val_mask).astype(np.int64)
    val_idx = np.flatnonzero(val_mask).astype(np.int64)
    if len(train_idx) != train_n or len(val_idx) != val_n:
        raise ValueError(
            "sim split 개수가 사전등록과 다르다 — "
            f"train {len(train_idx)} != {train_n} 또는 val {len(val_idx)} != {val_n}")
    return train_idx, val_idx


def validation_gate_indices(case: np.ndarray, *, n_sample: "int | None" = None,
                            gate_seed: int = 42, **split_contract) -> np.ndarray:
    """validation pool에서 뽑은 gate 표본을 원본 cache의 전역 인덱스로 반환한다."""
    sample_n = GATE_SAMPLE_N if n_sample is None else n_sample
    _train_idx, val_idx = sim_split_indices(case, **split_contract)
    local_idx = gate_sample_indices(len(val_idx), sample_n, gate_seed)
    return val_idx[local_idx]


def load_sim_cache_split(sim_path, case_path, **split_contract):
    """전체 sim cache를 검증해 원본 배열과 EXP-005 전역 split 인덱스를 반환한다."""
    require_source_name(sim_path, "sim_sem.npy")
    require_source_name(case_path, "sim_case.npy")
    sim = np.load(sim_path, mmap_mode="r")
    case = np.load(case_path, mmap_mode="r")
    expected_total = split_contract.get("expected_total", SIM_TOTAL_N)
    check_sem_array(sim, "sim_sem", expected_total)
    train_idx, val_idx = sim_split_indices(case, **split_contract)
    return sim, case, train_idx, val_idx


def sim_split_provenance(train_idx: np.ndarray, val_idx: np.ndarray, *,
                         total_n: "int | None" = None,
                         val_frac: "float | None" = None,
                         seed: "int | None" = None) -> dict:
    """split의 파라미터·개수·전역 인덱스 지문을 JSON 직렬화 가능한 형태로 묶는다."""
    train = np.asarray(train_idx, dtype=np.int64)
    val = np.asarray(val_idx, dtype=np.int64)
    total = SIM_TOTAL_N if total_n is None else total_n
    split_frac = SIM_SPLIT_VAL_FRAC if val_frac is None else val_frac
    split_seed = SIM_SPLIT_SEED if seed is None else seed
    if train.ndim != 1 or val.ndim != 1:
        raise ValueError("sim split 인덱스는 1차원이어야 한다")
    combined = np.concatenate([train, val])
    if (len(combined) != total or len(np.unique(combined)) != total
            or not np.array_equal(np.sort(combined), np.arange(total, dtype=np.int64))):
        raise ValueError("sim split 인덱스는 전체 cache 전역 인덱스를 중복 없이 분할해야 한다")
    return {
        "method": "map_level_split",
        "total_n": int(total),
        "train_n": int(len(train)),
        "val_n": int(len(val)),
        "val_frac": float(split_frac),
        "seed": int(split_seed),
        "train_indices_sha256": indices_sha256(train),
        "val_indices_sha256": indices_sha256(val),
    }


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def cyclegan_training_data_provenance(sim_path, case_path, real_path,
                                      train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
    """CycleGAN checkpoint가 결속할 세 입력과 고정 split provenance를 만든다."""
    provenance = {
        "inputs": {
            "sim_sem": _hashed_entry(sim_path),
            "sim_case": _hashed_entry(case_path),
            "real_sem": _hashed_entry(real_path),
        },
        "split": sim_split_provenance(train_idx, val_idx),
    }
    return validate_cyclegan_training_data(provenance)


def validate_cyclegan_training_data(provenance: dict) -> dict:
    """checkpoint 안 training_data의 필수 입력·해시·split 형태를 검증한다."""
    if not isinstance(provenance, dict) or set(provenance) != {"inputs", "split"}:
        raise GateFailedError("ckpt training_data는 inputs와 split만 가져야 한다")
    inputs = provenance.get("inputs")
    expected_inputs = {"sim_sem", "sim_case", "real_sem"}
    if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
        raise GateFailedError(
            f"ckpt training_data inputs가 {sorted(expected_inputs)}와 정확히 같아야 한다")
    for name, entry in inputs.items():
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise GateFailedError(f"ckpt training_data input:{name}의 path/sha256 형태가 잘못됐다")
        if not Path(str(entry["path"])).is_absolute():
            raise GateFailedError(f"ckpt training_data input:{name} path가 절대경로가 아니다")
        if not _SHA256_RE.fullmatch(str(entry["sha256"])):
            raise GateFailedError(f"ckpt training_data input:{name} sha256 형태가 잘못됐다")
    split = provenance.get("split")
    expected_split_keys = {
        "method", "total_n", "train_n", "val_n", "val_frac", "seed",
        "train_indices_sha256", "val_indices_sha256",
    }
    if not isinstance(split, dict) or set(split) != expected_split_keys:
        raise GateFailedError("ckpt training_data split provenance 형태가 잘못됐다")
    expected_values = {
        "method": "map_level_split", "total_n": SIM_TOTAL_N,
        "train_n": SIM_TRAIN_N, "val_n": SIM_VAL_N,
        "val_frac": SIM_SPLIT_VAL_FRAC, "seed": SIM_SPLIT_SEED,
    }
    drift = [key for key, value in expected_values.items() if split.get(key) != value]
    if drift:
        raise GateFailedError(f"ckpt training_data split 값이 사전등록과 다르다: {drift}")
    for key in ("train_indices_sha256", "val_indices_sha256"):
        if not _SHA256_RE.fullmatch(str(split.get(key, ""))):
            raise GateFailedError(f"ckpt training_data split {key} 형태가 잘못됐다")
    return provenance


def require_matching_training_data(stored: dict, current: dict) -> None:
    """resume state를 적용하기 전에 저장·현재 training_data가 정확히 같은지 확인한다."""
    validate_cyclegan_training_data(stored)
    validate_cyclegan_training_data(current)
    if stored != current:
        raise GateFailedError("resume checkpoint의 training_data가 현재 입력/split과 다르다")


def require_current_git_commit(claimed_commit: str, repo_root) -> str:
    """artifact 생성 시 받은 commit이 그 순간 알려진 repository HEAD와 정확히 같은지 확인한다."""
    if not _GIT_COMMIT_RE.fullmatch(str(claimed_commit)):
        raise ValueError("--git-commit은 40자리 16진 SHA여야 한다")
    repo = Path(repo_root).resolve()
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, encoding="utf-8").stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise ValueError(f"현재 repository HEAD를 확인할 수 없다: {repo}") from e
    if claimed_commit != head:
        raise ValueError(f"--git-commit이 현재 repository HEAD와 다르다: {claimed_commit} != {head}")
    return head


def _parabolic_subpixel(vec: np.ndarray, peak_idx: int, n: int) -> float:
    """정수 peak 주변 3점(순환 이웃)에 1차원 포물선을 맞춰 subpixel 위치를 추정한다.

    이웃 세 점이 평평하면(분모 0 — 상수 이미지 등 병적인 입력) 보정 없이 정수 peak를 그대로
    반환한다. NaN을 만들지 않기 위한 안전장치다.
    """
    left = vec[(peak_idx - 1) % n]
    center = vec[peak_idx]
    right = vec[(peak_idx + 1) % n]
    denom = left - 2.0 * center + right
    corr = 0.0 if denom == 0.0 else 0.5 * (left - right) / denom
    pos = peak_idx + corr
    half = n // 2
    if pos > half:
        pos -= n
    return float(pos)


def phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> "tuple[float, float]":
    """정규화 교차전력스펙트럼(phase correlation)으로 `a`→`b`의 정수+subpixel 이동량 `(dy, dx)`를 잰다.

    부호 규약: `b == np.roll(a, (dy, dx), axis=(0, 1))`이면 정확히 `(dy, dx)`를 돌려준다(주기의
    정확히 절반인 경계값은 `+half`와 `-half`가 같은 roll을 낳는 원천적 모호성이 있어 `+half`
    쪽을 고른다 — 어느 쪽을 골라도 롤 결과는 동일하다).

    동일 입력은 지름길로 `(0.0, 0.0)`을 정확히 반환한다(FFT 왕복의 부동소수 잡음이 섞이지
    않도록). 스펙트럼이 전부 0인 상수 이미지도 `eps`가 0/0을 막아 `(0.0, 0.0)`을 낸다 — NaN 없음.

    **알려진 사각지대**: 이 함수는 전역 위상만 보는 강체 이동(translation) 검출기다. 중심이
    고정된 대칭 팽창/축소(dilation/scale) — 예: 원판 구멍이 사방으로 고르게 커지는 것 — 는
    이동이 아니므로 대부분 `(0, 0)` 근방을 낸다(`test_phase_correlation_is_blind_to_symmetric_
    dilation`). 국소 왜곡과 비순환 subpixel 이동도 과소평가한다. 사전등록 gate의 잔여 위험이며,
    `local_shift_max`(국소 블록 매칭)가 진단 전용으로 기록하지만 판정을 대신하지 않는다
    (`LOCAL_PROBE_METHOD` 주석).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"a, b는 같은 shape의 2차원 배열이어야 한다 — 받은 {a.shape}, {b.shape}")
    if np.array_equal(a, b):
        return 0.0, 0.0

    h, w = a.shape
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fb * np.conj(fa)
    eps = 1e-12
    cross = cross / (np.abs(cross) + eps)
    r = np.fft.ifft2(cross).real

    py, px = np.unravel_index(np.argmax(r), r.shape)
    dy = _parabolic_subpixel(r[:, px], int(py), h)
    dx = _parabolic_subpixel(r[py, :], int(px), w)
    return dy, dx


def phase_correlation_shifts(orig: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """`(N, H, W)` 두 스택의 이미지별 부호 있는 이동량 `(dy, dx)` — shape `(N, 2)` float64.

    gate의 진단(diagnostics)이 방향성 편향(`mean_dy`/`mean_dx`)을 보려면 크기만으로는 부족해서
    부호 있는 값을 따로 노출한다. `shift_magnitudes`는 이 배열의 `hypot`이다.
    """
    orig = np.asarray(orig)
    moved = np.asarray(moved)
    if orig.shape != moved.shape or orig.ndim != 3:
        raise ValueError(f"orig, moved는 같은 shape의 (N,H,W)여야 한다 — 받은 "
                         f"{orig.shape}, {moved.shape}")
    out = np.empty((len(orig), 2), dtype=np.float64)
    for i in range(len(orig)):
        dy, dx = phase_correlation_shift(orig[i], moved[i])
        out[i, 0] = dy
        out[i, 1] = dx
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _parabolic_peak(cm: float, c: float, cp: float) -> float:
    """세 점 포물선의 꼭짓점 오프셋. 위로 볼록하지 않으면(평탄·병적) 보정하지 않는다."""
    d = cm - 2.0 * c + cp
    return 0.0 if d >= 0 else 0.5 * (cm - cp) / d


def block_match_shifts(a: np.ndarray, b: np.ndarray, tile: int = LOCAL_TILE,
                       search: int = LOCAL_SEARCH, min_std: float = LOCAL_MIN_STD) -> np.ndarray:
    """타일별 국소 변위 `(dy, dx)` — shape `(n_tiles, 2)`, 평탄 타일은 NaN.

    각 타일에서 `|d| <= search` 정수 후보마다 z-score 정규화 상관(NCC)을 재고 최댓값 주변을
    포물선으로 subpixel 보정한다. NCC라 아핀 밝기 변화(변환기의 정상 동작)에 불변이고, 탐색
    반경이 제한돼 백색화 phase correlation처럼 작은 타일에서 잡음 peak로 튀지 않는다.
    `b`는 반사 패딩으로 경계 밖을 읽는다. 부호 규약은 `phase_correlation_shift`와 같다:
    `b[p + d] == a[p]`(내용이 `+d`로 이동)이면 `d`를 돌려준다. 동일 입력은 정확히 0.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"a, b는 같은 shape의 2차원 배열이어야 한다 — 받은 {a.shape}, {b.shape}")
    h, w = a.shape
    if h % tile or w % tile:
        raise ValueError(f"shape {a.shape}이 타일 {tile}로 나누어떨어지지 않는다")
    same = np.array_equal(a, b)
    bp = np.pad(b, search, mode="reflect")
    k = 2 * search + 1
    out = []
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            ta = a[i:i + tile, j:j + tile]
            if ta.std() < min_std:
                out.append((np.nan, np.nan))
                continue
            if same:
                out.append((0.0, 0.0))
                continue
            za = _zscore(ta)
            score = np.empty((k, k))
            for dy in range(-search, search + 1):
                for dx in range(-search, search + 1):
                    tb = bp[i + search + dy:i + search + dy + tile,
                            j + search + dx:j + search + dx + tile]
                    score[dy + search, dx + search] = float((za * _zscore(tb)).mean())
            py, px = np.unravel_index(int(np.argmax(score)), score.shape)
            fy = _parabolic_peak(*score[py - 1:py + 2, px]) if 0 < py < k - 1 else 0.0
            fx = _parabolic_peak(*score[py, px - 1:px + 2]) if 0 < px < k - 1 else 0.0
            out.append((py - search + fy, px - search + fx))
    return np.asarray(out, dtype=np.float64)


def local_shift_max(orig: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """`(N, H, W)` 두 스택의 이미지별 **최대 타일 변위 크기** — 진단 전용 국소 probe의 입력.

    한 이미지 안에서 가장 많이 움직인 타일이 그 이미지의 기하 위반 정도다(국소 왜곡은 평균에
    묻힌다). 모든 타일이 평탄해 잴 수 없으면 NaN — probe가 "측정 불가"로 따로 센다.
    """
    orig = np.asarray(orig)
    moved = np.asarray(moved)
    if orig.shape != moved.shape or orig.ndim != 3:
        raise ValueError(f"orig, moved는 같은 shape의 (N,H,W)여야 한다 — 받은 "
                         f"{orig.shape}, {moved.shape}")
    out = np.empty(len(orig), dtype=np.float64)
    for n in range(len(orig)):
        mags = np.hypot(*block_match_shifts(orig[n], moved[n]).T)
        out[n] = np.nan if np.isnan(mags).all() else float(np.nanmax(mags))
    return out


def shift_magnitudes(orig: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """`(N, H, W)` 두 스택의 이미지별 이동량 크기(`hypot(dy, dx)`) — gate의 표본 통계 입력."""
    signed = phase_correlation_shifts(orig, moved)
    return np.hypot(signed[:, 0], signed[:, 1])


def measure_geometry(orig_u8: np.ndarray, moved_u8: np.ndarray) -> dict:
    """gate가 재는 기하 값을 한 번에 — 학습 gate와 번역 후 재검증이 **같은 함수**를 써야 둘이
    같은 것을 잰다. 입력은 `(N,72,48)` uint8 스택. 반환: `signed` `(N,2)`, `shifts` `(N,)`(전역
    이동 크기), `local` `(N,)`(`local_shift_max` — 진단 전용)."""
    check_sem_array(orig_u8, "gate 원본")
    check_sem_array(moved_u8, "gate 변환본")
    a = np.asarray(orig_u8, dtype=np.float64)
    b = np.asarray(moved_u8, dtype=np.float64)
    signed = phase_correlation_shifts(a, b)
    return {"signed": signed, "shifts": np.hypot(signed[:, 0], signed[:, 1]),
            "local": local_shift_max(a, b)}


def roundtrip_mae(orig_u8: np.ndarray, roundtrip_u8: np.ndarray) -> float:
    """`sim→real→sim` round-trip의 `[0, 1]` 정규화 MAE.

    이것은 **cycle-consistency 정합성 점검**이지 기하 보증이 아니다 — 생성기가 두 방향 모두
    항등에 가깝게 붕괴해도(모드 붕괴) MAE는 작게 나올 수 있다. 기하 보증은
    `phase_correlation_shift`/`evaluate_gate`의 이동량 기준이 맡는다.
    """
    a = np.asarray(orig_u8, dtype=np.float64)
    b = np.asarray(roundtrip_u8, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"orig_u8, roundtrip_u8은 같은 shape이어야 한다 — 받은 {a.shape}, {b.shape}")
    return float(np.mean(np.abs(a - b)) / 255.0)


def _local_probe(local: np.ndarray, n: int) -> dict:
    """미보정 국소 probe의 기록 — **판정에 관여하지 않는다**(`LOCAL_PROBE_METHOD` 주석).

    측정 불가(NaN) 이미지는 따로 세고 나머지로 통계를 낸다. 값이 없으면 `None`(NaN이 아니라 —
    JSON 왕복 뒤에도 `==` 대조가 성립해야 `recheck_gate`가 요약 조작을 잡는다)."""
    local = np.asarray(local, dtype=np.float64)
    if local.shape != (n,):
        raise ValueError(f"local은 ({n},) 형태여야 한다 — 받은 {local.shape}")
    finite = local[~np.isnan(local)]
    n_nan = int(n - len(finite))
    median = float(np.median(finite)) if len(finite) else None
    p95 = float(np.percentile(finite, 95)) if len(finite) else None
    flags = []
    if n_nan:
        flags.append(f"local_shift 측정 불가 {n_nan}장")
    if median is not None and median > SHIFT_MEDIAN_MAX:
        flags.append(f"local_shift_median {median} > {SHIFT_MEDIAN_MAX}")
    if p95 is not None and p95 > SHIFT_P95_MAX:
        flags.append(f"local_shift_p95 {p95} > {SHIFT_P95_MAX}")
    return {"role": LOCAL_PROBE_ROLE, "method": dict(LOCAL_PROBE_METHOD), "median": median,
            "p95": p95, "n_unmeasurable": n_nan, "flags": flags}


def evaluate_gate(shifts: np.ndarray, mae: float, *, local: "np.ndarray | None" = None,
                  signed: "np.ndarray | None" = None) -> dict:
    """기하 위생 gate 판정. 경계는 **포함**(`==` 임계값은 통과)이고, `NaN`은 항상 실패다.

    표본 크기가 `GATE_SAMPLE_N`이 아니면 애초에 사전등록된 gate가 아니므로 예외를 던진다 —
    표본 크기 자체가 사전등록의 일부다.

    **`passed`는 정확히 사전등록 3기준의 AND다**: `shift_median`, `shift_p95`(전역 phase
    correlation), `roundtrip_mae`. 하나라도 실패하면 downstream 없이 종료한다(사전등록 stop).

    `local`(이미지별 `local_shift_max`)을 주면 `local_probe`에 **진단 전용** 기록을 남긴다 —
    미보정이라 외관 전용 변환에 거짓 플래그를 올리고 타일 안 팽창을 못 보므로, 통과를 실패로도
    실패를 통과로도 바꾸지 않는다(`LOCAL_PROBE_METHOD` 주석). `diagnostics`(`shift_p99`,
    `shift_max`, `signed`가 주어졌을 때 `mean_dy`/`mean_dx`)도 같은 이유로 판정에 관여하지 않는다.
    """
    shifts = np.asarray(shifts, dtype=np.float64)
    n = len(shifts)
    if n != GATE_SAMPLE_N:
        raise ValueError(f"gate 표본 크기는 {GATE_SAMPLE_N}이어야 한다 — 받은 {n}")
    mae = float(mae)

    shift_nan = bool(np.isnan(shifts).any())
    mae_nan = bool(np.isnan(mae))
    median = float("nan") if shift_nan else float(np.median(shifts))
    p95 = float("nan") if shift_nan else float(np.percentile(shifts, 95))

    failures = []
    if shift_nan:
        failures.append("shift 배열에 NaN이 있다")
    else:
        if median > SHIFT_MEDIAN_MAX:
            failures.append(f"shift_median {median} > {SHIFT_MEDIAN_MAX}")
        if p95 > SHIFT_P95_MAX:
            failures.append(f"shift_p95 {p95} > {SHIFT_P95_MAX}")
    if mae_nan:
        failures.append("roundtrip_mae가 NaN이다")
    elif mae > ROUNDTRIP_MAE_MAX:
        failures.append(f"roundtrip_mae {mae} > {ROUNDTRIP_MAE_MAX}")

    diagnostics: dict = {
        "shift_p99": float("nan") if shift_nan else float(np.percentile(shifts, 99)),
        "shift_max": float("nan") if shift_nan else float(np.max(shifts)),
    }
    if signed is not None:
        signed = np.asarray(signed, dtype=np.float64)
        if signed.shape != (n, 2):
            raise ValueError(f"signed는 ({n}, 2) 형태여야 한다 — 받은 {signed.shape}")
        diagnostics["mean_dy"] = float(np.mean(signed[:, 0]))
        diagnostics["mean_dx"] = float(np.mean(signed[:, 1]))

    result = {
        "passed": len(failures) == 0,
        "n": n,
        "shift_median": median,
        "shift_p95": p95,
        "roundtrip_mae": mae,
        "thresholds": dict(GATE_THRESHOLDS),
        "failures": failures,
        "diagnostics": diagnostics,
    }
    if local is not None:
        result["local_probe"] = _local_probe(local, n)
    return result


# ── manifest — 덮어쓰기 거부 + 해시 검증 (불변이 아니라 "한 번 쓰고, 매번 재검증") ──

def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """파일 하나의 sha256 — 청크 단위로 읽어 큰 캐시(.npy)에서도 메모리를 안 먹는다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once_json(path, obj) -> None:
    """`path`가 이미 있으면 예외 — **덮어쓰기 거부**(overwrite-refusing)이지 불변(immutable)이
    아니다: 파일 자체를 잠그거나 보호하지 않고, 오직 "같은 경로에 두 번째로 쓰는 것"만 막는다.
    이후 변경 탐지는 이 함수가 아니라 `verify_manifest`/`require_gate_passed`의 **재해시 검증**이
    한다 — gate 결과·manifest는 한 번 쓰고, 쓸 때마다 그 내용을 다시 확인한다. 자유롭게 편집
    가능한 JSON의 hash는 인증 서명이 아니며, downstream checkpoint가 검증 시점 manifest 자체의
    sha256을 결속해 나중 변경을 탐지할 근거만 남긴다.

    `O_CREAT|O_EXCL`은 POSIX·Windows 모두 원자적이다(`ai_co_scientist.locks`와 같은 원리).
    Windows에서 막 unlink된 파일이 delete-pending 상태면 `EEXIST`가 아니라 `EACCES`
    (`PermissionError`)를 던지므로 이를 `FileExistsError`로 매핑한다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except (FileExistsError, PermissionError) as e:
        raise FileExistsError(f"이미 존재해 덮어쓸 수 없다(write-once): {path}") from e
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


class GateFailedError(RuntimeError):
    """기하 위생 gate를 통과하지 못했다 — downstream 진행(구조 학습·제출)을 막는 하드 스톱."""


GATE_SUMMARY_KEYS = ("passed", "shift_median", "shift_p95", "roundtrip_mae", "failures",
                     "thresholds", "local_probe")


def recheck_gate(gate: dict) -> dict:
    """저장된 gate를 **원본 per-image 값으로 다시 판정**한다 — 실패하면 `GateFailedError`.

    판정은 `shifts`(`GATE_SAMPLE_N`개)·`roundtrip_mae`만으로 완결된다 — 없으면 재판정할 수 없으므로
    거부한다. 진단 전용 `local_shifts`는 선택이지만, 있으면 `local_probe`를 재계산해 대조하고
    원본 없이 `local_probe`만 있으면 요약 불일치로 거부한다. 재판정이 통과여도 저장된 요약값
    (`GATE_SUMMARY_KEYS`)이 재계산과 **정확히** 같지 않으면 거부한다 — 요약만 고친 gate(예:
    median을 손으로 낮춘 JSON)가 원본과 따로 놀며 사람을 속이는 것을 막는다.
    `build_manifest`·`verify_manifest`·`require_gate_passed`가 공유한다.
    """
    shifts = gate.get("shifts")
    local = gate.get("local_shifts")
    mae = gate.get("roundtrip_mae")
    if not isinstance(shifts, list) or len(shifts) != GATE_SAMPLE_N or mae is None:
        raise GateFailedError(f"gate에 재평가할 원본 shifts({GATE_SAMPLE_N}개)/roundtrip_mae가 없다")
    if local is not None and (not isinstance(local, list) or len(local) != GATE_SAMPLE_N):
        raise GateFailedError(f"gate의 진단 원본 local_shifts가 {GATE_SAMPLE_N}개 목록이 아니다")
    recomputed = evaluate_gate(
        np.asarray(shifts, dtype=np.float64), float(mae),
        local=None if local is None else np.asarray(local, dtype=np.float64))
    if not recomputed["passed"]:
        raise GateFailedError(
            f"저장된 shifts/roundtrip_mae를 재평가하면 실패한다: {recomputed['failures']}")
    drift = [k for k in GATE_SUMMARY_KEYS if gate.get(k) != recomputed.get(k)]
    if drift:
        raise GateFailedError(f"gate의 저장된 요약값이 원본 재계산과 다르다: {drift}")
    return recomputed


def _hashed_entry(path) -> dict:
    # 절대경로 — 상대 레시피 경로(runtime/...)로 적으면 다른 cwd에서 검증할 때 "파일 없음"이 된다
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


REQUIRED_SOURCES = ("sim_sem", "sim_depth", "sim_case", "real_sem")
REQUIRED_OUTPUTS = ("sim_sem", "sim_depth", "sim_case")  # 번역본 + y/case 원본 사본
UNCHANGED_OUTPUTS = ("sim_depth", "sim_case")  # y는 원본 sim GT 그대로다 — 바이트가 같아야 한다


def _binding_problems(m: dict) -> list:
    """manifest 부품끼리의 결속 — 해시가 다 맞아도 다른 실행의 gate를 붙여 넣은 manifest를 잡는다.

    output이 `REQUIRED_OUTPUTS`와 정확히 같고, `UNCHANGED_OUTPUTS`는 같은 이름의 source와
    sha256이 같고(= "y는 변환 전 sim GT" 전제), gate의 `ckpt_sha256`·`report_id`가 manifest의
    ckpt·report_id와 같고, manifest·gate의 config가 사전등록과 같아야 한다.
    """
    problems = []
    outs = m.get("output_files") or {}
    srcs = m.get("source_files") or {}
    gate = m.get("gate") or {}
    if sorted(srcs) != sorted(REQUIRED_SOURCES):
        problems.append(f"source 목록이 {list(REQUIRED_SOURCES)}이어야 한다: {sorted(srcs)}")
    if sorted(outs) != sorted(REQUIRED_OUTPUTS):
        problems.append(f"output 목록이 {list(REQUIRED_OUTPUTS)}이어야 한다: {sorted(outs)}")
    if m.get("hypothesis") != "H6":
        problems.append(f"hypothesis가 'H6'이어야 한다: {m.get('hypothesis')!r}")
    if m.get("x_domain") != "sim_translated_to_real_appearance":
        problems.append("x_domain이 'sim_translated_to_real_appearance'이어야 한다")
    if m.get("y_source") != "sim_depth_gt":
        problems.append("y_source가 'sim_depth_gt'이어야 한다")
    if not _GIT_COMMIT_RE.fullmatch(str(m.get("git_commit", ""))):
        problems.append("git_commit은 40자리 16진 commit이어야 한다")
    for name in UNCHANGED_OUTPUTS:
        o, s = outs.get(name) or {}, srcs.get(name) or {}
        if not o.get("sha256") or o.get("sha256") != s.get("sha256"):
            problems.append(f"output:{name}이 source:{name}과 바이트가 다르다(y는 원본 그대로여야 한다)")
    if gate.get("ckpt_sha256") != (m.get("ckpt") or {}).get("sha256"):
        problems.append("gate의 ckpt_sha256이 manifest ckpt와 다르다(다른 실행의 gate)")
    if gate.get("report_id") != m.get("report_id"):
        problems.append(f"gate의 report_id({gate.get('report_id')!r})가 manifest"
                        f"({m.get('report_id')!r})와 다르다")
    case_entry = srcs.get("sim_case") or {}
    if gate.get("sim_case_sha256") != case_entry.get("sha256"):
        problems.append("gate의 sim_case_sha256이 manifest source:sim_case와 다르다")
    case_path = case_entry.get("path")
    if case_path:
        try:
            case = np.load(case_path, mmap_mode="r")
            train_idx, val_idx = sim_split_indices(case)
            expected_split = sim_split_provenance(train_idx, val_idx)
            expected_gate_sha = indices_sha256(validation_gate_indices(case))
            if gate.get("split") != expected_split:
                problems.append("gate의 split provenance가 source:sim_case와 다르다")
            if gate.get("indices_sha256") != expected_gate_sha:
                problems.append("gate의 indices_sha256이 validation 전역 표본과 다르다")
            if set(srcs) == set(REQUIRED_SOURCES):
                expected_training_data = cyclegan_training_data_provenance(
                    srcs["sim_sem"]["path"], srcs["sim_case"]["path"],
                    srcs["real_sem"]["path"], train_idx, val_idx)
                try:
                    require_matching_training_data(
                        gate.get("training_data"), expected_training_data)
                except GateFailedError as e:
                    problems.append(f"gate training_data가 manifest source와 다르다: {e}")
        except (OSError, ValueError, GateFailedError) as e:
            problems.append(f"source:sim_case split 검증 실패: {e}")
    for label, cfg in (("manifest", m.get("config")), ("gate", gate.get("config"))):
        try:
            validate_config(CycleGANConfig.from_dict(cfg if isinstance(cfg, dict) else {}))
        except ValueError as e:
            problems.append(f"{label} config가 사전등록과 다르다: {e}")
    return problems


def build_manifest(*, report_id: str, config: dict, gate: dict, ckpt_path,
                   source_files: "dict[str, object]", output_files: "dict[str, object]",
                   git_commit: str) -> dict:
    """H6 변환 산출물의 manifest. **실패한 gate로는 만들 수 없다** — 이 함수 자체가 하드 스톱의
    앞단이고, `require_gate_passed`가 뒷단이다(둘 다 있어야 만드는 쪽과 쓰는 쪽이 모두 막힌다).

    `source_files`/`output_files`는 `{이름: 경로}`. source와 ckpt는 절대경로로, **output은
    manifest가 놓일 디렉터리 기준 파일명으로** 적는다 — `translate_sim.py`는 `<out>.partial/`에
    쓰고 검증 뒤 `<out>`으로 rename하므로, 절대경로로 적으면 rename 직후 산출물이 전부 "파일
    없음"이 된다. 그래서 output은 모두 한 디렉터리(= manifest 디렉터리)에 있어야 한다.
    """
    if gate.get("passed") is not True:
        raise ValueError("gate가 실패했다 — 실패한 gate로는 manifest를 만들 수 없다")
    try:
        recheck_gate(gate)
    except GateFailedError as e:
        raise ValueError(f"manifest에 넣을 gate가 재검증을 통과하지 못한다: {e}") from e
    reject_test_paths([ckpt_path, *source_files.values(), *output_files.values()])
    out_dirs = {Path(p).resolve().parent for p in output_files.values()}
    if len(out_dirs) > 1:
        raise ValueError(f"output 파일은 한 디렉터리(manifest 디렉터리)에 있어야 한다 — "
                         f"받은 디렉터리 {sorted(str(d) for d in out_dirs)}")
    manifest = {
        "hypothesis": "H6",
        "report_id": report_id,
        "config": config,
        "gate": gate,
        "git_commit": git_commit,
        "x_domain": "sim_translated_to_real_appearance",
        "y_source": "sim_depth_gt",
        "ckpt": _hashed_entry(ckpt_path),
        "source_files": {name: _hashed_entry(p) for name, p in source_files.items()},
        "output_files": {name: {"path": Path(p).name, "sha256": sha256_file(p)}
                         for name, p in output_files.items()},
    }
    problems = _binding_problems(manifest)
    if problems:
        raise ValueError("manifest 결속 실패 — " + "; ".join(problems))
    return manifest


def verify_manifest(manifest_path) -> dict:
    """manifest를 쓸 때마다 **다시 믿을 이유를 확인**한다 — 파일 해시와 내용 계약 둘 다.

    1. 모든 파일(ckpt + source + output)의 sha256 재계산 대조.
    2. 계약: `hypothesis == "H6"`, output `path`는 manifest 디렉터리 안의 **맨 파일명**(구분자·
       `..` 없음 — manifest 밖을 가리키도록 고치는 것 차단), 어떤 경로도 `reject_test_paths`에
       걸리지 않음, 저장된 gate가 `recheck_gate`를 통과함.
    3. 결속: `_binding_problems` — 필수 output, y/case 무변경, gate↔ckpt·report_id·config.

    이 함수는 `translate_sim.py`가 승격 전에, `train_structure.py --cache-manifest`가 배열을 열기
    전에 부른다. JSON 자체는 인증되지 않으므로 downstream checkpoint에는 검증한 manifest 파일의
    절대경로와 sha256도 함께 결속한다.

    문제를 **전부** 나열한 뒤 `ValueError`를 던진다 — 하나 찾고 바로 멈추면 두 번째 결함이
    다음 실행까지 숨는다.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []

    if manifest.get("hypothesis") != "H6":
        problems.append(f"hypothesis가 'H6'이 아니다: {manifest.get('hypothesis')!r}")
    try:
        recheck_gate(manifest.get("gate") or {})
    except GateFailedError as e:
        problems.append(f"gate 재검증 실패: {e}")
    problems += _binding_problems(manifest)
    paths = [(manifest.get("ckpt") or {}).get("path", "")]
    paths += [e.get("path", "") for e in (manifest.get("source_files") or {}).values()]
    try:
        reject_test_paths(paths)
    except ValueError as e:
        problems.append(f"경로 위생: {e}")

    def _check(label: str, entry: dict, base: "Path | None" = None) -> None:
        if not entry.get("path") or not entry.get("sha256"):
            problems.append(f"{label}: path/sha256 항목이 없다")
            return
        p = Path(entry["path"]) if base is None else base / entry["path"]
        if not p.exists():
            problems.append(f"{label}: 파일 없음 ({p})")
            return
        actual = sha256_file(p)
        if actual != entry["sha256"]:
            problems.append(f"{label}: sha256 불일치 ({p})")

    _check("ckpt", manifest.get("ckpt") or {})
    for name, entry in (manifest.get("source_files") or {}).items():
        _check(f"source:{name}", entry)
    for name, entry in (manifest.get("output_files") or {}).items():
        rel = entry.get("path", "")
        if rel in ("", ".", "..") or Path(rel).name != rel or "/" in rel or "\\" in rel:
            problems.append(f"output:{name}: manifest 디렉터리 안의 맨 파일명이어야 한다 ({rel!r})")
            continue
        try:
            reject_test_paths([rel])
        except ValueError as e:
            problems.append(f"output:{name}: 경로 위생: {e}")
            continue
        _check(f"output:{name}", entry, base=manifest_path.parent)

    if problems:
        raise ValueError("manifest 검증 실패 — " + "; ".join(problems))
    return manifest


def resolve_structure_cache_provenance(cache_dir, manifest_path=None) -> dict:
    """구조 학습이 cache 배열을 열기 전에 H6 manifest와 도메인 provenance를 확정한다.

    manifest를 주지 않은 control은 기존 sim 도메인으로 남는다. 주었다면 선택한 cache 바로 안의
    ``manifest.json``만 허용하고, 본문/파일 해시를 재검증한 뒤 각 output이 실제로 그 cache의
    고정 파일명인지 확인한다.
    """
    cache = Path(cache_dir).resolve()
    if manifest_path is None:
        if (cache / "manifest.json").exists():
            raise ValueError(
                f"cache에 manifest.json이 있다 — --cache-manifest로 명시해야 한다: {cache}")
        return {
            "cache_dir": str(cache), "cache_manifest": None,
            "report_id": None, "hypothesis": None, "git_commit": None,
            "x_domain": "sim", "y_source": "sim_depth_gt",
            "output_files": None,
        }
    manifest_path = Path(manifest_path).resolve()
    expected = cache / "manifest.json"
    if manifest_path != expected:
        raise ValueError(
            f"--cache-manifest는 선택한 cache-dir의 manifest.json이어야 한다: "
            f"{manifest_path} != {expected}")
    if not manifest_path.is_file():
        raise ValueError(f"cache manifest 파일이 없다: {manifest_path}")
    manifest = verify_manifest(manifest_path)
    resolved_outputs = {}
    for name, entry in manifest["output_files"].items():
        actual = (manifest_path.parent / entry["path"]).resolve()
        wanted = cache / f"{name}.npy"
        if actual != wanted:
            raise ValueError(
                f"manifest output:{name}이 선택한 cache-dir의 고정 파일이 아니다: "
                f"{actual} != {wanted}")
        resolved_outputs[name] = {"path": str(actual), "sha256": entry["sha256"]}
    return {
        "cache_dir": str(cache),
        "cache_manifest": {
            "path": str(manifest_path), "sha256": sha256_file(manifest_path),
        },
        "report_id": manifest["report_id"],
        "hypothesis": manifest["hypothesis"],
        "git_commit": manifest["git_commit"],
        "x_domain": manifest["x_domain"],
        "y_source": manifest["y_source"],
        "output_files": resolved_outputs,
    }


def bind_structure_checkpoint(payload: dict, provenance: dict) -> dict:
    """best/save-epoch/resume checkpoint에 검증된 cache provenance를 빠짐없이 붙인다."""
    return {**payload, "data_provenance": provenance}


def structure_result_provenance(provenance: dict) -> dict:
    """구조 학습 최종 JSON의 도메인 라벨과, H6 arm이면 manifest 결속을 반환한다."""
    result = {"x_domain": provenance["x_domain"], "y_source": provenance["y_source"]}
    if provenance.get("cache_manifest") is not None:
        result["data_provenance"] = provenance
    return result


def require_matching_structure_provenance(stored: dict, current: dict) -> None:
    """resume state를 적용하기 전 저장된 cache/manifest provenance의 정확 일치를 요구한다."""
    if stored is None and current.get("cache_manifest") is None and current.get("x_domain") == "sim":
        return  # 명시적 legacy raw checkpoint 정책: manifest 없는 raw cache에서만 허용
    if stored != current:
        raise ValueError("resume checkpoint의 data_provenance가 현재 cache/manifest와 다르다")


def expected_ckpt_name(report_id: str) -> str:
    """`report_id`가 가리켜야 할 유일한 최종 ckpt 파일명 — 학습 스크립트의 규약과 1:1이다.

    resume용 중간 ckpt는 별도 파일(다른 이름)이라는 것이 전제다(`docs/experiment/
    H6-cyclegan-sim-to-real.md`와 학습 스크립트 계약). `require_gate_passed`가 이 이름과
    다른 파일을 거부해, resume ckpt로 gate를 우회하는 경로를 막는다.
    """
    return f"{report_id}-cyclegan.pt"


def check_ckpt_name(ckpt_path, report_id: str) -> None:
    """ckpt 파일명이 `expected_ckpt_name(report_id)`가 아니면 `GateFailedError`.

    순수 문자열 비교라 torch 이전에 부를 수 있다 — gate JSON이 아직 없는 `gate` 단계에서도
    resume ckpt(`*.resume.pt`)를 거부하려면 `require_gate_passed`와 별도로 있어야 한다.
    """
    expected = expected_ckpt_name(report_id)
    actual = Path(ckpt_path).name
    if actual != expected:
        raise GateFailedError(
            f"ckpt 파일명이 {expected!r}이어야 한다(중간 resume ckpt로는 gate를 통과할 수 "
            f"없다) — 받은 이름 {actual!r}")


def indices_sha256(idx: np.ndarray) -> str:
    """gate 표본 인덱스 배열의 sha256 지문 — `gate_sample_indices`가 실제로 재현됐는지 확인한다."""
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()


def require_gate_passed(gate_json_path, ckpt_path, *, report_id: str, sim_case_path=None,
                        sim_sem_path=None, real_sem_path=None) -> dict:
    """`translate_sim.py`가 torch를 import하기 **전**에 부르는 하드 스톱.

    저장된 `passed` 불리언은 **신뢰하지 않는다** — 저장된 per-image `shifts`와
    `roundtrip_mae`로 `recheck_gate`(`evaluate_gate` 재실행 + 요약값 정확 일치)를 돌려, 그
    결과가 통과인지로만 판단한다. 그 외 거부 조건:

    - `ckpt_path`의 파일명이 `expected_ckpt_name(report_id)`와 다름(중간 resume ckpt 우회 차단)
    - gate의 `report_id`가 인자로 준 것과 다름(다른 실행의 gate를 잘못 재사용하는 것 차단)
    - `ckpt_epoch`이 사전등록 총 epoch(`epochs_fixed+epochs_decay`)이 아님(중간 epoch 체크포인트
      우회 차단)
    - `config`가 `validate_config`를 통과하지 못함(가설과 다른 하이퍼파라미터로 학습한 것 차단)
    - 기록된 임계값이 현재 모듈 상수와 다름(코드가 바뀌었는데 오래된 gate를 재사용하는 것 차단)
    - 표본 크기가 `GATE_SAMPLE_N`이 아님
    - sim_case의 sha256이나 그 case로 재계산한 고정 train/validation split provenance가 다름
    - gate의 training_data가 현재 sim_sem/sim_case/real_sem 해시와 split provenance와 다름
    - `indices_sha256`이 validation global index pool에서 뽑은 고정 2,048장 지문과 다름
      (train 표본 또는 다른 validation 표본으로 gate를 통과시키는 것 차단)
    - 기록된 `ckpt_sha256`이 실제 `ckpt_path`와 다름(다른 체크포인트로 gate를 통과시키고 엉뚱한
      체크포인트로 변환하는 것 차단)
    - `sim_sem_sha256`이 실제 `sim_sem_path`와 다름(gate 당시와 다른 sim 캐시로 변환하는 것 차단)
    """
    gate_json_path = Path(gate_json_path)
    if not gate_json_path.exists():
        raise GateFailedError(f"gate 결과 파일이 없다: {gate_json_path}")
    gate = json.loads(gate_json_path.read_text(encoding="utf-8"))

    ckpt_path = Path(ckpt_path)
    check_ckpt_name(ckpt_path, report_id)

    if gate.get("report_id") != report_id:
        raise GateFailedError(
            f"gate의 report_id가 다르다: {gate.get('report_id')!r} != {report_id!r}")

    total_epochs = PREREGISTERED["epochs_fixed"] + PREREGISTERED["epochs_decay"]
    if gate.get("ckpt_epoch") != total_epochs:
        raise GateFailedError(
            f"gate의 ckpt_epoch이 사전등록 총 epoch({total_epochs})과 다르다: "
            f"{gate.get('ckpt_epoch')!r}")

    # 기록된 passed는 **필요조건이지 충분조건이 아니다** — False면 즉시 거부하고, True라도
    # 아래에서 원본 shifts/roundtrip_mae로 재평가해 진짜인지 다시 확인한다(조작된 True 방지).
    if gate.get("passed") is not True:
        raise GateFailedError(f"gate가 실패했다(passed={gate.get('passed')!r}): {gate_json_path}")

    config = gate.get("config")
    if not isinstance(config, dict):
        raise GateFailedError(f"gate에 config가 없다: {gate_json_path}")
    try:
        validate_config(CycleGANConfig.from_dict(config))
    except ValueError as e:
        raise GateFailedError(f"gate의 config가 사전등록과 다르다: {e}") from e

    expected_thresholds = GATE_THRESHOLDS
    if gate.get("thresholds") != expected_thresholds:
        raise GateFailedError(
            f"gate의 임계값이 현재 모듈 상수와 다르다: {gate.get('thresholds')} != {expected_thresholds}")
    if gate.get("n") != GATE_SAMPLE_N:
        raise GateFailedError(f"gate 표본 크기가 {GATE_SAMPLE_N}이 아니다: {gate.get('n')!r}")

    # 저장된 passed 불리언은 신뢰하지 않는다 — 원본 per-image 값으로 직접 재평가하고, 저장된
    # 요약값이 재계산과 정확히 같은지도 본다.
    try:
        recheck_gate(gate)
    except GateFailedError as e:
        raise GateFailedError(f"{e}: {gate_json_path}") from e

    sim_case_path = (gate_json_path.parent / "sim_case.npy"
                     if sim_case_path is None else Path(sim_case_path))
    if not sim_case_path.exists():
        raise GateFailedError(f"sim_case 파일이 없다: {sim_case_path}")
    try:
        case = np.load(sim_case_path, mmap_mode="r")
        train_idx, val_idx = sim_split_indices(case)
    except (OSError, ValueError) as e:
        raise GateFailedError(f"sim_case/split 검증 실패: {e}") from e

    actual_case_sha = sha256_file(sim_case_path)
    if gate.get("sim_case_sha256") != actual_case_sha:
        raise GateFailedError(
            f"gate의 sim_case_sha256이 실제 파일과 다르다: "
            f"{gate.get('sim_case_sha256')!r} != {actual_case_sha!r}")
    expected_split = sim_split_provenance(train_idx, val_idx)
    if gate.get("split") != expected_split:
        raise GateFailedError(
            f"gate의 split provenance가 실제 sim_case split과 다르다: "
            f"{gate.get('split')!r} != {expected_split!r}")

    sim_sem_path = (gate_json_path.parent / "sim_sem.npy"
                    if sim_sem_path is None else Path(sim_sem_path))
    real_sem_path = (gate_json_path.parent / "real_sem.npy"
                     if real_sem_path is None else Path(real_sem_path))
    if not sim_sem_path.exists() or not real_sem_path.exists():
        raise GateFailedError(
            f"현재 sim_sem/real_sem 파일이 없다: {sim_sem_path}, {real_sem_path}")
    current_training_data = cyclegan_training_data_provenance(
        sim_sem_path, sim_case_path, real_sem_path, train_idx, val_idx)
    require_matching_training_data(gate.get("training_data"), current_training_data)

    idx = validation_gate_indices(case)
    expected_idx_sha = indices_sha256(idx)
    if gate.get("indices_sha256") != expected_idx_sha:
        raise GateFailedError(
            f"gate의 indices_sha256이 사전등록 표본과 다르다: {gate.get('indices_sha256')!r} "
            f"!= {expected_idx_sha!r}")

    if not ckpt_path.exists():
        raise GateFailedError(f"ckpt 파일이 없다: {ckpt_path}")
    actual_sha = sha256_file(ckpt_path)
    if gate.get("ckpt_sha256") != actual_sha:
        raise GateFailedError(
            f"gate의 ckpt_sha256이 실제 ckpt와 다르다: {gate.get('ckpt_sha256')!r} != {actual_sha!r}")

    actual_sim_sha = sha256_file(sim_sem_path)
    if gate.get("sim_sem_sha256") != actual_sim_sha:
        raise GateFailedError(
            f"gate의 sim_sem_sha256이 실제 파일과 다르다: {gate.get('sim_sem_sha256')!r} "
            f"!= {actual_sim_sha!r}")

    return gate


# ── 지연 runtime 아키텍처 (torch는 여기서만 import) ────────────

_TORCH_HINT = "torch가 필요하다 — `uv run --group baseline`으로 baseline 그룹을 설치한 뒤 실행할 것"


def patchgan_output_shape(h: int, w: int, n_layers: int = 3) -> "tuple[int, int]":
    """70×70 PatchGAN(`build_discriminator`)의 출력 공간 해상도 — 순수 산술이라 torch 없이 검증 가능.

    구조: stride-2 conv(k4,p1) × `n_layers`(C64, C128, C256, ...) → stride-1 conv(k4,p1) 1번
    (C512) → stride-1 conv(k4,p1) 1번(1채널 출력). `72×48` 입력·`n_layers=3`이면 `(7, 4)`다.
    """
    def _conv_out(size: int, stride: int) -> int:
        return (size + 2 * 1 - 4) // stride + 1  # kernel=4, padding=1

    for _ in range(n_layers):
        h = _conv_out(h, 2)
        w = _conv_out(w, 2)
    for _ in range(2):  # C512(stride1) + 최종 1채널 출력 conv(stride1)
        h = _conv_out(h, 1)
        w = _conv_out(w, 1)
    return h, w


def build_generator(cfg: CycleGANConfig):
    """ResNet generator: `c7s1-ngf, d(2ngf), d(4ngf), n_res_blocks×R(4ngf), u(2ngf), u(ngf),
    c7s1-1, tanh` — InstanceNorm, reflection padding. `(B,1,72,48) -> (B,1,72,48)`.

    torch는 여기서만 import한다. 없으면 어느 그룹을 설치해야 하는지 말하는 `ImportError`를 던진다.
    """
    validate_config(cfg)
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(_TORCH_HINT) from e

    class _ResBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
                nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
            )

        def forward(self, x):
            return x + self.block(x)

    ngf = cfg.ngf
    layers = [
        nn.ReflectionPad2d(3), nn.Conv2d(cfg.in_channels, ngf, 7),
        nn.InstanceNorm2d(ngf), nn.ReLU(True),
        nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
        nn.InstanceNorm2d(ngf * 2), nn.ReLU(True),
        nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
        nn.InstanceNorm2d(ngf * 4), nn.ReLU(True),
    ]
    layers += [_ResBlock(ngf * 4) for _ in range(cfg.n_res_blocks)]
    layers += [
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(ngf * 2), nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(ngf), nn.ReLU(True),
        nn.ReflectionPad2d(3), nn.Conv2d(ngf, cfg.in_channels, 7), nn.Tanh(),
    ]
    return nn.Sequential(*layers)


def build_discriminator(cfg: CycleGANConfig):
    """70×70 PatchGAN: `C64(no-norm), C128, C256(stride2), C512(stride1), 1채널 출력`.

    InstanceNorm, LeakyReLU(0.2). `72×48` 입력 → `(B,1,7,4)`(`patchgan_output_shape` 참조).
    """
    validate_config(cfg)
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(_TORCH_HINT) from e

    ndf = cfg.ndf
    layers = [nn.Conv2d(cfg.in_channels, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
    ch = ndf
    for _ in range(cfg.n_disc_layers - 1):
        layers += [
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 2), nn.LeakyReLU(0.2, True),
        ]
        ch *= 2
    layers += [
        nn.Conv2d(ch, ch * 2, 4, stride=1, padding=1),
        nn.InstanceNorm2d(ch * 2), nn.LeakyReLU(0.2, True),
    ]
    ch *= 2
    layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
    return nn.Sequential(*layers)


def load_generators(ckpt_path, device):
    """최종 ckpt를 읽어 `(cfg, epoch, G_sim2real, G_real2sim, training_data)`를 반환한다.

    이름 검사(`check_ckpt_name`)는 부르기 **전에** 끝나 있어야 한다 — 여기는 torch.load 이후의
    내용 검사다: 저장된 config가 사전등록과 같고, epoch이 총 epoch(= 학습 완료)이며 training_data
    블록이 세 입력 해시와 고정 split 형태를 갖춰야 한다. 중간 resume ckpt나 다른 설정/입력 형태의
    ckpt는 `GateFailedError`.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover - torch 유무에 따라 한쪽만 탄다
        raise ImportError(_TORCH_HINT) from e
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = CycleGANConfig.from_dict(ckpt["config"])
    try:
        validate_config(cfg)
    except ValueError as e:
        raise GateFailedError(f"ckpt의 config가 사전등록과 다르다: {e}") from e
    if ckpt.get("epoch") != cfg.total_epochs:
        raise GateFailedError(
            f"ckpt의 epoch({ckpt.get('epoch')!r})이 총 epoch({cfg.total_epochs})이 아니다 — "
            "중간 resume ckpt이거나 학습이 끝나지 않았다")
    validate_cyclegan_training_data(ckpt.get("training_data"))
    gens = []
    for key in ("G_sim2real", "G_real2sim"):
        g = build_generator(cfg).to(device)
        g.load_state_dict(ckpt["state_dict"][key])
        g.eval()
        gens.append(g)
    return cfg, ckpt["epoch"], gens[0], gens[1], ckpt["training_data"]


def translate_u8(generator, arr_u8: np.ndarray, batch_size: int, device) -> np.ndarray:
    """`(N,72,48)` uint8을 배치 단위로 generator에 통과시켜 같은 shape의 uint8로 되돌린다.

    입력은 `to_signed`로 [-1,1], 출력(tanh)은 `signed_to_u8`로 반올림한다 — gate와 번역이 같은
    변환 경로를 써야 gate가 본 이미지와 downstream이 받는 이미지가 같다.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(_TORCH_HINT) from e
    check_sem_array(arr_u8, "translate_u8 입력")
    out = np.empty_like(arr_u8)
    with torch.no_grad():
        for start in range(0, len(arr_u8), batch_size):
            x = to_signed(np.ascontiguousarray(arr_u8[start:start + batch_size]))[:, None]
            check_batch_shape(x.shape, channels=1)
            y = generator(torch.from_numpy(x).to(device)).to("cpu").numpy()[:, 0]
            out[start:start + batch_size] = signed_to_u8(y)
    return out
