"""H5 — real 의사 라벨 자기학습의 계약 계층.

**이 모듈은 stdlib + numpy까지만 의존한다.** torch/cv2는 이 워크트리에 없다(dev 그룹만
sync) — teacher 추론·student 학습 루프는 `scripts/build_pseudo_labels.py`,
`scripts/train_self_training.py`에 남고, 그 두 스크립트가 torch를 지연 import하기 직전까지
통과해야 하는 계약(경로 검증·매니페스트·샘플러·arm parity)이 여기 있다.

H5 사전보고(`docs/experiment/H5-self-training-pseudo-label.md`)의 "실행 전 중단" 조건을
그대로 함수화한 것이 이 파일이다: pseudo-label 생성에 test 입력이 섞이면, 파일 수/shape가
안 맞으면, NaN/Inf가 있으면, 같은 명령의 해시가 다르면, arm 사이에 seed·데이터 구성 이외의
차이가 있으면 — 전부 `ContractError`로 막는다.
"""
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_co_scientist.sem import map_level_split

IMAGE_SHAPE = (72, 48)
REAL_TRAIN_NPY = "real_sem.npy"
TEST_NPY = "test_sem.npy"
EXPECTED_REAL_N = 60664
PSEUDO_SCHEMA = "h5-pseudo-label/v1"
STUDENT_SCHEMA = "h5-student/v1"
TEACHER_ADABN_SOURCE = "real"        # H5는 teacher AdaBN 소스를 real로 고정한다. test/realtest 금지
TEACHER_ADABN_SHUFFLE = 42
TEACHER_ADABN_BATCH = 512            # infer_decomposed.adapt_bn 기본값 — EXP-019 레시피
TEACHER_ADABN_DROP_LAST = False
# student의 sim X = EXP-005 teacher의 학습 분할 그대로 (depth-map 단위 20퍼센트 홀드아웃을 뺀
# 138,648장). 분할 시드는 arm seed와 **무관하게** 고정한다 — arm0b(seed 43)가 분할까지 바꾸면
# seed 대조군이 데이터 구성 차이를 섞는다. 홀드아웃은 X를 정의할 뿐 선택에 쓰지 않는다.
SIM_SPLIT = {"val_frac": 0.2, "seed": 42}
EXPECTED_SIM_TRAIN_N = 138648

STUDENT_HPARAMS = {
    "arch": "mlp", "width": 32, "loss": "l1", "optimizer": "adamw", "lr": 1e-3,
    "schedule": "cosine", "epochs": 15, "batch_size": 128, "init": "scratch",
    "blur_sigma": 0.0, "checkpoint": "final_epoch",
}  # holdout 선택 없음 — 마지막 epoch을 그대로 쓴다 (사전보고 "student" 절)

ARMS = {
    "arm0": {"data": "sim_only", "seed": 42},
    "arm0b": {"data": "sim_only", "seed": 43},
    "arm1": {"data": "sim_pseudo", "seed": 42},
}

# arm 사이에 달라도 되는 키. x_domain/y_source는 `data`를 도메인 라벨로 다시 적은 것뿐이라
# 여기 있다 — 빠지면 train이 실제로 쓰는 student manifest에서 parity가 항상 실패한다.
PARITY_ALLOWED_DIFFS = frozenset({
    "arm", "seed", "data", "sampler", "pseudo_manifest", "pseudo_labels_sha256", "out",
    "config_fingerprint", "x_domain", "y_source",
})
DIRTY_SUFFIX = "+dirty"
# 두 GPU 진입점(라벨 생성·학습)이 잡는 기계 단위 락 이름 — `locks.resource_lock`
GPU_LOCK = "gpu-0"

# canonical_json/check_arm_parity 내부에서만 쓰는 결측 표시자 — 합법적인 None/빈문자열과
# 절대 겹치지 않도록 문자열 안에 넣을 수 없는 NUL을 포함시킨다.
_MISSING = "\u0000__missing__\u0000"


class ContractError(ValueError):
    """H5 계약 위반. 전부 이 타입으로 던진다 — CLI는 이것만 잡아 `ap.error()`로 바꾼다."""


# ── 해시 / 정규 직렬화 ──────────────────────────────────────

def sha256_file(path, chunk: int = 1 << 20) -> str:
    """파일을 스트리밍으로 읽어 sha256 hex digest를 낸다 — 전체를 메모리에 올리지 않는다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj) -> str:
    """키 정렬 + 구분자 고정 — 같은 내용이면 항상 같은 바이트열이 나온다(해시·비교의 전제)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def git_head(cwd=None) -> str:
    """실행 트리의 HEAD. 커밋 안 된 변경이 있으면(미추적 파일 포함) `+dirty`를 붙인다.

    미추적 파일도 dirty다 — 새 스크립트가 커밋 전이면 HEAD는 그 코드를 담고 있지 않다.
    `runtime/`은 .gitignore라 산출물은 여기 걸리지 않는다.

    git이 없거나 실패하면 빈 문자열 — `check_arm_parity`가 빈 값과 dirty를 둘 다 거부한다.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=cwd).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True,
                               cwd=cwd).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return head + DIRTY_SUFFIX if dirty else head


def require_clean_commit(commit: str) -> str:
    """학습/라벨 생성 **전에** 부른다 — 비었거나 dirty면 어느 코드가 돌았는지 말할 수 없다."""
    if not commit or commit.endswith(DIRTY_SUFFIX):
        raise ContractError(
            f"실행 트리가 깨끗한 커밋이 아니다 ({commit or 'git 정보 없음'}) — 커밋되지 않은 "
            f"변경(미추적 파일 포함)을 정리하고 같은 커밋에서 모든 arm을 실행할 것")
    return commit


def sim_train_indices(case, expected_n=EXPECTED_SIM_TRAIN_N) -> np.ndarray:
    """student가 쓰는 sim 이미지 인덱스 — EXP-005와 같은 `map_level_split(case, 0.2, 42)`의 train.

    expected_n이 None이 아니면 장수를 강제한다(사전보고 "공통 X: sim train 138,648장").
    """
    va = map_level_split(np.asarray(case), SIM_SPLIT["val_frac"], SIM_SPLIT["seed"])
    idx = np.where(~va)[0].astype(np.int64)
    if expected_n is not None and len(idx) != expected_n:
        raise ContractError(f"sim train 장수가 {expected_n}이 아니다 — 받은 {len(idx)}")
    return idx


def sim_subset_record(idx: np.ndarray) -> dict:
    """sim X를 config에 박는 기록 — 인덱스 배열의 해시까지 넣어 arm parity가 비교하게 한다."""
    a = np.ascontiguousarray(idx, dtype=np.int64)
    return {"split": "map_level_split", **SIM_SPLIT, "n": int(len(a)),
            "indices_sha256": hashlib.sha256(a.tobytes()).hexdigest()}


# ── 입력 경로 계약 ──────────────────────────────────────────

def require_real_train_source(source, cache_dir=None, expected_n: int = EXPECTED_REAL_N) -> Path:
    """`source`가 real train SEM 배열임을 강제한다 (H5 "실행 전 중단": test 입력 금지).

    1) 파일명이 정확히 ``real_sem.npy``여야 한다. **파일명만** 본다 — pytest의 tmp_path는
       부모 디렉터리 이름에 ``test``를 흔히 포함하므로 경로 전체를 검사하면 오탐이 난다.
    2) cache_dir가 주어지고 거기 ``test_sem.npy``가 있으면, source가 그것과 동일 파일이거나
       (`os.path.samefile`) 바이트가 같으면(우회 복사) 거부한다.
    3) 로드해 shape/dtype/장수를 검증한다.
    """
    src = Path(source)
    if src.name != REAL_TRAIN_NPY:
        if "test" in src.name.lower():
            raise ContractError(
                f"real train source 파일명이 {REAL_TRAIN_NPY}가 아니다 (test 입력 금지): "
                f"{src.name}")
        raise ContractError(f"real train source 파일명이 {REAL_TRAIN_NPY}가 아니다: {src.name}")

    resolved = src.resolve()
    if cache_dir is not None:
        test_path = Path(cache_dir) / TEST_NPY
        if test_path.exists():
            same = False
            if resolved.exists():
                try:
                    same = os.path.samefile(resolved, test_path)
                except OSError:
                    same = False
                if not same:
                    same = sha256_file(resolved) == sha256_file(test_path)
            if same:
                raise ContractError(
                    f"real train source가 test 캐시({test_path})와 동일 내용이다 "
                    f"(test 입력 금지)")

    if not resolved.is_file():
        raise ContractError(f"real train source 파일이 없다: {resolved}")
    arr = np.load(resolved, mmap_mode="r")
    if arr.ndim != 3:
        raise ContractError(
            f"real train source는 (N, H, W) 3차원이어야 한다 — 받은 shape {arr.shape}")
    if tuple(arr.shape[1:]) != IMAGE_SHAPE:
        raise ContractError(
            f"real train source 이미지 shape이 {IMAGE_SHAPE}가 아니다 — 받은 "
            f"{tuple(arr.shape[1:])}")
    if len(arr) != expected_n:
        raise ContractError(
            f"real train source 장수가 {expected_n}가 아니다 — 받은 {len(arr)}")
    if arr.dtype != np.uint8:
        raise ContractError(f"real train source dtype이 uint8이 아니다 — 받은 {arr.dtype}")
    return resolved


def validate_pseudo_labels(arr, n_expected: int) -> dict:
    """pseudo-label 배열의 shape/dtype/유한값/범위를 검증하고 통계를 낸다.

    `arr`는 `s = (L-depth)/L`(정규화 구조)이므로 [0, 1] 밖의 값은 teacher가 망가졌다는
    신호다 — `docs/data-facts.md` §2, `sem.depth_to_s`의 보장을 그대로 옮긴다.
    """
    shape = tuple(arr.shape)
    expected_shape = (n_expected, *IMAGE_SHAPE)
    if shape != expected_shape:
        raise ContractError(
            f"pseudo label shape이 {expected_shape}가 아니다 — 받은 {shape}")
    if arr.dtype != np.float32:
        raise ContractError(f"pseudo label dtype이 float32가 아니다 — 받은 {arr.dtype}")

    a = np.asarray(arr)
    if not np.isfinite(a).all():
        raise ContractError("pseudo label에 NaN/Inf가 있다")
    mn = float(a.min())
    mx = float(a.max())
    if mn < 0.0 or mx > 1.0:
        raise ContractError(f"pseudo label 범위가 [0,1]을 벗어난다 — min={mn} max={mx}")
    mean = float(a.astype(np.float64).mean())
    return {"n": int(shape[0]), "shape": list(shape), "dtype": "float32",
            "min": mn, "max": mx, "mean": mean}


# ── manifest ────────────────────────────────────────────────

def build_pseudo_manifest(*, labels_path, source_path, teacher_ckpt, expected_n,
                          adabn_source: str = TEACHER_ADABN_SOURCE,
                          adabn_shuffle: int = TEACHER_ADABN_SHUFFLE,
                          batch_size: int = 512, source_commit: str = "",
                          runtime=None) -> dict:
    """생성된 pseudo-label의 (X, y) 출처를 기록하는 manifest dict를 만든다.

    타임스탬프를 넣지 않는다 — 같은 입력이면 경로만 다른 두 번의 빌드가 `compare_pseudo_manifests`
    상 완전히 같아야(빈 diff) 하기 때문이다. `runtime`(device·torch·CUDA 버전)은 라벨 바이트를
    결정하는 환경이라 기록하고 비교한다 — 해시가 다를 때 원인을 말할 수 있게.
    """
    if adabn_source != TEACHER_ADABN_SOURCE:
        raise ContractError(
            f"H5는 teacher AdaBN source를 {TEACHER_ADABN_SOURCE!r}로 고정한다 — 받은 "
            f"{adabn_source!r}")
    if adabn_shuffle != TEACHER_ADABN_SHUFFLE:
        raise ContractError(
            f"H5는 teacher AdaBN shuffle seed를 {TEACHER_ADABN_SHUFFLE}로 고정한다 — 받은 "
            f"{adabn_shuffle}")

    source = require_real_train_source(source_path, cache_dir=None, expected_n=expected_n)
    labels_path = Path(labels_path).resolve()
    teacher_path = Path(teacher_ckpt).resolve()

    labels = np.load(labels_path, mmap_mode="r")
    stats = validate_pseudo_labels(labels, expected_n)

    return {
        "schema": PSEUDO_SCHEMA,
        "x_domain": "real",
        "x_desc": "real train SEM (AdaBN·의사 라벨 생성에만 사용, real test는 사용하지 않음)",
        "y_source": "pseudo_label",
        "y_desc": "EXP-005 계열 PlainMLP teacher가 real train에 생성한 연속값 soft pseudo-label s",
        "teacher": {
            "ckpt": str(teacher_path),
            "sha256": sha256_file(teacher_path),
            "adabn_source": adabn_source,
            "adabn_shuffle": adabn_shuffle,
            "adabn_batch": TEACHER_ADABN_BATCH,
            "adabn_drop_last": TEACHER_ADABN_DROP_LAST,
        },
        "source": {"path": str(source), "sha256": sha256_file(source), "n": expected_n},
        "labels": {"path": str(labels_path), "sha256": sha256_file(labels_path), **stats},
        "batch_size": batch_size,
        "source_commit": source_commit,
        "runtime": runtime or {},
    }


def write_manifest(path, manifest: dict) -> Path:
    """manifest를 JSON으로 쓴다. **덮어쓰지 않는다** — 이미 있으면 ContractError."""
    p = Path(path)
    if p.exists():
        raise ContractError(f"manifest가 이미 존재한다 — 덮어쓰지 않는다: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    p.write_text(text, encoding="utf-8")
    return p


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_pseudo_manifest(manifest_path, *, teacher_ckpt=None) -> dict:
    """manifest가 가리키는 파일들을 다시 해싱·검증해 변조/누락을 잡는다."""
    manifest = load_manifest(manifest_path)
    if manifest.get("schema") != PSEUDO_SCHEMA:
        raise ContractError(
            f"manifest schema가 {PSEUDO_SCHEMA}가 아니다 — 받은 {manifest.get('schema')}")

    claims = {"x_domain": (manifest.get("x_domain"), "real"),
              "y_source": (manifest.get("y_source"), "pseudo_label"),
              "teacher.adabn_source": (manifest["teacher"].get("adabn_source"),
                                       TEACHER_ADABN_SOURCE),
              "teacher.adabn_shuffle": (manifest["teacher"].get("adabn_shuffle"),
                                        TEACHER_ADABN_SHUFFLE),
              "labels.n": (manifest["labels"].get("n"), manifest["source"].get("n"))}
    for key, (got, want) in claims.items():
        if got != want:
            raise ContractError(f"manifest {key}가 H5 계약과 다르다: {got!r} != {want!r}")

    labels_path = Path(manifest["labels"]["path"])
    source_path = Path(manifest["source"]["path"])
    teacher_path = Path(manifest["teacher"]["ckpt"])
    require_real_train_source(source_path, expected_n=manifest["source"]["n"])

    if not labels_path.exists():
        raise ContractError(f"labels 파일이 없다: {labels_path}")
    if sha256_file(labels_path) != manifest["labels"]["sha256"]:
        raise ContractError(f"labels.sha256 불일치 — 파일이 변조됐다: {labels_path}")

    if not source_path.exists():
        raise ContractError(f"source 파일이 없다: {source_path}")
    if sha256_file(source_path) != manifest["source"]["sha256"]:
        raise ContractError(f"source.sha256 불일치 — 파일이 변조됐다: {source_path}")

    if not teacher_path.exists():
        raise ContractError(f"teacher ckpt 파일이 없다: {teacher_path}")
    if sha256_file(teacher_path) != manifest["teacher"]["sha256"]:
        raise ContractError(f"teacher.sha256 불일치 — 파일이 변조됐다: {teacher_path}")

    labels = np.load(labels_path, mmap_mode="r")
    validate_pseudo_labels(labels, manifest["labels"]["n"])

    if teacher_ckpt is not None:
        want = sha256_file(Path(teacher_ckpt))
        if want != manifest["teacher"]["sha256"]:
            raise ContractError(
                f"teacher.sha256이 주어진 --teacher-ckpt와 다르다: {teacher_ckpt}")

    return manifest


def _flatten(obj, prefix: str = "") -> dict:
    """중첩 dict를 점(.) 구분 키로 평탄화한다. dict가 아닌 리프만 담는다."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    else:
        out[prefix] = obj
    return out


_PATH_ONLY_KEYS = frozenset({"labels.path", "source.path", "teacher.ckpt"})


def compare_pseudo_manifests(a: dict, b: dict) -> list:
    """경로 필드를 무시하고 두 manifest의 내용을 비교한다. 빈 리스트 == 같은 명령의 재현."""
    fa = _flatten(a)
    fb = _flatten(b)
    keys = set(fa) | set(fb)
    diffs = []
    for k in keys:
        if k in _PATH_ONLY_KEYS:
            continue
        if canonical_json(fa.get(k, _MISSING)) != canonical_json(fb.get(k, _MISSING)):
            diffs.append(k)
    return sorted(diffs)


def require_reproduced(a: dict, b: dict) -> None:
    diffs = compare_pseudo_manifests(a, b)
    if diffs:
        raise ContractError(f"동일 명령 재현 실패 — 달라진 필드: {', '.join(diffs)}")


# ── 경로 분리 / teacher 복사 방지 ────────────────────────────

def reject_aliases(**paths) -> None:
    """서로 다른 이름의 경로 인자가 실제로는 같은 파일을 가리키면 거부한다.

    `None` 값은 (선택 인자가 안 쓰였다는 뜻이므로) 검사에서 뺀다.
    """
    resolved = {name: Path(p).resolve() for name, p in paths.items() if p is not None}
    names = list(resolved)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if resolved[names[i]] == resolved[names[j]]:
                raise ContractError(
                    f"{names[i]}와(과) {names[j]}가 같은 경로를 가리킨다: {resolved[names[i]]}")


def reject_teacher_copy(path, teacher_sha256: str) -> None:
    """`path`가 이미 존재하고 teacher checkpoint의 바이트 복사본이면 거부한다.

    student는 scratch에서 시작한다(사전보고 "student" 절) — out/resume이 teacher와 같은
    가중치면 사실상 이어학습이 되어 arm 비교가 깨진다.
    """
    p = Path(path)
    if p.exists() and p.is_file() and sha256_file(p) == teacher_sha256:
        raise ContractError(
            f"{p}가 teacher checkpoint의 바이트 복사본이다 — student는 scratch에서 "
            f"시작해야 한다")


def student_resume_path(out) -> Path:
    """`train_structure.py --resume`과 같은 규칙 — 별도 파일, out을 덮어쓰지 않는다."""
    return Path(out).with_suffix(".resume.pt")


def student_manifest_path(out) -> Path:
    return Path(out).with_suffix(".manifest.json")


# ── 샘플러 ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RealSamplerSpec:
    """sim 전량 + real 일부를 매 epoch 섞는 샘플러의 명세.

    real이 sim을 과대표집하지 않도록(사전보고 "arm 1") ``real_per_epoch <= n_sim``을 강제한다.
    """
    n_sim: int
    n_real: int
    real_per_epoch: int
    seed: int

    @classmethod
    def build(cls, n_sim: int, n_real: int, real_per_epoch=None, seed: int = 42):
        if n_sim <= 0:
            raise ContractError(f"n_sim은 양수여야 한다 — 받은 {n_sim}")
        if real_per_epoch is None:
            real_per_epoch = min(n_real, n_sim)
        if real_per_epoch < 0:
            raise ContractError(f"real_per_epoch은 음수일 수 없다 — 받은 {real_per_epoch}")
        if real_per_epoch > n_real:
            raise ContractError(
                f"real_per_epoch({real_per_epoch})이 n_real({n_real})보다 크다")
        if real_per_epoch > n_sim:
            raise ContractError(
                f"real_per_epoch({real_per_epoch})이 n_sim({n_sim})보다 크다 — H5는 real이 "
                f"sim을 과대표집하지 않도록 epoch당 n_sim 이하로 제한한다")
        if n_real == 0 and real_per_epoch > 0:
            raise ContractError("n_real이 0인데 real_per_epoch가 0보다 크다")
        return cls(n_sim=n_sim, n_real=n_real, real_per_epoch=real_per_epoch, seed=seed)

    def to_dict(self) -> dict:
        return {"n_sim": self.n_sim, "n_real": self.n_real,
                "real_per_epoch": self.real_per_epoch, "seed": self.seed,
                "replacement": False, "rng": "numpy.default_rng([seed, epoch])"}

    def epoch_indices(self, epoch: int) -> np.ndarray:
        """이 epoch의 인덱스 배열 — sim은 전량 정확히 한 번, real은 비복원 표집.

        `[seed, epoch]`를 시드로 쓰므로 (spec, epoch)에 대해 결정적이고, epoch이 다르면
        (일반적으로) 다른 순열이 나온다.
        """
        rng = np.random.default_rng([self.seed, epoch])
        sim_part = np.arange(self.n_sim, dtype=np.int64)
        if self.real_per_epoch > 0:
            chosen = rng.choice(self.n_real, self.real_per_epoch, replace=False)
            real_part = self.n_sim + chosen.astype(np.int64)
            combined = np.concatenate([sim_part, real_part])
        else:
            combined = sim_part
        return rng.permutation(combined).astype(np.int64)


# ── arm / student config ───────────────────────────────────

def arm_spec(arm: str) -> dict:
    try:
        return dict(ARMS[arm])
    except KeyError:
        raise ContractError(
            f"알 수 없는 arm: {arm!r} — {sorted(ARMS)} 중 하나여야 한다") from None


def student_config(arm: str, sampler: RealSamplerSpec, *, pseudo_manifest_sha256=None,
                   source_commit: str = "", sim_subset=None) -> dict:
    """arm 하나의 전체 학습 설정. `config_fingerprint`가 나머지 전체를 요약한다."""
    spec = arm_spec(arm)
    data = spec["data"]
    seed = spec["seed"]

    if data == "sim_only":
        if sampler.real_per_epoch != 0:
            raise ContractError(
                f"{arm}은 sim_only인데 sampler.real_per_epoch가 0이 아니다: "
                f"{sampler.real_per_epoch}")
        if pseudo_manifest_sha256 is not None:
            raise ContractError(f"{arm}은 sim_only인데 pseudo_manifest_sha256이 주어졌다")
    elif data == "sim_pseudo":
        if sampler.real_per_epoch <= 0:
            raise ContractError(
                f"{arm}은 sim_pseudo인데 sampler.real_per_epoch가 0 이하다: "
                f"{sampler.real_per_epoch}")
        if not pseudo_manifest_sha256:
            raise ContractError(f"{arm}은 sim_pseudo인데 pseudo_manifest_sha256이 없다")
    else:  # pragma: no cover - ARMS가 아닌 한 도달하지 않는다
        raise ContractError(f"알 수 없는 data 종류: {data!r}")

    rest = {"schema": STUDENT_SCHEMA, "arm": arm, "data": data, "seed": seed,
             **STUDENT_HPARAMS, "sampler": sampler.to_dict(),
             "pseudo_manifest": pseudo_manifest_sha256, "source_commit": source_commit,
             "sim_subset": sim_subset}
    cfg = dict(rest)
    cfg["config_fingerprint"] = fingerprint(rest)
    return cfg


# student_config가 지문에 넣는 키 — 그 뒤에 train이 덧붙이는 out/runtime/x_domain 등은 제외.
_CONFIG_KEYS = frozenset({"schema", "arm", "data", "seed", *STUDENT_HPARAMS, "sampler",
                          "pseudo_manifest", "source_commit", "sim_subset"})


def check_arm_parity(configs: list) -> None:
    """arm 사이에 seed·데이터 구성(`PARITY_ALLOWED_DIFFS`) 이외의 차이가 없는지 강제한다.

    사전보고 예산 조건: "모든 arm은 같은 구현 commit에서 실행한다" — `source_commit`이
    비어 있으면(커밋 정보를 못 읽었으면) 그 자체로 비교가 무의미하므로 거부한다.
    """
    if len(configs) < 2:
        raise ContractError("parity 검사는 최소 2개 config가 필요하다")
    arms = [c.get("arm") for c in configs]
    if len(set(arms)) != len(arms):
        raise ContractError(f"같은 arm이 두 번 들어왔다: {arms}")

    all_keys = set()
    for c in configs:
        all_keys |= set(c.keys())

    bad = []
    for k in sorted(all_keys):
        if k in PARITY_ALLOWED_DIFFS:
            continue
        values = [canonical_json(c.get(k, _MISSING)) for c in configs]
        if len(set(values)) > 1:
            bad.append(k)
    if bad:
        raise ContractError(f"arm 사이에 seed/데이터 구성 외 설정이 다르다: {', '.join(bad)}")

    # 키 차이를 먼저 보고해야 메시지가 원인(lr 등)을 가리킨다 — 지문 불일치는 그 다음이다.
    for c in configs:
        spec = arm_spec(c.get("arm"))
        if (c.get("seed"), c.get("data")) != (spec["seed"], spec["data"]):
            raise ContractError(f"{c.get('arm')}의 seed/data가 ARMS와 다르다")
        rest = {k: v for k, v in c.items() if k in _CONFIG_KEYS}
        if fingerprint(rest) != c.get("config_fingerprint"):
            raise ContractError(f"{c.get('arm')}의 config_fingerprint가 내용과 맞지 않는다")

    n_sims = {c.get("sampler", {}).get("n_sim") for c in configs}
    if len(n_sims) > 1:
        raise ContractError(f"sampler.n_sim이 arm마다 다르다: {sorted(map(str, n_sims))}")

    commits = {c.get("source_commit") for c in configs}
    if (len(commits) > 1 or "" in commits or None in commits
            or any(str(c).endswith(DIRTY_SUFFIX) for c in commits)):
        raise ContractError(
            f"source_commit이 arm마다 다르거나 비어 있거나 dirty다 — 모든 arm은 같은 깨끗한 "
            f"구현 commit에서 "
            f"실행해야 한다: {sorted(map(str, commits))}")
