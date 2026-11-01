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
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_co_scientist.locks import GPU_LOCK as GPU_LOCK
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
# student 학습이 읽는 sim 캐시 배열 — plan/train config에 각각의 sha256을 박는다
SIM_INPUT_NAMES = ("sim_sem", "sim_depth", "sim_case")
# 한 노출 라운드(=epoch)에 sim 전량 뒤에 더 보는 장수. arm1은 real pseudo 전량 60,664장,
# 대조군은 sim을 같은 수만큼 비복원으로 더 본다 — 세 arm의 노출 수·optimizer step 수가 같아야
# pseudo-label 효과가 학습량 차이와 섞이지 않는다 (라운드당 199,312장, 15라운드 23,370 step).
ROUND_EXTRA_N = EXPECTED_REAL_N

STUDENT_HPARAMS = {
    "arch": "mlp", "width": 32, "loss": "l1", "optimizer": "adamw", "lr": 1e-3,
    "schedule": "cosine", "epochs": 15, "batch_size": 128, "init": "scratch",
    "blur_sigma": 0.0, "checkpoint": "final_epoch",
    # cosine은 전역 optimizer step 단위로 T_max=total_optimizer_steps까지 내려간다. 라운드의
    # 마지막 배치는 버리지 않는다(drop_last=False) — 모든 노출이 학습에 들어간다.
    "schedule_step": "optimizer_step", "drop_last": False,
}  # holdout 선택 없음 — 마지막 epoch을 그대로 쓴다 (사전보고 "student" 절)

ARMS = {
    "arm0": {"data": "sim_only", "seed": 42},
    "arm0b": {"data": "sim_only", "seed": 43},
    "arm1": {"data": "sim_pseudo", "seed": 42},
}

# arm 사이에 달라도 되는 키. x_domain/y_source는 `data`를 도메인 라벨로 다시 적은 것뿐이라
# 여기 있다 — 빠지면 train이 실제로 쓰는 student manifest에서 parity가 항상 실패한다.
# sim/real_presentations·extra_sampler_seed는 sampler(허용 차이)에서 유도되는 값이라 여기 있고,
# 유도가 맞는지는 `check_arm_parity`가 따로 재계산한다. out_sha256은 arm마다 다른 산출물이다.
PARITY_ALLOWED_DIFFS = frozenset({
    "arm", "seed", "data", "sampler", "pseudo_manifest", "pseudo_labels_sha256", "out",
    "config_fingerprint", "x_domain", "y_source", "sim_presentations", "real_presentations",
    "extra_sampler_seed", "out_sha256",
})
DIRTY_SUFFIX = "+dirty"
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


def sim_inputs_record(cache_dir) -> dict:
    """student가 읽는 sim 캐시 3종(`SIM_INPUT_NAMES`)의 sha256·shape·dtype.

    경로는 넣지 않는다 — 같은 내용이면 어느 캐시에서 plan을 만들어도 지문이 같아야 하고,
    plan 뒤에 캐시 내용이 바뀌면 train의 지문 비교가 그것을 잡는다.
    """
    rec = {}
    for name in SIM_INPUT_NAMES:
        path = Path(cache_dir) / f"{name}.npy"
        if not path.is_file():
            raise ContractError(f"sim 입력 파일이 없다 ({name}): {path}")
        arr = np.load(path, mmap_mode="r")
        rec[name] = {"file": path.name, "sha256": sha256_file(path),
                     "shape": [int(d) for d in arr.shape], "dtype": str(arr.dtype)}
        del arr  # Windows: mmap 핸들이 남으면 같은 파일을 다시 쓰지 못한다
    return rec


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


def atomic_write(path, write_fn, *, overwrite: bool = False) -> Path:
    """`write_fn(tmp)`이 같은 디렉터리의 임시 파일에 다 쓴 뒤에만 `path`로 드러낸다.

    도중에 죽으면 `path`는 없거나(새 파일) 이전 내용 그대로다 — 반쯤 쓴 ckpt/라벨이 다음
    실행에 "이미 존재함"이나 재개 상태로 읽히지 않는다. 임시 파일은 원래 확장자로 끝난다
    (`np.save`가 `.npy`를 덧붙이지 않게).

    overwrite=False: 하드링크로 드러낸다 — 대상이 이미 있으면 링크가 원자적으로 실패하므로
    확인과 쓰기 사이에 끼어든 파일도 덮어쓰지 않는다. overwrite=True: `os.replace`.
    """
    p = Path(path)
    if not overwrite and p.exists():
        raise ContractError(f"이미 존재한다 — 덮어쓰지 않는다: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.stem}.{os.getpid()}-{uuid.uuid4().hex[:12]}.tmp{p.suffix}")
    try:
        write_fn(tmp)
        if overwrite:
            os.replace(tmp, p)
        else:
            try:
                os.link(tmp, p)
            except FileExistsError:
                raise ContractError(f"이미 존재한다 — 덮어쓰지 않는다: {p}") from None
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return p


def write_manifest(path, manifest: dict) -> Path:
    """manifest를 JSON으로 원자적으로 쓴다. **덮어쓰지 않는다** — 이미 있으면 ContractError."""
    p = Path(path)
    if p.exists():
        raise ContractError(f"manifest가 이미 존재한다 — 덮어쓰지 않는다: {p}")
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return atomic_write(p, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_pseudo_manifest(manifest_path, *, teacher_ckpt=None) -> dict:
    """manifest가 가리키는 파일들을 다시 해싱·검증해 변조/누락을 잡는다.

    해시만이 아니라 manifest의 **주장**도 본다: H5가 고정한 AdaBN 레시피, 그리고 라벨
    통계(shape·min·max·mean)가 파일에서 다시 계산한 값과 같은지. 필드가 빠진 manifest는
    KeyError 트레이스백이 아니라 ContractError다.
    """
    manifest = load_manifest(manifest_path)
    try:
        return _verify_pseudo_manifest(manifest, teacher_ckpt)
    except (KeyError, TypeError, AttributeError) as e:
        raise ContractError(f"pseudo-label manifest에 필수 필드가 없다: {e!r}") from None


def _verify_pseudo_manifest(manifest: dict, teacher_ckpt) -> dict:
    if manifest.get("schema") != PSEUDO_SCHEMA:
        raise ContractError(
            f"manifest schema가 {PSEUDO_SCHEMA}가 아니다 — 받은 {manifest.get('schema')}")

    claims = {"x_domain": (manifest.get("x_domain"), "real"),
              "y_source": (manifest.get("y_source"), "pseudo_label"),
              "teacher.adabn_source": (manifest["teacher"].get("adabn_source"),
                                       TEACHER_ADABN_SOURCE),
              "teacher.adabn_shuffle": (manifest["teacher"].get("adabn_shuffle"),
                                        TEACHER_ADABN_SHUFFLE),
              "teacher.adabn_batch": (manifest["teacher"].get("adabn_batch"),
                                      TEACHER_ADABN_BATCH),
              "teacher.adabn_drop_last": (manifest["teacher"].get("adabn_drop_last"),
                                          TEACHER_ADABN_DROP_LAST),
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
    stats = validate_pseudo_labels(labels, manifest["labels"]["n"])
    del labels  # Windows: mmap 핸들이 남으면 같은 파일을 옮기거나 지우지 못한다
    for key, got in stats.items():
        claimed = manifest["labels"].get(key, _MISSING)
        if claimed != got:
            raise ContractError(f"manifest labels.{key}가 파일에서 다시 계산한 값과 다르다: "
                                f"{claimed!r} != {got!r}")

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


# 재현을 주장하려면 양쪽 manifest에 **비어 있지 않게** 있어야 하는 필드 — 양쪽이 똑같이
# 빠져 있으면 diff는 비지만, 무엇이 재현됐는지 말할 근거가 없다.
_REPRO_REQUIRED = ("schema", "labels.path", "labels.sha256", "labels.n", "source.sha256",
                   "teacher.sha256", "teacher.adabn_source", "teacher.adabn_shuffle",
                   "teacher.adabn_batch", "batch_size", "source_commit", "runtime.device",
                   "runtime.torch")


def require_reproduced(a: dict, b: dict) -> None:
    """a, b가 **경로만 다른 두 번의 독립 빌드**이고 내용이 같음을 강제한다.

    파일 재해시는 여기서 하지 않는다 — CLI `compare`가 먼저 양쪽을 `verify_pseudo_manifest`로
    통과시킨 뒤 부른다.
    """
    for name, m in (("A", a), ("B", b)):
        flat = _flatten(m)
        missing = [k for k in _REPRO_REQUIRED if flat.get(k, _MISSING) in (None, "", _MISSING)]
        if missing:
            raise ContractError(f"{name} manifest에 재현 판정에 필요한 필드가 없다: "
                                f"{', '.join(missing)}")
        if str(m["source_commit"]).endswith(DIRTY_SUFFIX):
            raise ContractError(f"{name} manifest의 source_commit이 dirty다: "
                                f"{m['source_commit']}")
    if Path(a["labels"]["path"]).resolve() == Path(b["labels"]["path"]).resolve():
        raise ContractError("두 manifest가 같은 labels 파일을 가리킨다 — 독립된 두 빌드가 "
                            "아니면 재현이라 부를 수 없다")
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


def check_resume_state(out, *, resume: bool) -> bool:
    """train이 GPU 락을 잡기 **전에** 부른다. 반환: 재개 상태에서 이어갈지.

    fail-closed — 어느 쪽으로도 조용히 넘어가지 않는다:
      * student manifest나 out이 이미 있으면 거부 (끝난 실행이거나 남의 파일이다)
      * --resume인데 재개 파일이 없으면 거부 (처음부터 새로 학습하지 않는다)
      * --resume이 없는데 재개 파일이 있으면 거부 (중단된 실행의 상태를 덮어쓰지 않는다)
    """
    out = Path(out)
    resume_path, manifest_path = student_resume_path(out), student_manifest_path(out)
    if manifest_path.exists():
        raise ContractError(f"학생 manifest가 이미 존재한다 (덮어쓰지 않는다): {manifest_path}")
    if out.exists():
        raise ContractError(f"--out이 이미 존재한다 (덮어쓰지 않는다): {out}")
    if resume and not resume_path.is_file():
        raise ContractError(f"--resume인데 재개 상태가 없다: {resume_path} — 새로 시작하려면 "
                            f"--resume 없이 실행할 것")
    if not resume and resume_path.exists():
        raise ContractError(f"중단된 실행의 재개 상태가 있다: {resume_path} — 이어가려면 "
                            f"--resume, 버리려면 직접 치울 것 (덮어쓰지 않는다)")
    return resume


# ── 샘플러 ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RealSamplerSpec:
    """sim 전량 + 추가분(real pseudo 또는 sim 비복원 추가 표집)을 매 라운드 섞는 샘플러 명세.

    real이 sim을 과대표집하지 않도록(사전보고 "arm 1") ``real_per_epoch <= n_sim``을 강제한다.
    대조군은 ``extra_sim_per_epoch``만큼 sim을 비복원으로 한 번 더 봐서 arm1과 라운드당 노출
    수를 맞춘다. 추가분은 `extra_sampler_seed` 스트림, 섞기는 `seed` 스트림에서 뽑는다.
    """
    n_sim: int
    n_real: int
    real_per_epoch: int
    seed: int
    extra_sim_per_epoch: int = 0
    extra_sampler_seed: int = 42

    @classmethod
    def build(cls, n_sim: int, n_real: int, real_per_epoch=None, seed: int = 42,
              extra_sim_per_epoch: int = 0, extra_sampler_seed=None):
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
        if not 0 <= extra_sim_per_epoch <= n_sim:
            raise ContractError(
                f"extra_sim_per_epoch({extra_sim_per_epoch})는 0 이상 n_sim({n_sim}) 이하여야 "
                f"한다 — 라운드 안에서 비복원 추가 표집이다")
        if extra_sim_per_epoch and real_per_epoch:
            raise ContractError("real_per_epoch와 extra_sim_per_epoch를 함께 쓸 수 없다 — "
                                "추가분은 real pseudo(arm1) 또는 sim(대조군) 한쪽이다")
        return cls(n_sim=n_sim, n_real=n_real, real_per_epoch=real_per_epoch, seed=seed,
                   extra_sim_per_epoch=extra_sim_per_epoch,
                   extra_sampler_seed=seed if extra_sampler_seed is None else extra_sampler_seed)

    @property
    def presentations_per_epoch(self) -> int:
        return self.n_sim + self.real_per_epoch + self.extra_sim_per_epoch

    def to_dict(self) -> dict:
        return {"n_sim": self.n_sim, "n_real": self.n_real,
                "real_per_epoch": self.real_per_epoch, "seed": self.seed,
                "extra_sim_per_epoch": self.extra_sim_per_epoch,
                "extra_sampler_seed": self.extra_sampler_seed,
                "presentations_per_epoch": self.presentations_per_epoch,
                "replacement": False,
                "rng": "extra: numpy.default_rng([extra_sampler_seed, epoch, 1]); "
                       "order: numpy.default_rng([seed, epoch])"}

    def epoch_indices(self, epoch: int) -> np.ndarray:
        """이 라운드의 인덱스 배열 — sim 전량 정확히 한 번 + 추가분(비복원) 한 번.

        real 추가분은 `n_sim + j`, sim 추가분은 sim 인덱스 그대로다. (spec, epoch)에 대해
        결정적이고 epoch이 다르면 (일반적으로) 다른 추가분·순서가 나온다.
        """
        extra_rng = np.random.default_rng([self.extra_sampler_seed, epoch, 1])
        parts = [np.arange(self.n_sim, dtype=np.int64)]
        if self.real_per_epoch > 0:
            chosen = extra_rng.choice(self.n_real, self.real_per_epoch, replace=False)
            parts.append(self.n_sim + chosen.astype(np.int64))
        if self.extra_sim_per_epoch > 0:
            parts.append(extra_rng.choice(self.n_sim, self.extra_sim_per_epoch,
                                          replace=False).astype(np.int64))
        rng = np.random.default_rng([self.seed, epoch])
        return rng.permutation(np.concatenate(parts)).astype(np.int64)


def exposure_plan(sampler: RealSamplerSpec, *, epochs: int, batch_size: int) -> dict:
    """노출 라운드 수와 배치 크기로 정해지는 학습량 — 세 arm의 step 수가 같다는 근거.

    라운드마다 DataLoader가 `ceil(presentations/batch)` step을 돈다 (drop_last=False).
    """
    ppr = sampler.presentations_per_epoch
    steps = math.ceil(ppr / batch_size)
    return {"exposure_rounds": epochs, "presentations_per_round": ppr,
            "steps_per_round": steps, "total_optimizer_steps": steps * epochs,
            "sim_presentations": (sampler.n_sim + sampler.extra_sim_per_epoch) * epochs,
            "real_presentations": sampler.real_per_epoch * epochs,
            "extra_sampler_seed": sampler.extra_sampler_seed}


# ── arm / student config ───────────────────────────────────

def arm_spec(arm: str) -> dict:
    try:
        return dict(ARMS[arm])
    except KeyError:
        raise ContractError(
            f"알 수 없는 arm: {arm!r} — {sorted(ARMS)} 중 하나여야 한다") from None


def student_config(arm: str, sampler: RealSamplerSpec, *, pseudo_manifest_sha256=None,
                   source_commit: str = "", sim_subset=None, sim_inputs=None) -> dict:
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
        if sampler.extra_sim_per_epoch <= 0:
            raise ContractError(
                f"{arm}은 sim_only인데 extra_sim_per_epoch가 0이다 — 대조군도 라운드마다 sim을 "
                f"추가로 봐서 arm1과 optimizer step 수를 맞춰야 한다")
    elif data == "sim_pseudo":
        if sampler.real_per_epoch <= 0:
            raise ContractError(
                f"{arm}은 sim_pseudo인데 sampler.real_per_epoch가 0 이하다: "
                f"{sampler.real_per_epoch}")
        if not pseudo_manifest_sha256:
            raise ContractError(f"{arm}은 sim_pseudo인데 pseudo_manifest_sha256이 없다")
        if sampler.extra_sim_per_epoch:
            raise ContractError(f"{arm}은 sim_pseudo인데 extra_sim_per_epoch가 0이 아니다")
    else:  # pragma: no cover - ARMS가 아닌 한 도달하지 않는다
        raise ContractError(f"알 수 없는 data 종류: {data!r}")

    exposure = exposure_plan(sampler, epochs=STUDENT_HPARAMS["epochs"],
                             batch_size=STUDENT_HPARAMS["batch_size"])
    rest = {"schema": STUDENT_SCHEMA, "arm": arm, "data": data, "seed": seed,
            **STUDENT_HPARAMS, "sampler": sampler.to_dict(), **exposure,
            "pseudo_manifest": pseudo_manifest_sha256, "source_commit": source_commit,
            "sim_subset": sim_subset, "sim_inputs": sim_inputs}
    cfg = dict(rest)
    cfg["config_fingerprint"] = fingerprint(rest)
    return cfg


_EXPOSURE_KEYS = ("exposure_rounds", "presentations_per_round", "steps_per_round",
                  "total_optimizer_steps", "sim_presentations", "real_presentations",
                  "extra_sampler_seed")

# student_config가 지문에 넣는 키 — 그 뒤에 train이 덧붙이는 out/runtime/x_domain 등은 제외.
_CONFIG_KEYS = frozenset({"schema", "arm", "data", "seed", *STUDENT_HPARAMS, "sampler",
                          *_EXPOSURE_KEYS, "pseudo_manifest", "source_commit", "sim_subset",
                          "sim_inputs"})


def _sampler_from_dict(d: dict) -> RealSamplerSpec:
    return RealSamplerSpec.build(d["n_sim"], d["n_real"], real_per_epoch=d["real_per_epoch"],
                                 seed=d["seed"],
                                 extra_sim_per_epoch=d.get("extra_sim_per_epoch", 0),
                                 extra_sampler_seed=d.get("extra_sampler_seed"))


def _require_sim_inputs(c: dict) -> None:
    rec = c.get("sim_inputs")
    ok = isinstance(rec, dict) and all(
        isinstance(rec.get(n), dict) and isinstance(rec[n].get("sha256"), str)
        and len(rec[n]["sha256"]) == 64 for n in SIM_INPUT_NAMES)
    if not ok:
        raise ContractError(f"{c.get('arm')}의 sim_inputs에 {', '.join(SIM_INPUT_NAMES)} "
                            f"sha256이 없다 — plan은 sim 캐시 해시를 기록해야 한다")


def build_student_manifest(cfg: dict, *, out, runtime: dict, optimizer_steps_done: int) -> dict:
    """학습이 끝난 뒤 쓰는 student manifest — cfg + 산출 ckpt 해시 + 실제로 돈 step 수.

    step 수가 계획과 다르면 쓰지 않는다: 중간에 멈춘 학습이 끝난 것처럼 기록되면 안 된다.
    """
    if optimizer_steps_done != cfg["total_optimizer_steps"]:
        raise ContractError(
            f"optimizer_steps_done({optimizer_steps_done})이 계획한 total_optimizer_steps("
            f"{cfg['total_optimizer_steps']})와 다르다")
    is_arm1 = cfg["data"] == "sim_pseudo"
    out = Path(out)
    return {**cfg, "out": str(out), "out_sha256": sha256_file(out),
            "optimizer_steps_done": optimizer_steps_done,
            "x_domain": "sim+real" if is_arm1 else "sim",
            "y_source": "sim_depth_gt+pseudo_label" if is_arm1 else "sim_depth_gt",
            "metric": None, "note": "judged by leaderboard only",
            # 학습 환경 — parity가 비교한다 (arm0은 CPU, arm1은 GPU 같은 차이를 잡는다)
            "runtime": runtime}


def verify_student_artifacts(manifest: dict) -> None:
    """학습 후 manifest가 가리키는 ckpt가 기록된 해시 그대로인지 재확인한다 (plan은 건너뜀)."""
    if "out" not in manifest:
        return
    out = Path(manifest["out"])
    if not out.is_file():
        raise ContractError(f"{manifest.get('arm')}의 out 파일이 없다: {out}")
    if sha256_file(out) != manifest.get("out_sha256"):
        raise ContractError(f"{manifest.get('arm')}의 out_sha256 불일치 — 학습 뒤 ckpt가 "
                            f"바뀌었다: {out}")
    if manifest.get("optimizer_steps_done") != manifest.get("total_optimizer_steps"):
        raise ContractError(f"{manifest.get('arm')}의 optimizer_steps_done이 "
                            f"total_optimizer_steps와 다르다")


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
        # 노출·step 필드가 sampler와 하이퍼파라미터에서 실제로 유도되는지 — 모든 arm이 같은
        # 거짓 값을 적으면 위의 값 비교로는 잡히지 않는다.
        try:
            want = exposure_plan(_sampler_from_dict(c["sampler"]), epochs=c["epochs"],
                                 batch_size=c["batch_size"])
        except (KeyError, TypeError) as e:
            raise ContractError(f"{c.get('arm')}의 sampler 기록이 불완전하다: {e!r}") from None
        wrong = [k for k in _EXPOSURE_KEYS if c.get(k, _MISSING) != want[k]]
        if wrong:
            raise ContractError(f"{c.get('arm')}의 {', '.join(wrong)}가 sampler에서 유도한 "
                                f"값과 다르다")
        _require_sim_inputs(c)

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
