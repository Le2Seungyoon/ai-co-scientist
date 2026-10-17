"""H7 — DANN 특징 도메인 정렬의 순수 로직 (config·GRL 스케줄·분할·매니페스트).

**이 모듈은 torch를 import하지 않는다.** 이 워크트리는 dev 그룹만 sync하므로 torch가 없다 —
`scripts/train_dann.py`가 실제 학습 루프를 갖고, 여기 있는 것은 arm A/B가 `lambda_max` 외에는
정확히 같은 코드 경로를 타는지를 **행동으로** 증명할 수 있는 부분이다(분할·배치 순서·GRL 계수·
manifest parity). torch가 필요한 여섯 함수(`grad_reverse`, `make_discriminator`,
`dann_step_losses`, `bn_buffers_snapshot`, `bn_buffers_restore`, `domain_probe_auc`)만 함수 본문
안에서 lazy import한다 — 이 파일을 import하는 것 자체가 torch를 끌어오면 안 된다(테스트가
서브프로세스로 확인한다).

## H7 사전등록 조건과의 대응 (docs/experiment/H7-dann-feature-alignment.md)

- arm A(`lambda_max=0`)와 arm B(`lambda_max=1`)의 차이는 **오직** `lambda_max`뿐이어야 한다 —
  sampler/augmentation RNG를 모델 초기화 RNG와 분리하고(`epoch_rng`), `step_plan`이 lambda와
  무관하게 만들고, `build_manifest`의 `parity_key`가 `ARM_FIELDS` 밖 필드 전부의 동일성을
  증명한다. `check_arm_parity`가 이걸 실행 전에 기계적으로 검증한다.
- 구조 회귀는 sim train만, domain 판별은 sim+real train만 본다 — real test는 절대 들어가면 안
  된다(`assert_domain_sources`). test 파일이 조용히 섞이는 것이 이 실험 전체를 무효로 만드는
  가장 값싼 실패 모드다.
- 사전등록된 sim train 138,648장은 원본 캐시(173,304장) 전체가 아니라 EXP-005 분할
  (`sem.map_level_split(case, 0.2, seed=42)`)의 train 쪽이다 — `sim_structure_split`이 그 분할을
  재현한다. domain probe의 sim 쪽은 그 분할의 **holdout**(34,656장)에서만 뽑는다
  (`sim_probe_indices`) — train 집합과 겹치면 probe가 학습에 쓰인 이미지를 다시 보게 된다.
"""
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ai_co_scientist import sem

# ── 상수 ────────────────────────────────────────────────────

# sim 캐시 전체(173,304장) 중 EXP-005 구조 분할(sem.map_level_split, val_frac=0.2, seed=42)의
# train 쪽(138,648장)만 구조 회귀에 쓴다 — 사전등록 X. seed 42는 EXP-005와 동일해야 재현된다.
SIM_SPLIT_VAL_FRAC = 0.2
SIM_SPLIT_SEED = 42
PREREG_N_SIM_TOTAL = 173_304
PREREG_N_SIM_TRAIN = 138_648
PREREG_N_SIM = PREREG_N_SIM_TRAIN  # 하위 호환 별칭 — "sim train 개수"를 가리킨다
PREREG_N_REAL = 60_664

ARMS = {"A": 0.0, "B": 1.0}  # arm → lambda_max. arm 사이에 달라도 되는 **유일한** 지식.

DOMAIN_SOURCES = ("sim_sem.npy", "real_sem.npy")  # domain 판별 입력. real test는 여기 없다.
STRUCTURE_SOURCES = ("sim_sem.npy", "sim_depth.npy", "sim_case.npy")

# manifest 키 중 arm A/B가 달라도 되는 것들 — parity_key 계산에서 제외된다. `lambda_digest`는
# lambda 배열 자체의 해시라 lambda_max가 다르면 당연히 다르다(스케줄 **공식**은 grl_schedule에).
# `source`(git HEAD·dirty 경로)도 여기 있다 — arm A와 B 사이에 docs/registry 커밋이 끼거나
# resume 전에 무관한 커밋이 생겨도 parity가 깨지면 안 된다. 코드 동일성은 parity_key 안의
# `environment.code_sha256`(파일별 해시)이 증명한다.
ARM_FIELDS = frozenset({"arm", "lambda_max", "lambda_digest", "report_id", "outputs",
                        "source"})

# 코드 fingerprint가 해시하는 파일(저장소 루트 기준). 학습 결과를 바꿀 수 있는 코드 전부 —
# git HEAD만으로는 dirty 트리의 코드를 증명하지 못한다.
CODE_PATHS = (
    "src/ai_co_scientist/dann.py", "src/ai_co_scientist/sem.py",
    "src/ai_co_scientist/adabn.py", "src/ai_co_scientist/locks.py",
    "scripts/train_dann.py", "scripts/train_structure.py", "pyproject.toml", "uv.lock",
)
# `git status`로 dirty 여부를 보는 범위 — runtime/·docs/의 변화는 학습 코드가 아니다.
CODE_STATUS_SCOPE = ("src", "scripts", "pyproject.toml", "uv.lock")

# epoch_rng의 스트림 이름 → 고정 정수. **Python hash()를 쓰지 않는다** — 프로세스마다 salt가
# 달라 재현 불가능해진다(PYTHONHASHSEED 미고정 시).
_STREAM_IDS = {"sim": 0, "real": 1, "probe": 2}

_ALLOWED_SOURCE_NAMES = frozenset(STRUCTURE_SOURCES) | frozenset(DOMAIN_SOURCES)


# ── 설정 ────────────────────────────────────────────────────

@dataclass(frozen=True)
class DannConfig:
    """arm 하나의 전체 학습 설정. `for_arm`으로만 만든다 — `lambda_max`를 손으로 못 바꾸게 막는다."""

    arm: str
    lambda_max: float
    seed: int = 42
    split_seed: int = 0
    probe_frac: float = 0.2
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 10.0
    arch: str = "mlp"
    width: int = 32
    feature_dim: int = 128
    disc_hidden: int = 256
    schedule: str = "cosine"
    augmentation: str = "none"
    amp: bool = False

    @classmethod
    def for_arm(cls, arm: str, **overrides) -> "DannConfig":
        if arm not in ARMS:
            raise ValueError(f"알 수 없는 arm: {arm!r} (허용: {sorted(ARMS)})")
        if "lambda_max" in overrides:
            raise ValueError("lambda_max는 arm이 결정한다 — overrides로 줄 수 없다 (H7 parity 조건)")
        return cls(arm=arm, lambda_max=ARMS[arm], **overrides)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_arm(arm: str, lambda_max: float) -> None:
    """`arm`과 `lambda_max`가 `ARMS`와 정확히 일치하는지 — CLI가 `--arm`/`--lambda-max`를 따로
    받는다면(AMENDMENT 2), 둘이 어긋나는 조합(가령 arm B에 lambda_max 0.0)을 실행 직전에
    잡아야 한다. `for_arm`은 lambda_max 자체를 못 받게 막지만, CLI는 재확인용으로 값을
    명시적으로 받으므로 검증 지점이 따로 필요하다."""
    if arm not in ARMS:
        raise ValueError(f"알 수 없는 arm: {arm!r} (허용: {sorted(ARMS)})")
    if lambda_max != ARMS[arm]:
        raise ValueError(
            f"arm {arm!r}의 lambda_max는 {ARMS[arm]}이어야 한다 — 받은 값 {lambda_max}")


# ── GRL 스케줄 ──────────────────────────────────────────────

def ganin_lambda(progress: float, lambda_max: float, gamma: float = 10.0) -> float:
    """Ganin schedule: `lambda_max * (2/(1+exp(-gamma*p)) - 1)`, `p`는 [0,1]로 클램프.

    p=0에서, 그리고 lambda_max=0이면 항상 정확히 0.0 — arm A(대조군)가 GRL을 완전히 죽인다.
    """
    p = min(1.0, max(0.0, float(progress)))
    return float(lambda_max) * (2.0 / (1.0 + np.exp(-float(gamma) * p)) - 1.0)


def progress(step: int, total_steps: int) -> float:
    """전역 진행률 `step/total_steps`. `step`은 `[0, total_steps)` 안이어야 한다 — 그 밖의
    값은 스텝 카운팅이 어긋났다는 뜻이라 조용히 클램프하지 않고 raise한다(클램프는
    `ganin_lambda` 안, 진행률이 유효하다는 전제 위에서만 한다)."""
    if not (0 <= step < total_steps):
        raise ValueError(f"step은 [0, {total_steps})여야 한다 — 받은 값 {step}")
    return step / total_steps


def lambda_schedule(total_steps: int, lambda_max: float, gamma: float = 10.0) -> np.ndarray:
    """`total_steps`개 스텝 전체에 대한 GRL 계수 배열(float64) — `dann_step_losses`가 매 스텝
    쓸 `lam`을 학습 시작 전에 통째로 고정한다(재현성: 스텝 i의 lam이 실행마다 같다)."""
    return np.array(
        [ganin_lambda(progress(i, total_steps), lambda_max, gamma) for i in range(total_steps)],
        dtype=np.float64)


# ── 분할 ────────────────────────────────────────────────────

def sim_structure_split(case: np.ndarray) -> np.ndarray:
    """sim 캐시 전체에서 구조 회귀용 **train** 마스크(이미지 단위) — EXP-005 분할의 재현.

    `sem.map_level_split`은 val 마스크를 주므로 여기서는 그 보수를 train으로 쓴다. depth-map
    쌍(2k,2k+1)은 그 함수가 이미 쌍 단위로 묶어 반환하므로 여기서 다시 처리하지 않는다.
    """
    return ~sem.map_level_split(case, SIM_SPLIT_VAL_FRAC, SIM_SPLIT_SEED)


def real_domain_split(site: np.ndarray, y: np.ndarray, probe_frac: float = 0.2,
                      seed: int = 0) -> np.ndarray:
    """real train을 site 단위로 층화 분할한 이미지 단위 **probe** 마스크.

    `sem.site_split`을 그대로 위임한다 — 사이트 단위 홀드아웃 로직을 두 번 짜면 한쪽만
    바뀌는 드리프트가 생긴다(`coding-patterns.md` → 두 번째 사본이 추출 지점).
    """
    return sem.site_split(site, y, probe_frac, seed)


def epoch_rng(seed: int, stream: str, epoch: int) -> np.random.Generator:
    """`(seed, stream, epoch)`에서 결정적이고 스트림끼리 독립인 RNG.

    Stateless: resume이 RNG 상태를 들고 다닐 필요가 없다 — epoch만 알면 그 epoch의 배치 순서를
    처음부터 재계산할 수 있다. 모델 초기화 RNG(torch)와는 별도 축이라 절대 섞이지 않는다.
    """
    if stream not in _STREAM_IDS:
        raise ValueError(f"알 수 없는 stream: {stream!r} (허용: {sorted(_STREAM_IDS)})")
    ss = np.random.SeedSequence(int(seed), spawn_key=(_STREAM_IDS[stream], int(epoch)))
    return np.random.default_rng(ss)


def sim_batches(sim_idx: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """한 epoch의 sim 배치 계획. `(len//batch_size, batch_size)`, 순열, drop_last."""
    perm = rng.permutation(np.asarray(sim_idx))
    n_batches = len(perm) // batch_size
    return perm[: n_batches * batch_size].reshape(n_batches, batch_size)


def real_batches(real_idx: np.ndarray, batch_size: int, n_steps: int,
                 rng: np.random.Generator) -> np.ndarray:
    """한 epoch의 real 배치 계획. `(n_steps, batch_size)` — real이 sim보다 적을 수 있으므로
    소진되면 **같은 rng**에서 새 순열을 뽑아 이어 붙인다(사이클마다 전부 한 번씩 나온 뒤 반복)."""
    real_idx = np.asarray(real_idx)
    total = n_steps * batch_size
    out = np.empty(total, dtype=real_idx.dtype)
    filled = 0
    while filled < total:
        perm = rng.permutation(real_idx)
        take = min(len(perm), total - filled)
        out[filled:filled + take] = perm[:take]
        filled += take
    return out.reshape(n_steps, batch_size)


def step_plan(n_sim_idx: np.ndarray, real_idx: np.ndarray, cfg: DannConfig,
             epoch: int) -> tuple[np.ndarray, np.ndarray]:
    """한 epoch의 `(sim_batches, real_batches)`. `cfg.lambda_max`/`cfg.arm`과 무관해야 한다 —
    arm A/B가 정확히 같은 스텝 순서로 학습한다는 H7 parity 조건."""
    sim_rng = epoch_rng(cfg.seed, "sim", epoch)
    sb = sim_batches(n_sim_idx, cfg.batch_size, sim_rng)
    real_rng = epoch_rng(cfg.seed, "real", epoch)
    rb = real_batches(real_idx, cfg.batch_size, sb.shape[0], real_rng)
    return sb, rb


def sim_probe_indices(holdout_mask: np.ndarray, case: np.ndarray, real_probe_groups: np.ndarray,
                      seed: int = 0) -> np.ndarray:
    """domain probe의 sim 쪽 표본 — **구조 회귀 holdout**에서, real probe와 **그룹별로 대칭**되게
    뽑는다(AMENDMENT 2 §sim_probe_indices).

    `real_probe_groups`는 real probe 이미지들의 그룹(`sem.GROUPS` 인덱스, 0..3)이다. 그룹 g는
    같은 배경 레벨 L을 공유하는 sim Case g+1과 1:1 대응한다(`sem.CASE_LEVEL`). 그룹별로
    `2*floor(count_g/2)`장을 그 그룹의 sim holdout에서 depth-map 쌍째로 뽑는다 — 그룹을 안 갈라
    한 덩어리로 뽑으면 레벨이 쏠린 구성으로 probe AUC를 재는 것이라 domain 판별이 레벨 신호에
    편승할 수 있다.
    """
    mask = np.asarray(holdout_mask, dtype=bool)
    case = np.asarray(case)
    if len(mask) != len(case):
        raise ValueError(f"holdout_mask와 case 길이가 다르다: {len(mask)} vs {len(case)}")
    if len(mask) % 2 != 0:
        raise ValueError(f"holdout_mask 길이가 짝수가 아니다(depth-map 쌍 가정 위반): {len(mask)}")
    if not np.array_equal(mask[::2], mask[1::2]):
        raise ValueError("holdout_mask가 depth-map 쌍 경계와 어긋난다 — 쌍이 함께 hold되지 않았다")
    if not np.array_equal(case[::2], case[1::2]):
        raise ValueError("case가 depth-map 쌍 경계와 어긋난다 — 쌍이 같은 Case를 공유하지 않는다")

    pair_mask, pair_case = mask[::2], case[::2]
    groups, counts = np.unique(np.asarray(real_probe_groups), return_counts=True)
    count_by_group = dict(zip(groups.tolist(), counts.tolist()))

    rng = np.random.default_rng(seed)
    chosen_pairs = []
    for g in range(len(sem.GROUPS)):
        n_pairs = count_by_group.get(g, 0) // 2
        if n_pairs == 0:
            continue
        pool = np.where(pair_mask & (pair_case == g + 1))[0]
        if n_pairs > len(pool):
            raise ValueError(
                f"그룹 {g}(Case {g + 1})의 sim holdout 풀이 부족하다 — "
                f"{n_pairs}쌍 요청, {len(pool)}쌍 가용")
        chosen_pairs.append(rng.choice(pool, n_pairs, replace=False))

    pairs = (np.concatenate(chosen_pairs) if chosen_pairs else np.array([], dtype=np.int64))
    idx = np.concatenate([2 * pairs, 2 * pairs + 1]).astype(np.int64)
    return np.sort(idx)


def probe_order(sim_probe_idx, real_probe_idx, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """두 probe를 **한 번** 섞은 고정 순서. 정렬된 채로 배치하면 real 배치 하나가 glob 순서상
    사이트 4개 남짓, sim 배치 하나가 Case 하나로 채워진다 — BN batch-stat 모드의 probe에서는
    배치 구성이 특징을 바꾸므로 AUC가 도메인이 아니라 배치 구성을 재게 된다. 학습 배치처럼
    무작위로 섞되 `probe` 스트림으로 고정해 매 epoch·두 arm이 같은 배치를 본다."""
    rng = epoch_rng(seed, "probe", 0)
    return rng.permutation(np.asarray(sim_probe_idx)), rng.permutation(np.asarray(real_probe_idx))


def probe_batches(idx: np.ndarray, batch_size: int) -> list:
    """`idx`를 주어진 순서 그대로 `batch_size` 단위 연속 조각으로 나눈다 — 인덱스를 하나도
    버리지 않는다.

    `batch_size`보다 짧은 꼬리는 **바로 앞 배치에 합친다**(별도 배치로 남기지 않는다) — encoder
    BN을 train 모드로 돌리는 probe forward(AMENDMENT 1 §5)에서 배치 크기 1은
    `BatchNorm.forward`가 raise한다. 전체 길이가 `batch_size` 이하면 배치 하나.
    """
    idx = np.asarray(idx)
    n = len(idx)
    if n < 2:
        raise ValueError(f"probe 배치는 인덱스가 최소 2개 있어야 한다 — 받은 개수 {n}")
    if n <= batch_size:
        return [idx.copy()]
    n_batches = n // batch_size
    batches = [idx[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]
    tail = idx[n_batches * batch_size:]
    if len(tail) > 0:
        batches[-1] = np.concatenate([batches[-1], tail])
    return batches


def assert_disjoint(a, b, what: str) -> None:
    """`a`와 `b`의 인덱스 집합이 겹치면 `what`을 이름 붙여 raise — probe/train 누수를 실행 전에
    잡는 마지막 그물이다(`sim_probe_indices`/`real_domain_split`이 이미 분리하지만, 호출부가
    잘못된 배열을 넘기는 실수까지는 못 막는다)."""
    inter = set(np.asarray(a).tolist()) & set(np.asarray(b).tolist())
    if inter:
        raise ValueError(f"{what} — 서로 겹치는 인덱스가 있다: {sorted(inter)}")


# ── 지표 ────────────────────────────────────────────────────

def roc_auc(scores, labels) -> float:
    """Mann-Whitney U를 통한 ROC AUC, 동순위는 평균 순위로 처리. 순수 numpy(scikit-learn 없음).

    한쪽 클래스가 없으면 정의되지 않으므로 raise — 무음으로 0.5나 nan을 돌려주면 domain probe가
    "정렬됐다"로 오독될 수 있다.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc는 두 클래스가 모두 있어야 한다")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks_sorted[i:j + 1] = ranks_sorted[i:j + 1].mean()
        i = j + 1
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ranks_sorted

    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ── 무결성 가드 ─────────────────────────────────────────────

def assert_domain_sources(names) -> None:
    """domain 판별 입력이 `DOMAIN_SOURCES`(sim_sem/real_sem)뿐인지 — real test나 구조 전용
    소스(sim_depth/sim_case)가 섞이면 실험이 조용히 무효가 된다."""
    bad = [n for n in names if n not in DOMAIN_SOURCES]
    if bad:
        raise ValueError(f"domain 학습에 허용되지 않은 source: {bad} (허용: {DOMAIN_SOURCES})")


def check_counts(n_sim_total: int, n_sim_train: int, n_real: int) -> None:
    """사전등록된 캐시 개수와 정확히 같은지. 하나라도 다르면 캐시가 갱신됐거나 분할이 어긋난 것."""
    expected = (PREREG_N_SIM_TOTAL, PREREG_N_SIM_TRAIN, PREREG_N_REAL)
    got = (n_sim_total, n_sim_train, n_real)
    if got != expected:
        raise ValueError(
            f"사전등록 개수와 다르다 — sim_total {n_sim_total}(기대 {PREREG_N_SIM_TOTAL}), "
            f"sim_train {n_sim_train}(기대 {PREREG_N_SIM_TRAIN}), "
            f"real {n_real}(기대 {PREREG_N_REAL})")


def file_fingerprint(path) -> dict:
    """`{"name", "size_bytes", "sha256"}` — 1 MiB 청크 스트리밍(캐시 파일이 메모리보다 클 수 있다)."""
    p = Path(path)
    h = hashlib.sha256()
    size = 0
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return {"name": p.name, "size_bytes": size, "sha256": h.hexdigest()}


def array_digest(*arrays) -> str:
    """배열들의 dtype+shape+bytes에 대한 sha256 — split/plan 마스크를 manifest에 넣을 다이제스트."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(str(a.dtype).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def code_fingerprint(root, paths=CODE_PATHS, status_scope=CODE_STATUS_SCOPE,
                     run=subprocess.run) -> dict:
    """학습 코드의 fingerprint — git HEAD + 범위 안의 dirty 경로 + 코드 파일별 sha256.

    HEAD만 기록하면 커밋 안 된 수정으로 돈 실행이 그 커밋의 결과로 둔갑한다. 파일 해시는 git이
    없거나 실패해도 남으므로 두 arm의 코드가 같았다는 증거는 git 없이도 선다. `run`은 테스트가
    git을 대체하는 주입점이다.
    """
    root = Path(root)

    def git(*args):
        try:
            return run(["git", *args], cwd=root, capture_output=True, text=True,
                       check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all", "--", *status_scope)
    dirty_paths = sorted(line[3:] for line in (status or "").splitlines() if line.strip())
    return {
        "git_commit": head.strip() if head else "unknown",
        "git_status_ok": status is not None,
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "files_sha256": {rel: file_fingerprint(root / rel)["sha256"] for rel in paths},
    }


def assert_clean_code(fp: dict) -> None:
    """실제 학습은 커밋된 깨끗한 코드에서만 — registry의 source_commit이 사실이어야 한다."""
    if fp["git_commit"] == "unknown" or not fp["git_status_ok"]:
        raise RuntimeError("git HEAD/status를 읽지 못했다 — source commit을 증명할 수 없다")
    if fp["dirty"]:
        raise RuntimeError(f"커밋 안 된 학습 코드 변경이 있다: {fp['dirty_paths']}")


# ── manifest ────────────────────────────────────────────────

def _strip_arm_fields(manifest: dict) -> dict:
    """parity_key 계산용 — ARM_FIELDS(+ config 안의 lambda_max/arm) + parity_key 자체를 뺀 사본."""
    stripped = {k: v for k, v in manifest.items() if k not in ARM_FIELDS and k != "parity_key"}
    cfg = dict(stripped.get("config", {}))
    cfg.pop("lambda_max", None)
    cfg.pop("arm", None)
    stripped["config"] = cfg
    return stripped


def _parity_key(manifest: dict) -> str:
    canon = json.dumps(_strip_arm_fields(manifest), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def run_contract(cfg: DannConfig) -> dict:
    """arm과 무관한 실행 계약 — GRL/LR 스케줄 공식, epoch/step 의미, RNG 스트림, probe 정책,
    overwrite 정책. manifest와 `--dry-plan`이 이 함수 하나를 공유하므로 실행 전에 본 계약과
    실행이 기록한 계약이 다를 수 없다. parity_key 안에 들어간다."""
    return {
        "grl_schedule": {
            "formula": "lambda = lambda_max * (2 / (1 + exp(-gamma * p)) - 1)",
            "progress": "p = global_step / total_steps, global_step in [0, total_steps)",
            "granularity": "one value per optimizer step, precomputed before training",
            "gamma": cfg.gamma,
        },
        "lr_schedule": {
            "optimizer": "AdamW(model + discriminator, one param group, torch defaults)",
            "lr": cfg.lr,
            "scheduler": "CosineAnnealingLR",
            "t_max": cfg.epochs,
            "eta_min": 0.0,
            "step": "once per epoch, after the epoch's last optimizer step",
        },
        "epoch_semantics": {
            "steps_per_epoch": "floor(n_sim_train / batch_size), drop_last",
            "step": "one sim batch and one real batch, forwarded separately (never concatenated)",
            "real_batches": "real domain indices cycled, reshuffled from the same stream per cycle",
            "loss": "l1(structure, sim only) + mean(bce_sim, bce_real)",
        },
        "rng": {
            "sampler": "numpy SeedSequence(seed, spawn_key=(stream_id, epoch)), stateless",
            "streams": dict(_STREAM_IDS),
            "model_init": "seed_everything(seed), then make_model, then make_discriminator",
            "sim_structure_split": (f"sem.map_level_split(val_frac={SIM_SPLIT_VAL_FRAC}, "
                                    f"seed={SIM_SPLIT_SEED}) complement"),
            "real_probe_split": f"sem.site_split(val_frac={cfg.probe_frac}, seed={cfg.split_seed})",
            "sim_probe": f"numpy default_rng({cfg.split_seed}) over sim holdout pairs by group",
            "probe_order": "epoch_rng(seed, 'probe', 0): one permutation of sim, then of real",
        },
        "probe": {
            "sim": "structure-split holdout pairs, per-group counts matched to the real probe",
            "real": "site-level held-out real train sites",
            "forward": ("encoder BN in batch-stat mode, other modules eval, one domain per "
                        "batch in a fixed shuffled order, short tail merged; BN buffers "
                        "restored afterwards"),
            "when": "after every epoch; logged only, never used for selection",
        },
        "overwrite_policy": (
            "fresh run refuses if any output exists; --resume needs the manifest and refuses "
            "if ckpt/disc_ckpt/result_json exist; manifest/result_json/ckpt/disc_ckpt are "
            "create-once; resume_ckpt is replaced atomically once per epoch"),
    }


def build_manifest(cfg: DannConfig, *, report_id: str, sources: list, n_sim_train: int,
                   n_real_domain: int, n_real_probe: int, n_sim_probe: int, split_digest: str,
                   plan_digest: str, lambda_digest: str, steps_per_epoch: int, outputs: dict,
                   optimizer_signature: dict, environment: dict,
                   source: dict | None = None) -> dict:
    """arm 하나의 실행 계약 전체를 담는 manifest.

    `optimizer_signature`/`environment`는 **parity_key 안에** 들어간다 — 두 arm이 파라미터
    그룹 구성이나 실행 환경(torch/cuda 버전, 디바이스)까지 같다는 것을 기계적으로 증명해야
    "차이는 lambda_max뿐"이라는 H7 조건이 선다.
    """
    for s in sources:
        if s["name"] not in _ALLOWED_SOURCE_NAMES:
            raise ValueError(
                f"허용되지 않은 source: {s['name']} (허용: {sorted(_ALLOWED_SOURCE_NAMES)})")

    manifest = {
        "hypothesis": "H7",
        "report_id": report_id,
        "arm": cfg.arm,
        "lambda_max": cfg.lambda_max,
        "config": cfg.to_dict(),
        "x_domain": {"structure": "sim", "domain": ["sim", "real_train"]},
        "y_source": {"structure": "sim_depth_gt", "domain": "domain_label(sim=0,real=1)"},
        "metric": {"name": "leaderboard_rmse", "x_domain": "real", "y_source": "real_depth_gt"},
        "mechanism_metric": {
            "name": "domain_probe_auc",
            "x_domain": "sim+real_train_probe_sites",
            "y_source": "domain_label",
            "note": "리더보드 판정을 대체하지 않는다",
            "not_for_selection": True,
            "epoch": "final (all epochs logged)",
        },
        "checkpoint_selection": "final_epoch (no holdout selection)",
        "counts": {
            "n_sim_train": n_sim_train,
            "n_real_domain": n_real_domain,
            "n_real_probe": n_real_probe,
            "n_sim_probe": n_sim_probe,
        },
        "sources": sources,
        "split_digest": split_digest,
        "plan_digest": plan_digest,
        "lambda_digest": lambda_digest,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * cfg.epochs,
        "outputs": outputs,
        "optimizer_signature": optimizer_signature,
        "environment": environment,
        **run_contract(cfg),
        "source": source,
    }
    # JSON 왕복으로 정규화한다 — 메모리 manifest의 tuple과 파일에서 읽은 list가 `!=`로 갈려
    # 같은 실행을 parity 불일치로 판정하는 일을 막는다(`--parity-with`가 정확히 그 비교다).
    manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
    manifest["parity_key"] = _parity_key(manifest)
    return manifest


def _diff_paths(a: dict, b: dict, prefix: str = "") -> list:
    """두 dict가 다른 키의 dotted-path 목록. 값이 둘 다 dict면 재귀적으로 파고든다."""
    diffs = []
    for k in sorted(set(a) | set(b)):
        path = f"{prefix}.{k}" if prefix else k
        if k not in a or k not in b:
            diffs.append(path)
            continue
        va, vb = a[k], b[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(_diff_paths(va, vb, path))
        elif va != vb:
            diffs.append(path)
    return diffs


def check_arm_parity(man_a: dict, man_b: dict) -> None:
    """arm A/B manifest가 `ARM_FIELDS`(와 config.arm/lambda_max) 말고는 완전히 같은지 검증.

    parity_key 일치만으로도 이미 함의되지만, 해시 하나에 기대는 대신 실제 차이 나는 키를
    나열해 raise한다 — 실행 전 중단 시 무엇이 달랐는지 사람이 바로 읽을 수 있어야 한다.
    `outputs`처럼 ARM_FIELDS에 속한 top-level 키는 **내부까지 통째로** 허용된다 — 체크포인트
    경로가 arm마다 다른 건 당연하다. `_strip_arm_fields`가 parity_key 계산과 똑같은 방식으로
    그 키들을 걷어내므로 두 검사가 같은 정의를 공유한다(따로 만들면 한쪽만 갱신되는 드리프트).
    """
    arm_a, arm_b = man_a.get("arm"), man_b.get("arm")
    if arm_a == arm_b:
        raise ValueError(f"두 manifest의 arm이 같다({arm_a!r}) — A/B 비교가 아니다")

    diffs = _diff_paths(_strip_arm_fields(man_a), _strip_arm_fields(man_b))
    parity_ok = man_a.get("parity_key") == man_b.get("parity_key")
    if diffs or not parity_ok:
        raise ValueError(f"arm 간 parity 불일치 — ARM_FIELDS 밖에서 달라진 키: {sorted(diffs)}")



def check_manifest_files(path_a, path_b) -> None:
    """두 arm의 manifest 파일을 읽어(변조 검사 포함) `check_arm_parity`로 대조한다 — 실행 전
    중단 조건("arm 사이의 manifest·seed·학습 step·optimizer 상태가 달라졌으면 폐기")의 파일 단위
    진입점. `train_dann.py check-parity`와 `--parity-with`가 이것을 부른다."""
    check_arm_parity(read_manifest(path_a), read_manifest(path_b))


def _tmp_path(p: Path) -> Path:
    """게시 전 임시 파일. pid를 붙여 동시에 도는 두 프로세스의 임시 파일이 겹치지 않게 한다."""
    return p.with_name(f".{p.name}.{os.getpid()}.tmp")


def publish_once(path, write_fn) -> Path:
    """`write_fn(tmp)`로 임시 파일을 **끝까지** 쓴 뒤 `path`에 한 번만, 원자적으로 게시한다.

    이미 있으면 `FileExistsError` — 쓰기 전에 한 번, 게시 순간에 한 번 확인한다. 게시는
    `os.link`(원자적이고 배타적: 대상이 있으면 실패)다. 하드링크를 못 거는 파일시스템에서만
    존재 재확인 후 `os.replace`로 물러선다 — 그 틈의 경합은 호출부가 쥔 `gpu-0` 락이 막는다.
    크래시가 나도 `path`에는 반쯤 쓰인 파일이 남지 않는다(임시 파일만 남는다).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"이미 존재한다(한 번만 쓰는 산출물): {p}")
    tmp = _tmp_path(p)
    try:
        write_fn(tmp)
        try:
            os.link(tmp, p)
        except FileExistsError:
            raise
        except OSError:
            if p.exists():
                raise FileExistsError(f"이미 존재한다(한 번만 쓰는 산출물): {p}") from None
            os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
    return p


def replace_atomic(path, write_fn) -> Path:
    """`write_fn(tmp)` 후 `os.replace` — 재개 체크포인트처럼 매 epoch 갱신되는 파일용. 쓰는
    도중 죽어도 직전 epoch의 완전한 파일이 남는다(`torch.save`를 제자리에 하면 잘린 파일이
    남아 resume 자체가 불가능해진다)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(p)
    try:
        write_fn(tmp)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
    return p


def write_json_exclusive(path, obj) -> Path:
    """`obj`를 정준 JSON(sort_keys, ensure_ascii=False, indent=2)으로 **불변** 기록한다 —
    경로가 이미 있으면 `FileExistsError`가 그대로 전파된다(재시도하지 않는다). manifest와 결과
    JSON이 둘 다 이 함수 하나를 쓴다 — "한 번 쓰면 안 바뀐다"는 보장이 실행 계약 파일 전부에
    같은 강도로 걸려야 한다. 원자성은 `publish_once`에 위임한다.

    바이트로 쓴다(`write_bytes`) — 텍스트 모드는 Windows에서 `\n`을 `\r\n`으로 바꿔
    저장된 바이트가 `json.dumps`의 결과와 달라진다.
    """
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    return publish_once(path, lambda tmp: Path(tmp).write_bytes(data))


def write_manifest(path, manifest: dict) -> Path:
    """manifest를 불변 기록한다. 학습 시작 전에 이미 있는 manifest는 실수로 덮어쓰면 안 되는
    실행 계약이다 — `write_json_exclusive`에 위임(원자성 로직은 한 곳)."""
    return write_json_exclusive(path, manifest)


def output_paths(out) -> dict:
    """`out`(구조 회귀 ckpt 경로, `.pt`)에서 이 실행의 5개 출력 경로를 전부 유도한다.

    `ckpt`(=out)만 `train_structure.load_model`/`infer_decomposed`가 읽는 추론 계약이고, 나머지
    넷(`disc_ckpt`, `resume_ckpt`, `manifest`, `result_json`)은 학습 부산물이다 — 이름을 여기
    한 곳에서만 정하면 `check_output_policy`와 스크립트가 서로 다른 경로를 계산해 어긋나는
    사고가 안 난다.
    """
    p = Path(out)
    if p.suffix != ".pt":
        raise ValueError(f"out 확장자는 .pt여야 한다 — 받은 값 {out}")
    base = p.parent / p.stem
    return {
        "ckpt": p,
        "disc_ckpt": base.parent / f"{base.name}.disc.pt",
        "resume_ckpt": base.parent / f"{base.name}.resume.pt",
        "manifest": base.parent / f"{base.name}.manifest.json",
        "result_json": base.parent / f"{base.name}.result.json",
    }


def check_output_policy(paths: dict, resume: bool) -> None:
    """학습 시작 전 출력 경로 상태를 검증한다.

    새 실행(`resume=False`)은 5개 경로 중 하나라도 있으면 실행하지 않는다 — 사고로 다른
    실행의 산출물을 덮어쓰는 것을 막는다. `--resume`은 **끝난** 실행을 다시 돌리는 게 아니라
    **중단된** 실행을 이어가는 것이므로: manifest는 반드시 있어야 하고(이미 이 실행이 시작은
    했다는 증거), 완료 표식인 `result_json`은 절대 있으면 안 된다(가장 마지막에 쓰인다 — 있으면
    이미 끝났다는 뜻). `ckpt`/`disc_ckpt`는 마지막 epoch 뒤 게시 도중 죽은 경우에만 있을 수
    있고, 그때는 스크립트가 재개점이 마지막 epoch인지 확인한 뒤 이미 게시된 파일을 건너뛴다
    (`--resume` 없이는 어떤 산출물도 있으면 안 된다). `resume_ckpt`는 있어도 없어도 된다.
    """
    if not resume:
        existing = [str(v) for v in paths.values() if Path(v).exists()]
        if existing:
            raise FileExistsError(f"출력 경로가 이미 존재한다(새 실행이어야 한다): {existing}")
        return
    if not Path(paths["manifest"]).exists():
        raise FileNotFoundError(f"--resume인데 manifest가 없다: {paths['manifest']}")
    if Path(paths["result_json"]).exists():
        raise FileExistsError(f"이미 끝난 실행은 resume하지 않는다: {paths['result_json']}")


def read_manifest(path) -> dict:
    """manifest를 읽고 저장된 `parity_key`가 재계산과 일치하는지 확인 — 안 맞으면 변조로 본다."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = manifest.get("parity_key")
    recomputed = _parity_key(manifest)
    if stored != recomputed:
        raise ValueError(f"manifest 변조 감지 — parity_key 불일치: {path}")
    return manifest


# ── torch-lazy ──────────────────────────────────────────────

def grad_reverse(x, lam: float):
    """Gradient Reversal Layer — forward는 identity, backward는 `-lam` 배. `autograd.Function`을
    함수 본문 안에서 만들어 이 모듈이 torch를 top-level에서 요구하지 않게 한다."""
    import torch

    class _GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.lam, None

    return _GradReverse.apply(x, lam)


def make_discriminator(feature_dim: int = 128, hidden: int = 256):
    """domain discriminator: `Linear→ReLU→Linear→ReLU→Linear(1)`. BatchNorm을 넣지 않는다 —
    sim/real 배치를 분리 순전파하는데 BN이 있으면 배치 통계 자체가 도메인을 누설한다."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(feature_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def dann_step_losses(model, disc, x_sim, s_sim, x_real, lam: float) -> dict:
    """한 스텝의 `{l1, domain, total}` 손실.

    sim/real을 **절대 concat하지 않는다** — 합쳐서 한 번에 순전파하면 BN이 있는 backbone에서
    배치 통계가 두 도메인을 섞어버린다. 지금 encoder(PlainMLP)부터 이미 BatchNorm1d를 7개 갖고
    있어 이 위험은 가정이 아니라 현재 경로의 사실이다 — 이 계약은 backbone을 바꿔도 유지된다.
    `lam=0`이면 `grad_reverse`의 backward가 0을 곱해 encoder로 가는 domain
    gradient만 정확히 0이 된다 — disc 자신의 gradient는 살아 있다(두 arm이 같은 optimizer
    상태 shape를 갖는다는 조건).
    """
    import torch
    import torch.nn.functional as F

    b_sim = x_sim.shape[0]
    b_real = x_real.shape[0]

    f_sim = model.encoder(x_sim.reshape(b_sim, -1))
    s_hat = model.out(model.decoder(f_sim)).view_as(s_sim)
    l1 = F.l1_loss(s_hat, s_sim)

    f_real = model.encoder(x_real.reshape(b_real, -1))

    d_sim = disc(grad_reverse(f_sim, lam))
    d_real = disc(grad_reverse(f_real, lam))
    domain_sim = F.binary_cross_entropy_with_logits(d_sim, torch.zeros_like(d_sim))
    domain_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    domain = (domain_sim + domain_real) / 2

    return {"l1": l1, "domain": domain, "total": l1 + domain}


def bn_buffers_snapshot(model) -> dict:
    """BN 층 이름 → `{running_mean, running_var, num_batches_tracked}` 텐서 스냅샷.

    domain probe를 encoder BN **train 모드**로 돌리면 그 forward가 running stat을 바꾼다 —
    probe는 관찰 전용이라 모델을 건드리면 안 되므로, 돌기 전 스냅샷 → 돈 뒤 복원한다.

    BN 층 목록은 `adabn.bn_modules`에 위임한다 — BN 타입 튜플을 여기서 다시 선언하면 한쪽만
    갱신되는 드리프트가 생긴다(`coding-patterns.md` → 두 번째 사본이 추출 지점). 이 import도
    함수 본문 안에서만 일어나므로 이 모듈이 torch를 top-level로 끌어오는 일은 없다.
    """
    from ai_co_scientist.adabn import bn_modules

    return {
        name: {
            "running_mean": m.running_mean.clone(),
            "running_var": m.running_var.clone(),
            "num_batches_tracked": (m.num_batches_tracked.clone()
                                    if m.num_batches_tracked is not None else None),
        }
        for name, m in bn_modules(model)
    }


def bn_buffers_restore(model, snap: dict) -> None:
    """`bn_buffers_snapshot`이 찍은 값으로 BN 버퍼를 제자리로 되돌린다."""
    modules = dict(model.named_modules())
    for name, buf in snap.items():
        m = modules[name]
        m.running_mean.copy_(buf["running_mean"])
        m.running_var.copy_(buf["running_var"])
        if buf["num_batches_tracked"] is not None and m.num_batches_tracked is not None:
            m.num_batches_tracked.copy_(buf["num_batches_tracked"])


def domain_probe_auc(model, disc, sources, batch_size: int, to_tensor) -> float:
    """고정 domain probe의 AUC — 기전 관찰값, 체크포인트 선택에는 쓰지 않는다.

    `sources`는 `[(array, idx, label), ...]`(sim=0, real=1)이고 `to_tensor(array, chunk)`가
    디바이스 위 `(B,1,H,W)` 텐서를 만든다. 두 도메인을 **따로** 배치한다 — 섞으면 BN 배치
    통계가 도메인을 섞는다. encoder BN만 batch-stat(train) 모드로 두고 나머지는 eval이다
    (dropout 등 확률적 층이 probe를 흔들지 못하게). 짧은 꼬리는 `probe_batches`가 앞 배치에
    합쳐 BN이 크기 1 배치를 보지 않는다. BN 버퍼와 두 모듈의 train/eval 상태는 예외가 나도
    `finally`에서 되돌린다 — probe가 학습 상태를 바꾸면 arm 사이 parity가 probe 구현에 의존하게
    된다.
    """
    import torch

    from ai_co_scientist.adabn import bn_modules

    snap = bn_buffers_snapshot(model)
    modes = (model.training, disc.training)
    scores, labels = [], []
    try:
        model.eval()
        for _, m in bn_modules(model):
            m.train()
        disc.eval()
        with torch.no_grad():
            for arr, idx, label in sources:
                for chunk in probe_batches(idx, batch_size):
                    x = to_tensor(arr, chunk)
                    f = model.encoder(x.reshape(x.shape[0], -1))
                    scores.append(disc(f).float().cpu().numpy().ravel())
                    labels.append(np.full(len(chunk), float(label)))
    finally:
        bn_buffers_restore(model, snap)
        model.train(modes[0])
        disc.train(modes[1])
    return roc_auc(np.concatenate(scores), np.concatenate(labels))
