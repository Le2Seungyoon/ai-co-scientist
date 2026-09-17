# 추론 단계 분리 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ŝ(구조 성분)를 한 번만 GPU로 뽑아 덤프해 두고, 레벨·τ 후처리 가설을 GPU 없이 병렬로 재조립·평가할 수 있게 한다.

**Architecture:** `infer_decomposed.py`의 조립 루프에서 GPU 구간(ŝ 산출)과 CPU 구간(레벨 결정 → `d = L·(1−s)` → PNG → zip)을 분리한다. 조립·인코딩·zip 쓰기 로직은 `src/ai_co_scientist/`로 옮겨 **두 진입점이 같은 함수를 부른다**. 새 CPU 진입점은 numpy + stdlib만 쓰므로 torch/cv2 없는 워크트리에서 돈다.

**Tech Stack:** Python 3.12 · numpy(본 의존성) · stdlib `zlib`/`struct`/`zipfile` · torch·opencv는 `baseline` 그룹(GPU 경로 전용) · pytest

**Spec:** `docs/superpowers/specs/2026-09-13-split-infer-stages-design.md`

## Global Constraints

- **`src/ai_co_scientist/`의 새 코드는 numpy + stdlib만 쓴다.** 본 의존성은 `numpy` · `httpx` · `pyyaml` 뿐이고 `torch` · `opencv-python`은 `baseline` 그룹이다. 워크트리는 dev 그룹만 sync하므로 둘 다 없다.
- **테스트는 오프라인**: GPU·캐시·네트워크·실데이터 없이 합성 데이터로 돈다 (`.agents/rules/testing.md`).
- **커밋은 개발자가 한다**: 기본은 `stage + diff only` (`.agents/rules/git-workflow.md` → Commits). 각 태스크의 마지막 단계는 스테이징까지이고, 커밋 명령은 제안만 한다. 이번 세션에서 직접 git이 명시적으로 허가되면 그때 실행한다.
- **커밋 메시지는 영어 단문 제목**, 본문 불릿 없음. **AI attribution 금지** (`Co-Authored-By` / `Generated with` → PreToolUse 훅이 deny).
- **언어**: `src/`의 docstring·주석은 한국어 · `.agents/**`와 훅 deny 메시지는 영어 · `docs/**`는 한국어 (`.agents/rules/docs.md`).
- **ruff line-length 100.**
- **`scripts/`는 얇게, 로직은 `src/`로** (`.agents/rules/architecture.md` → CLI / logic separation).
- 훅 테스트(`.agent-hooks/test_*.py`)는 pytest가 수집하지 않는다(`testpaths = ["tests"]`). stdlib만 쓰고 직접 실행한다.

---

## File Structure

| 파일 | 책임 | 변경 |
|---|---|---|
| `src/ai_co_scientist/submission.py` | PNG 인코딩/디코딩, zip 쓰기, 제출 검증 | `encode_png_gray8`, `write_submission_zip` 추가 |
| `src/ai_co_scientist/sem.py` | 순수 SEM/depth 로직 (torch 없음) | `assemble_depth` 추가 |
| `scripts/infer_decomposed.py` | GPU 추론 CLI | 조립을 공유 함수로 교체 · `--dump-structure` / `--dump-level-proba` 추가 · `cv2` 제거 |
| `scripts/assemble_submission.py` | **신규** CPU 재조립 CLI | 생성 |
| `.agent-hooks/block_runtime_commands.py` | 실험 명령 게이트 | 등급 분리 · 탈출구 축소 |
| `tests/`, `.agent-hooks/test_*.py` | 계약 고정 | 각 태스크에서 추가 |
| `.agents/rules/*.md`, `config.yaml` | 규칙·설정 | Task 8에서 일괄 |

---

### Task 1: stdlib PNG 그레이스케일 인코더

기존 `decode_png_gray8`의 역함수를 만든다. cv2 없이 제출본을 쓸 수 있어야 CPU 진입점이 워크트리에서 돈다.

**Files:**
- Modify: `src/ai_co_scientist/submission.py`
- Test: `tests/test_submission.py`

**Interfaces:**
- Consumes: 없음 (기존 `PNG_SIGNATURE`, `decode_png_gray8` 재사용)
- Produces: `encode_png_gray8(arr: np.ndarray, level: int = 6) -> bytes` — `(H, W) uint8` → 8bit 그레이 non-interlaced PNG 바이트

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_submission.py` 끝에 추가:

```python
def test_encode_png_gray8_roundtrips():
    """인코더 출력이 기존 디코더로 원본 배열까지 복원돼야 한다."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(72, 48), dtype=np.uint8)
    out = decode_png_gray8(encode_png_gray8(img))
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_encode_png_gray8_is_deterministic():
    """같은 입력은 같은 바이트 — 두 경로의 zip 바이트 동일성이 여기에 걸려 있다."""
    img = np.arange(72 * 48, dtype=np.int64).reshape(72, 48).astype(np.uint8)
    assert encode_png_gray8(img) == encode_png_gray8(img)


def test_encode_png_gray8_rejects_wrong_shape_or_dtype():
    """조용히 넘기지 않는다 — 잘못된 배열이 제출본이 되면 슬롯 하나가 날아간다."""
    with pytest.raises(ValueError):
        encode_png_gray8(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_png_gray8(np.zeros((4, 4), dtype=np.float32))


def test_encode_png_gray8_matches_cv2_decode_when_available():
    """cv2가 읽을 수 있는 PNG여야 한다 — 채점 측 디코더가 무엇일지 모른다."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(16, 24), dtype=np.uint8)
    buf = np.frombuffer(encode_png_gray8(img), dtype=np.uint8)
    assert np.array_equal(cv2.imdecode(buf, cv2.IMREAD_UNCHANGED), img)
```

`tests/test_submission.py` 상단 import에 `encode_png_gray8`을 추가한다 (기존 import 줄에 이름만 더한다).

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_submission.py -k encode -v`
Expected: FAIL — `ImportError: cannot import name 'encode_png_gray8'`

- [ ] **Step 3: 최소 구현을 쓴다**

`src/ai_co_scientist/submission.py`의 `decode_png_gray8` **앞에** 추가:

```python
def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    """길이 + 타입 + 데이터 + CRC32 — PNG 청크 1개."""
    return (struct.pack(">I", len(payload)) + ctype + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))


def encode_png_gray8(arr: np.ndarray, level: int = 6) -> bytes:
    """(H, W) uint8 → 8bit 그레이 non-interlaced PNG 바이트. `decode_png_gray8`의 역이다.

    **필터는 0(None)만 쓴다.** 제출본은 한 번 쓰고 채점만 되므로 압축률보다 **결정성**이
    중요하다 — 같은 입력이 같은 바이트를 내야 두 조립 경로의 동일성을 바이트로 비교할 수 있다.

    cv2를 쓰지 않는 이유는 `decode_png_gray8`과 같다: `opencv-python`은 `baseline` 그룹이라
    워크트리(dev 그룹만 sync)에는 없다. 조립이 옵션 의존성에 묶이면 CPU 진입점이 돌지 않는다.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"(H, W) 2차원 배열이어야 한다 — 받은 형태 {a.shape}")
    if a.dtype != np.uint8:
        raise ValueError(f"uint8이어야 한다 — 받은 dtype {a.dtype}")
    height, width = a.shape
    raw = np.zeros((height, width + 1), dtype=np.uint8)
    raw[:, 1:] = a  # 행마다 filter 바이트 0이 앞에 붙는다
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return PNG_SIGNATURE + b"".join([
        _png_chunk(b"IHDR", ihdr),
        _png_chunk(b"IDAT", zlib.compress(raw.tobytes(), level)),
        _png_chunk(b"IEND", b""),
    ])
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_submission.py -v`
Expected: PASS (기존 테스트 포함 전부)

- [ ] **Step 5: 스테이징**

```bash
git add src/ai_co_scientist/submission.py tests/test_submission.py
# 제안 메시지: Add stdlib PNG grayscale encoder for submission images
```

---

### Task 2: numpy 조립 함수 `assemble_depth`

`d = L·(1−s)`와 τ 클램프를 torch에서 numpy로 옮긴다. GPU 경로와 CPU 경로가 이 함수 하나를 공유한다.

**Files:**
- Modify: `src/ai_co_scientist/sem.py`
- Test: `tests/test_sem.py`

**Interfaces:**
- Consumes: 없음
- Produces: `ai_co_scientist.sem.assemble_depth(structure: np.ndarray, levels: np.ndarray, tau: float = 0.0) -> np.ndarray` — `(N, H, W)` 또는 `(N, 1, H, W)` float ŝ + `(N,)` float 레벨 → `(N, H, W) uint8`. 다른 모듈에서는 `from ai_co_scientist.sem import assemble_depth` 로 직접 import한다 (`tests/test_sem.py`만 모듈 관용구를 쓴다)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_sem.py` 끝에 추가:

```python
def test_assemble_depth_reconstructs_background_level():
    """s=0인 픽셀은 정확히 배경 레벨 L이어야 한다 (docs/data-facts.md §1-2의 전제)."""
    s = np.zeros((2, 3, 4), dtype=np.float32)
    levels = np.array([140.0, 170.0], dtype=np.float32)
    d = sem.assemble_depth(s, levels)
    assert d.dtype == np.uint8
    assert d.shape == (2, 3, 4)
    assert (d[0] == 140).all()
    assert (d[1] == 170).all()


def test_assemble_depth_applies_reparameterization():
    """d = L*(1-s) — s=0.5, L=160이면 80."""
    s = np.full((1, 2, 2), 0.5, dtype=np.float32)
    d = sem.assemble_depth(s, np.array([160.0], dtype=np.float32))
    assert (d == 80).all()


def test_assemble_depth_tau_clamps_background_to_exact_level():
    """tau 미만의 s는 0으로 눌려 배경이 정확히 L에 붙는다. tau=0이면 클램프 없음."""
    s = np.array([[[0.01, 0.9]]], dtype=np.float32)
    assert sem.assemble_depth(s, np.array([100.0], dtype=np.float32), tau=0.05)[0, 0, 0] == 100
    assert sem.assemble_depth(s, np.array([100.0], dtype=np.float32), tau=0.0)[0, 0, 0] == 99


def test_assemble_depth_accepts_channel_dim():
    """구조 회귀기 출력은 (N,1,H,W)다 — 채널 축을 받아들여야 한다."""
    s = np.zeros((2, 1, 3, 4), dtype=np.float32)
    d = sem.assemble_depth(s, np.array([150.0, 150.0], dtype=np.float32))
    assert d.shape == (2, 3, 4)


def test_assemble_depth_clips_out_of_range():
    """s<0 이나 s>1 이 들어와도 uint8 범위를 벗어난 값을 쓰지 않는다."""
    s = np.array([[[-1.0, 2.0]]], dtype=np.float32)
    d = sem.assemble_depth(s, np.array([200.0], dtype=np.float32))
    assert d[0, 0, 0] == 255
    assert d[0, 0, 1] == 0


def test_assemble_depth_rejects_length_mismatch():
    with pytest.raises(ValueError):
        sem.assemble_depth(np.zeros((3, 2, 2), dtype=np.float32),
                       np.array([140.0, 150.0], dtype=np.float32))
```

**import을 추가하지 않는다.** `tests/test_sem.py`는 `from ai_co_scientist import sem` 으로 모듈을 통째로 받아 `sem.xxx` 로 부르는 파일이다 — 위 테스트도 그 관용구를 따라 `sem.assemble_depth(...)` 로 쓴다 (`np`·`pytest`는 이미 import돼 있다).

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_sem.py -k assemble -v`
Expected: FAIL — `AttributeError: module 'ai_co_scientist.sem' has no attribute 'assemble_depth'`

- [ ] **Step 3: 최소 구현을 쓴다**

`src/ai_co_scientist/sem.py` 끝에 추가:

```python
def assemble_depth(structure: np.ndarray, levels: np.ndarray, tau: float = 0.0) -> np.ndarray:
    """d̂ = L̂·(1 − ŝ). ŝ < τ 는 0으로 클램프해 배경을 정확히 L̂에 붙인다 (τ=0이면 클램프 없음).

    GPU 경로(`scripts/infer_decomposed.py`)와 CPU 재조립 경로(`scripts/assemble_submission.py`)가
    **이 함수 하나를 공유한다.** 조립이 두 벌이 되면 어느 쪽이 과거 점수를 낸 경로인지 말할 수
    없게 된다.

    structure: (N, H, W) 또는 (N, 1, H, W) — 구조 회귀기 출력 ŝ
    levels:    (N,) — 이미지별 배경 레벨 L̂ (LEVELS의 값)
    반환:      (N, H, W) uint8
    """
    s = np.asarray(structure, dtype=np.float32)
    if s.ndim == 4 and s.shape[1] == 1:
        s = s[:, 0]
    if s.ndim != 3:
        raise ValueError(f"(N, H, W) 또는 (N, 1, H, W)여야 한다 — 받은 형태 {s.shape}")
    lv = np.asarray(levels, dtype=np.float32)
    if lv.ndim != 1 or len(lv) != len(s):
        raise ValueError(f"levels는 (N,)이어야 한다 — 구조 {len(s)}장, levels {lv.shape}")
    if tau > 0:
        s = np.where(s < tau, np.float32(0.0), s)
    d = np.round(lv.reshape(-1, 1, 1) * (1.0 - s))
    return np.clip(d, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_sem.py -v`
Expected: PASS

- [ ] **Step 5: 스테이징**

```bash
git add src/ai_co_scientist/sem.py tests/test_sem.py
# 제안 메시지: Add numpy depth assembly shared by both inference paths
```

---

### Task 3: 결정적 제출 zip 쓰기 `write_submission_zip`

zip 엔트리의 타임스탬프를 고정해 **같은 입력이면 같은 zip 바이트**가 나오게 한다. 현재 `zf.write(path)`는 파일 mtime을 헤더에 넣어 실행할 때마다 바이트가 달라진다 — 그러면 두 경로의 동일성을 바이트로 비교할 수 없다.

**Files:**
- Modify: `src/ai_co_scientist/submission.py`
- Test: `tests/test_submission.py`

**Interfaces:**
- Consumes: `encode_png_gray8` (Task 1)
- Produces: `write_submission_zip(depth: np.ndarray, names: list[str], zip_path, work_dir=None) -> int` · 상수 `ZIP_DATE_TIME`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_submission.py`에 추가:

```python
def test_write_submission_zip_is_byte_reproducible(tmp_path):
    """두 번 써서 바이트가 같아야 한다 — 경로 등가성 검증이 여기에 의존한다."""
    rng = np.random.default_rng(2)
    depth = rng.integers(0, 256, size=(3, 8, 6), dtype=np.uint8)
    names = ["000000.png", "000001.png", "000002.png"]
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    assert write_submission_zip(depth, names, a) == 3
    write_submission_zip(depth, names, b)
    assert a.read_bytes() == b.read_bytes()


def test_write_submission_zip_contents_decode_back(tmp_path):
    """zip 안의 PNG가 원본 배열로 복원돼야 한다."""
    rng = np.random.default_rng(3)
    depth = rng.integers(0, 256, size=(2, 8, 6), dtype=np.uint8)
    names = ["a.png", "b.png"]
    path = tmp_path / "s.zip"
    write_submission_zip(depth, names, path)
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == names
        for i, name in enumerate(names):
            assert np.array_equal(decode_png_gray8(zf.read(name)), depth[i])


def test_write_submission_zip_work_dir_receives_same_bytes(tmp_path):
    """work_dir는 zip에 들어간 것과 같은 바이트를 남긴다 (검수용 사본)."""
    depth = np.zeros((1, 4, 4), dtype=np.uint8)
    work = tmp_path / "work"
    path = tmp_path / "s.zip"
    write_submission_zip(depth, ["x.png"], path, work_dir=work)
    with zipfile.ZipFile(path) as zf:
        assert (work / "x.png").read_bytes() == zf.read("x.png")


def test_write_submission_zip_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        write_submission_zip(np.zeros((2, 4, 4), dtype=np.uint8), ["only.png"],
                             tmp_path / "s.zip")
```

상단 import에 `write_submission_zip`을 추가한다. `zipfile` import가 없으면 추가한다.

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_submission.py -k write_submission -v`
Expected: FAIL — `ImportError: cannot import name 'write_submission_zip'`

- [ ] **Step 3: 최소 구현을 쓴다**

`src/ai_co_scientist/submission.py`에 추가 (`encode_png_gray8` 아래):

```python
# zip 엔트리 타임스탬프 고정값. 파일 mtime을 쓰면 같은 입력도 실행마다 다른 바이트가 나와
# 두 조립 경로의 동일성을 바이트로 비교할 수 없다. (1980-01-01은 zip 포맷의 하한)
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def write_submission_zip(depth: np.ndarray, names, zip_path, work_dir=None) -> int:
    """(N, H, W) uint8 depth → 제출 zip. 반환: 쓴 장수.

    `work_dir`을 주면 같은 PNG 바이트를 파일로도 남긴다(검수용). **zip마다 다른 디렉터리를
    써야 한다** — 공유하면 파일명이 test_names.json에서 오므로 모든 실행이 동일해, 두 조립이
    병렬로 돌 때 서로의 PNG를 덮어써 zip에 다른 모델 출력이 섞인다. 점수는 나오지만 그게
    무엇의 점수인지 알 수 없게 되는 최악의 실패다.
    """
    names = list(names)
    if len(depth) != len(names):
        raise ValueError(f"장수 불일치: depth {len(depth)}장, names {len(names)}개")
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for img, name in zip(depth, names):
            png = encode_png_gray8(img)
            zf.writestr(zipfile.ZipInfo(name, date_time=ZIP_DATE_TIME), png)
            if work_dir is not None:
                (work_dir / name).write_bytes(png)
    return len(names)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_submission.py -v`
Expected: PASS

- [ ] **Step 5: 스테이징**

```bash
git add src/ai_co_scientist/submission.py tests/test_submission.py
# 제안 메시지: Add reproducible submission zip writer with fixed timestamps
```

---

### Task 4: `infer_decomposed`를 공유 조립으로 바꾸고 덤프 플래그를 붙인다

**Files:**
- Modify: `scripts/infer_decomposed.py` (`reconstruct_and_zip` 187-215 · `main` 218-325)
- Test: `tests/test_train_manifest.py`

**Interfaces:**
- Consumes: `assemble_depth` (Task 2), `write_submission_zip` (Task 3)
- Produces: `predict_structure(model, cache, lut=None, batch=512) -> np.ndarray` — `(N, H, W) float32` ŝ · CLI 플래그 `--dump-structure` / `--dump-level-proba`

- [ ] **Step 1: 실패하는 소스 계약 테스트를 쓴다**

`tests/test_train_manifest.py`에 추가:

```python
def test_infer_decomposed_does_not_import_cv2():
    # 조립이 옵션 의존성(baseline 그룹)에 묶이면 CPU 진입점과 코드를 공유할 수 없다
    source = _script("infer_decomposed.py")
    assert "import cv2" not in source
    assert "cv2." not in source


def test_infer_decomposed_uses_shared_assembly():
    # 조립이 두 벌이 되면 어느 경로가 과거 점수를 냈는지 말할 수 없게 된다
    source = _script("infer_decomposed.py")
    assert "assemble_depth" in source
    assert "write_submission_zip" in source


def test_infer_decomposed_exposes_dump_flags():
    # ŝ와 레벨 사후확률을 덤프할 수 없으면 CPU 재조립 경로에 입력이 없다
    source = _script("infer_decomposed.py")
    assert '"--dump-structure"' in source
    assert '"--dump-level-proba"' in source
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_train_manifest.py -v`
Expected: FAIL — `assert "import cv2" not in source`

- [ ] **Step 3: 구현한다**

(a) import 정리: `import cv2` 줄을 삭제하고, `ai_co_scientist` import 블록에 조립 함수를 더한다.

```python
from ai_co_scientist.sem import (
    GROUPS, LEVELS, assemble_depth, load_labels, pixel_features, qda_log_posterior,
    smooth_levels, softmax, viterbi_levels,
)
from ai_co_scientist.submission import write_submission_zip
```

(b) `reconstruct_and_zip`(187-215)을 **두 함수로 나눈다.** 기존 함수는 지운다.

```python
@torch.no_grad()
def predict_structure(model, cache: Path, lut=None, batch: int = 512) -> np.ndarray:
    """test SEM 전량 → ŝ (N, H, W) float32. **여기까지가 GPU 구간이다.**

    이후의 레벨 결정·τ 클램프·조립·zip은 전부 ŝ를 읽기만 하는 CPU 연산이라
    `scripts/assemble_submission.py`가 GPU 없이 되풀이할 수 있다.
    """
    model.eval()
    sem = np.load(cache / "test_sem.npy", mmap_mode="r")
    out = np.empty((len(sem), H, W), dtype=np.float32)
    for s in range(0, len(sem), batch):
        a = np.asarray(sem[s:s + batch])
        if lut is not None:
            a = lut[a]  # 구조 모델 입력만 변환 — 레벨 분류기는 원본 real 특징을 쓴다
        x = a.astype(np.float32)[:, None] / 255.0
        sp = model(torch.from_numpy(x).to(DEVICE))
        out[s:s + len(x)] = sp.reshape(-1, H, W).cpu().numpy()
    return out


def reconstruct_and_zip(structure: np.ndarray, cache: Path, cls: np.ndarray, tau: float,
                        zip_path: Path) -> int:
    """ŝ + 레벨 → 제출 zip. GPU를 쓰지 않는다 (조립은 `assemble_depth`가 한다)."""
    names = json.loads((cache / "test_names.json").read_text(encoding="utf-8"))
    levels = np.array(LEVELS, dtype=np.float32)[cls]
    depth = assemble_depth(structure, levels, tau)
    return write_submission_zip(depth, names, zip_path,
                                work_dir=cache.parent / "submission_work" / zip_path.stem)
```

(c) CLI 플래그를 추가한다 (`--adabn-dump` 인자 옆):

```python
    ap.add_argument("--dump-structure", default="",
                    help="구조 성분 ŝ를 이 경로에 .npy로 덤프한다 (N,H,W) float32. "
                         "CPU 재조립(scripts/assemble_submission.py)의 입력")
    ap.add_argument("--dump-level-proba", default="",
                    help="레벨 사후확률을 이 경로에 .npy로 덤프한다 (N,4) float32. "
                         "--level-source mean_only는 사후확률이 없어 거부된다")
```

(d) `main()`에서 레벨 예측 분기를 고친다. 덤프를 요구하면 사후확률이 필요하므로, `--level-hmm`이 아니어도 `return_proba=True`로 부른다. `if args.level_hmm:` / `else:` 블록(282-295)을 아래로 교체한다:

```python
    want_proba = bool(args.level_hmm or args.dump_level_proba)
    if want_proba and args.level_source == "mean_only":
        ap.error("--level-source mean_only는 사후확률을 내지 않는다 — "
                 "--level-hmm / --dump-level-proba는 cnn 또는 qda에서만 쓸 수 있다")
    if want_proba:
        cls, diag, proba = fit_predict_levels(Path(args.data_dir), cache, args.level_source,
                                              args.level_ckpt, return_proba=True)
    else:
        cls, diag = fit_predict_levels(Path(args.data_dir), cache, args.level_source,
                                       args.level_ckpt)
        proba = None
    if args.dump_level_proba:
        Path(args.dump_level_proba).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_level_proba, proba.astype(np.float32))
        print(f"레벨 사후확률 덤프 → {args.dump_level_proba} {proba.shape}", flush=True)
    if args.level_hmm:
        before = cls.copy()
        cls = viterbi_levels(proba, a=args.level_hmm_a)
        changed = int((before != cls).sum())
        diag = {**_diag(args.level_source, cls), "hmm_a": args.level_hmm_a,
                "changed": changed, "changed_frac": round(changed / len(cls), 4)}
        print(f"레벨 Viterbi(a={args.level_hmm_a}): {changed}장 변경 "
              f"({100 * changed / len(cls):.2f} percent)", flush=True)
```

(e) 조립 호출부(`n = reconstruct_and_zip(model, cache, cls, args.tau, Path(args.submit), lut)`)를 교체한다:

```python
    structure = predict_structure(model, cache, lut)
    if args.dump_structure:
        Path(args.dump_structure).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_structure, structure)
        print(f"구조 성분 덤프 → {args.dump_structure} {structure.shape}", flush=True)
    n = reconstruct_and_zip(structure, cache, cls, args.tau, Path(args.submit))
```

(f) 마지막 진단 JSON의 `"adabn_dump"` 줄 옆에 덤프 경로를 추가한다:

```python
        "dump_structure": args.dump_structure or None,
        "dump_level_proba": args.dump_level_proba or None,
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/ -q`
Expected: PASS — 전체 스위트. 특히 `test_train_manifest.py`의 기존 계약(`cnn` arm · `predict_levels_cnn`에 AdaBN 없음 · 리더보드 타깃 선언)이 함께 통과해야 한다.

Run: `uv run ruff check scripts/ src/`
Expected: 위반 없음

- [ ] **Step 5: 스테이징**

```bash
git add scripts/infer_decomposed.py tests/test_train_manifest.py
# 제안 메시지: Split GPU structure prediction from CPU assembly in infer_decomposed
```

---

### Task 5: CPU 재조립 진입점 `scripts/assemble_submission.py`

torch도 cv2도 import하지 않는다. 이것이 워크트리에서 병렬로 도는 진입점이다.

**Files:**
- Create: `scripts/assemble_submission.py`
- Test: `tests/test_assemble_submission.py` (신규)

**Interfaces:**
- Consumes: `assemble_depth`, `write_submission_zip`, `smooth_levels`, `viterbi_levels`, `LEVELS`, `GROUPS`
- Produces: CLI. stdout 마지막 줄에 진단 JSON

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_assemble_submission.py` 생성:

```python
"""CPU 재조립 진입점의 계약 — torch/cv2 없이 돌고, 레벨 후처리 인자가 실제로 결과를 바꾼다.

이 진입점은 워크트리(dev 그룹만 sync — torch도 cv2도 없다)에서 병렬로 도는 것이 존재 이유다.
그래서 **import 계약**이 성능 못지않게 중요하다.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from ai_co_scientist.submission import decode_png_gray8

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "assemble_submission.py"


def _fixture(tmp_path, n=6, h=4, w=3):
    """ŝ 덤프 · 레벨 사후확률 덤프 · 파일명 목록을 합성한다."""
    rng = np.random.default_rng(0)
    structure = rng.random((n, h, w), dtype=np.float32)
    proba = np.full((n, 4), 0.1, dtype=np.float32)
    proba[:, 0] = 0.7
    proba[3, :] = [0.1, 0.7, 0.1, 0.1]  # 런 한가운데의 고립된 한 장 — 평활 대상
    names = [f"{i:06d}.png" for i in range(n)]
    np.save(tmp_path / "s.npy", structure)
    np.save(tmp_path / "p.npy", proba)
    (tmp_path / "names.json").write_text(json.dumps(names), encoding="utf-8")
    return structure, proba, names


def _run(tmp_path, *extra):
    out = tmp_path / "sub.zip"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--structure", str(tmp_path / "s.npy"),
         "--level-proba", str(tmp_path / "p.npy"),
         "--names", str(tmp_path / "names.json"),
         "--submit", str(out), *extra],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return out, json.loads(proc.stdout.strip().splitlines()[-1])


def test_produces_readable_zip_without_torch_or_cv2(tmp_path):
    _fixture(tmp_path)
    out, diag = _run(tmp_path)
    assert diag["n"] == 6
    with zipfile.ZipFile(out) as zf:
        assert len(zf.namelist()) == 6
        assert decode_png_gray8(zf.read("000000.png")).shape == (4, 3)


def test_source_imports_neither_torch_nor_cv2():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import cv2" not in source


def test_level_smooth_changes_the_isolated_image(tmp_path):
    """k=3 최빈값 필터는 런 한가운데 고립된 한 장을 흡수해야 한다."""
    _fixture(tmp_path)
    _, plain = _run(tmp_path)
    _, smoothed = _run(tmp_path, "--level-smooth", "3")
    assert smoothed["level_diag"]["changed"] == 1
    assert plain["level_smooth"] == 0
    assert smoothed["level_smooth"] == 3


def test_smooth_and_hmm_are_mutually_exclusive(tmp_path):
    """평활기를 두 번 겹치면 어느 쪽이 점수를 움직였는지 분리되지 않는다."""
    _fixture(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--structure", str(tmp_path / "s.npy"),
         "--level-proba", str(tmp_path / "p.npy"),
         "--names", str(tmp_path / "names.json"),
         "--submit", str(tmp_path / "x.zip"),
         "--level-smooth", "3", "--level-hmm"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "함께 쓸 수 없다" in proc.stderr


def test_declares_leaderboard_target():
    """리더보드의 y는 숨은 real depth GT다 — average_depth로 잘못 라벨링하면 안 된다."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"x_domain": "real"' in source
    assert '"y_source": "real_depth_gt"' in source
    assert "real_average_depth" not in source
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_assemble_submission.py -v`
Expected: FAIL — 스크립트 파일이 없어 `returncode != 0`

- [ ] **Step 3: 구현한다**

`scripts/assemble_submission.py` 생성:

```python
"""CPU 재조립 — 덤프된 ŝ와 레벨 사후확률로 제출 zip을 다시 만든다 (GPU 불필요).

`scripts/infer_decomposed.py`가 GPU로 한 번 뽑아 둔 ŝ를 읽기만 하므로, 레벨 축 후처리 가설
(`--level-smooth` k · `--level-hmm` a)과 배경 클램프 τ를 **몇 개든 병렬로** 되풀이할 수 있다.
구조 성분 ŝ는 이 인자들 중 어느 것에도 영향받지 않는다 — 그것이 분리가 성립하는 이유다.

**torch도 cv2도 import하지 않는다.** 워크트리는 dev 그룹만 sync하므로 둘 다 없고, 이 진입점이
거기서 도는 것이 존재 이유다. 조립·인코딩은 `ai_co_scientist`의 공유 함수가 한다 — GPU 경로와
같은 코드다.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.sem import GROUPS, LEVELS, assemble_depth, smooth_levels, viterbi_levels
from ai_co_scientist.submission import write_submission_zip


def _diag(pred: np.ndarray) -> dict:
    dist = Counter(pred.tolist())
    return {"test_class_dist": {GROUPS[c]: dist.get(c, 0) for c in range(len(GROUPS))},
            "test_class_frac": {GROUPS[c]: round(dist.get(c, 0) / len(pred), 4)
                                for c in range(len(GROUPS))}}


def main() -> int:
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True,
                    help="infer_decomposed --dump-structure가 남긴 ŝ (.npy, (N,H,W) float32). "
                         "기본값을 두지 않는다 — 워크트리에서 상대 기본값은 빈 runtime/으로 "
                         "조용히 풀린다")
    ap.add_argument("--level-proba", required=True,
                    help="infer_decomposed --dump-level-proba가 남긴 사후확률 (.npy, (N,4))")
    ap.add_argument("--names", required=True, help="runtime/cache/test_names.json 경로")
    ap.add_argument("--submit", required=True, help="생성할 제출 zip 경로")
    ap.add_argument("--work-dir", default="", help="PNG 사본을 남길 디렉터리 (비우면 남기지 않음)")
    ap.add_argument("--tau", type=float, default=0.0, help="배경 클램프 임계 (0=끕)")
    ap.add_argument("--level-smooth", type=int, default=0,
                    help="파일 순서를 따라 레벨 예측을 최빈값 필터로 평활 (0=끕, EXP-019는 9)")
    ap.add_argument("--level-hmm", action="store_true",
                    help="레벨 예측을 4상태 Viterbi로 복호한다 (--level-smooth의 대안)")
    ap.add_argument("--level-hmm-a", type=float, default=0.974, help="Viterbi 자기전이 확률")
    args = ap.parse_args()

    if args.level_hmm and args.level_smooth > 1:
        ap.error("--level-hmm과 --level-smooth는 함께 쓸 수 없다 — 평활기를 두 번 겹치면 "
                 "어느 쪽이 점수를 움직였는지 분리되지 않는다. 하나만 고를 것")

    structure = np.load(args.structure, mmap_mode="r")
    proba = np.load(args.level_proba)
    names = json.loads(Path(args.names).read_text(encoding="utf-8"))
    if not (len(structure) == len(proba) == len(names)):
        raise SystemExit(f"장수 불일치 — 구조 {len(structure)} · 사후확률 {len(proba)} · "
                         f"파일명 {len(names)}")

    cls = proba.argmax(1)
    diag = _diag(cls)
    if args.level_hmm:
        before = cls.copy()
        cls = viterbi_levels(proba, a=args.level_hmm_a)
        changed = int((before != cls).sum())
        diag = {**_diag(cls), "hmm_a": args.level_hmm_a, "changed": changed,
                "changed_frac": round(changed / len(cls), 4)}
        print(f"레벨 Viterbi(a={args.level_hmm_a}): {changed}장 변경", flush=True)
    elif args.level_smooth > 1:
        before = cls.copy()
        cls = smooth_levels(cls, args.level_smooth)
        changed = int((before != cls).sum())
        diag = {**_diag(cls), "smoothed": args.level_smooth, "changed": changed,
                "changed_frac": round(changed / len(cls), 4)}
        print(f"레벨 평활(k={args.level_smooth}): {changed}장 변경", flush=True)

    levels = np.array(LEVELS, dtype=np.float32)[cls]
    depth = assemble_depth(structure, levels, args.tau)
    n = write_submission_zip(depth, names, Path(args.submit),
                             work_dir=Path(args.work_dir) if args.work_dir else None)
    print(f"제출본 {n}장 → {args.submit}", flush=True)

    print(json.dumps({
        "x_domain": "real", "y_source": "real_depth_gt",
        "metric": {"name": "leaderboard_rmse", "x_domain": "real", "y_source": "real_depth_gt"},
        "structure": args.structure, "level_proba": args.level_proba,
        "tau": args.tau, "level_smooth": args.level_smooth, "level_hmm": bool(args.level_hmm),
        "level_hmm_a": args.level_hmm_a if args.level_hmm else None,
        "reconstruct": "d = L * (1 - s)", "levels": list(LEVELS),
        "n": n, "zip": args.submit, "level_diag": diag,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_assemble_submission.py -v`
Expected: PASS

Run: `uv run ruff check scripts/assemble_submission.py`
Expected: 위반 없음

- [ ] **Step 5: 스테이징**

```bash
git add scripts/assemble_submission.py tests/test_assemble_submission.py
# 제안 메시지: Add CPU-only submission assembly entry point
```

---

### Task 6: 경로 등가성 테스트

두 경로가 **같은 zip 바이트**를 내는지 합성 데이터로 고정한다. 이것이 조립 단일화의 실질적 보증이다.

**Files:**
- Test: `tests/test_assemble_submission.py` (Task 5에서 만든 파일에 추가)

**Interfaces:**
- Consumes: `assemble_depth`, `write_submission_zip`, `scripts/assemble_submission.py`
- Produces: 없음

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_assemble_submission.py`에 추가:

```python
def test_two_paths_produce_identical_zip_bytes(tmp_path):
    """GPU 경로의 조립과 CPU 진입점의 조립이 바이트까지 같아야 한다.

    GPU forward 자체는 재현할 수 없으므로, 그 **출력** ŝ를 고정한 뒤 이후 구간만 비교한다.
    조립이 두 벌이 되면(다른 인코더·다른 반올림·다른 zip 타임스탬프) 여기서 즉시 깨진다.
    """
    from ai_co_scientist.sem import LEVELS, assemble_depth
    from ai_co_scientist.submission import write_submission_zip

    structure, proba, names = _fixture(tmp_path)

    # 경로 ① — infer_decomposed.reconstruct_and_zip이 하는 것과 같은 순서
    cls = proba.argmax(1)
    levels = np.array(LEVELS, dtype=np.float32)[cls]
    direct = tmp_path / "direct.zip"
    write_submission_zip(assemble_depth(structure, levels, 0.0), names, direct)

    # 경로 ② — CPU 진입점
    viacli, _ = _run(tmp_path)

    assert direct.read_bytes() == viacli.read_bytes()


def test_structure_dump_roundtrips_bit_exactly(tmp_path):
    """float32 저장·로드가 ŝ를 비트 단위로 보존해야 한다.

    float16(절반 용량)을 버린 근거를 고정한다: d = L*(1-s)에서 L이 최대 170이라 s의 오차
    ~0.0005가 d에서 ~0.085가 되어 round()의 경계를 흔든다.
    """
    from ai_co_scientist.sem import assemble_depth

    rng = np.random.default_rng(7)
    s = rng.random((200, 4, 3), dtype=np.float32)
    path = tmp_path / "s32.npy"
    np.save(path, s)
    assert np.load(path).dtype == np.float32
    assert np.array_equal(np.load(path), s)

    # float16으로 저장했다면 조립 결과가 실제로 달라진다 — 버린 근거를 숫자로 남긴다.
    # L=170은 LEVELS의 최댓값이라 s 오차가 d로 가장 크게 증폭되는 조건이다.
    levels = np.full(len(s), 170.0, dtype=np.float32)
    lossy = s.astype(np.float16).astype(np.float32)
    differing = int((assemble_depth(s, levels) != assemble_depth(lossy, levels)).sum())
    assert differing > 0, (
        "float16 왕복이 이 표본에서 조립 결과를 전혀 바꾸지 않았다 — 표본을 넓히거나, "
        "float32를 고집하는 근거를 다시 재야 한다")


def test_tau_survives_both_paths(tmp_path):
    """τ가 CPU 진입점에서도 같은 클램프를 낸다 — 인자가 한쪽에만 먹으면 조용히 갈린다."""
    from ai_co_scientist.sem import LEVELS, assemble_depth
    from ai_co_scientist.submission import write_submission_zip

    structure, proba, names = _fixture(tmp_path)
    cls = proba.argmax(1)
    levels = np.array(LEVELS, dtype=np.float32)[cls]
    direct = tmp_path / "direct_tau.zip"
    write_submission_zip(assemble_depth(structure, levels, 0.5), names, direct)

    viacli, _ = _run(tmp_path, "--tau", "0.5")
    assert direct.read_bytes() == viacli.read_bytes()
```

- [ ] **Step 2: 실패 또는 통과를 확인한다**

Run: `uv run pytest tests/test_assemble_submission.py -k two_paths -v`
Expected: PASS. **여기서 실패하면 Task 2~5 중 한 곳에서 두 경로가 갈렸다는 뜻이다** — 진행하지 말고 원인을 찾는다 (인코더·반올림·zip 타임스탬프 순).

- [ ] **Step 3: 스테이징**

```bash
git add tests/test_assemble_submission.py
# 제안 메시지: Pin byte equality between GPU and CPU assembly paths
```

---

### Task 7: 게이트 등급 분리와 탈출구 축소

**Files:**
- Modify: `.agent-hooks/block_runtime_commands.py`
- Test: `.agent-hooks/test_block_runtime_commands.py`

**Interfaces:**
- Consumes: 없음
- Produces: 상수 `REGISTRY_WRITERS` · `EXCLUSIVE` · `GUARDED` (= 둘의 합) · `guarded_hit`이 `(script, kind)` 반환

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`.agent-hooks/test_block_runtime_commands.py`의 기존 시나리오 뒤에 추가한다. 이 파일은 stdlib만 쓰고 `failures` 리스트에 모아 마지막에 보고하는 방식이므로 그 관용구를 따른다 — 기존 헬퍼 `run(root, command, env_extra=None)`은 `(stdout, returncode)`를 준다.

```python
# --- 등급 분리: 레지스트리 없는 트리에서 무엇이 풀리고 무엇이 남는가 -----------------
with tempfile.TemporaryDirectory() as root:  # registry.jsonl 없음 = 워크트리 흉내
    out, _ = run(root, "uv run python scripts/probe_level.py --cache-dir ../../x/runtime/cache")
    if out.strip():
        failures.append("probe_level이 여전히 denied — 읽기 전용이라 풀렸어야 한다")

    out, _ = run(root, "uv run python scripts/assemble_submission.py --submit a.zip")
    if out.strip():
        failures.append("assemble_submission이 denied — GUARDED 대상이 아니다")

    out, _ = run(root, "uv run python scripts/exp.py new --title x")
    if not out.strip():
        failures.append("exp.py가 통과 — 레지스트리 포크를 막아야 한다")

    out, _ = run(root, "uv run python scripts/exp.py new --title x",
                 env_extra={"ACS_RUNTIME_EXEMPT": "because I said so"})
    if not out.strip():
        failures.append("exp.py가 탈출구로 뚫렸다 — 센티넬이 생기면 게이트가 영구 해제된다")

    for script in ("train_level.py", "train_structure.py", "infer_decomposed.py",
                   "dacon_submit.py"):
        out, _ = run(root, f"uv run python scripts/{script} --submit a.zip",
                     env_extra={"ACS_RUNTIME_EXEMPT": "measured one-off"})
        if out.strip():
            failures.append(f"{script}가 탈출구로도 막혔다 — 축소가 과하게 번졌다")
```

`probe_level`이 `GUARDED`에서 빠지므로, 기존 테스트 중 `probe_level`이 denied되기를 기대하는 시나리오가 있으면 함께 고친다.

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run python .agent-hooks/test_block_runtime_commands.py`
Expected: FAIL — "probe_level이 여전히 denied", "exp.py가 탈출구로 뚫렸다"

- [ ] **Step 3: 구현한다**

`.agent-hooks/block_runtime_commands.py`에서 `GUARDED`를 등급으로 나누고, deny 사유와 탈출구 적용 범위를 분리한다.

```python
# The registry's single truth. `report_id` comes from `len(records)`, so a second tree starts
# over at 1 and the two truths can never be merged -- the escape hatch does NOT cover this:
# writing here CREATES the sentinel, which would permanently unlock the gate in that tree.
REGISTRY_WRITERS = ("scripts/exp.py",)

# Exclusive resources: the single 8 GB GPU, and the finite DACON submission quota.
EXCLUSIVE = (
    "scripts/train_level.py",
    "scripts/train_structure.py",
    "scripts/infer_decomposed.py",
    "scripts/dacon_submit.py",
)

# `scripts/probe_level.py` is deliberately absent: it reads with mmap_mode="r" and writes
# nothing under runtime/, so it has no registry-fork path. With no cache in the tree `np.load`
# fails loudly -- there is no silent-corruption route to guard against.
# `scripts/assemble_submission.py` is absent for the same reason: it writes only its own tree.
GUARDED = REGISTRY_WRITERS + EXCLUSIVE
```

`guarded_hit`이 등급도 돌려주게 고친다:

```python
def guarded_hit(command):
    """Return (guarded script, kind) the command *executes*, or (None, None).

    `kind` is "registry" or "exclusive" -- the deny reason and whether the escape hatch
    applies both turn on it.
    """
    for script in GUARDED:
        escaped = re.escape(script).replace("/", r"[/\\]")
        pattern = r"(?:^|[;&|\n])\s*(?:uv\s+run|python[\w.]*)\b[^;&|\n]*" + escaped
        if re.search(pattern, command):
            return script, ("registry" if script in REGISTRY_WRITERS else "exclusive")
    return None, None
```

사유 문구를 둘로 나눈다 (`REASON` 상수를 교체):

```python
REASON_REGISTRY = (
    "This tree has no {sentinel} -- it is a git worktree, and `runtime/` is gitignored so it "
    "was never copied. Running `{hit}` here would fork the registry: report_id comes from "
    "len(records), so a second tree starts over at 1 and the two truths can never be merged. "
    "Run registry commands in the MAIN worktree, where the registry lives. "
    "There is NO escape hatch for this one: writing here would create {sentinel} and thereby "
    "unlock every other gate in this tree for good. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
)

REASON_EXCLUSIVE = (
    "This tree has no {sentinel} -- it is a git worktree, and `runtime/` is gitignored so it "
    "was never copied. `{hit}` needs an exclusive resource the MAIN worktree owns: the single "
    "8 GB GPU, or the finite DACON submission quota. This lane is for additive code, offline "
    "tests, and CPU reassembly (`scripts/assemble_submission.py`, `scripts/probe_level.py`), "
    "which are not guarded here. "
    "Escape hatch: set {var}=\"<reason>\" for this command. "
    "-- .agents/rules/architecture.md -> Parallel execution contract"
)
```

`main()`의 끝부분을 교체한다:

```python
    command = (payload.get("tool_input") or {}).get("command") or ""
    hit, kind = guarded_hit(command)
    if not hit:
        return

    if os.path.exists(os.path.join(project_root(), SENTINEL)):
        return

    sentinel = SENTINEL.replace("\\", "/")
    if kind == "registry":
        deny(REASON_REGISTRY.format(sentinel=sentinel, hit=hit))
        return

    if os.environ.get(EXEMPT_VAR, "").strip():
        return

    deny(REASON_EXCLUSIVE.format(sentinel=sentinel, hit=hit, var=EXEMPT_VAR))
```

모듈 docstring의 `Governed` / `Escape` 항목도 함께 고친다 — 탈출구가 이제 `EXCLUSIVE`에만 적용된다는 사실이 문서화되어야 한다.

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run python .agent-hooks/test_block_runtime_commands.py`
Expected: 모든 검사 통과, rc=0

Run: `uv run pytest tests/test_harness_generated.py -v`
Expected: PASS — 규칙 링크 스캔이 새 경로를 검사한다

- [ ] **Step 5: 스테이징**

```bash
git add .agent-hooks/block_runtime_commands.py .agent-hooks/test_block_runtime_commands.py
# 제안 메시지: Grade the runtime gate by reason and narrow the escape hatch
```

---

### Task 8: 레지스트리 provenance 필드

병렬 실행에서는 코디네이터(`main`)가 기록하고 코드는 워커 브랜치에 있다. 어느 코드가 그 숫자를 냈는지 레코드가 말해야 한다.

**Files:**
- Modify: `src/ai_co_scientist/registry.py` (`new_report` 118-158)
- Modify: `scripts/exp.py` (`new` 서브파서)
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: 없음
- Produces: `new_report(..., source_branch: str = "", source_commit: str = "")` · 레코드의 `source` 키 (`{"branch": str, "commit": str}` 또는 `None`)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_registry.py`에 추가한다. 이 파일은 현재 `import pytest` 와 `from ai_co_scientist import registry` 만 갖고 있으므로 **상단에 `import json` 을 추가**한다 (세 번째 테스트가 쓴다).

```python
def test_new_report_records_source_when_given(tmp_path):
    """워커 브랜치에서 돈 실험은 그 브랜치/커밋으로 기록돼야 한다 — 코디네이터의 HEAD가 아니라."""
    path = tmp_path / "r.jsonl"
    rec = registry.new_report(
        title="t", x_domain="real", x_desc="x", y_source="real_group_label", y_desc="y",
        model="m", method="me", purpose="p", metric_name="acc",
        metric_x_domain="real", metric_y_source="real_group_label",
        source_branch="feature/postproc-k", source_commit="deadbeef", path=path)
    assert rec["source"] == {"branch": "feature/postproc-k", "commit": "deadbeef"}
    assert registry.get(rec["report_id"], path)["source"]["branch"] == "feature/postproc-k"


def test_new_report_source_is_none_when_omitted(tmp_path):
    """optional 필드다 — 기존 호출부는 바뀌지 않는다."""
    path = tmp_path / "r.jsonl"
    rec = registry.new_report(
        title="t", x_domain="sim", x_desc="x", y_source="sim_depth_gt", y_desc="y",
        model="m", method="me", purpose="p", metric_name="rmse",
        metric_x_domain="sim", metric_y_source="sim_depth_gt", path=path)
    assert rec["source"] is None


def test_records_without_source_still_load(tmp_path):
    """기존 20건에는 source 키가 없다 — 로드가 깨지면 안 된다."""
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({
        "report_id": "EXP-001", "created": "2026-07-30T00:00:00", "title": "old",
        "x": {"domain": "sim", "desc": "d"}, "y": {"source": "sim_depth_gt", "desc": "d"},
        "model": "m", "method": "me", "purpose": "p",
        "metric": {"name": "rmse", "x_domain": "sim", "y_source": "sim_depth_gt",
                   "matches_target": False, "warning": "w"},
        "val": None, "lb": None, "verdict": "",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    assert registry.get("EXP-001", path)["title"] == "old"
    assert registry.load_all(path)[0].get("source") is None
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_registry.py -k source -v`
Expected: FAIL — `TypeError: new_report() got an unexpected keyword argument 'source_branch'`

- [ ] **Step 3: 구현한다**

`src/ai_co_scientist/registry.py`의 `new_report` 시그니처에 두 인자를 더한다:

```python
def new_report(*, title, x_domain, x_desc, y_source, y_desc, model, method, purpose,
               metric_name, metric_x_domain, metric_y_source,
               source_branch="", source_commit="", path=None) -> dict:
```

docstring 아래에 설명을 더하고, `record` 딕셔너리의 `"verdict": ""` **앞에** 추가한다:

```python
            # 실험을 낸 코드의 출처. 병렬 워크트리에서는 기록하는 쪽(코디네이터)과 실행한
            # 쪽(워커 브랜치)이 다르므로, 기록자의 HEAD를 쓰면 조용히 틀린다. 워커가 자기
            # 트리에서 읽은 값을 그대로 넣는다. 기존 레코드에는 이 키가 없다 — optional이다.
            "source": ({"branch": source_branch, "commit": source_commit}
                       if (source_branch or source_commit) else None),
```

`scripts/exp.py`의 `new` 서브파서에 인자를 더한다:

```python
    new.add_argument("--source-branch", default="",
                     help="실험을 낸 코드의 브랜치 — 워커가 자기 워크트리에서 읽은 값")
    new.add_argument("--source-commit", default="",
                     help="실험을 낸 코드의 커밋 — 기록자의 HEAD가 아니라 실행한 트리의 HEAD")
```

그리고 `scripts/exp.py:59-63`의 `new_report(...)` 호출부에 인자를 전달한다. **이 파일은 파싱 결과를 `a` 라는 이름으로 받는다** (`args`가 아니다):

```python
        rec = registry.new_report(
            title=a.title, x_domain=a.x_domain, x_desc=a.x_desc,
            y_source=a.y_source, y_desc=a.y_desc, model=a.model, method=a.method,
            purpose=a.purpose, metric_name=a.metric_name,
            metric_x_domain=a.metric_x, metric_y_source=a.metric_y,
            source_branch=a.source_branch, source_commit=a.source_commit)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS

- [ ] **Step 5: 스테이징**

```bash
git add src/ai_co_scientist/registry.py scripts/exp.py tests/test_registry.py
# 제안 메시지: Record source branch and commit on experiment pre-reports
```

---

### Task 9: 규칙·문서 갱신과 A2A 잔재 제거

코드가 착지한 뒤에 한다 — `check_rule_links.py`가 규칙이 가리키는 경로의 실재를 검사하므로, 문서가 먼저 가면 스위트가 깨진다.

**Files:**
- Modify: `.agents/rules/architecture.md` (Parallel execution contract · `## No protocol layer` 삭제)
- Modify: `.agents/rules/enforcement.md` (This project's gates 표)
- Modify: `config.yaml` (첫 줄 주석)
- Modify: `tests/test_config.py` (`test_no_a2a_leftovers` 삭제)

**Interfaces:**
- Consumes: Task 5의 `scripts/assemble_submission.py` · Task 7의 등급 분리
- Produces: 없음

- [ ] **Step 1: A2A 잔재를 지운다**

`tests/test_config.py`에서 `test_no_a2a_leftovers` 함수를 통째로 삭제한다. 금지 목록의 `executor`가 현재 에이전트 레인 이름이라, 나중에 executor 설정이 필요해지면 옛 잔재 테스트가 엉뚱하게 막는 잠복 충돌도 함께 사라진다.

`config.yaml` 첫 줄을 바꾼다:

```yaml
# 실험 하네스 설정.
```

`.agents/rules/architecture.md`에서 `## No protocol layer` 절(마지막 절)을 통째로 삭제한다.

- [ ] **Step 2: 실행 계약을 갱신한다**

`.agents/rules/architecture.md`의 Parallel execution contract 표 아래, `engineer`/`executor` 문단 뒤에 더한다 (영어):

```markdown
**CPU reassembly is a fourth class.** `scripts/assemble_submission.py` rebuilds a submission from
a dumped ŝ and level posterior with numpy and stdlib only -- no torch, no cv2, so it runs in a
worktree, whose `uv sync` brings the dev group alone. Level post-processing (`--level-smooth`,
`--level-hmm`, `--tau`) moves no structure component, so N of these run in parallel off ONE GPU
inference. They write into their own tree and never call `scripts/exp.py`: the coordinator issues
the pre-report, dispatches the `report_id`, and records the result, so `report_id = len(records)`
has no fork path across trees. A worker reports its own `git rev-parse HEAD` and branch, and the
coordinator records THOSE -- never its own.

**Never create `runtime/registry.jsonl` in a worktree.** That file's existence IS the runtime
gate's sentinel; once it exists the gate is unlocked in that tree for good. This is why the
escape hatch does not cover `scripts/exp.py` (`enforcement.md` -> This project's gates).

**Workers build submissions; they never submit one.** The leaderboard is the only verdict and its
slots are finite, so N parallel zips cannot all be spent. A worker runs `verify_submission()` in
its own tree -- that one is not guarded, and it catches a broken zip before a slot pays for it --
and reports its pre-selection numbers. Ranking for the level axis is real train + `site_split`
holdout accuracy, which is real->real and is how `k=9` was chosen; it **ranks, it does not
judge** (real has 2,836 runs against test's 1,046, so the optimum does not transfer). The
coordinator submits the top one. One sweep is ONE pre-report: `exp.py result` merges into the
existing `val` by default, so per-arm numbers stack without a new schema.
```

- [ ] **Step 3: 게이트 표를 갱신한다**

`.agents/rules/enforcement.md`의 "Runtime-command gate" 행을 두 행으로 나눈다:

```markdown
| Registry-write gate | PreToolUse hook (`block_runtime_commands.py`) | `scripts/exp.py` runs only where `runtime/registry.jsonl` exists | none — deny (an exemption here would create the sentinel and unlock the tree) |
| Exclusive-resource gate | PreToolUse hook (`block_runtime_commands.py`) | the four GPU / submission-quota scripts run only where `runtime/registry.jsonl` exists | `ACS_RUNTIME_EXEMPT="<reason>"` |
```

- [ ] **Step 4: 전체 스위트와 크기 예산을 확인한다**

Run: `uv run pytest tests/ -q`
Expected: PASS

Run: `uv run python .agent-hooks/check_rules_size.py` (또는 파일을 한 번 편집해 PostToolUse 훅이 돌게 한다)
Expected: `architecture.md`·`enforcement.md`가 ~150줄 예산 안. 넘으면 `refactor-agent-rules` 스킬로 판단한다 — 무작정 줄이지 않는다.

Run: `uv run python .agent-hooks/test_block_runtime_commands.py`
Expected: rc=0

- [ ] **Step 5: 스테이징**

```bash
git add .agents/rules/architecture.md .agents/rules/enforcement.md config.yaml tests/test_config.py
# 제안 메시지: Update execution contract for CPU reassembly and drop A2A leftovers
```

---

### Task 10: 실제 레시피 회귀 — 착지 증거 (main 체크아웃에서만)

오프라인 테스트는 "두 경로가 같다"만 증명한다. "새 경로가 과거 채택 레시피를 재현한다"는 실제 ckpt와 캐시가 필요하고 그것은 main에만 있다. **이 태스크는 워크트리에서 실행할 수 없다.**

> **개정 (Task 4 리뷰 결과, 2026-09-14).** 최초 계획은 리팩터 **이후**의 두 경로끼리만 비교했다.
> 그러면 조립이 torch에서 numpy로, 인코더가 cv2에서 stdlib으로 옮겨간 것을 아무도 검증하지
> 않는다 — 두 경로가 같은 새 코드를 쓰므로 함께 틀려도 일치한다. 리뷰어가 이 구멍을 지적했고,
> **`runtime/submissions/EXP-019-smooth9.zip` 이 남아 있다**: LB 3.0493을 낸 그 제출본이고,
> **리팩터 이전 cv2 + torch 코드가 만든 것**이다. 이것이 진짜 기준선이므로 비교 대상에 넣는다.

**Files:**
- 코드 변경 없음. 산출물: 증거 기록

**Interfaces:**
- Consumes: Task 4·5의 진입점
- Produces: 픽셀 일치 여부 판정

- [ ] **Step 1: EXP-019 레시피를 기존 CLI로 돌리고 ŝ와 사후확률을 덤프한다**

인자는 EXP-019 레코드에서 읽은 것이다 — "구조·AdaBN은 EXP-016과 동일(EXP-005 ckpt, adabn real shuffle 42, tau 0). 레벨 분류기도 EXP-013 그대로 — 바뀌는 것은 예측 후처리 하나뿐".

```bash
uv run python scripts/infer_decomposed.py \
  --ckpt runtime/ckpt/EXP-005-structure.pt \
  --level-source cnn --level-ckpt runtime/ckpt/EXP-013-level-cnn.pt \
  --adabn real --adabn-shuffle 42 --tau 0 --level-smooth 9 \
  --submit runtime/submissions/regress-oneshot.zip \
  --dump-structure runtime/cache/regress-structure.npy \
  --dump-level-proba runtime/cache/regress-proba.npy
```

**이 실행 자체가 첫 번째 검증이다.** EXP-019 레코드는 평활이 바꾼 장수를 `changed_imgs: 554`
(`changed_frac: 0.0213`), 레벨 분포를 `test_class_frac: [0.2502, 0.2498, 0.2502, 0.2499]`로
기록하고 있다. stdout의 `레벨 평활(k=9): N장 변경`이 **554가 아니면** 레벨 경로가 이미 갈린
것이므로 Step 2로 넘어가지 않는다.

`uv run python scripts/exp.py show EXP-019`로 레코드를 직접 확인할 수 있다 — **레코드가 정본이다.**

- [ ] **Step 2: 같은 후처리를 CPU 경로로 되풀이한다**

```bash
uv run python scripts/assemble_submission.py \
  --structure runtime/cache/regress-structure.npy \
  --level-proba runtime/cache/regress-proba.npy \
  --names runtime/cache/test_names.json \
  --level-smooth 9 \
  --submit runtime/submissions/regress-twostage.zip
```

여기서도 `레벨 평활(k=9): 554장 변경`이 나와야 한다. Step 1과 이 값이 다르면 레벨 결정이
두 경로에서 갈렸다는 뜻이다 — 픽셀 비교로 가기 전에 여기서 멈춘다.

- [ ] **Step 3: 픽셀 일치를 확인한다**

바이트가 아니라 **픽셀**로 비교한다 — 인코더가 cv2에서 stdlib으로 바뀌었으므로 과거 zip과 바이트는 다르고, 채점되는 것은 픽셀이다.

**세 개를 비교한다.** 기준선은 리팩터 이전 코드가 만든 과거 제출본이다.

```bash
uv run python - <<'PY'
import zipfile
import numpy as np
from ai_co_scientist.submission import decode_png_gray8, verify_submission

HIST = "runtime/submissions/EXP-019-smooth9.zip"       # 리팩터 이전 cv2+torch 산출물 (LB 3.0493)
ONE = "runtime/submissions/regress-oneshot.zip"        # 리팩터 이후 GPU 경로
TWO = "runtime/submissions/regress-twostage.zip"       # 리팩터 이후 CPU 재조립

zips = {k: zipfile.ZipFile(v) for k, v in (("hist", HIST), ("one", ONE), ("two", TWO))}
names = sorted(zips["hist"].namelist())
for k, z in zips.items():
    assert sorted(z.namelist()) == names, f"{k}: 파일 목록이 다르다 ({len(z.namelist())}장)"

mismatch = {"hist_vs_one": [], "one_vs_two": []}
for n in names:
    h = decode_png_gray8(zips["hist"].read(n))
    o = decode_png_gray8(zips["one"].read(n))
    t = decode_png_gray8(zips["two"].read(n))
    if not np.array_equal(h, o):
        mismatch["hist_vs_one"].append(n)
    if not np.array_equal(o, t):
        mismatch["one_vs_two"].append(n)

print(f"검사 {len(names)}장")
for k, v in mismatch.items():
    print(f"  {k}: 불일치 {len(v)}장" + (f" — 예 {v[:5]}" if v else ""))
print(verify_submission(TWO))
PY
```

Expected: `검사 25988장`, **두 비교 모두 불일치 0장**, 그리고 `verify_submission`이 통과.

- `hist_vs_one` 불일치 0 = **리팩터가 숫자를 바꾸지 않았다** (torch→numpy 조립, cv2→stdlib 인코더).
  이것이 과거 리더보드 점수와의 유일한 연결고리다.
- `one_vs_two` 불일치 0 = 두 경로가 같다.

**어느 쪽이든 1장이라도 어긋나면 착지를 중단한다.** `hist_vs_one`이 어긋나면 리팩터가 산술을
바꾼 것이고, `one_vs_two`가 어긋나면 CPU 진입점이 GPU 경로와 갈린 것이다 — 원인이 다르므로
어느 쪽이 깨졌는지부터 확인한다.

바이트가 아니라 **픽셀**로 비교하는 이유: 인코더가 cv2에서 stdlib으로 바뀌었고, zip
타임스탬프도 고정값으로 바뀌었다. 채점되는 것은 픽셀이다.

- [ ] **Step 4: 증거를 기록한다**

`uv run python scripts/exp.py` 로 새 선보고를 내지 않는다 — 이것은 실험이 아니라 리팩터링 증거다. 출력(장수·불일치 0·`verify_submission` 결과)을 PR 설명에 붙인다.

- [ ] **Step 5: 임시 산출물을 지운다**

```bash
rm runtime/cache/regress-structure.npy runtime/cache/regress-proba.npy
rm runtime/submissions/regress-oneshot.zip runtime/submissions/regress-twostage.zip
```

---

## 착지 전 점검

- [ ] `uv run pytest tests/ -q` 전부 통과
- [ ] `uv run python .agent-hooks/test_block_runtime_commands.py` rc=0
- [ ] `uv run ruff check src/ scripts/ tests/` 위반 없음
- [ ] Task 10의 픽셀 불일치 0장
- [ ] `git fetch origin main && git merge origin/main` 후 위 전부 재확인 (`.agents/rules/git-workflow.md` → Merge main before opening a PR)
- [ ] `self-review.md` 실행, 두 절반 모두 PR 설명에 포함
