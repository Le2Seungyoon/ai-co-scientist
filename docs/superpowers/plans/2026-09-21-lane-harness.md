# 병렬 레인 하네스 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 가설 하나 = 파일 하나 = 작성자 하나로 원장을 쪼개고, 기계 단위 자원 락과 오케스트레이션 규칙을 갖춘 뒤, **실제 Orca 레인 두 개로 그것이 도는 것을 측정한다.**

**Architecture:** 원장을 `docs/experiment/H<id>-*.md`로 분할해 레인이 코디네이터를 거치지 않고 자기 파일에 직접 쓰게 한다(섹션별 단일 작성자 → git이 자동 머지). 레지스트리는 저장소로 남고 `hypothesis` 필드로 가설과 조인된다. GPU·제출 슬롯은 저장소 **바깥** 기계 단위 락으로 배타화한다. 검증은 `2026-09-13`이 연 CPU 후처리 공간에서, 실제 레인 두 개로 한다.

**Tech Stack:** Python 3.12 · numpy(본 의존성) · stdlib `os`/`tempfile`/`contextlib` · pytest · Orca CLI **1.4.206** · Windows 11

**Spec:** `docs/superpowers/specs/2026-09-21-lane-harness-design.md`

## Global Constraints

- **`src/ai_co_scientist/`의 새 코드는 numpy + stdlib만.** `torch`·`opencv-python`은 `baseline` 그룹이고 워크트리는 dev 그룹만 sync한다.
- **테스트는 오프라인**: GPU·네트워크·`data/`·실제 `runtime/` 미접근. 합성 데이터만.
- **언어**: `src/`·`scripts/`의 docstring·주석은 **한국어** · `.agents/**`와 훅 deny 메시지는 **영어** · `docs/**`는 **한국어** (`.agents/rules/docs.md`).
- **ruff `line-length = 100`.** 명령은 저장소 안에서 `uv run`으로.
- **`scripts/`는 얇게, 로직은 `src/`로** (`architecture.md` → CLI / logic separation).
- **커밋 메시지는 영어 단문 제목**, 본문 불릿 없음. **AI attribution 금지** (`Co-Authored-By` / `Generated with` → PreToolUse 훅이 deny).
- **지시 파일은 ~150줄 예산** (`check_rules_size.py`, advisory). 넘으면 잘라내지 말고 보고할 것.
- **Part C는 Orca 코디네이터 세션에서만** 실행된다 — `$ORCA_TERMINAL_HANDLE`이 있어야 Run에 바인딩된다.

---

## File Structure

| 파일 | 책임 | 변경 |
|---|---|---|
| `src/ai_co_scientist/locks.py` | 크로스플랫폼 배타 락 프리미티브 | **신규** |
| `src/ai_co_scientist/registry.py` | 실험 기록소 | `locked()`를 `locks.py`로 위임 · `hypothesis` 필드 |
| `scripts/exp.py` | 기록소 CLI | `--hypothesis` 인자 |
| `docs/experiment/` | 가설 카드 (분할 원장) | **신규 디렉터리** + `_TEMPLATE.md` |
| `.agents/rules/experiment-ledger.md` | 분할 원장 규약 | **신규** |
| `.agents/rules/orca-parallel.md` | Orca lifecycle | **1.4.206 실측으로 재작성** |
| `scripts/dump_level_proba.py` | real 레벨 사후확률 덤프 (GPU, main 전용) | **신규** |
| `scripts/sweep_level_smoothing.py` | 레벨 평활 스윕 (CPU 전용, 레인이 실행) | **신규** |
| `AGENTS.md` | 규칙 전달 표 | 새 규칙 2개 등재 |

---

### Task 1: 기계 단위 자원 락

`registry.locked()`의 획득 루프를 자원 이름으로 일반화한다. **기본 경로가 저장소 바깥이어야** 워크트리와 main이 같은 자원을 두고 실제로 다툰다.

**Files:**
- Create: `src/ai_co_scientist/locks.py`
- Modify: `src/ai_co_scientist/registry.py` (`locked` 39-90 부근)
- Test: `tests/test_locks.py` (신규)

**Interfaces:**
- Consumes: 없음
- Produces:
  - `file_lock(lock_path, *, timeout: float, stale: float = 120.0)` — contextmanager. 획득 실패 시 `ResourceBusy`
  - `resource_lock(name: str, *, timeout: float = 0.0, root=None)` — contextmanager. 기본 경로 `tempfile.gettempdir()/ai-co-scientist-locks/<name>.lock`
  - `ResourceBusy(RuntimeError)`
  - `LOCK_STALE = 120.0`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_locks.py` 생성:

```python
"""자원 락 — 배타성과 스테일 회수를 **행동으로** 검사한다 (오프라인, 합성 경로).

`flock`은 이 호스트에 없다. `O_CREAT|O_EXCL`이 그 자리를 채우므로, 이 파일이 재는 것은
"두 번째 획득이 실제로 거부되는가"와 "죽은 보유자의 락이 회수되는가"다.
"""
import os
import time
from pathlib import Path

import pytest

from ai_co_scientist.locks import ResourceBusy, file_lock, resource_lock


def test_second_acquire_fails_immediately_at_zero_timeout(tmp_path):
    """timeout=0은 큐잉하지 않고 즉시 거부한다 — flock -w 0과 같은 의미."""
    lock = tmp_path / "gpu.lock"
    with file_lock(lock, timeout=0.0):
        with pytest.raises(ResourceBusy):
            with file_lock(lock, timeout=0.0):
                pass


def test_lock_is_released_on_exit(tmp_path):
    lock = tmp_path / "gpu.lock"
    with file_lock(lock, timeout=0.0):
        pass
    with file_lock(lock, timeout=0.0):  # 재획득 가능해야 한다
        pass
    assert not lock.exists()


def test_lock_is_released_when_the_body_raises(tmp_path):
    """예외로 빠져나가도 락이 남으면 그 자원은 영구 점유된다."""
    lock = tmp_path / "gpu.lock"
    with pytest.raises(ZeroDivisionError):
        with file_lock(lock, timeout=0.0):
            1 / 0
    assert not lock.exists()


def test_stale_lock_is_reclaimed(tmp_path):
    """죽은 프로세스가 남긴 락은 회수된다 — 아니면 재부팅까지 자원이 잠긴다."""
    lock = tmp_path / "gpu.lock"
    lock.write_text("99999", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    with file_lock(lock, timeout=0.0, stale=120.0):
        pass


def test_fresh_lock_is_not_reclaimed(tmp_path):
    """스테일 판정이 너무 공격적이면 살아있는 보유자를 밀어낸다."""
    lock = tmp_path / "gpu.lock"
    lock.write_text("99999", encoding="utf-8")
    with pytest.raises(ResourceBusy):
        with file_lock(lock, timeout=0.0, stale=120.0):
            pass


def test_resource_lock_default_root_is_outside_the_repo(tmp_path):
    """워크트리마다 다른 파일을 잠그면 락이 아니라 장식이다 —
    `architecture.md`가 이미 이름 붙인 `registry.locked()`의 트리별 경로 결함."""
    import tempfile

    from ai_co_scientist.config import project_root

    with resource_lock("probe-default-root") as held:
        assert Path(tempfile.gettempdir()) in Path(held).parents
        assert project_root() not in Path(held).parents


def test_resource_lock_excludes_by_name(tmp_path):
    with resource_lock("gpu-0", root=tmp_path):
        with pytest.raises(ResourceBusy):
            with resource_lock("gpu-0", root=tmp_path):
                pass
        with resource_lock("dacon-slot", root=tmp_path):  # 다른 자원은 막히지 않는다
            pass
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_locks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ai_co_scientist.locks'`

- [ ] **Step 3: 구현한다**

`src/ai_co_scientist/locks.py` 생성:

```python
"""크로스플랫폼 배타 락 — 자원 하나에 실행 하나.

`flock`은 이 호스트(Windows)에 없다. `O_CREAT|O_EXCL`은 POSIX·Windows 모두에서 원자적이라
별도 의존성 없이 같은 보장을 준다. 획득 루프는 `registry.locked()`가 쓰던 것을 자원 이름으로
일반화한 것이며, **Windows 전용 실패 경로를 그대로 들고 온다** — 8스레드 x 40회 타격에서
재현됐고, 안 잡으면 그 스레드의 작업이 통째로 소실된다.

**기본 경로가 저장소 바깥인 것이 핵심이다.** `project_root()`는 `__file__` 기준이라 워크트리마다
다른 경로를 준다. 레지스트리에는 그것이 의도지만(복제 금지), GPU와 제출 슬롯은 모든 워크트리가
같은 자원을 두고 다투므로 기계 단위 경로여야 한다.
"""
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_STALE = 120.0  # 이보다 오래된 락은 죽은 프로세스가 남긴 것으로 보고 회수한다
_LOCK_DIRNAME = "ai-co-scientist-locks"


class ResourceBusy(RuntimeError):
    """다른 보유자가 자원을 들고 있다. `timeout=0`에서는 즉시 난다."""


@contextmanager
def file_lock(lock_path, *, timeout: float, stale: float = LOCK_STALE):
    """이 경로를 배타적으로 잡는다. 보유 중 경로를 yield한다.

    timeout=0은 큐잉 없이 즉시 실패한다 — `flock -w 0`과 같은 의미다. 못 잡은 쪽은 기다리며
    그렇게 말해야 하고, 다른 자원으로 옮기면 안 된다(그것이 두 작업을 한 카드에 올리는 경로다).
    """
    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            # PermissionError는 **Windows 전용 경로**다. 막 unlink된 파일이 delete-pending
            # 상태면 O_CREAT|O_EXCL이 EEXIST가 아니라 EACCES를 던진다 — POSIX 가정으로 짜면
            # 놓친다. `exists()`와 `stat()` 사이에 해제되면 FileNotFoundError(TOCTOU).
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue  # 방금 해제됐다 — 즉시 재시도
            if age > stale:
                lock.unlink(missing_ok=True)  # 죽은 프로세스가 남긴 락 회수
                continue
            if time.monotonic() > deadline:
                raise ResourceBusy(f"자원이 사용 중이다({timeout}초 대기): {lock}")
            time.sleep(0.05)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield lock
    finally:
        lock.unlink(missing_ok=True)


def resource_lock(name: str, *, timeout: float = 0.0, root=None):
    """이름 붙은 기계 단위 자원을 잡는다 — `gpu-0`, `dacon-slot`.

    root를 주지 않으면 저장소 바깥(tempfile.gettempdir())에 놓는다. 그래야 워크트리에서 잡은
    락이 main의 실행을 실제로 막는다.
    """
    base = Path(root) if root is not None else Path(tempfile.gettempdir()) / _LOCK_DIRNAME
    return file_lock(base / f"{name}.lock", timeout=timeout)
```

`src/ai_co_scientist/registry.py`에서 획득 루프를 위임한다. `LOCK_TIMEOUT` / `LOCK_STALE` 상수는 그대로 두고, `locked`의 본문만 교체한다:

```python
from ai_co_scientist.locks import file_lock


@contextmanager
def locked(path=None):
    """기록소 갱신 직렬화 — **읽기와 쓰기를 함께 감싸야** 한다.

    sub-agent를 병렬로 돌리면 두 에이전트가 같은 `len(records)`를 보고 **같은 report_id**를
    발급하고, 나중 write가 앞선 선보고를 통째로 덮어쓴다(실측 확인). `_write_all`이 파일 전체를
    다시 쓰는 load-modify-write이므로 락 없이는 append조차 안전하지 않다.

    획득 루프는 `locks.file_lock`에 있다. **경로는 의도적으로 트리별이다** — 기록소는 워크트리가
    복제하지 않는 자원이고, 기계 단위 자원은 `locks.resource_lock`이 맡는다.
    """
    with file_lock(_path(path).with_suffix(".lock"), timeout=LOCK_TIMEOUT, stale=LOCK_STALE):
        yield
```

`registry.py`에서 더 이상 쓰이지 않게 된 `os`·`time` import는 다른 사용처가 없을 때만 지운다 — 지우기 전에 `grep -n "os\.\|time\." src/ai_co_scientist/registry.py`로 확인한다.

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_locks.py tests/test_registry.py -v`
Expected: PASS — 신규 7개 + 기존 registry 테스트 전부. **기존 동시성 테스트가 깨지면 위임이 잘못된 것이다.**

Run: `uv run ruff check src/ tests/`
Expected: 위반 없음

- [ ] **Step 5: 커밋**

```bash
git add src/ai_co_scientist/locks.py src/ai_co_scientist/registry.py tests/test_locks.py
git commit -m "Add machine-scoped resource locks outside the repository"
```

---

### Task 2: 레지스트리 ↔ 가설 조인 키

지금 가설과 실행의 연결은 `docs/hypotheses.md`의 **손으로 유지하는 표 한 줄**이다. 레인이 병렬로 쓰면 그 표가 충돌 지점이고, 빠뜨려도 아무도 모른다.

**Files:**
- Modify: `src/ai_co_scientist/registry.py` (`new_report`)
- Modify: `scripts/exp.py` (`new` 서브파서와 호출부)
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: 없음
- Produces: `new_report(..., hypothesis: str, ...)` — **필수 키워드**. 레코드의 `hypothesis` 키(문자열)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_registry.py`에 추가한다 (이 파일은 `import pytest`와 `from ai_co_scientist import registry`, 그리고 `BASE` 딕셔너리를 이미 갖고 있다):

```python
def test_new_report_records_the_hypothesis(tmp_path):
    """가설 ↔ 실행 조인은 손으로 유지하는 표가 아니라 레코드가 들어야 한다."""
    path = tmp_path / "r.jsonl"
    rec = registry.new_report(**BASE, hypothesis="H12", path=path)
    assert rec["hypothesis"] == "H12"
    assert registry.get(rec["report_id"], path)["hypothesis"] == "H12"


def test_new_report_refuses_without_a_hypothesis(tmp_path):
    """조인 없는 레코드는 조인에 보이지 않고 아무도 눈치채지 못한다 —
    조용한 실패이므로 관례가 아니라 거부로 막는다."""
    path = tmp_path / "r.jsonl"
    with pytest.raises(TypeError):
        registry.new_report(**BASE, path=path)


def test_new_report_refuses_a_blank_hypothesis(tmp_path):
    """빈 문자열을 통과시키면 필수 인자가 형식뿐인 것이 된다."""
    path = tmp_path / "r.jsonl"
    with pytest.raises(ValueError):
        registry.new_report(**BASE, hypothesis="   ", path=path)


def test_many_runs_may_answer_one_hypothesis(tmp_path):
    """관계는 다대다다 — EXP-010은 3-arm을 한 항목으로 기록했다."""
    path = tmp_path / "r.jsonl"
    a = registry.new_report(**BASE, hypothesis="H12", path=path)
    b = registry.new_report(**BASE, hypothesis="H12", path=path)
    assert a["report_id"] != b["report_id"]
    ids = [r["report_id"] for r in registry.load_all(path) if r["hypothesis"] == "H12"]
    assert ids == [a["report_id"], b["report_id"]]
```

**기존 테스트가 깨진다** — `BASE`로 `new_report`를 부르는 모든 케이스가 이제 `hypothesis`를 요구한다. 그 호출부에 `hypothesis="H0"`를 더한다. 이것은 의도된 계약 변경이며, 깨지는 것이 곧 게이트가 작동한다는 증거다.

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL — `TypeError: new_report() got an unexpected keyword argument 'hypothesis'`

- [ ] **Step 3: 구현한다**

`new_report` 시그니처에 **필수 키워드**로 더한다 (기존 `source_branch`/`source_commit` 앞):

```python
def new_report(*, title, x_domain, x_desc, y_source, y_desc, model, method, purpose,
               metric_name, metric_x_domain, metric_y_source, hypothesis,
               source_branch="", source_commit="", path=None) -> dict:
```

docstring 첫 줄 뒤에 한 줄 더한다: `가설 id(`hypothesis`)는 필수다 — 조인 없는 레코드는 조인에 보이지 않는다.`

기존 `_require` 검증 옆에 더한다:

```python
    hypothesis = _require("hypothesis", hypothesis)
```

record 딕셔너리에서 `"created"` 바로 다음 줄에 더한다:

```python
            # 가설 ↔ 실행 조인 키. 다대다다 — 한 실행이 여러 가설에 답할 수 있고, 한 가설이
            # 여러 실행을 갖는다. 손으로 유지하는 표를 대신하므로 선택이 아니라 필수다.
            "hypothesis": hypothesis,
```

`scripts/exp.py`의 `new` 서브파서에 더한다:

```python
    new.add_argument("--hypothesis", required=True,
                     help="이 실험이 답하는 가설 id (docs/experiment/H<id>-*.md)")
```

그리고 `scripts/exp.py:59-63` 부근의 호출부에 `hypothesis=a.hypothesis,`를 더한다. **이 파일은 파싱 결과를 `a`로 받는다**(`args`가 아니다).

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/ -q`
Expected: PASS — 전체 스위트

Run: `uv run ruff check src/ scripts/ tests/`
Expected: 위반 없음

- [ ] **Step 5: 커밋**

```bash
git add src/ai_co_scientist/registry.py scripts/exp.py tests/test_registry.py
git commit -m "Require a hypothesis id on every experiment pre-report"
```

---

### Task 3: 분할 원장 — 디렉터리·템플릿·규약

**Files:**
- Create: `docs/experiment/_TEMPLATE.md`
- Create: `.agents/rules/experiment-ledger.md`
- Modify: `docs/hypotheses.md` (머리말 — 역할 경계)
- Modify: `AGENTS.md` (규칙 전달 표)

**Interfaces:**
- Consumes: Task 2의 `hypothesis` 필드
- Produces: `docs/experiment/H<id>-<slug>.md` 규약

- [ ] **Step 1: 템플릿을 만든다**

`docs/experiment/_TEMPLATE.md`:

```markdown
---
id: H00
status: 계획            # 계획 → 진행중 → 측정됨 → 판정
verdict: 미검증          # 미검증 / 채택 / 조건부 / 기각 / 결론 없음
axis: 레벨               # 오차 예산의 어느 성분인가 — sim 구조 / 갭 / 레벨
lane: null              # dispatch 시 레인 이름
registry: []            # 이 가설에 답한 report_id 목록 (EXP-0NN)
---

# H00. <한 줄 가설>

## 질문
<무엇을 묻는가. 한두 문장.>

## 조건
<언제 · 무엇으로 재는가. X와 y의 도메인, 모집단, 모델·설정, seed 정책.
 이전 실험과 비교하려면 무엇이 같아야 하는지까지.>

## 무엇이 답인가
<**dispatch 전에** 박아 둔다. 어떤 수치가 어떻게 나오면 채택/조건부/기각인가.
 **"돌리지 않을 조건"도 함께 쓴다** — 그것이 있어야 안 돌리고 닫을 수 있다.
 사후에 기준을 옮기지 않기 위한 절이다.>

## 결과
<report_id로 가리킨다. 판정을 읽는 데 필요한 대표 수치만 옮겨 적는다 —
 전체 수치는 기록소가 원본이다.>

| report_id | arm | 지표 | 비고 |
|---|---|---|---|

## 관찰
<수치 옆에 붙는 해석. 왜 그렇게 보이는가, 무엇이 잡음인가.>

## 판정 · 미검증
**판정**: <채택 / 조건부(무엇이 해소되면) / 기각(근거) / 결론 없음(밴드를 함께 적는다 —
 "무해하다"로 읽지 않는다)>

**미검증**: <재보지 못한 것과 그 이유. 교란이 남았으면 여기.>

## 이관 범위
<이 결과를 어디까지 재사용할 수 있는가. merge 후 오케스트레이터가 적는다.>
```

- [ ] **Step 2: 규약 파일을 쓴다**

`.agents/rules/experiment-ledger.md` (영어, ~80줄 목표):

```markdown
# Experiment ledger (docs/experiment/)

Where a hypothesis lives while it is being tested, and how parallel lanes write results without
fighting over one file. `runtime/registry.jsonl` owns the numbers; this file owns the document
side and the join between them.

## One hypothesis, one file, one writer at a time

`docs/experiment/H<id>-<slug>.md`, Korean, from `docs/experiment/_TEMPLATE.md`. **An id prefix,
not a date** — a hypothesis is revisited, and a date would lie about when it was last true
(specs keep dates: they are decisions made at one moment and never edited again).

| Section | Written by | When |
|---|---|---|
| 질문 · 조건 · 무엇이 답인가 | orchestrator | before dispatch, on `main` |
| 결과 · 관찰 · 판정 · 미검증 | the lane | during/after its run |
| 이관 범위 | orchestrator | after merge |

**A lane touches only its own hypothesis file.** Not another lane's, not the backlog. Two lanes
therefore never edit the same path, so git merges them without a conflict — the property depends
on the single-writer rule, not on the directory.

**The branch point is the snapshot.** A lane cut from commit X already holds every hypothesis
file as of X; copying the ledger into the task spec would recreate the divergence this layout
removes.

## No shared index

Status lives in each file's front-matter; `ls docs/experiment/` plus that is the view. A
hand-maintained index puts every lane back on one hunk — the exact conflict the split removes.
When a table is genuinely wanted, **generate it** and give it a freshness test, never hand-write
it (`enforcement.md`).

`docs/hypotheses.md` is the **error budget and the backlog** — what has not been tried and what
was rejected. A dispatched hypothesis moves to its own file and is closed there.

## The registry is the original; the file quotes it

- **Numbers, conditions, the code commit: `runtime/registry.jsonl`.** The file copies only the
  headline figures a reader needs to follow the verdict.
- **Judgment, plan, transfer scope: this file.**
- **If it is not in the registry, it is not a result.**

## The join key

Every pre-report carries `hypothesis=<id>`, so the link is queryable both ways: which runs tested
H12, and why a run exists at all. A report names exactly one hypothesis; a hypothesis accumulates
many reports (EXP-010 recorded a 3-arm sweep as one entry, under one id).
`registry.new_report` refuses without it, and `tests/test_registry.py` pins that — an untagged record is invisible to the join and
nothing else notices, which is why it is a refusal and not a convention.

## Write the "don't run" condition too

A hypothesis that pins a symptom on something the optimiser or the model can absorb is usually
already answered. 「무엇이 답인가」 states the condition for running **and** the condition for
closing it unrun — the second is what makes a hypothesis cost nothing.

## One lane, one hypothesis

Two lanes on one hypothesis is a dispatch bug, not a merge bug: the single-writer property above
is what the automatic merge depends on. The orchestrator writes 질문 · 조건 · 무엇이 답인가
**before** dispatch, on `main`, so pre-registration is proved by commit order. How a lane is
dispatched and driven: `orca-parallel.md`.
```

- [ ] **Step 3: 백로그의 역할을 명시한다**

`docs/hypotheses.md` 머리말의 "역할 분담" 문장 옆에 한 줄 더한다:

```markdown
> **진행 중인 가설은 여기 없다.** dispatch된 가설은 `docs/experiment/H<id>-*.md`로 나가고
> (`.agents/rules/experiment-ledger.md`), 이 파일은 **오차 예산과 백로그** — 아직 안 한 것과
> 기각된 것 — 를 담는다. 레인이 병렬로 쓰는 것은 각자의 가설 파일이지 이 표가 아니다.
```

- [ ] **Step 4: 규칙을 등재한다**

`AGENTS.md`의 규칙 전달 표에 한 행 더한다 (기존 행들과 같은 형식, 영어):

```markdown
| `experiment-ledger.md` | when writing `docs/experiment/**`, dispatching an experiment lane, or asking where a result or judgment belongs — one hypothesis one file, the single-writer sections, the join key |
```

- [ ] **Step 5: 검증하고 커밋한다**

Run: `uv run pytest tests/test_harness_generated.py -v`
Expected: PASS — 규칙 링크 스캔이 새 파일 경로를 검사한다

Run: `uv run python .agent-hooks/check_rules_size.py` (또는 규칙 파일을 한 번 편집해 PostToolUse 훅이 돌게 한다)
Expected: `experiment-ledger.md`가 ~150줄 예산 안

```bash
git add docs/experiment/_TEMPLATE.md .agents/rules/experiment-ledger.md docs/hypotheses.md AGENTS.md
git commit -m "Split the experiment ledger into one file per hypothesis"
```

---

### Task 4: 오케스트레이션 규칙 — `orca-parallel.md` 재작성

**현재 `orca-parallel.md`는 버전이 밀린 것이 아니라 다른 메커니즘을 설명한다.** 2026-09-21에 CLI 표면을 확인했다(1.4.206): 문서가 적은 `dispatch --to <worker handle> --inject`(이미 열린 터미널에 주입) 자리에 supervised worker 계열이 들어와 있다. **낡은 lifecycle을 `.agents/rules/`에 남기는 것은 없는 것보다 나쁘다** — 에이전트가 그것을 따르기 때문이다.

**Files:**
- Modify: `.agents/rules/orca-parallel.md` (전면 재작성)
- Modify: `AGENTS.md` (해당 행의 트리거 문구)
- Modify: `../CLAUDE.md` (오케스트레이터 컨테이너 — **저장소 밖**)

**Interfaces:**
- Consumes: Task 3의 `experiment-ledger.md`
- Produces: 없음

- [ ] **Step 1: 실측된 표면만 남긴다**

`orca-parallel.md`를 다시 쓴다. **측정하지 않은 것을 적지 않는다** — Task 7이 왕복을 실측하기 전까지, 이 문서는 "CLI가 이렇게 생겼다"까지만 말하고 동작 주장은 하지 않는다. 문서 머리에 버전과 날짜를 박는다:

````markdown
# Orca — running experiments across parallel sessions

> **Orca-only**, CLI surface read on **1.4.206 / Windows, 2026-09-21**. Behaviour claims carry
> their own measurement date; anything unmeasured says so. Sub-agents stay the default — reach
> for a second *session* only when the work must outlive a turn, hold its own approval gate, or
> hold the GPU while this session keeps planning.

## The lifecycle

```bash
orca orchestration run-create --objective "<what this batch is for>" --json   # binds THIS terminal
orca orchestration worker-start --spec "<task spec>" --agent claude \
    --worktree new-top-level --repo path:<repo> --base-branch <ref> \
    --name lane-<x> --display-name <branch> --setup run --json
orca orchestration check --wait --timeout-ms 45000 --json                     # collect
orca orchestration worker-release --dispatch <dispatch_id> --json             # after it settles
```

`worker-start` creates the worktree, launches the agent and injects the spec in one action —
there is no separate "open a terminal, then dispatch into it" step. `coordinator-start` is
retired; the worker contract now arrives as an Orca skill.

**The coordinator is a terminal, not a person.** `run-create` binds the terminal that runs it,
so every later `check` reads that Run. `$ORCA_TERMINAL_HANDLE` names it; after a reconnect,
resolve it from `orca terminal list` instead of trusting the variable.

## Reading the waiter

`check --wait` emits JSON keepalive lines on **stderr** every 15 s (`_keepalive`), which is how a
caller tells a live wait from a hung one. Filter them out when merging streams. `_heartbeat` is a
deprecated alias.

**Read `ok` before the payload.** A Run holds one active waiter; a second `check --wait` is
refused, and a parser that reaches straight for the count renders that refusal as "nothing has
arrived yet".

## A silent wait is the coordinator's failure

While the coordinator blocks, the person who asked sees nothing and cannot tell a running lane
from a stuck one. **Bound every wait and report at each expiry** — elapsed time, the lane's
liveness verdict, and, when it needs something, the one action that unblocks it. Never re-enter
a wait silently. An unverifiable verdict means *unknown*, never *running*.

**A permission prompt is the one thing waiting cannot resolve.** While one is open the session is
not merely unwatched but **blocked**: messages queue until its next tool round, which does not
come until a person answers. Surface it the moment it appears.

## What goes in a spec

**Open with the approval scope.** State that the user approved this lane, list what is approved
and what is not, and tell the lane to route new questions through the preamble's `ask`. Measured
in the sibling repo (custflow, 2026-09-21, same worker-start and one variable): without the scope
both lanes **asked at their own window and sat idle**; with it both started within a minute.

**Approval relayed after the fact does not work, and the lane is right to refuse it** — it is a
quote the lane cannot verify, and a coordinator rule that makes its own relays authoritative is
an agent granting itself authority. The scope arrives as part of the task, never as a correction
to it.

Put in the spec only the task, the domain limits, the file domain it owns, and the hypothesis
file it writes (`experiment-ledger.md`). **Do not copy the ledger in** — the branch point is
already the snapshot.

## Not measured — treat as open

Everything above is the CLI surface plus the sibling repo's measurements. **This repo has not yet
measured a round trip on 1.4.206.** Until it has, no claim here about delivery, injection or
worker lifetime is this repo's own.
````

- [ ] **Step 2: 컨테이너의 워크트리 경로를 고친다**

`orca worktree create` / `worker-start --worktree new-top-level`은 경로를 `<base>/<repo dir basename>/<name>`으로 강제하고 `<base>`만 설정 가능하다. 오케스트레이터 `CLAUDE.md`는 `.worktrees/<name>`으로 문서화돼 있어 어긋난다.

`../CLAUDE.md`(저장소 **밖**, 컨테이너)에서 워크트리 경로를 `.worktrees/ai-co-scientist/<name>`으로 고치고, 이유를 한 줄 적는다: 손으로 만든 워크트리는 Orca에 기록이 남지 않아 세션·보드·`worktree list` 어디에도 안 보인다.

**이 파일은 git 저장소가 아니다** — 커밋 대상이 아니므로 변경 사실을 보고서에 남긴다.

- [ ] **Step 3: 검증하고 커밋한다**

Run: `uv run pytest tests/test_harness_generated.py -v`
Expected: PASS — 링크 스캔·생성 레인 신선도·하네스 패리티

Run: 규칙 파일 크기 확인 — `orca-parallel.md`가 ~150줄 예산 안

```bash
git add .agents/rules/orca-parallel.md AGENTS.md
git commit -m "Rewrite the Orca lifecycle against the 1.4.206 CLI surface"
```

---

### Task 5: real 레벨 사후확률 덤프

레인은 torch 없이 돌아야 한다(워크트리는 dev 그룹만 sync). 그러므로 CNN 예측을 **main에서 한 번 덤프**하고 레인은 그것을 읽는다 — `2026-09-13`의 ŝ 덤프와 같은 패턴이다.

**Files:**
- Modify: `scripts/infer_decomposed.py` (`predict_levels_cnn`)
- Create: `scripts/dump_level_proba.py`
- Test: `tests/test_train_manifest.py`

**Interfaces:**
- Consumes: 없음
- Produces: `predict_levels_cnn(cache, ckpt, batch=512, return_proba=False, npy="test_sem.npy")` · `scripts/dump_level_proba.py` CLI

- [ ] **Step 1: 실패하는 소스 계약 테스트를 쓴다**

`tests/test_train_manifest.py`에 추가:

```python
def test_predict_levels_cnn_takes_a_source_array():
    # 레벨 사후확률을 real에서도 뽑을 수 있어야 CPU 레인이 스윕할 입력이 생긴다
    source = _script("infer_decomposed.py")
    fn = source.split("def predict_levels_cnn(")[1].split("\ndef ")[0]
    assert "npy" in fn, "소스 배열을 고를 수 없다 — test_sem.npy에 고정돼 있다"


def test_dump_level_proba_declares_its_domain():
    # real train SEM → real group label. 리더보드 타깃(real_depth_gt)이 아니다
    source = _script("dump_level_proba.py")
    assert '"x_domain": "real"' in source
    assert '"y_source": "real_group_label"' in source
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_train_manifest.py -v`
Expected: FAIL — `dump_level_proba.py` 없음 · `npy`가 함수 본문에 없음

- [ ] **Step 3: 구현한다**

`scripts/infer_decomposed.py`의 `predict_levels_cnn` 시그니처와 로드 줄만 고친다. **AdaBN을 걸지 않는다는 성질을 건드리지 않는다** — `test_level_cnn_gets_no_adabn`이 이 함수 본문에 `adapt_bn`이 없음을 검사한다.

```python
@torch.no_grad()
def predict_levels_cnn(cache: Path, ckpt: str, batch: int = 512,
                       return_proba: bool = False,
                       npy: str = "test_sem.npy") -> np.ndarray | tuple:
```

docstring에 한 줄 더한다: `npy로 소스 배열을 고른다 — real train(real_sem.npy)에서 뽑으면 레벨 축 후처리의 사전 선별 지표가 된다.`

본문의 로드 줄을 바꾼다:

```python
    sem = np.load(cache / npy, mmap_mode="r")
```

`scripts/dump_level_proba.py` 생성:

```python
"""레벨 사후확률 덤프 — CPU 레인이 평활 스윕에 쓸 입력을 GPU에서 한 번 뽑는다.

`2026-09-13`의 ŝ 덤프와 같은 패턴이다: GPU가 필요한 부분을 한 번만 치르고, 그 산출물을
읽기 전용으로 공유해 후처리 가설을 병렬로 돌린다. LevelCNN은 torch라 워크트리에서 돌지
않으므로, 이 스크립트는 **main 체크아웃 전용**이다.

X = real train SEM · y = 폴더명 Depth_{110,120,130,140}(주최측 실측 라벨). 둘 다 real이라
이 프로젝트에서 드물게 도메인 정합 검증이 성립한다 — 다만 리더보드 대리는 아니다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_co_scientist.config import ensure_utf8_console  # noqa: E402
from infer_decomposed import predict_levels_cnn  # noqa: E402


def main() -> int:
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="runtime/cache")
    ap.add_argument("--level-ckpt", default="runtime/ckpt/EXP-013-level-cnn.pt")
    ap.add_argument("--npy", default="real_sem.npy", help="캐시 안의 소스 배열")
    ap.add_argument("--out", required=True, help="사후확률 .npy 출력 경로 (N,4) float32")
    args = ap.parse_args()

    _, proba = predict_levels_cnn(Path(args.cache_dir), args.level_ckpt,
                                  return_proba=True, npy=args.npy)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, proba.astype(np.float32))

    print(json.dumps({
        "x_domain": "real", "y_source": "real_group_label",
        "source_npy": args.npy, "level_ckpt": args.level_ckpt,
        "out": str(out), "shape": list(proba.shape),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/ -q`
Expected: PASS — 전체. 특히 `test_level_cnn_gets_no_adabn`과 `test_infer_decomposed_supports_cnn_level_source`가 함께 통과해야 한다.

Run: `uv run ruff check scripts/ tests/`
Expected: 위반 없음

**실제 덤프는 돌리지 않는다** — GPU와 캐시가 필요하고, 그것은 Task 8의 준비 단계다.

- [ ] **Step 5: 커밋**

```bash
git add scripts/infer_decomposed.py scripts/dump_level_proba.py tests/test_train_manifest.py
git commit -m "Dump real-train level posteriors for CPU-only smoothing sweeps"
```

---

### Task 6: 레벨 평활 스윕 — CPU 전용

레인이 실제로 실행할 계산. **torch도 cv2도 import하지 않는다.**

**Files:**
- Create: `scripts/sweep_level_smoothing.py`
- Test: `tests/test_sweep_level_smoothing.py` (신규)

**Interfaces:**
- Consumes: Task 5의 사후확률 덤프 · `sem.smooth_levels` · `sem.viterbi_levels` · `sem.site_split` · `sem.score_classes` · `sem.load_labels`
- Produces: CLI. stdout 마지막 줄에 결과 JSON

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_sweep_level_smoothing.py` 생성:

```python
"""레벨 평활 스윕 — torch 없이 돌고, 런 구조가 실제로 평활에 영향을 주는지 본다.

이 스크립트는 워크트리(dev 그룹만 sync — torch 없음)에서 레인이 돌리는 것이 존재 이유다.
그래서 **import 계약**이 수치 못지않게 중요하다.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_level_smoothing.py"


def _fixture(tmp_path, n_sites=40, per_site=8):
    """사이트 단위 합성 데이터. 사이트 하나는 라벨 하나를 공유한다(실제 구조와 같다)."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), n_sites // 4)
    rng.shuffle(y)
    site = np.repeat(np.arange(n_sites), per_site)
    labels = np.repeat(y, per_site)

    proba = np.full((len(labels), 4), 0.05, dtype=np.float32)
    proba[np.arange(len(labels)), labels] = 0.85
    flip = rng.choice(len(labels), size=len(labels) // 12, replace=False)  # 고립 오분류
    for i in flip:
        wrong = (labels[i] + 1) % 4
        proba[i] = 0.05
        proba[i, wrong] = 0.85
    proba /= proba.sum(1, keepdims=True)

    np.save(tmp_path / "proba.npy", proba.astype(np.float32))
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "site.npy", site)
    return proba, labels, site


def _run(tmp_path, *extra):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--proba", str(tmp_path / "proba.npy"),
         "--labels", str(tmp_path / "labels.npy"),
         "--site", str(tmp_path / "site.npy"), *extra],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_source_imports_neither_torch_nor_cv2():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import cv2" not in source


def test_smoothing_improves_accuracy_on_isolated_errors(tmp_path):
    """런 한가운데 고립된 오분류는 최빈값 필터가 흡수해야 한다."""
    _fixture(tmp_path)
    out = _run(tmp_path, "--arm", "smooth", "--values", "1,5")
    got = {a["value"]: a["holdout_accuracy"] for a in out["arms"]}
    assert got[5] > got[1]


def test_reports_the_run_count_so_comparability_is_visible(tmp_path):
    """런 길이 분포가 test와 다르면 최적 k가 옮겨가지 않는다 — 읽는 사람이 판단해야 한다."""
    _fixture(tmp_path)
    out = _run(tmp_path, "--arm", "smooth", "--values", "5")
    assert out["run_count"] > 0
    assert out["n"] == 320


def test_seed_is_reproducible(tmp_path):
    """같은 시드는 같은 순열 — 재현 가능한 실험의 최소 조건."""
    _fixture(tmp_path)
    a = _run(tmp_path, "--arm", "smooth", "--values", "5", "--seed", "7")
    b = _run(tmp_path, "--arm", "smooth", "--values", "5", "--seed", "7")
    assert a["arms"] == b["arms"]


def test_hmm_arm_runs_and_reports(tmp_path):
    _fixture(tmp_path)
    out = _run(tmp_path, "--arm", "hmm", "--values", "0.9,0.974")
    assert [a["value"] for a in out["arms"]] == [0.9, 0.974]
    assert all("holdout_accuracy" in a for a in out["arms"])


def test_declares_its_domain(tmp_path):
    """real SEM → real group label. 리더보드 타깃이 아니다 — 사전 선별 지표다."""
    _fixture(tmp_path)
    out = _run(tmp_path, "--arm", "smooth", "--values", "5")
    assert out["x_domain"] == "real"
    assert out["y_source"] == "real_group_label"
    assert out["is_leaderboard_proxy"] is False
```

- [ ] **Step 2: 실패를 확인한다**

Run: `uv run pytest tests/test_sweep_level_smoothing.py -v`
Expected: FAIL — 스크립트 파일이 없어 `returncode != 0`

- [ ] **Step 3: 구현한다**

`scripts/sweep_level_smoothing.py` 생성:

```python
"""레벨 평활 스윕 — 덤프된 사후확률에 창 k 또는 Viterbi a를 걸고 홀드아웃 정확도를 잰다.

**GPU를 쓰지 않는다.** 레벨 축 후처리는 사후확률만 읽으므로, 사후확률을 한 번 덤프해 두면
값마다 다른 레인이 동시에 돌 수 있다 — 이 스크립트가 그 레인이 실행하는 것이다.
torch도 cv2도 import하지 않는다(워크트리는 dev 그룹만 sync한다).

**이것은 사전 선별 지표이지 판정이 아니다.** 리더보드가 유일한 판정이고, 이 프록시는 두 번 다
낙관이었다(EXP-013→014에서 2.55pp, EXP-019에서 3.55pp 할인). 읽는 것은 **순위**이지 절대값이
아니다.

**런 구조는 만들어진 것이다.** test는 파일 순서가 곧 내용 순서라 런이 생기지만, real train은
그렇지 않다. 그래서 사이트를 시드로 섞어 이어붙여 런을 만든다 — 사이트 하나는 라벨 하나를
공유하므로 사이트 길이만큼의 런이 생긴다. 결과의 `run_count`를 test의 런 수와 나란히 읽어야
비교 가능성이 판단된다.

**예측은 in-sample이다.** 사후확률은 real train으로 학습된 분류기가 real train에 낸 것이라
절대 정확도는 낙관 편향돼 있다. 두 arm이 같은 편향을 공유하므로 **arm 사이의 순위**만 읽는다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.sem import score_classes, site_split, smooth_levels, viterbi_levels


def build_run_order(site: np.ndarray, seed: int) -> np.ndarray:
    """사이트를 시드로 섞어 이어붙인 인덱스 순열. 사이트 내부 순서는 보존한다."""
    rng = np.random.default_rng(seed)
    ids = np.unique(site)
    rng.shuffle(ids)
    return np.concatenate([np.flatnonzero(site == s) for s in ids])


def count_runs(labels_in_order: np.ndarray) -> int:
    """인접 라벨이 바뀌는 지점의 수 + 1 = 런의 개수."""
    if len(labels_in_order) == 0:
        return 0
    return int((np.diff(labels_in_order) != 0).sum()) + 1


def main() -> int:
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba", required=True,
                    help="레벨 사후확률 .npy (N,4). scripts/dump_level_proba.py의 산출물. "
                         "기본값을 두지 않는다 — 워크트리에서 상대 기본값은 빈 runtime/으로 "
                         "조용히 풀린다")
    ap.add_argument("--labels", required=True, help="정답 클래스 .npy (N,) int")
    ap.add_argument("--site", required=True, help="사이트 id .npy (N,) int")
    ap.add_argument("--arm", required=True, choices=("smooth", "hmm"))
    ap.add_argument("--values", required=True,
                    help="쉼표로 구분한 값 — smooth면 창 k(정수), hmm이면 자기전이 a(실수)")
    ap.add_argument("--seed", type=int, default=0, help="사이트 섞기 시드")
    ap.add_argument("--val-frac", type=float, default=0.2, help="사이트 단위 홀드아웃 비율")
    args = ap.parse_args()

    proba = np.load(args.proba)
    labels = np.load(args.labels)
    site = np.load(args.site)
    if not (len(proba) == len(labels) == len(site)):
        raise SystemExit(f"장수 불일치 — 사후확률 {len(proba)} · 라벨 {len(labels)} · "
                         f"사이트 {len(site)}")

    order = build_run_order(site, args.seed)
    held = site_split(site, labels, args.val_frac, args.seed)   # 이미지 단위 마스크
    base_cls = proba.argmax(1)

    arms = []
    for raw in args.values.split(","):
        raw = raw.strip()
        value = int(raw) if args.arm == "smooth" else float(raw)
        ordered = (smooth_levels(base_cls[order], value) if args.arm == "smooth"
                   else viterbi_levels(proba[order], a=value))
        cls = np.empty_like(base_cls)
        cls[order] = ordered                      # 원래 인덱스로 되돌린다
        changed = int((cls != base_cls).sum())
        scored = score_classes(cls[held], labels[held])
        arms.append({
            "value": value,
            "holdout_accuracy": scored["accuracy"],
            "holdout_adjacent_ok": scored["adjacent_ok"],
            "changed": changed,
            "changed_frac": round(changed / len(cls), 4),
        })

    print(json.dumps({
        "x_domain": "real", "y_source": "real_group_label",
        "is_leaderboard_proxy": False,
        "arm": args.arm, "seed": args.seed, "val_frac": args.val_frac,
        "n": int(len(labels)), "n_holdout": int(held.sum()),
        "run_count": count_runs(labels[order]),
        "baseline_holdout_accuracy": score_classes(base_cls[held], labels[held])["accuracy"],
        "arms": arms,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 통과를 확인한다**

Run: `uv run pytest tests/test_sweep_level_smoothing.py -v`
Expected: PASS

Run: `uv run ruff check scripts/ tests/`
Expected: 위반 없음

- [ ] **Step 5: 커밋**

```bash
git add scripts/sweep_level_smoothing.py tests/test_sweep_level_smoothing.py
git commit -m "Add a CPU-only level smoothing sweep for parallel lanes"
```

---

### Task 7: Orca 왕복 실측 — 레인 하나

**Orca 코디네이터 세션에서만.** 레인 둘을 띄우기 전에 **하나로 왕복을 잰다** — 두 개가 동시에 실패하면 원인이 병렬성인지 lifecycle인지 분리되지 않는다.

**Files:**
- Modify: `.agents/rules/orca-parallel.md` (측정 결과 기록)
- 코드 변경 없음

**Interfaces:**
- Consumes: Task 4의 lifecycle 문서
- Produces: 실측된 `run_id` · `dispatch_id` · 왕복 소요시간

- [ ] **Step 1: 코디네이터를 바인딩한다**

```bash
orca orchestration run-create --objective "lane harness round-trip probe (1.4.206)" --json
```

`result.run.id`를 기록한다. `$ORCA_TERMINAL_HANDLE`이 이 Run의 코디네이터가 된다.

- [ ] **Step 2: 워커 하나를 띄운다 — spec은 승인 범위로 시작한다**

spec 본문 앞에 승인 범위를 둔다(`orca-parallel.md` → What goes in a spec). 이 프로브의 범위는 좁다:

```
## 사용자 승인이 났다
승인됨: 이 워크트리 안에서 읽기, `uv run pytest tests/test_sweep_level_smoothing.py -q` 실행,
        결과를 worker_done의 outcome으로 반환.
승인되지 않음: 커밋, push, 실험 명령(scripts/exp.py · train_* · infer_decomposed · dacon_submit),
        runtime/ 쓰기, 다른 파일 수정.
새 질문은 preamble의 `ask`로 코디네이터에게 보낸다 — 자기 창에서 묻지 않는다.

## 과제
이 워크트리에서 `uv run pytest tests/test_sweep_level_smoothing.py -q`를 실행하고,
통과/실패와 테스트 수를 outcome에 담아 worker_done으로 보고한다. 그 외에는 아무것도 하지 않는다.
```

```bash
orca orchestration worker-start --spec "<위 spec>" --agent claude \
  --worktree new-top-level --repo path:ai-co-scientist --base-branch feature/lane-harness \
  --name lane-probe --display-name feature/lane-probe --setup run --json
```

`result.dispatch.id`와 생성된 워크트리 경로를 기록한다.

- [ ] **Step 3: bound 대기로 수집한다**

```bash
orca orchestration check --wait --timeout-ms 45000 --json
```

**만료마다 보고한다** — 경과시간, 레인의 liveness 판정, 막혔다면 그것을 푸는 행동. `ok`를 먼저 읽고 payload를 읽는다. 조용히 재진입하지 않는다.

- [ ] **Step 4: 실측을 문서에 기록한다**

`orca-parallel.md`의 "Not measured — treat as open" 절을 실제로 잰 것으로 교체한다. 기록할 것: 왕복 소요시간 · `worker_done`이 dispatch/task를 자동 완료하는가 · 어떤 스코프의 `check`가 메시지를 보는가(`--run` 바인딩 vs `--terminal`) · 워크트리가 어디에 만들어졌는가 · `--setup run`이 실제로 무엇을 했는가.

**측정하지 않은 것은 계속 "미측정"으로 남긴다.**

- [ ] **Step 5: 워커를 정리하고 커밋한다**

```bash
orca orchestration worker-release --dispatch <dispatch_id> --json
orca worktree rm --worktree "path:<the worktree>" --force
```

```bash
git add .agents/rules/orca-parallel.md
git commit -m "Record the measured Orca round trip on 1.4.206"
```

---

### Task 8: 레인 둘 — 병렬 실행

**Orca 코디네이터 세션에서만.** 여기가 이 계획의 인수 시험이다.

> **개정 (Task 3 리뷰가 드러낸 계획 결함, 2026-09-21).** 최초 계획은 레인 둘이 **같은 H12
> 파일**의 결과 절을 쓰게 했다. 그것은 `experiment-ledger.md`가 "dispatch 버그"로 규정한
> 바로 그 경우이고 — 검증 실험이 검증 대상 규칙을 위반한다 — 머지 충돌이 구조적으로 발생해
> Task 9의 통과 조건(충돌 0)이 성립하지 않는다. **가설을 둘로 쪼갠다**: 레인 하나당 가설 하나,
> 파일 하나. 검증은 오히려 강해진다 — 서로 다른 파일을 쓰는 두 레인의 머지가 자동이어야 한다.

**Files:**
- Create: `docs/experiment/H12-level-smoothing-window.md` (레인 A가 결과를 쓴다)
- Create: `docs/experiment/H13-viterbi-vs-mode-filter.md` (레인 B가 결과를 쓴다)
- 각 레인은 **자기 파일만** 건드린다

**Interfaces:**
- Consumes: Task 3의 템플릿 · Task 5·6의 스크립트 · Task 7의 실측된 lifecycle
- Produces: 각자 자기 레인의 결과가 담긴 가설 파일 두 개

- [ ] **Step 1: 사후확률을 덤프한다 (main, GPU 1회)**

```bash
uv run python scripts/dump_level_proba.py \
  --npy real_sem.npy \
  --out runtime/cache/real-level-proba.npy
```

`real_sem.npy`는 60,664장이다. 출력은 `(60664, 4) float32` ≈ 971 KB.

라벨과 사이트 배열도 같은 캐시에 저장해 레인이 읽게 한다:

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np
from ai_co_scientist.sem import load_labels
cache = Path("runtime/cache")
n = len(np.load(cache / "real_sem.npy", mmap_mode="r"))
y, site = load_labels(Path("data"), n)
np.save(cache / "real-labels.npy", y)
np.save(cache / "real-site.npy", site)
print(n, y.shape, site.shape)
PY
```

- [ ] **Step 2: 가설 파일을 dispatch 전에 커밋한다**

템플릿에서 **두 개**를 만들고 각각 **질문 · 조건 · 무엇이 답인가**만 채운다. 사전등록은 커밋 순서로 증명되므로 **dispatch 전에 커밋한다.**

| 파일 | 질문 | arm |
|---|---|---|
| `H12-level-smoothing-window.md` | 최빈값 필터의 최적 창이 EXP-019가 쓴 k=9인가 | `--arm smooth --values 3,5,7,9,11` |
| `H13-viterbi-vs-mode-filter.md` | Viterbi 복호가 고정폭 최빈값 필터를 이기는가 | `--arm hmm --values 0.95,0.974,0.99` |

H13이 별개 가설인 근거는 코드에 이미 있다 — `viterbi_levels`의 독스트링이 *"최빈값 필터는 창 크기를 통해 '런은 최소 k/2보다 길다'를 **강제**하므로 경계에서 손해를 본다(k=31이 k=9보다 나쁜 이유). Viterbi는 같은 사전지식을 **비용**으로 넣는다"*라고 적고, `a`의 기본값 0.974를 *"실측이 아니라 실험이 스윕할 하이퍼파라미터"*라고 명시한다. 기전이 다르므로 판정도 따로 난다.

**두 파일 모두의 「무엇이 답인가」에 반드시 들어갈 것:**
- 읽는 것은 **arm 사이의 순위**이지 절대 정확도가 아니다 — 사후확률은 real train에 대한 in-sample 예측이고, 이 프록시는 두 번 다 낙관이었다(2.55pp, 3.55pp)
- **돌리지 않을 조건**: `run_count`가 test의 1,046과 자릿수가 다르면 비교 가능성이 없으므로 그대로 닫는다
- 리더보드에 올리지 않는다 — 제출 슬롯 0

H13에만 추가로: 두 arm은 **같은 사후확률 덤프**를 읽으므로 비교가 성립한다. 다른 덤프를 읽었다면 그 사실을 「미검증」에 적고 판정하지 않는다.

```bash
git add docs/experiment/H12-level-smoothing-window.md docs/experiment/H13-viterbi-vs-mode-filter.md
git commit -m "Pre-register H12 and H13 before dispatching their lanes"
```

- [ ] **Step 3: 레인 둘을 띄운다**

각 spec은 승인 범위로 시작하고, **자기 arm과 자기가 쓸 파일만** 말한다. 원장을 복사해 넣지 않는다 — 분기점이 이미 스냅샷이다.

레인 A(H12, `--arm smooth --values 3,5,7,9,11`)와 레인 B(H13, `--arm hmm --values 0.95,0.974,0.99`)를
각각 — **각 레인은 자기 가설 파일 하나만 건드린다**:

```bash
orca orchestration worker-start --spec "<레인별 spec>" --agent claude \
  --worktree new-top-level --repo path:ai-co-scientist --base-branch feature/lane-harness \
  --name lane-h12-smooth --display-name feature/h12-smooth --setup run --json
```

레인 spec 전문 (레인 A 기준 — B는 arm·가설 파일·브랜치명을 H13 쪽으로 바꾼다):

```
## 사용자 승인이 났다
승인됨: 이 워크트리 안에서 읽기·쓰기, `uv run python scripts/sweep_level_smoothing.py` 실행,
        **자기 가설 파일 하나**(레인 A는 docs/experiment/H12-level-smoothing-window.md,
        레인 B는 docs/experiment/H13-viterbi-vs-mode-filter.md)의 「결과」·「관찰」·
        「판정 · 미검증」 세 절에 쓰기, 이 워크트리 브랜치에 commit, `git push`.
승인되지 않음: 다른 절(질문·조건·무엇이 답인가·이관 범위) 수정, **다른 레인의 가설 파일**을
        포함한 다른 파일 수정,
        scripts/exp.py 호출, train_*/infer_decomposed/dacon_submit 실행, runtime/ 쓰기,
        리더보드 제출, main 또는 다른 레인 브랜치에 대한 어떤 작업.
새 질문은 preamble의 `ask`로 코디네이터에게 보낸다 — 자기 창에서 묻지 않는다.

## 과제
`uv run python scripts/sweep_level_smoothing.py --arm smooth --values 3,5,7,9,11` 을 실행한다.
입력 세 개는 main 체크아웃의 덤프를 **읽기 전용**으로 가리킨다:
  --proba  ../../ai-co-scientist/runtime/cache/real-level-proba.npy
  --labels ../../ai-co-scientist/runtime/cache/real-labels.npy
  --site   ../../ai-co-scientist/runtime/cache/real-site.npy
(경로가 맞지 않으면 추측하지 말고 `ask`로 묻는다. np.load가 시끄럽게 실패하므로 조용한
오염 경로는 없다.)

결과 JSON을 자기 가설 파일(레인 A: H12-level-smoothing-window.md)의 「결과」 표에 옮기고,
「관찰」과 「판정 · 미검증」을 쓴다. report_id는 EXP-<코디네이터가 발급한 번호>다 —
이미 발급돼 있으니 scripts/exp.py를 부르지 않는다.

「판정」에 반드시 지킬 것: **arm 사이의 순위만 읽는다.** 사후확률은 real train에 대한
in-sample 예측이라 절대 정확도가 낙관 편향돼 있고, 이 프록시는 과거 두 번 다 낙관이었다
(2.55pp, 3.55pp 할인). 결과의 run_count를 test의 1,046과 나란히 적어 비교 가능성을 드러낸다.
run_count가 1,046과 자릿수가 다르면 「판정」은 "결론 없음"이고 그 이유를 적는다.
「미검증」에 재보지 못한 것을 남긴다.

끝나면 커밋하고 push한 뒤 worker_done으로 보고한다.
```

- [ ] **Step 4: bound 대기로 수집한다**

```bash
orca orchestration check --wait --timeout-ms 45000 --json
```

만료마다 경과시간과 liveness를 보고한다. **두 레인이 같은 파일의 다른 절을 쓴다** — 그것이 이 시험의 대상이다.

- [ ] **Step 5: 락 배타성을 확인한다**

레인이 도는 동안 코디네이터에서:

```bash
uv run python - <<'PY'
from ai_co_scientist.locks import ResourceBusy, resource_lock
with resource_lock("probe-exclusion"):
    try:
        with resource_lock("probe-exclusion"):
            print("FAIL: 두 번째 획득이 통과했다")
    except ResourceBusy:
        print("OK: 배타 성립")
PY
```

**워크트리에서 잡은 락이 main에서도 막는지**를 함께 확인한다 — 레인 중 하나에 `resource_lock("probe-cross-tree")`를 잡게 하고, 그 사이 코디네이터에서 같은 이름을 잡아 `ResourceBusy`가 나는지 본다. 여기서 실패하면 락 경로가 여전히 트리별이라는 뜻이다.

---

### Task 9: 머지와 판정

**Files:**
- Modify: `docs/experiment/H12-level-smoothing-window.md` · `docs/experiment/H13-viterbi-vs-mode-filter.md` (각각 이관 범위)
- Modify: `.agents/rules/orca-parallel.md` (병렬 실측)

**Interfaces:**
- Consumes: Task 8의 두 레인 브랜치
- Produces: 충돌 없는 머지 · 판정

- [ ] **Step 1: 두 레인을 머지한다**

```bash
git fetch origin
git merge --no-ff feature/h12-smooth
git merge --no-ff feature/h13-hmm
```

**충돌이 나면 그것이 발견이다** — 단일 작성자 분할이 성립하지 않았다는 뜻이므로, 해결하지 말고 어느 절을 누가 썼는지 기록한다. 충돌 0이 이 시험의 통과 조건이다.

- [ ] **Step 2: 결과를 기록소에 넣는다 (코디네이터)**

각 arm의 결과를 `exp.py result`로 기록한다. `exp.py result`는 기본이 **기존 val에 병합**이므로 arm별 결과가 순차로 쌓인다 — 스윕 하나 = 선보고 하나.

- [ ] **Step 3: 판정과 이관 범위를 쓴다**

가설 파일의 「이관 범위」를 오케스트레이터가 채운다. **순위만 읽는다** — 절대 정확도는 in-sample이고 프록시는 낙관이다. `run_count`를 test의 1,046과 나란히 적어 비교 가능성을 읽는 사람이 판단하게 한다.

- [ ] **Step 4: 병렬 실측을 문서에 남긴다**

`orca-parallel.md`에 기록한다: 레인 둘이 동시에 도는 동안 무엇이 성립했는가 · 머지 충돌 수 · 락이 트리를 가로질러 배타를 걸었는가 · 승인 범위를 앞에 둔 spec이 실제로 질문 없이 착수했는가(custflow의 측정이 이 저장소에서도 재현되는가).

- [ ] **Step 5: 정리하고 커밋한다**

```bash
orca orchestration worker-release --dispatch <각 dispatch_id> --json
orca worktree rm --worktree "path:<각 워크트리>" --force
```

```bash
git add docs/experiment/H12-level-smoothing-window.md docs/experiment/H13-viterbi-vs-mode-filter.md \n        .agents/rules/orca-parallel.md
git commit -m "Record the H12 and H13 verdicts and the measured parallel lane run"
```

---

## 착지 전 점검

- [ ] `uv run pytest tests/ -q` 전부 통과
- [ ] `uv run python .agent-hooks/test_block_runtime_commands.py` rc=0
- [ ] `uv run ruff check src/ scripts/ tests/ .agent-hooks/` 위반 없음
- [ ] Task 8의 머지 충돌 **0**
- [ ] 락이 워크트리 ↔ main을 가로질러 배타를 걸었다
- [ ] `orca-parallel.md`에 미측정으로 남은 항목이 실제로 미측정인 것만이다
- [ ] `git fetch origin main && git merge origin/main` 후 위 전부 재확인
- [ ] `self-review.md` 실행, 두 절반 모두 PR 설명에 포함
