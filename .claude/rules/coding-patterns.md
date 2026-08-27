---
paths:
  - src/**
  - scripts/**
---
# Coding Patterns

## Before writing new code (the rule that generates the rest)

**Find the nearest existing analog and copy its conventions** — file naming, layout, import order,
config loading, return shape. Most rules below are just instances of this. When unsure, don't
invent — read a sibling file. Canonical analogs:

- Reading config → `from ai_co_scientist.config import load_config` — the single loader;
  never open `config.yaml` directly.
- New experiment CLI → mirror `scripts/exp.py`: argparse subcommands, `ensure_utf8_console()` first,
  마지막 줄에 기계가 읽을 JSON 한 줄.
- Tests → `tests/<module>/test_*.py`, mirroring the source package (see `testing.md`).

Prefer a proven library over a hand-rolled implementation. Don't borrow a library's metric/API name
for a different custom implementation.

**1:1 correspondence is the design default** — artifacts describing the same thing map 1:1 in name
and unit, so the counterpart's location is derivable without search (visibility + maintainability):
- `tests/<pkg>/` ↔ `src/ai_co_scientist/<pkg>/`
Adding one side of a pair without the other is a smell — wire both in the same change, and add a
correspondence test when the mapping is enumerable.

## Style & configuration

- **Secrets** → `.env` (copy `.env.example`), read via `ai_co_scientist/config.py` `load_dotenv()`.
  Never hardcode or commit secrets.
- **Tunable values** (paths · target values · train defaults) → `config.yaml`, not code.
- Imports/ordering: stdlib → third-party → local, blank-line separated. Type hints on signatures.
  Korean docstrings/comments are the norm in `src/` — match the file you're editing.
- Don't couple logic to specific value names/counts — behave off the config lists / thresholds.

## Gotchas (non-obvious — mirroring won't catch these)

- **`load_dotenv()` is called only by the real backend** (`backends/dacon.py`)
  — tests must never depend on `.env`. Offline isolation goes through
  `COSCIENTIST_DACON_FAKE_HTTP` and `tmp_path`, so the suite passes with zero API keys and no
  `.env` file.
- **Windows console is cp949** — any new process entrypoint must call `ensure_utf8_console()`
  (from `ai_co_scientist/config.py`) before printing, or unicode (`—`, `→`) crashes with
  UnicodeEncodeError. **Call it before `ArgumentParser()`, not after `parse_args()`** — argparse
  prints `--help` itself and exits, so a later call never runs. Four scripts had `--help` broken
  this way and nobody noticed, because the normal path calls it early enough. If a `help=` string
  contains an em dash, `--help` is the only thing that crashes.
- **Never `path.split('/')`** on glob results — Windows returns backslashes. Use `pathlib`.
  (This exact bug broke the official DACON baseline notebook on Windows.)
- **`DataLoader(num_workers>0)` needs an `if __name__ == "__main__":` guard** on Windows
  (spawn, not fork) — see `scripts/legacy/baseline_sem_depth.py`. It also **flakes intermittently**
  (`OSError: [Errno 22] Invalid argument` / truncated pickle) with mmap-backed datasets — use
  `--num-workers 0` for reliable local screening runs; workers are fine on Linux.
- **Runtime values consumed inside `DataLoader` workers must be dataset attributes, not module
  globals** — Windows spawn workers don't inherit globals set in `main()` (a global histmatch
  reference read as `None` in the worker). Pass it via the `Dataset.__init__` so it's pickled.
- **Windows file locks: catch `PermissionError` as well as `FileExistsError`.** A just-unlinked
  file can sit in *delete-pending* state, and `os.open(O_CREAT|O_EXCL)` then raises `EACCES`, not
  `EEXIST` — a POSIX-shaped retry loop drops straight through and the caller's work is lost
  (measured: 1 of 8 concurrent registry writes vanished). Guard the `stat()` too: `exists()` →
  `stat()` is a TOCTOU race that raises `FileNotFoundError` under contention. Both bugs were
  invisible in single-threaded runs and surfaced only as an **intermittent** suite failure —
  if a test fails ~1 run in 6, suspect a race, not a fluke.
- **`seed_everything` must set `cudnn.benchmark = True`, not just `deterministic = True`.**
  With `benchmark=False`, cuDNN requests a fresh workspace per call and OOMs *outside* PyTorch's
  allocator — an smp backbone died at step ~195 while PyTorch held only 1.3 GiB of 8 GiB, and
  `nvidia-smi` showed the GPU idle. The error surfaces as `CUDA out of memory` or
  `CUDNN_STATUS_INTERNAL_ERROR` in `backward()`, so it reads like a batch-size problem and isn't.
  Copy `train_structure.py`'s version whole; dropping the line is the bug.
- **The whole PC dying mid-training is NOT a CUDA OOM — check the event log before touching the
  batch size.** A CUDA OOM raises `torch.cuda.OutOfMemoryError`; the process dies and the machine
  lives. A machine-wide reboot is a driver bugcheck. Diagnose first:
  `Get-WinEvent -FilterHashtable @{LogName='System'} | Where-Object {$_.Id -in 41,1001,6008}`
  (run it from a `.ps1` file — bash mangles `$_`). This bit us: a 0x10E
  VIDEO_MEMORY_MANAGEMENT_INTERNAL bugcheck at **13 % VRAM use** was read as an OOM, so EXP-011 was
  re-run at bs64 — silently adding a BN-statistics confound the pre-report then had to disclaim.
  bs128 was fine all along.
- **Don't set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on Windows.** It's Linux-only
  (PyTorch routes allocation through the CUDA VMM API), and on WDDM that path goes straight through
  the video memory manager that throws 0x10E. It is never the fix for an apparent OOM here — the
  two real causes are the `cudnn.benchmark` line above and genuine batch size.
- **Long training on Windows needs a per-epoch resume checkpoint** (`train_structure.py --resume`).
  Keep it in a **separate file** from the best ckpt: `out` is the inference contract
  (`arch` + `state_dict`, read by `load_model`) and mixing optimizer/scaler state into it breaks
  backward compatibility with EXP-005's bare-`state_dict` ckpt.

> For layer boundaries (what agents may import), see `architecture.md`. For tests, see
> `testing.md`.
