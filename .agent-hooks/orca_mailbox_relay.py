#!/usr/bin/env python3
"""Forward lane messages from a filesystem mailbox onto the Orca bus.

WHY THIS EXISTS (measured 2026-09-21, Orca 1.4.206, Windows)
    Orca's worker contract tells every lane to report with `orca orchestration send`. A Codex lane
    on this host cannot: its sandbox refuses to *launch* that binary. Reproduced directly --
    `codex sandbox -- <abs>\\orca.exe --version` fails `CreateProcessAsUserW: 5 (access denied)`,
    and granting `sandbox_permissions=["disk-full-read-access"]` does NOT help, because the denial
    is at process creation, not at a read. Only `sandbox_mode="danger-full-access"` lifted it, and
    that drops the sandbox for every command the lane runs.

    So the lane writes its message to a file instead, inside a directory its sandbox already
    trusts, and this relay -- which runs in the COORDINATOR session, outside the sandbox -- puts it
    on the bus. Injection into the lane already works, so this closes the round trip without
    touching anyone's security boundary. Details: `.agents/rules/harness.md` -> Codex cannot reach
    the Orca CLI.

THE MAILBOX PATH IS EXPLICIT, NEVER DERIVED
    `--mailbox` is required. A default resolved from `__file__` would give a different directory
    per worktree, which is exactly the flaw `architecture.md` names for `registry.locked()` -- and
    the whole point here is that several lanes and one coordinator share ONE directory. It must
    also sit under a path the lane's sandbox trusts (a project root), so a machine-level temp
    directory is wrong for the same reason it was right for `locks.py`.

MESSAGE FORMAT
    One JSON object per file, written atomically (`.tmp` then rename) so the relay never reads a
    half-written message. Required keys mirror what `orca orchestration send` needs and the lane
    already has them all from its injected preamble:

        from, dispatch_capability, type, task_id, dispatch_id

    Optional: subject, body, outcome, phase.

DELIVERY IS AT-LEAST-ONCE, AND SAYS SO
    A message moves to `sending/` before the send and to `sent/` after it. A crash between those
    two leaves it in `sending/`, where it is visible rather than lost -- the relay never retries it
    on its own, because a silent re-send would double-report a `worker_done`. Recovering one is a
    person's decision: read it and move it back.

Tests: `test_orca_mailbox_relay.py`, beside this file. Python 3.8 compatible (bare `python`).
"""
import argparse
import json
import os
import subprocess
import sys
import time

REQUIRED = ("from", "dispatch_capability", "type", "task_id", "dispatch_id")
OPTIONAL_FLAGS = (
    ("subject", "--subject"),
    ("body", "--body"),
    ("outcome", "--outcome"),
    ("phase", "--phase"),
)


def build_command(msg, orca="orca"):
    """JSON message -> argv for `orca orchestration send`. Raises ValueError when a key is missing.

    argv, never a shell string: a body carries arbitrary prose (and this repo's is Korean), and
    a lane must never be able to end the quoting and start a command.
    """
    missing = [k for k in REQUIRED if not str(msg.get(k, "")).strip()]
    if missing:
        raise ValueError("필수 키 누락: {0}".format(", ".join(missing)))
    argv = [
        orca, "orchestration", "send",
        "--from", str(msg["from"]),
        "--dispatch-capability", str(msg["dispatch_capability"]),
        "--type", str(msg["type"]),
        "--task-id", str(msg["task_id"]),
        "--dispatch-id", str(msg["dispatch_id"]),
    ]
    for key, flag in OPTIONAL_FLAGS:
        value = msg.get(key)
        if value is not None and str(value).strip():
            argv += [flag, str(value)]
    return argv + ["--json"]


def _subdirs(mailbox):
    return (
        os.path.join(mailbox, "sending"),
        os.path.join(mailbox, "sent"),
        os.path.join(mailbox, "failed"),
    )


def deliver_one(path, mailbox, orca="orca", runner=None):
    """Send one message file. Returns (ok, detail). Moves the file to sending/ then sent/failed/."""
    sending, sent, failed = _subdirs(mailbox)
    for d in (sending, sent, failed):
        os.makedirs(d, exist_ok=True)

    name = os.path.basename(path)
    staged = os.path.join(sending, name)
    try:
        os.replace(path, staged)
    except OSError as exc:              # another relay took it, or it vanished
        return False, "이동 실패(다른 릴레이가 가져갔을 수 있다): {0}".format(exc)

    try:
        # Accept both plain UTF-8 and the BOM-prefixed UTF-8 commonly emitted by PowerShell.
        with open(staged, encoding="utf-8-sig") as fh:
            msg = json.load(fh)
        if not isinstance(msg, dict):
            raise ValueError("message must be a JSON object")
        argv = build_command(msg, orca)
    except (ValueError, OSError) as exc:
        os.replace(staged, os.path.join(failed, name))
        return False, "읽기/검증 실패: {0}".format(exc)

    run = runner or (lambda a: subprocess.run(a, capture_output=True, text=True))
    proc = run(argv)
    if getattr(proc, "returncode", 1) == 0:
        os.replace(staged, os.path.join(sent, name))
        return True, "{0} 전달".format(msg.get("type"))
    os.replace(staged, os.path.join(failed, name))
    detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
    return False, "orca send 실패(rc={0}): {1}".format(proc.returncode, detail[:300])


def scan(mailbox):
    """Message files waiting at the top level, oldest first. Subdirectories are state, not inbox."""
    if not os.path.isdir(mailbox):
        return []
    out = []
    for name in os.listdir(mailbox):
        path = os.path.join(mailbox, name)
        if name.endswith(".json") and os.path.isfile(path):
            out.append(path)
    return sorted(out, key=lambda p: (os.path.getmtime(p), p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mailbox", required=True,
                    help="레인들이 쓰는 메일박스 디렉터리. 기본값을 두지 않는다 — 워크트리마다 "
                         "다른 경로로 풀리면 공유가 깨지고, 레인의 샌드박스가 신뢰하는 "
                         "프로젝트 경로 아래여야 한다")
    ap.add_argument("--orca", default="orca", help="orca 실행 파일 (기본: PATH의 orca)")
    ap.add_argument("--interval", type=float, default=3.0, help="폴링 간격(초)")
    ap.add_argument("--once", action="store_true",
                    help="한 번만 훑고 끝낸다 — 감사나 테스트용")
    args = ap.parse_args()

    if not args.once:
        sys.stderr.write("[relay] watching {0} every {1}s\n".format(args.mailbox, args.interval))
    delivered = 0
    while True:
        for path in scan(args.mailbox):
            ok, detail = deliver_one(path, args.mailbox, args.orca)
            delivered += 1 if ok else 0
            # stdout is the record: one line per message, so a coordinator can tail it.
            print(json.dumps({"file": os.path.basename(path), "ok": ok, "detail": detail},
                             ensure_ascii=False), flush=True)
        if args.once:
            return 0 if delivered or not scan(args.mailbox) else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
