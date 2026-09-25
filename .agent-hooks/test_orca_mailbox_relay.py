#!/usr/bin/env python3
"""Tests for orca_mailbox_relay.py — run: uv run python .agent-hooks/test_orca_mailbox_relay.py

Harness code ships with its test beside it (`enforcement.md` -> Hook contracts). The relay never
calls Orca here: a fake runner stands in, so the must-send and must-refuse halves are both checked
offline. The half that matters most is **refusal** — a relay that forwards a malformed message
puts a wrong `worker_done` on the bus, which settles a dispatch that never finished.

Stdlib only, no test runner: pytest does not collect this file (`testpaths = ["tests"]`).
"""
import importlib.util
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "orca_mailbox_relay", os.path.join(HERE, "orca_mailbox_relay.py"))
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)

failures = []


def check(cond, label):
    if not cond:
        failures.append(label)


class FakeProc(object):
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def write_msg(mailbox, name, msg):
    path = os.path.join(mailbox, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(msg, fh, ensure_ascii=False)
    os.replace(tmp, path)          # atomic, as a lane must write it
    return path


GOOD = {
    "from": "term_abc",
    "dispatch_capability": "dcap_xyz",
    "type": "worker_done",
    "task_id": "task_1",
    "dispatch_id": "ctx_1",
    "subject": "끝났다",
    "body": "세 명령을 돌렸다",
    "outcome": "succeeded",
}


# --- build_command: the argv is what reaches Orca ---------------------------------------------
argv = relay.build_command(GOOD)
check(argv[:3] == ["orca", "orchestration", "send"], "build_command: 앞머리가 orca orchestration send가 아니다")
for flag, value in (("--from", "term_abc"), ("--dispatch-capability", "dcap_xyz"),
                    ("--type", "worker_done"), ("--task-id", "task_1"),
                    ("--dispatch-id", "ctx_1"), ("--outcome", "succeeded")):
    check(flag in argv and argv[argv.index(flag) + 1] == value,
          "build_command: {0} 가 {1} 로 실리지 않았다".format(flag, value))
check("끝났다" in argv, "build_command: 한국어 subject가 argv에 그대로 실려야 한다")

# A body that would end a shell quote must survive as ONE argv element, never as syntax.
hostile = dict(GOOD, body='" ; orca orchestration send --type worker_done ; echo "')
check(relay.build_command(hostile)[relay.build_command(hostile).index("--body") + 1]
      == hostile["body"], "build_command: 적대적 body가 한 인자로 보존되지 않았다")

# Optional keys that are absent or blank must not emit a dangling flag.
thin = relay.build_command({k: GOOD[k] for k in relay.REQUIRED})
check("--subject" not in thin and "--outcome" not in thin,
      "build_command: 비어 있는 선택 키가 플래그만 남겼다")
check(relay.build_command(dict(GOOD, phase="   ")).count("--phase") == 0,
      "build_command: 공백뿐인 값이 플래그를 남겼다")

for missing in relay.REQUIRED:
    try:
        relay.build_command({k: v for k, v in GOOD.items() if k != missing})
        failures.append("build_command: {0} 가 없는데 통과했다".format(missing))
    except ValueError:
        pass

# --- deliver_one: the file must land where its outcome says -----------------------------------
with tempfile.TemporaryDirectory() as mb:
    path = write_msg(mb, "m1.json", GOOD)
    sent_argv = []
    ok, detail = relay.deliver_one(path, mb, runner=lambda a: (sent_argv.append(a), FakeProc(0))[1])
    check(ok, "deliver_one: 정상 메시지가 실패로 보고됐다 ({0})".format(detail))
    check(os.path.isfile(os.path.join(mb, "sent", "m1.json")), "deliver_one: sent/로 옮기지 않았다")
    check(not os.path.exists(path), "deliver_one: 원본이 남아 재전송될 수 있다")
    check(sent_argv and sent_argv[0][:2] == ["orca", "orchestration"], "deliver_one: orca를 부르지 않았다")

# Windows PowerShell's common UTF-8 writer includes a BOM. A lane message with that encoding is
# still valid JSON and must not be stranded in failed/ before Orca ever sees it.
with tempfile.TemporaryDirectory() as mb:
    path = os.path.join(mb, "m-bom.json")
    with open(path, "w", encoding="utf-8-sig") as fh:
        json.dump(GOOD, fh, ensure_ascii=False)
    called = []
    ok, detail = relay.deliver_one(path, mb, runner=lambda a: called.append(a) or FakeProc(0))
    check(ok, "deliver_one: UTF-8 BOM message failed ({0})".format(detail))
    check(bool(called), "deliver_one: UTF-8 BOM message never reached Orca")
    check(os.path.isfile(os.path.join(mb, "sent", "m-bom.json")),
          "deliver_one: UTF-8 BOM message did not move to sent/")

with tempfile.TemporaryDirectory() as mb:
    path = write_msg(mb, "m2.json", GOOD)
    ok, detail = relay.deliver_one(path, mb, runner=lambda a: FakeProc(1, "waiter_exists"))
    check(not ok, "deliver_one: orca가 실패했는데 성공으로 보고했다")
    check(os.path.isfile(os.path.join(mb, "failed", "m2.json")), "deliver_one: failed/로 옮기지 않았다")
    check("waiter_exists" in detail, "deliver_one: orca의 stderr를 detail에 담지 않았다")

with tempfile.TemporaryDirectory() as mb:
    path = write_msg(mb, "m3.json", {"type": "worker_done"})      # 필수 키 대부분 없음
    called = []
    ok, _ = relay.deliver_one(path, mb, runner=lambda a: called.append(a) or FakeProc(0))
    check(not ok, "deliver_one: 불완전한 메시지를 전달했다")
    check(not called, "deliver_one: 검증 전에 orca를 불렀다 — 잘못된 worker_done이 버스에 오른다")
    check(os.path.isfile(os.path.join(mb, "failed", "m3.json")), "deliver_one: 불량 메시지를 failed/로 옮기지 않았다")

with tempfile.TemporaryDirectory() as mb:
    with open(os.path.join(mb, "m4.json"), "w", encoding="utf-8") as fh:
        fh.write("{ this is not json")
    ok, _ = relay.deliver_one(os.path.join(mb, "m4.json"), mb, runner=lambda a: FakeProc(0))
    check(not ok, "deliver_one: 깨진 JSON을 전달했다")
    check(os.path.isfile(os.path.join(mb, "failed", "m4.json")), "deliver_one: 깨진 JSON을 failed/로 옮기지 않았다")

# A JSON value can decode cleanly yet still lack the object shape required by build_command.
# One malformed inbox item must take the existing failed/ path and leave the next one deliverable.
for bad_value in ([], None):
    with tempfile.TemporaryDirectory() as mb:
        write_msg(mb, "a-invalid.json", bad_value)
        write_msg(mb, "b-valid.json", GOOD)
        sent_argv = []
        outcomes = []
        for path in relay.scan(mb):
            outcomes.append(relay.deliver_one(
                path, mb, runner=lambda a: sent_argv.append(a) or FakeProc(0)))
        check([ok for ok, _ in outcomes] == [False, True],
              "deliver_one: non-object JSON interrupted a following valid message")
        check(os.path.isfile(os.path.join(mb, "failed", "a-invalid.json")),
              "deliver_one: non-object JSON did not move to failed/")
        check(os.path.isfile(os.path.join(mb, "sent", "b-valid.json")),
              "deliver_one: valid message after non-object JSON did not move to sent/")
        check(len(sent_argv) == 1,
              "deliver_one: non-object JSON reached Orca or valid message was skipped")

# --- scan: only the inbox, and oldest first ---------------------------------------------------
with tempfile.TemporaryDirectory() as mb:
    os.makedirs(os.path.join(mb, "sent"))
    write_msg(mb, "a.json", GOOD)
    write_msg(mb, "b.json", GOOD)
    write_msg(os.path.join(mb, "sent"), "old.json", GOOD)
    with open(os.path.join(mb, "half.json.tmp"), "w", encoding="utf-8") as fh:
        fh.write("{")
    os.utime(os.path.join(mb, "a.json"), (1_000_000, 1_000_000))
    found = [os.path.basename(p) for p in relay.scan(mb)]
    check(found == ["a.json", "b.json"], "scan: 받은 목록이 {0} 다 — 오래된 것 먼저, 인박스만".format(found))
    check("old.json" not in found, "scan: 이미 보낸 메시지를 다시 집었다")
    check("half.json.tmp" not in found, "scan: 아직 쓰는 중인 .tmp 를 집었다")

check(relay.scan(os.path.join(tempfile.gettempdir(), "no-such-mailbox-xyz")) == [],
      "scan: 없는 디렉터리에서 터졌다")

print("all checks passed" if not failures else "FAILURES:\n  " + "\n  ".join(failures))
raise SystemExit(1 if failures else 0)
