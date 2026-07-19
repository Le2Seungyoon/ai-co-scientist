"""전체 기동 스크립트. --skeleton (M1: PM↔Research 왕복 1회) / --cycle N (M2: 사이클 N바퀴)."""
import argparse
import asyncio
import contextlib
import subprocess
import sys
import time

import httpx

from ai_co_scientist.core.config import ensure_utf8_console, load_config

# a2a-sdk 버전에 따라 카드 경로가 다름 — 둘 다 시도
CARD_PATHS = ["/.well-known/agent.json", "/.well-known/agent-card.json"]


def spawn_agent(name: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", f"ai_co_scientist.agents.{name}.server"],
    )


def wait_for_agent(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for path in CARD_PATHS:
            try:
                r = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
        time.sleep(0.2)
    raise TimeoutError(f"포트 {port} 에이전트가 {timeout}초 내에 뜨지 않음")


CYCLE_AGENTS = ["research", "analysis", "coder", "executor", "critic", "harness_engineer"]


def _cleanup(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # SIGTERM 무시/지연 시 강제 종료 — 고아 프로세스 방지
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)


def run_skeleton_mode() -> None:
    from ai_co_scientist.agents.pm.skeleton import run_skeleton

    cfg = load_config()
    research = spawn_agent("research")
    try:
        wait_for_agent(cfg["agents"]["research"]["port"])
        asyncio.run(run_skeleton())
    finally:
        _cleanup([research])


def run_cycle_mode(n: int) -> None:
    from ai_co_scientist.agents.pm.cycle import run_cycles

    cfg = load_config()
    procs: list[subprocess.Popen] = []
    try:
        # 스폰 도중 실패해도 이미 추가된 프로세스는 finally의 _cleanup으로 정리됨
        for name in CYCLE_AGENTS:
            procs.append(spawn_agent(name))
        for name in CYCLE_AGENTS:
            wait_for_agent(cfg["agents"][name]["port"])
        asyncio.run(run_cycles(n))
    finally:
        _cleanup(procs)


def main(argv: list[str] | None = None) -> None:
    ensure_utf8_console()
    parser = argparse.ArgumentParser(description="ai-co-scientist 기동")
    parser.add_argument("--skeleton", action="store_true", help="M1 walking skeleton 실행")
    parser.add_argument("--cycle", type=int, metavar="N",
                        help="M2 사이클을 N바퀴 실행")
    args = parser.parse_args(argv)
    if args.skeleton:
        run_skeleton_mode()
    elif args.cycle is not None:
        if args.cycle < 1:
            parser.error("--cycle N은 1 이상이어야 함")
        run_cycle_mode(args.cycle)
    else:
        parser.error("--skeleton 또는 --cycle N 필요")


if __name__ == "__main__":
    main()
