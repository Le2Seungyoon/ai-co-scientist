"""Manual one-off Lightning AI GPU job submission CLI.

Thin wrapper around ai_co_scientist.mcp_servers.lightning.server's real submit/poll
path — the same code the Executor uses when config.yaml's mock.lightning is false.
Reads LIGHTNING_API_KEY / LIGHTNING_USER_ID / LIGHTNING_TEAMSPACE from .env via
load_dotenv(); useful to verify credentials/quota before wiring a full agent cycle.
"""

import argparse
import time

from ai_co_scientist.core.config import load_dotenv
from ai_co_scientist.mcp_servers.lightning.server import (
    _real_get_credits,
    _real_poll_job,
    _real_submit_job,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entrypoint_path", help="원격에서 실행할 로컬 Python 스크립트")
    parser.add_argument("--timeout-s", type=float, default=300.0,
                        help="스크립트 자체 실행 제한 시간(원격 기동 오버헤드는 별도 가산)")
    parser.add_argument("--poll-interval-s", type=float, default=15.0)
    args = parser.parse_args()

    load_dotenv()
    print("잔여 크레딧:", _real_get_credits())

    sub = _real_submit_job(args.entrypoint_path, args.timeout_s)
    print("제출됨:", sub)

    while True:
        job = _real_poll_job(sub["job_id"])
        print("상태:", job["status"])
        if job["status"] != "running":
            print(job)
            break
        time.sleep(args.poll_interval_s)


if __name__ == "__main__":
    main()
