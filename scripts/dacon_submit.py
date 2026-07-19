"""Manual one-off DACON submission CLI.

Thin wrapper around ai_co_scientist.mcp_servers.dacon.server._real_submit — the
same code path the Analysis Agent uses when config.yaml's mock.dacon is false.
Reads DACON_API_TOKEN / DACON_CPT_ID / DACON_TEAM_NAME from .env via load_dotenv();
CLI flags override them for one-off use.
"""

import argparse
import os

from ai_co_scientist.core.config import load_dotenv
from ai_co_scientist.mcp_servers.dacon.server import _real_submit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--cpt-id", default=None, help="default: DACON_CPT_ID in .env")
    parser.add_argument("--team-name", default=None, help="default: DACON_TEAM_NAME in .env")
    parser.add_argument("--memo", default="")
    parser.add_argument("--cycle-id", type=int, default=0, help="cycle 밖 수동 제출은 0")
    args = parser.parse_args()

    load_dotenv()
    if args.cpt_id:
        os.environ["DACON_CPT_ID"] = args.cpt_id
    if args.team_name:
        os.environ["DACON_TEAM_NAME"] = args.team_name

    result = _real_submit(args.file_path, args.cycle_id, args.memo)
    print(result)


if __name__ == "__main__":
    main()
