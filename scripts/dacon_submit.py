"""DACON 제출 CLI.

--report-id를 주면 memo 앞에 report_id를 붙여 리더보드 행 ↔ 기록소 항목을 이어붙인다.
점수는 API가 주지 않으므로, 리더보드에서 확인한 뒤 `scripts/exp.py lb`로 기입한다.
"""
import argparse
import os

from ai_co_scientist.backends import dacon
from ai_co_scientist.config import ensure_utf8_console


def main():
    ensure_utf8_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--report-id", default="", help="기록소 report_id (예: EXP-001)")
    parser.add_argument("--memo", default="")
    parser.add_argument("--cpt-id", default=None, help="기본값: .env의 DACON_CPT_ID")
    parser.add_argument("--team-name", default=None, help="기본값: .env의 DACON_TEAM_NAME")
    args = parser.parse_args()

    if args.cpt_id:
        os.environ["DACON_CPT_ID"] = args.cpt_id
    if args.team_name:
        os.environ["DACON_TEAM_NAME"] = args.team_name

    memo = f"[{args.report_id}] {args.memo}".strip() if args.report_id else args.memo
    print(dacon.submit(args.file_path, memo))


if __name__ == "__main__":
    main()
