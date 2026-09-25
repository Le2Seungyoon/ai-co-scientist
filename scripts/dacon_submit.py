"""DACON 제출 CLI.

--report-id를 주면 memo 앞에 report_id를 붙여 리더보드 행 ↔ 기록소 항목을 이어붙인다.
점수는 API가 주지 않으므로, 리더보드에서 확인한 뒤 `scripts/exp.py lb`로 기입한다.

제출 전 `verify_submission()`이 **하드 프리체크**로 돈다: 파일 수 25,988 · 이미지별 max가
{140,150,160,170} · 벗어남 0.00 percent (docs/data-facts.md §1-2). 실패하면 POST하지 않고
종료 코드 2로 끝난다 — 제출 횟수는 유한하고, 깨진 zip 한 번이 슬롯 하나다.
접수되지 않은 응답(`isSubmitted: false` — 횟수 초과·형식 오류)도 종료 코드 1이다.
"""
import argparse
import json
import os
import sys

from ai_co_scientist.backends import dacon
from ai_co_scientist.config import ensure_utf8_console
from ai_co_scientist.locks import DACON_LOCK, ResourceBusy, resource_lock
from ai_co_scientist.submission import EXPECTED_FILES, verify_submission

EXIT_VERIFY_FAILED = 2
EXIT_NOT_SUBMITTED = 1


def main() -> int:
    ensure_utf8_console()  # argparse가 help를 찍기 **전**에 (cp949 콘솔)
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--report-id", default="", help="기록소 report_id (예: EXP-001)")
    parser.add_argument("--memo", default="")
    parser.add_argument("--cpt-id", default=None, help="기본값: .env의 DACON_CPT_ID")
    parser.add_argument("--team-name", default=None, help="기본값: .env의 DACON_TEAM_NAME")
    parser.add_argument("--expected-files", type=int, default=EXPECTED_FILES,
                        help="제출 zip에 있어야 할 PNG 수 (기본 25,988)")
    parser.add_argument("--verify-only", action="store_true",
                        help="검증만 하고 제출하지 않는다 (제출 횟수를 쓰지 않는 예행)")
    args = parser.parse_args()

    check = verify_submission(args.file_path, expected_files=args.expected_files)
    print(f"제출본 검증: {check.summary()}", flush=True)
    if not check.ok:
        for e in check.errors:
            print(f"  검증 실패: {e}", file=sys.stderr)
        print(json.dumps({"submitted": False, "verified": False,
                          "verify": check.to_dict()}, ensure_ascii=False))
        return EXIT_VERIFY_FAILED
    if args.verify_only:
        print(json.dumps({"submitted": False, "verified": True,
                          "verify": check.to_dict()}, ensure_ascii=False))
        return 0

    if args.cpt_id:
        os.environ["DACON_CPT_ID"] = args.cpt_id
    if args.team_name:
        os.environ["DACON_TEAM_NAME"] = args.team_name

    memo = f"[{args.report_id}] {args.memo}".strip() if args.report_id else args.memo
    try:
        with resource_lock(DACON_LOCK):
            result = dacon.submit(args.file_path, memo)
    except ResourceBusy as e:
        print(f"{DACON_LOCK} 사용 중: {e}", file=sys.stderr)
        print(json.dumps({"submitted": False, "verified": True, "busy": True,
                          "verify": check.to_dict()}, ensure_ascii=False))
        return EXIT_NOT_SUBMITTED
    if not result.get("isSubmitted"):
        print(f"제출되지 않음: {result.get('detail')}", file=sys.stderr)
    print(json.dumps({"submitted": bool(result.get("isSubmitted")), "verified": True,
                      "response": result, "verify": check.to_dict()}, ensure_ascii=False))
    return 0 if result.get("isSubmitted") else EXIT_NOT_SUBMITTED


if __name__ == "__main__":
    sys.exit(main())
