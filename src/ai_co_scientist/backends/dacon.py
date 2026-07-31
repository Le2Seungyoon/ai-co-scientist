"""DACON 공개 제출 API 래퍼.

점수는 동기로 반환되지 않는다 — 접수 여부만 확인 가능하고, 실제 public/private 점수는
사람이 리더보드에서 읽어 기록소에 기입한다(`scripts/exp.py lb`). 그래서 memo에 report_id를
넣어 리더보드 행과 기록소 항목을 이어붙인다.
"""
import os

import httpx

from ai_co_scientist.config import load_dotenv

API_URL = "https://openapi.dacon.io/submission"
_TIMEOUT_S = 120.0

_ERROR_DETAIL = {
    1: "csv file header error. csv 파일의 헤더 오류",
    2: "data parsing error. 데이터 파싱 오류",
    3: "number of row error. row 개수 오류",
    4: "other submission value error. 기타 제출값 오류",
    21: "data type error. 데이터 타입 오류",
    901: "csv file's header doesn't match. csv 파일의 헤더가 올바르지 않습니다",
    902: "error occurs in data parsing. 데이터 파싱간 에러가 발생했습니다",
    903: "number of row doesn't match. row 개수가 맞지 않습니다",
    904: "submission value occurs error. 기타 제출값에 오류가 발생했습니다",
    9021: "data must not have Character. 데이터에 문자가 들어갈 수 없습니다.",
}


def decode_message(message: str, data: object) -> tuple[bool, str]:
    """API 응답 message/data → (접수됨?, 사람이 읽을 상세)."""
    if message == "ok":
        return True, "Success"
    if message == "over_max_count":
        return False, "대회 기간 중 제출 가능한 최대 횟수가 초과되었습니다"
    if message == "day_max_count":
        return False, "일일 제출 가능한 최대 횟수가 초과되었습니다"
    if message == "wrong":
        return False, _ERROR_DETAIL.get(data, f"wrong (code={data})")
    return False, f"unknown response: {message}"


def submit(file_path: str, memo: str = "") -> dict:
    """제출 zip 업로드. COSCIENTIST_DACON_FAKE_HTTP가 있으면 HTTP 호출을 대체(오프라인 테스트)."""
    load_dotenv()
    token = os.environ.get("DACON_API_TOKEN")
    cpt_id = os.environ.get("DACON_CPT_ID")
    team_name = os.environ.get("DACON_TEAM_NAME")
    if not (token and cpt_id and team_name):
        raise RuntimeError(
            "DACON_API_TOKEN/DACON_CPT_ID/DACON_TEAM_NAME 필요 (.env 참고, README 제출 API 활성화)")

    fake = os.environ.get("COSCIENTIST_DACON_FAKE_HTTP")
    if fake is not None:
        message, _, code = fake.partition(":")
        data = int(code) if code else None
    else:
        form = {"cpt_id": cpt_id, "team_name": team_name,
                "file_name": os.path.basename(file_path), "memo": memo,
                "api_token": token, "api_version": 2.0}
        with open(file_path, "rb") as f:
            resp = httpx.post(API_URL, data=form, files={"file": f}, timeout=_TIMEOUT_S)
        body = resp.json()
        message, data = body.get("message"), body.get("data")

    is_submitted, detail = decode_message(message, data)
    return {"isSubmitted": is_submitted, "detail": detail, "memo": memo}
