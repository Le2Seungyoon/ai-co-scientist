"""DACON MCP 서버 — mock: holdout 채점(public/private 분할), real: 공개 제출 API 업로드.

config.yaml의 mock.dacon(또는 테스트용 COSCIENTIST_DACON_MOCK 오버라이드)로 분기한다.
real 경로는 DACON이 점수를 동기 반환하지 않아(§ README "DACON 제출 API 활성화" 참고)
제출 접수 여부만 확인 가능 — 점수는 사람이 리더보드에서 직접 확인한다."""
import csv
import os
import uuid
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from ai_co_scientist.core.config import load_config, load_dotenv, project_root
from ai_co_scientist.mcp_servers.dacon import db

mcp = FastMCP("dacon")

_API_URL = "https://openapi.dacon.io/submission"
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


def _db_path() -> str:
    return os.environ.get("COSCIENTIST_DACON_DB") or str(
        project_root() / load_config()["paths"]["dacon_db"])


def _toy_dir() -> Path:
    return Path(os.environ.get("COSCIENTIST_TOY_DATA") or (
        project_root() / load_config()["paths"]["toy_data_dir"]))


def _is_mock() -> bool:
    override = os.environ.get("COSCIENTIST_DACON_MOCK")
    if override is not None:
        return override == "1"
    return bool(load_config()["mock"]["dacon"])


def _decode_message(message: str, data: object) -> tuple[bool, str]:
    if message == "ok":
        return True, "Success"
    if message == "over_max_count":
        return False, "Over max submission count of Competition. 대회 기간 중 제출 가능한 최대 횟수가 초과되었습니다"
    if message == "day_max_count":
        return False, "Over max submission count of Daily. 일일 제출 가능한 최대 횟수가 초과되었습니다."
    if message == "wrong":
        return False, _ERROR_DETAIL.get(data, f"wrong (code={data})")
    return False, f"unknown response: {message}"


def _real_submit(file_path: str, cycle_id: int, memo: str = "") -> dict:
    """실제 DACON 공개 제출 API(POST /submission) 호출. dacon_submit_api wheel과
    동일한 요청 형식(multipart form)이라 그 패키지 자체는 필요 없다."""
    load_dotenv()
    token = os.environ.get("DACON_API_TOKEN")
    cpt_id = os.environ.get("DACON_CPT_ID")
    team_name = os.environ.get("DACON_TEAM_NAME")
    if not (token and cpt_id and team_name):
        raise RuntimeError(
            "DACON_API_TOKEN/DACON_CPT_ID/DACON_TEAM_NAME 필요 (.env 참고, README '제출 API 활성화')")

    fake = os.environ.get("COSCIENTIST_DACON_FAKE_HTTP")
    if fake is not None:
        message, data = (fake.split(":", 1) + [None])[:2]
        if data is not None:
            data = int(data)
    else:
        form = {"cpt_id": cpt_id, "team_name": team_name,
                "file_name": os.path.basename(file_path), "memo": memo,
                "api_token": token, "api_version": 2.0}
        with open(file_path, "rb") as f:
            resp = httpx.post(_API_URL, data=form, files={"file": f}, timeout=120.0)
        body = resp.json()
        message, data = body.get("message"), body.get("data")

    is_submitted, detail = _decode_message(message, data)
    submission_id = uuid.uuid4().hex[:12]
    db.save_submission(
        _db_path(), submission_id=submission_id, cycle_id=cycle_id,
        public=None, private=None,
        status="submitted" if is_submitted else "rejected", detail=detail)
    return {"submission_id": submission_id, "public_score": None,
            "isSubmitted": is_submitted, "detail": detail}


def _mse(pairs: list[tuple[float, float]]) -> float:
    return round(sum((y - p) ** 2 for y, p in pairs) / len(pairs), 6)  # 부동소수 노이즈 제거용 고정 정밀도


def _mock_submit(predictions_path: str, cycle_id: int) -> dict:
    holdout = _toy_dir() / "holdout.csv"
    if not holdout.exists():
        raise FileNotFoundError(f"holdout 없음: {holdout}")
    with open(holdout, encoding="utf-8") as f:
        truth = [float(row["y"]) for row in csv.DictReader(f)]
    with open(predictions_path, encoding="utf-8") as f:
        preds = [float(row["pred"]) for row in csv.DictReader(f)]
    if len(truth) != len(preds):
        raise ValueError(f"행 수 불일치: holdout {len(truth)} vs preds {len(preds)}")
    half = len(truth) // 2
    if half == 0:
        raise ValueError(f"채점 불가: holdout 행 수 부족 ({len(truth)}행 — 최소 2행 필요)")
    pairs = list(zip(truth, preds))
    public = _mse(pairs[:half])
    private = _mse(pairs[half:])
    submission_id = uuid.uuid4().hex[:12]
    db.save_submission(_db_path(), submission_id=submission_id, cycle_id=cycle_id,
                       public=public, private=private)
    return {"submission_id": submission_id, "public_score": public}


@mcp.tool()
def submit(predictions_path: str, cycle_id: int) -> dict:
    """제출. mock: holdout 채점(public=전반부, private=후반부 MSE, predictions는 holdout과
    '행 순서'로 대응). real: predictions_path를 실제 제출 zip으로 보고 DACON에 업로드
    (점수는 동기 반환되지 않음 — public_score는 항상 None, get_score로 접수 상태만 확인)."""
    if _is_mock():
        return _mock_submit(predictions_path, cycle_id)
    return _real_submit(predictions_path, cycle_id)


@mcp.tool()
def get_score(submission_id: str) -> dict:
    """제출 상태/점수 조회. mock은 항상 채점된 public/private을 반환.
    real은 접수 상태(status/detail)만 반환 — 실제 점수는 리더보드에서 직접 확인해야 함."""
    row = db.get_submission(_db_path(), submission_id)
    if row is None:
        raise KeyError(f"submission 없음: {submission_id}")
    if row["status"] != "scored":
        return {"public_score": None, "private_score": None,
                "status": row["status"], "detail": row["detail"]}
    return {"public_score": row["public"], "private_score": row["private"]}


if __name__ == "__main__":
    mcp.run()
