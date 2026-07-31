"""DACON 백엔드 — 오프라인 테스트. COSCIENTIST_DACON_FAKE_HTTP로 실제 HTTP를 대체한다."""
import pytest

from ai_co_scientist.backends import dacon


def test_decode_message_ok():
    assert dacon.decode_message("ok", None) == (True, "Success")


def test_decode_message_over_max_count():
    is_submitted, detail = dacon.decode_message("over_max_count", None)
    assert is_submitted is False
    assert "최대 횟수" in detail


def test_decode_message_wrong_maps_known_code():
    is_submitted, detail = dacon.decode_message("wrong", 901)
    assert is_submitted is False
    assert detail == "csv file's header doesn't match. csv 파일의 헤더가 올바르지 않습니다"


def test_decode_message_wrong_unknown_code_keeps_code():
    is_submitted, detail = dacon.decode_message("wrong", 4242)
    assert is_submitted is False
    assert "4242" in detail


def test_submit_requires_credentials(tmp_path, monkeypatch):
    for key in ("DACON_API_TOKEN", "DACON_CPT_ID", "DACON_TEAM_NAME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(dacon, "load_dotenv", lambda: None)  # .env 유출 차단
    f = tmp_path / "sub.zip"
    f.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="DACON_API_TOKEN"):
        dacon.submit(str(f))


def test_submit_fake_http_success(tmp_path, monkeypatch):
    monkeypatch.setenv("DACON_API_TOKEN", "t")
    monkeypatch.setenv("DACON_CPT_ID", "c")
    monkeypatch.setenv("DACON_TEAM_NAME", "n")
    monkeypatch.setenv("COSCIENTIST_DACON_FAKE_HTTP", "ok")
    f = tmp_path / "sub.zip"
    f.write_bytes(b"x")
    result = dacon.submit(str(f), memo="EXP-001")
    assert result == {"isSubmitted": True, "detail": "Success", "memo": "EXP-001"}


def test_submit_fake_http_rejected_with_code(tmp_path, monkeypatch):
    monkeypatch.setenv("DACON_API_TOKEN", "t")
    monkeypatch.setenv("DACON_CPT_ID", "c")
    monkeypatch.setenv("DACON_TEAM_NAME", "n")
    monkeypatch.setenv("COSCIENTIST_DACON_FAKE_HTTP", "wrong:903")
    f = tmp_path / "sub.zip"
    f.write_bytes(b"x")
    result = dacon.submit(str(f))
    assert result["isSubmitted"] is False
    assert "row 개수" in result["detail"]
