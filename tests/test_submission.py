"""제출 zip 검증 — 오프라인(합성 zip, data/ runtime/ 미접근).

검사기가 실제로 **판별**하는지를 본다: 정상 zip은 통과, 파일 수가 다르면 실패, max가
{140,150,160,170} 밖이면 실패. 검사기를 무력화하는 가장 조용한 실패(모두 통과)를 막는 것이
이 파일의 목적이다.
"""
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from ai_co_scientist.submission import EXPECTED_FILES, decode_png_gray8, encode_png_gray8, verify_submission

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


# ── 합성 PNG 인코더 (stdlib) ─────────────────────────────────
# 디코더와 독립적으로 필터 타입 0~4를 만들어 넣기 위해 테스트 쪽에 둔다.

def _chunk(ctype: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + ctype + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))


def _filter_row(cur: np.ndarray, prior: np.ndarray, ftype: int) -> bytes:
    out = bytearray()
    for x in range(len(cur)):
        a = int(cur[x - 1]) if x else 0
        b = int(prior[x])
        c = int(prior[x - 1]) if x else 0
        if ftype == 0:
            pred = 0
        elif ftype == 1:
            pred = a
        elif ftype == 2:
            pred = b
        elif ftype == 3:
            pred = (a + b) >> 1
        else:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
        out.append((int(cur[x]) - pred) & 0xFF)
    return bytes(out)


def _encode_png_gray8_with_filters(img: np.ndarray, ftype: int = 0) -> bytes:
    h, w = img.shape
    raw = bytearray()
    prior = np.zeros(w, dtype=np.uint8)
    for y in range(h):
        raw.append(ftype)
        raw += _filter_row(img[y], prior, ftype)
        prior = img[y]
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\x0a" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(raw))) + _chunk(b"IEND", b""))


def _image(max_value: int, h: int = 6, w: int = 5) -> np.ndarray:
    """max가 정확히 max_value인 이미지 (나머지는 그보다 작은 값)."""
    img = np.full((h, w), max_value // 2, dtype=np.uint8)
    img[0, 0] = max_value
    return img


def _make_zip(tmp_path: Path, maxima, name="sub.zip", ftype: int = 0) -> Path:
    zp = tmp_path / name
    with zipfile.ZipFile(zp, "w") as zf:
        for i, m in enumerate(maxima):
            zf.writestr(f"img_{i:05d}.png", _encode_png_gray8_with_filters(_image(m), ftype))
    return zp


# ── 디코더 ──────────────────────────────────────────────────

@pytest.mark.parametrize("ftype", [0, 1, 2, 3, 4])
def test_decode_roundtrip_every_filter_type(ftype):
    """PNG 필터 5종을 모두 복원한다. cv2가 쓴 제출본은 적응형 필터라 0만 지원하면
    현장에서 오탐/미탐이 난다."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 171, size=(9, 7), dtype=np.uint8)
    assert np.array_equal(decode_png_gray8(_encode_png_gray8_with_filters(img, ftype)), img)


def test_decode_rejects_non_png():
    with pytest.raises(ValueError, match="PNG 서명"):
        decode_png_gray8(b"not a png at all")


def test_decode_matches_cv2_when_available():
    """실제 제출본을 만드는 인코더(cv2.imwrite)의 출력을 그대로 읽는지 교차 검증."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    img = rng.integers(0, 171, size=(72, 48), dtype=np.uint8)
    buf = cv2.imencode(".png", img)[1].tobytes()
    assert np.array_equal(decode_png_gray8(buf), img)


# ── 검증기 ──────────────────────────────────────────────────

def test_good_zip_passes(tmp_path):
    zp = _make_zip(tmp_path, [140, 150, 160, 170])
    r = verify_submission(zp, expected_files=4)
    assert r.ok is True
    assert (r.n_files, r.n_outside, r.percent_outside) == (4, 0, 0.0)
    assert r.maxima == [140, 150, 160, 170]
    assert r.errors == []


def test_wrong_file_count_fails(tmp_path):
    zp = _make_zip(tmp_path, [140, 150, 160])
    r = verify_submission(zp, expected_files=4)
    assert r.ok is False and r.n_files == 3
    assert any("파일 수 3 != 기대 4" in e for e in r.errors)
    assert r.n_outside == 0  # 픽셀은 멀쩡하다 — 실패 사유가 섞이지 않는지 확인


def test_out_of_set_maximum_fails(tmp_path):
    zp = _make_zip(tmp_path, [140, 150, 160, 165])
    r = verify_submission(zp, expected_files=4)
    assert r.ok is False
    assert r.maxima == [140, 150, 160, 165]
    assert (r.n_outside, r.percent_outside) == (1, 25.0)
    assert any("165" in e for e in r.errors)


def test_unreadable_member_counts_as_failure(tmp_path):
    """파싱 실패를 조용히 건너뛰면 검사기가 무력해진다 — 실패로 세는지 고정한다."""
    zp = _make_zip(tmp_path, [140, 150, 160])
    with zipfile.ZipFile(zp, "a") as zf:
        zf.writestr("img_00003.png", b"garbage")
    r = verify_submission(zp, expected_files=4)
    assert r.ok is False and r.n_files == 4 and r.n_outside == 1
    assert r.unreadable and "img_00003.png" in r.unreadable[0]


def test_missing_zip_fails_without_raising(tmp_path):
    r = verify_submission(tmp_path / "nope.zip", expected_files=4)
    assert r.ok is False and r.n_files == 0
    assert any("zip이 없다" in e for e in r.errors)


def test_default_expected_files_is_25988():
    assert EXPECTED_FILES == 25988


def test_default_allowed_maxima_are_the_four_levels(tmp_path):
    """허용 집합의 기본값이 docs/data-facts.md §1의 4개인지 — 기본 호출로 확인."""
    zp = _make_zip(tmp_path, [140, 171])
    r = verify_submission(zp, expected_files=2)
    assert r.allowed_maxima == [140, 150, 160, 170]
    assert r.ok is False and r.n_outside == 1


# ── CLI 프리체크 (HTTP는 가짜, 네트워크 없음) ────────────────

def _load_cli():
    sys.path.insert(0, str(SCRIPTS))
    import dacon_submit  # noqa: PLC0415
    return dacon_submit


def _creds(monkeypatch, fake="ok"):
    monkeypatch.setenv("DACON_API_TOKEN", "t")
    monkeypatch.setenv("DACON_CPT_ID", "c")
    monkeypatch.setenv("DACON_TEAM_NAME", "n")
    monkeypatch.setenv("COSCIENTIST_DACON_FAKE_HTTP", fake)


def test_cli_blocks_post_when_verification_fails(tmp_path, monkeypatch):
    """검증 실패 zip은 POST되지 않고 종료 코드 2. submit이 불렸으면 실패한다."""
    cli = _load_cli()
    _creds(monkeypatch)
    called = []
    monkeypatch.setattr(cli.dacon, "submit", lambda *a, **k: called.append(a) or {})
    zp = _make_zip(tmp_path, [140, 165])
    monkeypatch.setattr(sys, "argv", ["dacon_submit.py", str(zp), "--expected-files", "2"])
    assert cli.main() == cli.EXIT_VERIFY_FAILED
    assert called == []


def test_cli_rejected_submission_exits_nonzero(tmp_path, monkeypatch):
    """접수 거부(`isSubmitted: false`)가 성공과 구별되는지 — 예전엔 둘 다 exit 0이었다."""
    cli = _load_cli()
    _creds(monkeypatch, fake="day_max_count")
    zp = _make_zip(tmp_path, [140, 150])
    monkeypatch.setattr(sys, "argv", ["dacon_submit.py", str(zp), "--expected-files", "2"])
    assert cli.main() == cli.EXIT_NOT_SUBMITTED


def test_cli_accepted_submission_exits_zero(tmp_path, monkeypatch):
    cli = _load_cli()
    _creds(monkeypatch, fake="ok")
    zp = _make_zip(tmp_path, [140, 150])
    monkeypatch.setattr(sys, "argv", ["dacon_submit.py", str(zp), "--expected-files", "2",
                                      "--report-id", "EXP-999"])
    assert cli.main() == 0


def test_cli_verify_only_does_not_submit(tmp_path, monkeypatch):
    cli = _load_cli()
    _creds(monkeypatch)
    called = []
    monkeypatch.setattr(cli.dacon, "submit", lambda *a, **k: called.append(a) or {})
    zp = _make_zip(tmp_path, [140, 150])
    monkeypatch.setattr(sys, "argv", ["dacon_submit.py", str(zp), "--expected-files", "2",
                                      "--verify-only"])
    assert cli.main() == 0
    assert called == []


# ── 인코더 ──────────────────────────────────────────────────

def test_encode_png_gray8_roundtrips():
    """인코더 출력이 기존 디코더로 원본 배열까지 복원돼야 한다."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(72, 48), dtype=np.uint8)
    out = decode_png_gray8(encode_png_gray8(img))
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_encode_png_gray8_is_deterministic():
    """같은 입력은 같은 바이트 — 두 경로의 zip 바이트 동일성이 여기에 걸려 있다."""
    img = np.arange(72 * 48, dtype=np.int64).reshape(72, 48).astype(np.uint8)
    assert encode_png_gray8(img) == encode_png_gray8(img)


def test_encode_png_gray8_rejects_wrong_shape_or_dtype():
    """조용히 넘기지 않는다 — 잘못된 배열이 제출본이 되면 슬롯 하나가 날아간다."""
    with pytest.raises(ValueError):
        encode_png_gray8(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_png_gray8(np.zeros((4, 4), dtype=np.float32))


def test_encode_png_gray8_matches_cv2_decode_when_available():
    """cv2가 읽을 수 있는 PNG여야 한다 — 채점 측 디코더가 무엇일지 모른다."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(16, 24), dtype=np.uint8)
    buf = np.frombuffer(encode_png_gray8(img), dtype=np.uint8)
    assert np.array_equal(cv2.imdecode(buf, cv2.IMREAD_UNCHANGED), img)
