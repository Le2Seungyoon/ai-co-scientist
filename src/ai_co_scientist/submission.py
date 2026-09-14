"""제출 zip 검증 — 파일 수와 **이미지별 최대값**을 실제로 읽어 확인한다.

왜 필요한가: `d = L·(1 − s)`이고 `s ∈ [0,1]`이므로 **한 장의 max는 반드시 배경 레벨 L**이고,
L은 4택1이다 (docs/data-facts.md §1-2 — sim depth GT 20,000장에서 unique max = [140 150 160 170],
`d > L` 0.000%). 따라서 max가 {140,150,160,170} 밖인 이미지는 재구성 파이프라인이 깨졌다는 뜻이다.

`scripts/infer_decomposed.py`가 찍던 `제출본 25988장`은 `len(test_names.json)`이라 **무조건**
25,988이 나온다 — zip 안을 본 적이 없으므로 검증이 아니다. 여기서 zip을 열어 세고, 픽셀을 읽는다.

**PNG 디코딩은 stdlib(zlib) + numpy로 한다.** cv2는 `baseline` 그룹이라 기본 환경에 없을 수
있는데, 제출 직전 검증이 옵션 의존성 때문에 건너뛰어지면 검증이 없는 것과 같다.
"""
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai_co_scientist.sem import LEVELS

# test 이미지 수 (docs/data-facts.md · runtime/cache/test_names.json 길이와 같아야 한다)
EXPECTED_FILES = 25988
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunks(data: bytes):
    """PNG 청크 스트림 → (type, payload). 길이/서명이 어긋나면 ValueError."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("PNG 서명이 아니다")
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        if len(payload) != length:
            raise ValueError("PNG 청크가 잘렸다")
        yield ctype, payload
        pos += 12 + length  # length + type + data + crc


def decode_png_gray8(data: bytes) -> np.ndarray:
    """8-bit 그레이스케일 non-interlaced PNG → (H, W) uint8 배열.

    제출본은 `cv2.imwrite`가 쓴 8bit 1채널이다. 다른 형식이면 조용히 넘기지 않고 예외를 던진다 —
    읽지 못한 파일을 통과로 세면 검사기가 무력해진다 (`.claude/rules/enforcement.md`).
    """
    header, idat = None, bytearray()
    for ctype, payload in _png_chunks(data):
        if ctype == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload[:13])
        elif ctype == b"IDAT":
            idat += payload
        elif ctype == b"IEND":
            break
    if header is None:
        raise ValueError("IHDR 청크가 없다")
    width, height, depth, color, _comp, _filt, interlace = header
    if (depth, color, interlace) != (8, 0, 0):
        raise ValueError(f"지원하지 않는 PNG 형식 (depth={depth}, color={color}, "
                         f"interlace={interlace}) — 8bit 그레이 non-interlaced만 읽는다")

    raw = zlib.decompress(bytes(idat))
    stride = width + 1  # 행마다 filter 바이트 1개가 앞에 붙는다
    if len(raw) != stride * height:
        raise ValueError(f"IDAT 길이 불일치: {len(raw)} != {stride * height}")

    out = np.zeros((height, width), dtype=np.uint8)
    prior = np.zeros(width, dtype=np.int32)
    for y in range(height):
        ftype = raw[y * stride]
        line = np.frombuffer(raw, dtype=np.uint8, count=width, offset=y * stride + 1)
        cur = line.astype(np.int32)
        if ftype == 0:  # None
            pass
        elif ftype == 2:  # Up — 좌측 의존이 없어 벡터화된다
            cur = (cur + prior) & 0xFF
        else:  # Sub / Average / Paeth — 좌측 픽셀에 의존하므로 순차 계산
            for x in range(width):
                a = cur[x - 1] if x else 0
                b = prior[x]
                c = prior[x - 1] if x else 0
                if ftype == 1:
                    cur[x] = (cur[x] + a) & 0xFF
                elif ftype == 3:
                    cur[x] = (cur[x] + ((a + b) >> 1)) & 0xFF
                elif ftype == 4:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    cur[x] = (cur[x] + pred) & 0xFF
                else:
                    raise ValueError(f"알 수 없는 PNG 필터 타입 {ftype}")
        out[y] = cur.astype(np.uint8)
        prior = cur
    return out


@dataclass
class SubmissionCheck:
    """검증 결과. `ok`가 False면 제출하지 않는다."""
    ok: bool
    n_files: int
    expected_files: int
    n_images: int
    maxima: list[int]
    allowed_maxima: list[int]
    n_outside: int
    percent_outside: float
    unreadable: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "n_files": self.n_files, "expected_files": self.expected_files,
                "n_images": self.n_images, "maxima": self.maxima,
                "allowed_maxima": self.allowed_maxima, "n_outside": self.n_outside,
                "percent_outside": self.percent_outside,
                "unreadable": self.unreadable[:10], "errors": self.errors}

    def summary(self) -> str:
        state = "PASS" if self.ok else "FAIL"
        return (f"[{state}] {self.n_files}/{self.expected_files}장 · max={self.maxima} "
                f"· 벗어남 {self.n_outside}장 ({self.percent_outside:.2f} percent)")


def verify_submission(zip_path: str | Path, expected_files: int = EXPECTED_FILES,
                      allowed_maxima=LEVELS, max_unreadable_listed: int = 10) -> SubmissionCheck:
    """제출 zip을 열어 파일 수·이미지별 max를 검사한다. 반환: `SubmissionCheck`.

    통과 조건 세 가지가 모두 성립해야 `ok=True`:
      1. 멤버 수 == expected_files (기본 25,988)
      2. 발견된 max 집합 ⊆ {140,150,160,170}
      3. 벗어난 이미지 0장 (0.00 percent)
    읽지 못한 PNG는 **통과가 아니라 실패**로 센다 — 파싱 실패를 조용히 건너뛰면 검사기가
    무력한 채로 초록을 낸다.
    """
    allowed = sorted(int(v) for v in allowed_maxima)
    path = Path(zip_path)
    errors: list[str] = []
    if not path.exists():
        return SubmissionCheck(False, 0, expected_files, 0, [], allowed, 0, 0.0,
                               errors=[f"zip이 없다: {path}"])

    found: set[int] = set()
    unreadable: list[str] = []
    n_outside = n_images = 0
    with zipfile.ZipFile(path) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        for name in names:
            try:
                img = decode_png_gray8(zf.read(name))
            except (ValueError, zlib.error, struct.error) as exc:
                if len(unreadable) < max_unreadable_listed:
                    unreadable.append(f"{name}: {exc}")
                n_outside += 1
                n_images += 1
                continue
            n_images += 1
            m = int(img.max())
            found.add(m)
            if m not in allowed:
                n_outside += 1

    n_files = len(names)
    percent_outside = round(100.0 * n_outside / n_images, 4) if n_images else 100.0
    if n_files != expected_files:
        errors.append(f"파일 수 {n_files} != 기대 {expected_files}")
    bad = sorted(found - set(allowed))
    if bad:
        errors.append(f"허용 밖 max 값 {bad} (허용 {allowed})")
    if unreadable:
        errors.append(f"읽을 수 없는 PNG {len(unreadable)}장 이상 — 예: {unreadable[0]}")
    if n_outside:
        errors.append(f"max가 허용 밖인 이미지 {n_outside}장 ({percent_outside:.2f} percent)")

    return SubmissionCheck(ok=not errors, n_files=n_files, expected_files=expected_files,
                           n_images=n_images, maxima=sorted(found), allowed_maxima=allowed,
                           n_outside=n_outside, percent_outside=percent_outside,
                           unreadable=unreadable, errors=errors)
