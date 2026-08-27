---
paths:
  - tests/**
---
# Testing

- Location & convention: `tests/test_*.py` (패키지 모듈은 `tests/<pkg>/test_*.py`).
  Run: `uv run pytest -q` (all) · `uv run pytest tests/test_registry.py -q` (one file) ·
  `-k <pattern>` (one test).
- Deterministic, **no network access** — API 키 없이 전부 통과해야 한다. DACON 백엔드는
  `COSCIENTIST_DACON_FAKE_HTTP`로 HTTP를 대체하고, 파일 상태는 `tmp_path`로 격리한다
  (레지스트리 테스트는 항상 `path=tmp_path/...`를 넘긴다 — 실제 `runtime/registry.jsonl`을
  건드리면 실험 기록이 오염된다).
- 학습 스크립트는 GPU/데이터가 필요하므로 단위테스트로 돌리지 않는다. 대신 **계약**을 검사한다:
  standalone 여부, 매니페스트의 X/y 도메인 선언, 폐기 기능 부재
  (`tests/test_train_manifest.py`).
- **No weak asserts**: 범위 검사만 하면 잘못된 구현이 통과한다. 정확한 값과 관계를 검증할 것.
- 회귀 테스트는 **어떤 버그를 고정하는지 주석으로** 남긴다.
