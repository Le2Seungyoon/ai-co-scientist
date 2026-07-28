"""Gemini/Gemma 어댑터 (M5 실물화) — httpx 직접 호출, LLMRouter가 위임.

- system prompt = load_rules(agent) 전문이 task_input["rules"]로 들어옴 + 역할별 출력 계약
- user message = task_input에서 rules를 제외한 컨텍스트를 섹션별로 조립
- 구조화 출력: response_mime_type=application/json (gemma-4 지원 확인됨)
- gemma-4는 thinking 모델 — parts에 thought=true 조각이 섞이므로 extract_text가 걸러낸다
- RPM: 요청 간 최소 간격(60/rpm초), 429/5xx는 백오프 재시도 (스펙 §3)
"""
import json
import os
import time

import httpx

from ai_co_scientist.core.config import load_config, load_dotenv

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# 역할별 출력 계약 — scenarios.py의 mock 반환 형식과 1:1 (핵심 계약)
_ROLE_FORMATS = {
    "research": (
        '{"hypothesis": {"statement": str, "single_variable": str, "rationale": str}, '
        '"design": {"change": str, "keep_fixed": [str], "expected_effect": str}}\n'
        "제약: 실험 변인은 정확히 하나(single_variable). single_variable은 keep_fixed에 "
        "절대 포함하지 말 것. keep_fixed에는 이번에 고정하는 다른 변인들을 나열."
    ),
    "coder": (
        '{"code": str}\n'
        "code는 아래 계약을 지키는 완결된 단일 파이썬 스크립트여야 한다:\n"
        "- 데이터: os.environ[\"COSCIENTIST_TOY_DATA\"] 디렉토리의 train.csv/val.csv/holdout.csv "
        "(컬럼 x,y — holdout의 y는 사용 금지). x 범위는 [-2,2], train 60행/val 20행/holdout 20행.\n"
        "- 허용 라이브러리: 표준 라이브러리 + numpy만. 네트워크·파일 다운로드 금지.\n"
        "- 결정적일 것(난수는 고정 seed). 전체 실행 4초 이내(무거운 탐색 금지).\n"
        "- holdout 예측을 Path(__file__).parent / \"predictions.csv\"에 (id,pred) 헤더로 저장.\n"
        "- 마지막 stdout 줄에 정확히 JSON 하나만 출력: "
        '{"train_mse": float, "val_mse": float, "predictions_path": str}\n'
        "- 실험설계(design)의 change 하나만 바꾸고 keep_fixed는 유지."
    ),
    "critic": (
        '{"target": str, "attacks": [str], "verdict": "pass" | "revise"}\n'
        "제약: 치명적 결함(근거 없음·계약 위반·논리 모순)일 때만 revise. 사소한 스타일 지적으로 "
        "revise하지 말 것. attacks는 각각 한 문장, 구체적 근거 포함."
    ),
    "harness_engineer": (
        '{"agent": str, "lesson": str}\n'
        "agent는 교훈을 반영할 대상 에이전트 이름 하나(pm/research/analysis/coder/executor/"
        "critic 중). lesson은 rules 파일에 append할 재발 방지 지침 한 줄."
    ),
}

_KNOWN_AGENTS = {"pm", "research", "analysis", "coder", "executor", "critic", "harness_engineer"}


def extract_text(response: dict) -> str:
    """generateContent 응답에서 thought가 아닌 text 조각을 이어붙인다."""
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError(f"후보 없음 (blockReason: {response.get('promptFeedback')})")
    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p["text"] for p in parts if p.get("text") and not p.get("thought"))
    if not text.strip():
        raise ValueError(f"빈 응답 (finishReason: {cand.get('finishReason')})")
    return text


def parse_json_block(text: str) -> dict:
    """JSON 모드 응답 파싱 — 서문/코드펜스/후미 잡텍스트 변형까지 흡수.

    gemma-4는 JSON 모드에서도 객체 뒤에 추가 텍스트를 붙이는 경우가 실측됨
    ("Extra data" — 사이클 실행 로그). 첫 번째 완전한 객체만 raw_decode로 취한다.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError(f"JSON 없음: {text[:200]}")
    try:
        out, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e} — {text[start:start + 200]}")
    if not isinstance(out, dict):
        raise ValueError(f"dict 아님: {type(out).__name__}")
    return out


def postprocess(role: str, raw: dict, task_input: dict) -> dict:
    """모델 응답을 scenarios.py 계약 형태로 강제 — 어긋나면 ValueError(재요청 트리거)."""
    if role == "research":
        hyp, design = raw.get("hypothesis"), raw.get("design")
        if not isinstance(hyp, dict) or not isinstance(design, dict):
            raise ValueError("hypothesis/design 누락")
        cycle_id = task_input["cycle_id"]
        hyp["cycle_id"] = cycle_id
        design["cycle_id"] = cycle_id
        kf = design.get("keep_fixed", [])
        design["keep_fixed"] = [str(v) for v in kf] if isinstance(kf, list) else [str(kf)]
        return {"hypothesis": hyp, "design": design}
    if role == "coder":
        code = raw.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code 누락")
        return {"code": code}
    if role == "critic":
        verdict = str(raw.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "revise"):
            raise ValueError(f"verdict 불량: {raw.get('verdict')!r}")
        attacks = raw.get("attacks", [])
        return {
            # target은 모델 답변 대신 입력의 draft_type을 신뢰 (환각 방지)
            "target": task_input.get("draft_type", "unknown"),
            "attacks": [str(a) for a in attacks] if isinstance(attacks, list) else [str(attacks)],
            "verdict": verdict,
        }
    if role == "harness_engineer":
        agent, lesson = str(raw.get("agent", "")).strip(), str(raw.get("lesson", "")).strip()
        if agent not in _KNOWN_AGENTS:
            raise ValueError(f"미등록 agent: {agent!r}")
        if not lesson:
            raise ValueError("lesson 누락")
        return {"agent": agent, "lesson": lesson}
    raise KeyError(f"미등록 role: {role}")


def build_messages(role: str, task_input: dict) -> tuple[str, str]:
    """(system, user) 프롬프트 조립 — rules 전문 + 출력 계약 / rules 제외 컨텍스트 섹션."""
    rules = task_input.get("rules", "")
    system = (
        f"{rules}\n\n## 출력 형식 (절대 규칙)\n"
        f"아래 스키마의 JSON 객체 하나만 출력한다. 설명·서문·코드펜스 금지.\n{_ROLE_FORMATS[role]}"
    )
    sections = []
    for key, value in task_input.items():
        if key == "rules" or value in ("", None, [], {}):
            continue
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        sections.append(f"### {key}\n{rendered}")
    return system, "\n\n".join(sections) or "(추가 컨텍스트 없음)"


class GeminiRouter:
    def __init__(self):
        load_dotenv()
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY 없음 — .env를 채우거나 mock.llm을 유지할 것")
        cfg = load_config()["llm"].get("gemini", {})
        self._model = cfg.get("model", "gemma-4-31b-it")
        self._role_models: dict = cfg.get("role_models", {})
        self._min_interval = 60.0 / cfg.get("rpm", 20)
        self._max_retries = cfg.get("max_retries", 3)
        # thinking 모델(gemma-4)은 응답까지 수 분 걸릴 수 있음 — read를 넉넉하게 (스모크 실측)
        self._timeout = cfg.get("timeout_s", 300)
        self._client = httpx.Client(
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            timeout=httpx.Timeout(connect=10, read=self._timeout, write=30, pool=30))
        self._last_call = 0.0

    def _generate(self, model: str, system: str, user: str) -> dict:
        """1회 API 호출 (RPM 간격 + 429/5xx 백오프). 응답 JSON(dict) 반환."""
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.4},
        }
        for attempt in range(self._max_retries + 1):
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self._client.post(f"{API_BASE}/{model}:generateContent", json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                # 네트워크 단절/read 타임아웃도 재시도 대상 (스모크에서 ReadTimeout 실측)
                if attempt < self._max_retries:
                    backoff = [5, 15, 30, 60][min(attempt, 3)]
                    print(f"[gemini] 전송 오류 {type(e).__name__} — {backoff}s 백오프 "
                          f"(시도 {attempt + 1}/{self._max_retries})", flush=True)
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"Gemini API 전송 실패: {type(e).__name__}: {e}") from e
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503) and attempt < self._max_retries:
                backoff = [5, 15, 30, 60][min(attempt, 3)]
                print(f"[gemini] HTTP {resp.status_code} — {backoff}s 백오프 "
                      f"(시도 {attempt + 1}/{self._max_retries})", flush=True)
                time.sleep(backoff)
                continue
            raise RuntimeError(f"Gemini API 실패 HTTP {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError("unreachable")

    def invoke(self, role: str, task_input: dict) -> dict:
        model = self._role_models.get(role, self._model)
        system, user = build_messages(role, task_input)
        last_err = ""
        for content_try in range(2):
            prompt = user if not last_err else (
                f"{user}\n\n### 직전 응답 형식 오류 — 스키마에 맞게 다시\n{last_err}")
            response = self._generate(model, system, prompt)
            try:
                return postprocess(role, parse_json_block(extract_text(response)), task_input)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = str(e)
                print(f"[gemini] {role} 응답 계약 위반(재요청 {content_try + 1}/2): {last_err[:200]}",
                      flush=True)
        raise RuntimeError(f"Gemini {role} 응답이 계약을 2회 연속 위반: {last_err[:300]}")
