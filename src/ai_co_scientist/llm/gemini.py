"""Gemini 어댑터 골격 (M5) — 실 연동은 검증 포함 사람 몫 (스펙 §7 M5).

설계 기록 (이월 ② — rules→프롬프트 조립):
- system prompt = load_rules(agent) 전문 (역할·전략·하드 제약·교훈이 그대로 지침이 됨)
- user message = task_input에서 rules를 제외한 컨텍스트(consensus/diagnoses/critique/error 등)를
  섹션별로 조립 — Langfuse류 {{var}} 치환 방식 권장 (emis-api 관례)
- 구조화 출력: 역할별 response_schema(JSON mode)로 core.schema 계약을 강제
- 모델 배분: config llm.gemini.role_models (저빈도 역할=Gemini, 고빈도=Gemma — 스펙 §7-⑥,
  실측 호출 빈도는 공유로그/wandb로 수집 후 확정)
- RPM/RPD: 요청 간 간격 제어 + 429 백오프, Gemma 폴백 (스펙 §3)
"""


class GeminiRouter:
    def __init__(self):
        # TODO(M5+): langchain-google-genai 의존성 추가 + ChatGoogleGenerativeAI 초기화
        pass

    def invoke(self, role: str, task_input: dict) -> dict:
        raise NotImplementedError(
            "Gemini 실 연동은 골격만 — 키/의존성 추가 후 위 설계 기록대로 구현")
