"""역할별 LLM 라우터 — mock(결정적 시나리오) | gemini(실 API, gemini.py 위임)."""
import os

from ai_co_scientist.core.config import load_config
from ai_co_scientist.llm.scenarios import SCENARIOS


class LLMRouter:
    def __init__(self, scenario: str | None = None):
        cfg = load_config()["llm"]
        # env가 config보다 우선 (COSCIENTIST_MOCKLLM_SCENARIO와 동일 패턴) —
        # config 기본은 mock으로 두고(테스트 오프라인 불변식) 실 기동만 env로 전환
        provider = os.environ.get("COSCIENTIST_LLM_PROVIDER") or cfg["provider"]
        if provider == "gemini":
            from ai_co_scientist.llm.gemini import GeminiRouter
            self._delegate = GeminiRouter()
        elif provider == "mock":
            self._delegate = None
        else:
            raise ValueError(f"알 수 없는 LLM provider: {provider}")
        self._scenario = (
            scenario
            or os.environ.get("COSCIENTIST_MOCKLLM_SCENARIO")
            or cfg["scenario"]
        )
        if self._scenario not in SCENARIOS:
            raise ValueError(f"알 수 없는 MockLLM 시나리오: {self._scenario}")
        self._counts: dict[str, int] = {}

    def invoke(self, role: str, task_input: dict) -> dict:
        """role의 시나리오 함수를 (호출 인덱스, 입력)으로 호출. 미등록 role이면 KeyError."""
        if self._delegate is not None:
            return self._delegate.invoke(role, task_input)
        idx = self._counts.get(role, 0)
        self._counts[role] = idx + 1
        return SCENARIOS[self._scenario][role](idx, task_input)
