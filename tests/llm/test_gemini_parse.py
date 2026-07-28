"""GeminiRouter 순수 헬퍼 테스트 — 네트워크 없음 (실 API 호출은 사람 몫)."""
import pytest

from ai_co_scientist.llm.gemini import build_messages, extract_text, parse_json_block, postprocess


def _response(parts):
    return {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]}


class TestExtractText:
    def test_skips_thought_parts(self):
        # gemma-4는 thinking 모델 — thought=true 조각이 답변 앞에 섞여 옴 (실측)
        resp = _response([
            {"text": "생각 과정...", "thought": True},
            {"text": '{"ok": '},
            {"text": "true}"},
        ])
        assert extract_text(resp) == '{"ok": true}'

    def test_no_candidates_raises(self):
        with pytest.raises(ValueError, match="후보 없음"):
            extract_text({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})

    def test_only_thoughts_raises(self):
        with pytest.raises(ValueError, match="빈 응답"):
            extract_text(_response([{"text": "...", "thought": True}]))


class TestParseJsonBlock:
    def test_plain_json(self):
        assert parse_json_block('{"a": 1}') == {"a": 1}

    def test_fenced_or_prefixed_json(self):
        assert parse_json_block('```json\n{"a": 1}\n```') == {"a": 1}
        assert parse_json_block('결과입니다: {"a": 1}') == {"a": 1}

    def test_trailing_extra_data(self):
        # gemma-4가 JSON 모드에서도 객체 뒤에 텍스트를 붙인 실측 사례 (사이클 2 재요청 원인)
        assert parse_json_block('{"code": "x = 1"} 위 코드는 설계를 반영합니다.') == {"code": "x = 1"}
        # 중첩 객체 뒤 잡텍스트 — rfind('}') 방식이면 잡텍스트의 } 없이도 실패했을 케이스
        assert parse_json_block('{"a": {"b": 2}}\nextra {') == {"a": {"b": 2}}

    def test_non_dict_raises(self):
        # 객체가 아예 없으면 "JSON 없음" — 어느 쪽이든 ValueError로 재요청 트리거가 계약
        with pytest.raises(ValueError):
            parse_json_block("[1, 2]")


class TestPostprocess:
    def test_research_injects_cycle_id(self):
        raw = {"hypothesis": {"statement": "s", "single_variable": "lr", "rationale": "r"},
               "design": {"change": "lr", "keep_fixed": ["deg"], "expected_effect": "e"}}
        out = postprocess("research", raw, {"cycle_id": 7})
        # cycle_id는 모델 응답이 아니라 task_input에서 강제 주입 (모델 환각 방지)
        assert out["hypothesis"]["cycle_id"] == 7
        assert out["design"]["cycle_id"] == 7

    def test_coder_missing_code_raises(self):
        with pytest.raises(ValueError, match="code 누락"):
            postprocess("coder", {"code": ""}, {})

    def test_critic_target_from_input_and_verdict_coerced(self):
        out = postprocess("critic", {"target": "환각된-타겟", "attacks": [], "verdict": "PASS"},
                          {"draft_type": "research_output"})
        assert out["target"] == "research_output"
        assert out["verdict"] == "pass"

    def test_critic_bad_verdict_raises(self):
        with pytest.raises(ValueError, match="verdict 불량"):
            postprocess("critic", {"attacks": [], "verdict": "maybe"}, {"draft_type": "x"})

    def test_harness_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="미등록 agent"):
            postprocess("harness_engineer", {"agent": "ghost", "lesson": "l"}, {})


class TestBuildMessages:
    def test_rules_in_system_not_user(self):
        system, user = build_messages("critic", {
            "rules": "## 역할\n공격 전용", "draft_type": "verdict", "draft": {"a": 1}})
        assert "공격 전용" in system and "출력 형식" in system
        assert "공격 전용" not in user
        assert "### draft_type" in user

    def test_empty_values_dropped(self):
        _, user = build_messages("research", {"rules": "r", "cycle_id": 1, "critique": ""})
        assert "critique" not in user
