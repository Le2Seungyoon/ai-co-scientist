import pytest

from ai_co_scientist.core.rules import load_rules

AGENTS = ["pm", "research", "analysis", "coder", "executor", "critic", "harness_engineer"]


@pytest.mark.parametrize("agent", AGENTS)
def test_load_rules_has_required_sections(agent):
    text = load_rules(agent)
    for section in ["## 역할", "## 전략", "## 하드 제약", "## 교훈"]:
        assert section in text, f"{agent}.md에 '{section}' 섹션 없음"


def test_unknown_agent_raises():
    with pytest.raises(FileNotFoundError):
        load_rules("nobody")


def test_append_lesson_and_dedup(tmp_path, monkeypatch):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "coder.md").write_text(
        "# Coder\n\n## 역할\nx\n\n## 교훈\n(Harness Engineer가 append)\n", encoding="utf-8")
    monkeypatch.setenv("COSCIENTIST_RULES_DIR", str(rules_dir))

    from ai_co_scientist.core.rules import append_lesson, load_rules
    assert append_lesson("coder", "impl_bug 재발 방지: traceback을 근거로만 수정") is True
    assert "impl_bug 재발 방지" in load_rules("coder")
    assert append_lesson("coder", "impl_bug 재발 방지: traceback을 근거로만 수정") is False  # 중복


def test_append_lesson_exact_line_match(tmp_path, monkeypatch):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "coder.md").write_text("# C\n\n## 교훈\n- 짧은 교훈이다\n", encoding="utf-8")
    monkeypatch.setenv("COSCIENTIST_RULES_DIR", str(rules_dir))
    from ai_co_scientist.core.rules import append_lesson
    # 기존 교훈의 부분 문자열이어도 정확한 라인이 아니면 append 되어야 한다
    assert append_lesson("coder", "짧은 교훈") is True
    assert append_lesson("coder", "짧은 교훈") is False  # 이제 정확 라인 존재 → 중복
