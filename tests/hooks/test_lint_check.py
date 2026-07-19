from hooks.lint_check import lint_check

GOOD = 'import json\nprint(json.dumps({"a": 1.0}))\n'
SYNTAX_ERROR = "def broken(:\n"
UNDEFINED_NAME = "import json\nprint(json.dumps({'a': undefined_var}))\n"


def _write(tmp_path, content):
    p = tmp_path / "snippet.py"
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_good_code_passes(tmp_path):
    ok, reason = lint_check(_write(tmp_path, GOOD))
    assert ok, reason


def test_syntax_error_fails_with_reason(tmp_path):
    ok, reason = lint_check(_write(tmp_path, SYNTAX_ERROR))
    assert not ok and "컴파일" in reason


def test_undefined_name_caught_by_ruff(tmp_path):
    ok, reason = lint_check(_write(tmp_path, UNDEFINED_NAME))
    assert not ok and "ruff" in reason
