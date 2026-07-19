from hooks.submission_guard import submission_guard


def test_first_experiment_allowed():
    ok, reason = submission_guard(0.5, best=None)
    assert ok and "첫" in reason


def test_improvement_allowed_and_regression_blocked():
    assert submission_guard(0.2, best={"val_mse": 0.3})[0] is True
    ok, reason = submission_guard(0.4, best={"val_mse": 0.3})
    assert not ok and "차단" in reason
