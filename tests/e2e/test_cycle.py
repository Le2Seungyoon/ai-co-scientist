import os
import shutil
import signal
import subprocess
import sys

import pytest

from ai_co_scientist.core.config import project_root
from ai_co_scientist.mcp_servers.shared_log import db


def _run_runner(args: list[str], env_extra: dict, timeout: int = 180):
    proc = subprocess.Popen(
        [sys.executable, "-m", "ai_co_scientist.runner", *args],
        env={**os.environ, **env_extra},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        stdout, stderr = proc.communicate()
        pytest.fail(f"runner가 {timeout}초 내에 끝나지 않음.\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return proc.returncode, stdout, stderr


def _isolated_env(tmp_path, **extra) -> dict:
    """모든 러너 호출의 DB를 tmp로 격리 — 실제 runtime/*.sqlite3 오염 방지.

    LOG/LIGHTNING/WANDB/DACON 4종 DB는 항상 tmp 기반으로 만든다(예전엔 toy 테스트
    2건만 WANDB_DB를 넘겨, 나머지 호출이 실제 runtime/wandb.sqlite3에 행을 누적시켰다
    — 실측 180행). RULES_DIR/TOY_DATA 등 호출부별 추가 env는 extra로 병합한다.
    """
    env = {
        "COSCIENTIST_LOG_DB": str(tmp_path / "log.sqlite3"),
        "COSCIENTIST_LIGHTNING_DB": str(tmp_path / "lightning.sqlite3"),
        "COSCIENTIST_WANDB_DB": str(tmp_path / "wandb.sqlite3"),
        "COSCIENTIST_DACON_DB": str(tmp_path / "dacon.sqlite3"),
    }
    env.update(extra)
    return env


def test_cycle_happy_path_three_rounds(tmp_path):
    env = _isolated_env(tmp_path)
    code, stdout, stderr = _run_runner(["--cycle", "3"], env)
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert stdout.count("outcome=ok") == 3
    assert len(db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="hypothesis")) == 3
    assert len(db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="diagnosis")) == 3


def test_cycle_executor_failure_replans_then_escalates(tmp_path):
    # replan_limit=2 → infra_event(infra_oom) 3건 ≥ harness.failure_repeat_threshold(2) → 트리거 발동.
    # append_lesson이 실제 rules/ 디렉토리를 오염시키지 않도록 COSCIENTIST_RULES_DIR로 tmp 복사본 격리.
    rules_dir = tmp_path / "rules"
    shutil.copytree(project_root() / "rules", rules_dir)
    env = _isolated_env(tmp_path, COSCIENTIST_INJECT_FAILURE="infra_oom",
                         COSCIENTIST_RULES_DIR=str(rules_dir))
    code, stdout, stderr = _run_runner(["--cycle", "1"], env)
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "outcome=escalated" in stdout
    # replan_limit=2 → research 3회(최초+재설계 2) 가설, infra_event 3회
    assert len(db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="hypothesis")) == 3
    assert len(db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="infra_event")) == 3


def test_m4_critic_revise_cycle(tmp_path):
    env = _isolated_env(tmp_path, COSCIENTIST_MOCKLLM_SCENARIO="critic_revise")
    code, stdout, stderr = _run_runner(["--cycle", "1"], env)
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "outcome=ok" in stdout
    # critic 1라운드 revise → research 재작업 → hypothesis 2건
    assert len(db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="hypothesis")) == 2


def test_m4_harness_appends_lesson_on_repeat_failure(tmp_path):
    rules_dir = tmp_path / "rules"
    shutil.copytree("rules", rules_dir)   # 실제 rules/ 골격 복사 (git 작업트리 보호)
    env = _isolated_env(tmp_path, COSCIENTIST_INJECT_FAILURE="infra_oom",
                         COSCIENTIST_RULES_DIR=str(rules_dir))
    code, stdout, stderr = _run_runner(["--cycle", "1"], env)
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "outcome=escalated" in stdout          # 기존 M2 계약 유지
    text = (rules_dir / "executor.md").read_text(encoding="utf-8")
    assert "재발 방지" in text                     # infra_event 3건 ≥ threshold 2 → 교훈 append


def test_cycle_coder_failure_retries_then_escalates(tmp_path):
    env = _isolated_env(tmp_path, COSCIENTIST_INJECT_FAILURE="impl_bug")
    code, stdout, stderr = _run_runner(["--cycle", "1"], env)
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "outcome=escalated" in stdout
    assert len(db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="hypothesis")) == 1


def test_m3_causal_chain_three_cycles(tmp_path):
    """가설→코드→실행→판정→컨센서스 갱신의 인과 사슬이 실제로 만들어지는지."""
    env = _isolated_env(tmp_path, COSCIENTIST_MOCKLLM_SCENARIO="default")
    code, stdout, stderr = _run_runner(["--cycle", "3"], env)
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert stdout.count("outcome=ok") == 3
    hyps = db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="hypothesis")
    assert len(hyps) == 3
    variables = {h["content"]["single_variable"] for h in hyps}
    assert len(variables) == 3                     # 사이클마다 다른 단일 변인
    assert len(db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="diagnosis")) == 3
    best = db.get_consensus(env["COSCIENTIST_LOG_DB"])["best_pipeline"]
    assert best["val_mse"] == pytest.approx(0.25)  # 0.35→0.30→0.25 개선 사슬


def test_m3_coder_selfcorrect_cycle_succeeds(tmp_path):
    env = _isolated_env(tmp_path, COSCIENTIST_MOCKLLM_SCENARIO="coder_selfcorrect")
    code, stdout, stderr = _run_runner(["--cycle", "1"], env)
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "outcome=ok" in stdout                  # 1차 버그 → self-correct → 성공


def _toy_env(tmp_path):
    toy_dir = tmp_path / "toy"
    subprocess.run([sys.executable, "-m", "ai_co_scientist.toy_task.generate", str(toy_dir)],
                   check=True)
    return _isolated_env(tmp_path, COSCIENTIST_TOY_DATA=str(toy_dir))


def test_m6_toy_task_real_learning_three_cycles(tmp_path):
    """toy 실학습 3사이클 — 가설이 실제 점수를 바꾸는 인과 사슬 (M6 완성 기준).

    실측(T7 데이터 재사용): idx0(deg1)=0.878103, idx1(deg2,α0)=0.431460,
    idx2(deg2,α0.1)=0.435921 → cycle2가 베스트, cycle3은 미개선. 컨센서스
    best_pipeline.val_mse는 cycle2의 값으로 고정된다.
    """
    env = _toy_env(tmp_path)
    code, stdout, stderr = _run_runner(
        ["--cycle", "3"], {**env, "COSCIENTIST_MOCKLLM_SCENARIO": "toy_task"})
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert stdout.count("outcome=ok") == 3
    best = db.get_consensus(env["COSCIENTIST_LOG_DB"])["best_pipeline"]
    assert best["val_mse"] == pytest.approx(0.431460, abs=1e-5)
    from ai_co_scientist.mcp_servers.wandb_tools import db as wandb_db
    assert len(wandb_db.query_runs(env["COSCIENTIST_WANDB_DB"])) == 3


def test_m6_overfit_triggers_dacon_submission(tmp_path):
    """과적합 의심 → guard 통과(첫 실험) → dacon 제출 → public/private 기록."""
    env = _toy_env(tmp_path)
    code, stdout, stderr = _run_runner(
        ["--cycle", "1"], {**env, "COSCIENTIST_MOCKLLM_SCENARIO": "toy_overfit"})
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "outcome=ok" in stdout
    diags = db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="diagnosis")
    analysis_diag = [d for d in diags if d["owner"] == "analysis"][0]
    assert analysis_diag["content"]["overfitting_suspected"] is True
    assert "public" in analysis_diag["content"]["diagnosis"]
    from ai_co_scientist.mcp_servers.dacon import db as dacon_db
    submissions = dacon_db.list_submissions(env["COSCIENTIST_DACON_DB"])
    assert len(submissions) == 1


def test_m3_executor_timeout_recovery(tmp_path):
    env = _isolated_env(tmp_path, COSCIENTIST_MOCKLLM_SCENARIO="coder_slow",
                         COSCIENTIST_RUN_TIMEOUT_S="1")
    code, stdout, stderr = _run_runner(["--cycle", "1"], env)
    assert code == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "outcome=ok" in stdout                  # 1차 타임아웃 → ×2 재시도 → 성공
    events = db.query_ledger(env["COSCIENTIST_LOG_DB"], record_type="infra_event")
    assert len(events) == 1 and events[0]["failure_category"] == "infra_timeout"
