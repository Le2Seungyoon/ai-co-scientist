"""Analysis 에이전트 (M3: 룰 기반 판정 + 컨센서스 갱신) — 메시지 모드."""
import sys

from ai_co_scientist.a2a.base import serve
from ai_co_scientist.agents.analysis.logic import build_verdict
from ai_co_scientist.core.config import load_config
from ai_co_scientist.core.mcp_client import mcp_session, tool_result_data
from ai_co_scientist.core.rules import load_rules
from ai_co_scientist.core.schema import RunResult, parse_payload, to_payload
from hooks.submission_guard import submission_guard

SHARED_LOG_SERVER = "ai_co_scientist.mcp_servers.shared_log.server"
DACON_SERVER = "ai_co_scientist.mcp_servers.dacon.server"


def _make_verdict(rr, best, gap, consensus_known: bool):
    """조회 실패(unknown)면 개선 판단을 보류하고 컨센서스 갱신 대상에서 제외한다."""
    verdict = build_verdict(rr, best, gap)
    if not consensus_known:
        return verdict.model_copy(update={
            "improved": False,
            "diagnosis": f"판단 보류: 컨센서스 조회 실패 (val_mse {rr.metrics['val_mse']:.4f})",
            "case_findings": [
                f"컨센서스 조회 실패 — 비교 기준 미상 (val_mse {rr.metrics['val_mse']:.4f})"
            ],
        })
    return verdict


async def _maybe_submit(rr: RunResult, best: dict | None, verdict):
    """오버피팅 의심 케이스의 제출 경로 — dacon은 shared_log와 다른 MCP 서버라 별도 세션.

    guard 통과 시에만 제출하고, 제출 성공/실패와 무관하게 verdict(판정) 자체는 그대로 반환한다
    (제출≠판정 — 이월 ①의 분리 원칙). 제출 실패는 격리해 diagnosis에 관측만 남긴다.
    """
    predictions = rr.artifacts.get("predictions_path", "")
    if not predictions:
        return verdict
    ok, reason = submission_guard(rr.metrics["val_mse"], best)
    if not ok:
        return verdict.model_copy(update={"diagnosis": verdict.diagnosis + f" — {reason}"})
    try:
        async with mcp_session(DACON_SERVER) as dacon_session:
            res = await dacon_session.call_tool(
                "submit", {"predictions_path": predictions, "cycle_id": rr.cycle_id})
            if res.isError:
                raise RuntimeError(res.content[0].text if res.content else "submit 실패")
            data = tool_result_data(res)
            if isinstance(data, dict):
                data = data.get("result", data)
            submission_id = data.get("submission_id")
            public = data.get("public_score")
            try:
                score_res = await dacon_session.call_tool(
                    "get_score", {"submission_id": submission_id})
                if score_res.isError:
                    raise RuntimeError(
                        score_res.content[0].text if score_res.content else "get_score 실패")
                score_data = tool_result_data(score_res)
                if isinstance(score_data, dict):
                    score_data = score_data.get("result", score_data)
                private = score_data.get("private_score")
            except Exception as e:  # noqa: BLE001 — 제출은 이미 DB에 기록됨: 조회 실패와 구분
                print(f"[analysis] dacon 점수 조회 실패(무시): {e}", file=sys.stderr)
                return verdict.model_copy(update={
                    "diagnosis": verdict.diagnosis + " — 제출됨(점수 조회 실패)",
                })
            if public is None or private is None:
                # real 제출은 점수를 동기 반환하지 않음 — 접수 상태만 기록 (리더보드에서 직접 확인)
                status = score_data.get("status", "?")
                detail = score_data.get("detail", "")
                return verdict.model_copy(update={
                    "diagnosis": verdict.diagnosis + f" — 제출 접수({status}): {detail}",
                })
            return verdict.model_copy(update={
                "diagnosis": verdict.diagnosis +
                f" — 제출 확인: public {public:.4f}/private {private:.4f}",
            })
    except Exception as e:  # noqa: BLE001 — dacon 격리: 제출 실패가 판정을 막지 않음 (관측만)
        print(f"[analysis] dacon 제출 실패(무시): {e}", file=sys.stderr)
        return verdict.model_copy(update={"diagnosis": verdict.diagnosis + " — 제출 실패(무시)"})


async def handle(payload: dict) -> dict:
    rr = parse_payload(payload)
    if not isinstance(rr, RunResult):
        raise ValueError(f"RunResult 아님: {payload.get('type')}")
    if "val_mse" not in rr.metrics:
        # 도메인 검증은 MCP 격리 try 밖에서 — 격리 except가 삼키지 않게
        raise ValueError(f"val_mse 없음: {sorted(rr.metrics)}")
    gap = load_config()["analysis"]["overfit_gap"]
    best = None
    consensus_known = False
    verdict = None
    try:
        async with mcp_session(SHARED_LOG_SERVER) as session:
            res = await session.call_tool("get_consensus", {})
            # tool 수준 실패는 예외가 아니라 isError 응답 — 명시 확인 (T5 교훈)
            if not res.isError:
                data = tool_result_data(res)
                if isinstance(data, dict):
                    data = data.get("result", data)
                best = (data or {}).get("best_pipeline")
                consensus_known = True
            # 이월 ⑤: 이번 사이클 자신의 컨센서스 기록은 비교/갱신 대상에서 제외
            # (revise 재실행 멱등 — consensus_known 자체는 유지, 판단 보류가 아님)
            if best is not None and best.get("cycle_id") == rr.cycle_id:
                best = None
            verdict = _make_verdict(rr, best, gap, consensus_known)
            if consensus_known and verdict.overfitting_suspected:
                verdict = await _maybe_submit(rr, best, verdict)
            if consensus_known and verdict.improved:
                res = await session.call_tool("update_consensus", {
                    "key": "best_pipeline",
                    "value": {"val_mse": rr.metrics["val_mse"], "cycle_id": rr.cycle_id},
                })
                # tool 수준 실패도 isError 응답 — 격리 유지, 관측만 (스펙 §9 M3 교훈)
                if res.isError:
                    print(f"[analysis] update_consensus tool 실패(무시): "
                          f"{res.content[0].text if res.content else ''}", file=sys.stderr)
            res = await session.call_tool("log_append", {
                "cycle_id": rr.cycle_id, "record_type": "diagnosis",
                "owner": "analysis", "content": verdict.model_dump(mode="json"),
            })
            # tool 수준 실패도 isError 응답 — 격리 유지, 관측만 (스펙 §9 M3 교훈)
            if res.isError:
                print(f"[analysis] log_append tool 실패(무시): "
                      f"{res.content[0].text if res.content else ''}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — 조회/기록 실패 격리: 판정 자체는 반환 (M2 이월 ①)
        print(f"[analysis] 공유로그 접근 실패(판정은 진행): {e}", file=sys.stderr)
        if verdict is None:
            verdict = _make_verdict(rr, best, gap, consensus_known)
    return to_payload(verdict)


def main() -> None:
    load_rules("analysis")
    serve("analysis", "결과 판정 담당 (M3 룰 기반)",
          load_config()["agents"]["analysis"]["port"], handle)


if __name__ == "__main__":
    main()
