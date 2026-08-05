"""실험 기록소 — 모든 실험의 단일 진실 소스.

`.claude/rules/workflow.md` → Experiment Pre-Report의 기계적 집행 지점이다.
선보고 5항목(X / y / 모델·하이퍼 / 방법론 / 목적)과 **판정지표의 (X, y)** 를 받지 않으면
실험을 등록할 수 없다. 지표 도메인이 타깃(real→real)과 다르면 경고를 박아둔다 —
sim SEM→sim depth 지표를 real validation으로 착각해 여러 실험을 헛돌린 실패의 재발 방지선.

저장: JSONL 1줄 = 실험 1건. 갱신은 load-modify-write (건수가 수백 규모라 단순함이 이득).
"""
import json
from datetime import datetime
from pathlib import Path

from ai_co_scientist.config import load_config, project_root

X_DOMAINS = ("sim", "real")
Y_SOURCES = ("sim_depth_gt", "real_average_depth", "real_group_label", "real_depth_gt",
             "pseudo_label")
# 실측 GT에서 온 y (pseudo_label은 모델 산출물이라 제외).
# real_group_label = train/SEM/Depth_{110,120,130,140}/ 폴더명 — 주최측이 부여한 real 라벨이라
#   average_depth와 동급의 실측 GT다. 배경 레벨 L을 결정하므로 depth map의 지배 성분이기도 하다.
# real_depth_gt = 리더보드가 채점에 쓰는 **숨은 real depth map**. 우리가 볼 수 없지만 이것이
#   진짜 타깃이다 — average_depth는 원본 전체 영상 기준이라 타깃이 아니다(docs/data-facts.md §4).
_REAL_Y = ("real_average_depth", "real_group_label", "real_depth_gt")

RESET_NOTICE = (
    "> **2026-07-29 리셋.** 이 날짜 이전의 ad-hoc 실험·결론은 모두 폐기(void) — "
    "인용하지 않는다. 이 파일만이 신뢰 가능한 실험 기록이다.\n"
)


def _default_path() -> Path:
    return project_root() / load_config()["paths"]["registry"]


def _path(path=None) -> Path:
    return Path(path) if path is not None else _default_path()


def metric_matches_target(metric_x_domain: str, metric_y_source: str) -> bool:
    """지표가 타깃 도메인(real 입력 + 실측 GT)을 재고 있는가."""
    return metric_x_domain == "real" and metric_y_source in _REAL_Y


def load_all(path=None) -> list[dict]:
    p = _path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_all(records: list[dict], path=None) -> None:
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def get(report_id: str, path=None) -> dict:
    for rec in load_all(path):
        if rec["report_id"] == report_id:
            return rec
    raise KeyError(f"기록소에 없는 report_id: {report_id}")


def _require(name: str, value: str) -> str:
    if not str(value).strip():
        raise ValueError(f"선보고 필수 항목 누락: {name}")
    return str(value).strip()


def new_report(*, title, x_domain, x_desc, y_source, y_desc, model, method, purpose,
               metric_name, metric_x_domain, metric_y_source, path=None) -> dict:
    """선보고 등록. 5항목 + 지표 도메인이 모두 있어야 report_id를 발급한다."""
    if x_domain not in X_DOMAINS:
        raise ValueError(f"x_domain은 {X_DOMAINS} 중 하나여야 한다: {x_domain}")
    if y_source not in Y_SOURCES:
        raise ValueError(f"y_source는 {Y_SOURCES} 중 하나여야 한다: {y_source}")
    if metric_x_domain not in X_DOMAINS:
        raise ValueError(f"metric_x_domain은 {X_DOMAINS} 중 하나여야 한다: {metric_x_domain}")
    if metric_y_source not in Y_SOURCES:
        raise ValueError(f"metric_y_source는 {Y_SOURCES} 중 하나여야 한다: {metric_y_source}")

    matches = metric_matches_target(metric_x_domain, metric_y_source)
    warning = "" if matches else (
        f"이 지표는 ({metric_x_domain}, {metric_y_source}) 도메인이라 "
        "타깃(real, real GT)을 검증하지 못한다 — 최종 판정은 리더보드로만 한다.")

    records = load_all(path)
    record = {
        "report_id": f"EXP-{len(records) + 1:03d}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "title": _require("title", title),
        "x": {"domain": x_domain, "desc": _require("x_desc", x_desc)},
        "y": {"source": y_source, "desc": _require("y_desc", y_desc)},
        "model": _require("model", model),
        "method": _require("method", method),
        "purpose": _require("purpose", purpose),
        "metric": {"name": _require("metric_name", metric_name),
                   "x_domain": metric_x_domain, "y_source": metric_y_source,
                   "matches_target": matches, "warning": warning},
        "val": None,
        "lb": None,
        "verdict": "",
    }
    _write_all(records + [record], path)
    return record


def _update(report_id: str, mutate, path=None) -> dict:
    records = load_all(path)
    for rec in records:
        if rec["report_id"] == report_id:
            mutate(rec)
            _write_all(records, path)
            return rec
    raise KeyError(f"기록소에 없는 report_id: {report_id}")


def record_result(report_id: str, val: dict, path=None) -> dict:
    return _update(report_id, lambda r: r.__setitem__("val", val), path)


def record_lb(report_id: str, public: float, private: float, path=None) -> dict:
    return _update(
        report_id,
        lambda r: r.__setitem__("lb", {"public": float(public), "private": float(private)}),
        path)


def set_verdict(report_id: str, verdict: str, path=None) -> dict:
    return _update(report_id, lambda r: r.__setitem__("verdict", verdict), path)


def render_markdown(path=None) -> str:
    """기록소 → docs용 markdown. 요약 테이블 + 상세."""
    records = load_all(path)
    out = ["# 실험 기록소 (Experiment Registry)", "",
           "> 이 파일은 `scripts/exp.py render`가 생성한다 — 직접 수정하지 말 것.",
           RESET_NOTICE,
           "## 요약", "",
           "| report_id | 제목 | X | y | 지표(타깃일치) | val | LB pub/priv | 판정 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in records:
        val = json.dumps(r["val"], ensure_ascii=False) if r["val"] else "-"
        lb = f"{r['lb']['public']} / {r['lb']['private']}" if r["lb"] else "-"
        mark = "✅" if r["metric"]["matches_target"] else "⚠️sim"
        out.append(
            f"| {r['report_id']} | {r['title']} | {r['x']['domain']} | {r['y']['source']} "
            f"| {r['metric']['name']} {mark} | {val} | {lb} | {r['verdict'] or '-'} |")

    out += ["", "## 상세", ""]
    for r in records:
        out += [f"### {r['report_id']} — {r['title']}", f"- **생성**: {r['created']}",
                f"- **X**: `{r['x']['domain']}` — {r['x']['desc']}",
                f"- **y**: `{r['y']['source']}` — {r['y']['desc']}",
                f"- **모델+하이퍼**: {r['model']}",
                f"- **방법론**: {r['method']}",
                f"- **목적**: {r['purpose']}",
                f"- **판정지표**: {r['metric']['name']} "
                f"(X={r['metric']['x_domain']}, y={r['metric']['y_source']})"]
        if r["metric"]["warning"]:
            out.append(f"  - ⚠️ {r['metric']['warning']}")
        val_line = json.dumps(r["val"], ensure_ascii=False) if r["val"] else "(미실행)"
        lb_line = "(미제출)"
        if r["lb"]:
            lb_line = f"public {r['lb']['public']} / private {r['lb']['private']}"
        out += [f"- **val**: {val_line}", f"- **LB**: {lb_line}",
                f"- **판정**: {r['verdict'] or '(미정)'}", ""]
    return "\n".join(out) + "\n"
