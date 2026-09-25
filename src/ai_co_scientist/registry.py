"""실험 기록소 — 모든 실험의 단일 진실 소스.

`.agents/rules/workflow.md` → Experiment Pre-Report의 기계적 집행 지점이다.
선보고 5항목(X / y / 모델·하이퍼 / 방법론 / 목적)과 **판정지표의 (X, y)** 를 받지 않으면
실험을 등록할 수 없다. 지표 도메인이 타깃(real→real)과 다르면 경고를 박아둔다 —
sim SEM→sim depth 지표를 real validation으로 착각해 여러 실험을 헛돌린 실패의 재발 방지선.

저장: JSONL 1줄 = 실험 1건. 갱신은 load-modify-write (건수가 수백 규모라 단순함이 이득).
"""
import json
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path

from ai_co_scientist.config import load_config, project_root
from ai_co_scientist.locks import LOCK_STALE, ResourceBusy, file_lock

LOCK_TIMEOUT = 30.0  # 초. 학습이 아니라 JSONL 갱신이므로 이보다 오래 걸릴 일이 없다
# LOCK_STALE은 locks.py의 정의를 그대로 쓴다 — 같은 값을 두 곳에서 정의하면 한쪽만 바뀌는 드리프트가 생긴다

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


@contextmanager
def locked(path=None):
    """기록소 갱신 직렬화 — **읽기와 쓰기를 함께 감싸야** 한다.

    sub-agent를 병렬로 돌리면 두 에이전트가 같은 `len(records)`를 보고 **같은 report_id**를
    발급하고, 나중 write가 앞선 선보고를 통째로 덮어쓴다(실측 확인). `_write_all`이 파일 전체를
    다시 쓰는 load-modify-write이므로 락 없이는 append조차 안전하지 않다.

    획득 루프는 `locks.file_lock`에 있다. **경로는 의도적으로 트리별이다** — 기록소는 워크트리가
    복제하지 않는 자원이고, 기계 단위 자원은 `locks.resource_lock`이 맡는다.

    `try`는 **획득 한 줄만** 감싼다. `yield`까지 감싸면 본문 안에서 난 `ResourceBusy`(중첩된
    `resource_lock` 등)가 이 락의 타임아웃으로 잘못 보고된다 — 엉뚱한 파일 이름을 댄 채로.
    """
    with ExitStack() as stack:
        try:
            stack.enter_context(
                file_lock(_path(path).with_suffix(".lock"), timeout=LOCK_TIMEOUT, stale=LOCK_STALE))
        except ResourceBusy as e:
            # 역호환성: 기존 호출부는 TimeoutError를 기대한다
            raise TimeoutError(
                f"기록소 락 대기 초과({LOCK_TIMEOUT}초): {_path(path).with_suffix('.lock')}") from e
        yield


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
    if value is None or not str(value).strip():
        raise ValueError(f"선보고 필수 항목 누락: {name}")
    return str(value).strip()


def new_report(*, title, x_domain, x_desc, y_source, y_desc, model, method, purpose,
               metric_name, metric_x_domain, metric_y_source, hypothesis,
               source_branch="", source_commit="", path=None) -> dict:
    """선보고 등록. 5항목 + 지표 도메인이 모두 있어야 report_id를 발급한다.
    가설 id(`hypothesis`)는 필수다 — 조인 없는 레코드는 조인에 보이지 않는다."""
    if x_domain not in X_DOMAINS:
        raise ValueError(f"x_domain은 {X_DOMAINS} 중 하나여야 한다: {x_domain}")
    if y_source not in Y_SOURCES:
        raise ValueError(f"y_source는 {Y_SOURCES} 중 하나여야 한다: {y_source}")
    if metric_x_domain not in X_DOMAINS:
        raise ValueError(f"metric_x_domain은 {X_DOMAINS} 중 하나여야 한다: {metric_x_domain}")
    if metric_y_source not in Y_SOURCES:
        raise ValueError(f"metric_y_source는 {Y_SOURCES} 중 하나여야 한다: {metric_y_source}")

    hypothesis = _require("hypothesis", hypothesis)

    matches = metric_matches_target(metric_x_domain, metric_y_source)
    warning = "" if matches else (
        f"이 지표는 ({metric_x_domain}, {metric_y_source}) 도메인이라 "
        "타깃(real, real GT)을 검증하지 못한다 — 최종 판정은 리더보드로만 한다.")

    # report_id가 `len(records)`에서 나오므로 **읽기와 쓰기를 한 락 안에서** 해야 한다.
    # 나누면 두 에이전트가 같은 ID를 발급받고 뒤 write가 앞 선보고를 덮어쓴다.
    with locked(path):
        records = load_all(path)
        record = {
            "report_id": f"EXP-{len(records) + 1:03d}",
            "created": datetime.now().isoformat(timespec="seconds"),
            # 가설 ↔ 실행 조인 키. 다대일이다 — 한 가설이 여러 실행을 가질 수 있으나(3-arm
            # sweep도 한 실행), 한 실행은 가설 하나만 가리킨다. 손으로 유지하는 표를 대신하므로
            # 선택이 아니라 필수다.
            "hypothesis": hypothesis,
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
            # 실험을 낸 코드의 출처. 병렬 워크트리에서는 기록하는 쪽(코디네이터)과 실행한
            # 쪽(워커 브랜치)이 다르므로, 기록자의 HEAD를 쓰면 조용히 틀린다. 워커가 자기
            # 트리에서 읽은 값을 그대로 넣는다. 기존 레코드에는 이 키가 없다 — optional이다.
            "source": ({"branch": source_branch, "commit": source_commit}
                       if (source_branch or source_commit) else None),
            "verdict": "",
        }
        _write_all(records + [record], path)
    return record


def _update(report_id: str, mutate, path=None) -> dict:
    with locked(path):  # read-modify-write 전체가 원자적이어야 한다
        records = load_all(path)
        for rec in records:
            if rec["report_id"] == report_id:
                mutate(rec)
                _write_all(records, path)
                return rec
    raise KeyError(f"기록소에 없는 report_id: {report_id}")


class ResultUpdate:
    """`record_result`의 결과 — 갱신된 기록 + **무엇이 바뀌었는지**.

    반환값이 기록만이면 파괴적 동작(덮어쓰기·통째 교체)이 호출자에게 보이지 않는다.
    소실될 수 있는 값은 여기에 담아 CLI가 반드시 출력하게 한다.
    """

    __slots__ = ("record", "added", "unchanged", "overwritten", "dropped")

    def __init__(self, record: dict, added: list[str], unchanged: list[str],
                 overwritten: dict, dropped: dict):
        self.record = record
        self.added = added
        self.unchanged = unchanged
        self.overwritten = overwritten  # {키: 이전 값} — --replace-key로 명시 허용된 덮어쓰기
        self.dropped = dropped  # {키: 이전 값} — replace=True로 통째 교체하며 버린 키

    @property
    def report_id(self) -> str:
        return self.record["report_id"]

    def __getitem__(self, key):
        # 기존 호출부(`record_result(...)["report_id"]`) 호환.
        return self.record[key]


def merge_val(existing, new: dict, *, replace_keys=()) -> tuple[dict, list, list, dict]:
    """실행 중 누적되는 `val` 매니페스트를 합친다 — **얕은(top-level) 병합**.

    깊은 병합은 하지 않는다. 같은 결함을 한 단계 아래로 옮길 뿐이고, 중첩 dict가 통째로
    바뀌었는지 일부만 바뀌었는지 호출자가 구분할 수 없게 된다.

    충돌 규칙: 같은 키가 이미 있고
      - 값이 **같으면** 통과(재실행 멱등성).
      - 값이 **다르면** 예외 — `replace_keys`에 그 키를 명시해야만 덮어쓴다.
    조용한 덮어쓰기는 통째 교체와 같은 결함이라 기본 경로에서 배제한다.

    반환: (합쳐진 dict, 추가된 키, 값이 같아 유지된 키, {덮어쓴 키: 이전 값})
    """
    if not isinstance(new, dict):
        raise ValueError(f"val은 dict여야 한다: {type(new).__name__}")
    if existing is None:
        existing = {}
    if not isinstance(existing, dict):
        raise ValueError(
            "기존 val이 dict가 아니라 병합할 수 없다 "
            f"({type(existing).__name__}) — 의도적 교체라면 replace=True를 쓸 것")

    allowed = set(replace_keys)
    conflicts = [k for k, v in new.items()
                 if k in existing and existing[k] != v and k not in allowed]
    if conflicts:
        raise ValueError(
            f"val 키 충돌: {sorted(conflicts)} — 기존 값과 다르다. "
            "덮어쓰려면 해당 키를 replace_keys(CLI: --replace-key)로 명시하거나, "
            "매니페스트 전체를 버릴 의도라면 replace=True(CLI: --replace)를 쓸 것")

    merged = dict(existing)
    added, unchanged, overwritten = [], [], {}
    for k, v in new.items():
        if k not in existing:
            added.append(k)
        elif existing[k] == v:
            unchanged.append(k)
            continue
        else:
            overwritten[k] = existing[k]
        merged[k] = v
    return merged, added, unchanged, overwritten


def record_result(report_id: str, val: dict, *, replace: bool = False, replace_keys=(),
                  path=None) -> ResultUpdate:
    """실행 결과 매니페스트를 기록한다 — **기본은 병합, 교체는 명시적으로만**.

    `val`은 한 실험의 생애 동안 누적되는 유일한 필드다(예: 학습 직후 `wall_clock`/`run`,
    제출 직후 `verify_only`). 통째 교체가 기본이었을 때 두 번째 `result` 호출이 첫 호출의
    기록을 조용히 지웠다 — EXP-020에서 제출 zip 검증 근거가 사라질 뻔했다.

    `replace=True`는 기존 매니페스트를 버린다. 버려진 키는 `ResultUpdate.dropped`로 돌려주니
    호출자는 반드시 그것을 드러내야 한다.

    락: 읽기·병합·쓰기가 모두 `_update` 안의 `locked()` 하나에 들어간다. 병합 대상을 락 밖에서
    읽으면 8건 중 2건만 살아남았던 그 경쟁 조건이 그대로 재현된다.
    """
    box: dict = {}

    def mutate(rec: dict) -> None:
        if replace:
            old = rec.get("val") or {}
            if isinstance(old, dict):
                box["dropped"] = {k: v for k, v in old.items() if k not in val}
            else:
                box["dropped"] = {"(이전 val)": old}
            box["added"], box["unchanged"], box["overwritten"] = list(val), [], {}
            rec["val"] = dict(val)
            return
        merged, added, unchanged, overwritten = merge_val(
            rec.get("val"), val, replace_keys=replace_keys)
        box.update(added=added, unchanged=unchanged, overwritten=overwritten, dropped={})
        rec["val"] = merged

    rec = _update(report_id, mutate, path)
    return ResultUpdate(rec, box["added"], box["unchanged"], box["overwritten"], box["dropped"])


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
           "| report_id | 가설 | 제목 | X | y | 지표(타깃일치) | val | LB pub/priv | 판정 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        val = json.dumps(r["val"], ensure_ascii=False) if r["val"] else "-"
        lb = f"{r['lb']['public']} / {r['lb']['private']}" if r["lb"] else "-"
        mark = "✅" if r["metric"]["matches_target"] else "⚠️sim"
        # 가설 20건(2026-09-21 이전)에는 이 키 자체가 없다 — 빈 칸도 "None"도 찍지 않는다.
        hypothesis = r.get("hypothesis") or "-"
        out.append(
            f"| {r['report_id']} | {hypothesis} | {r['title']} | {r['x']['domain']} | {r['y']['source']} "
            f"| {r['metric']['name']} {mark} | {val} | {lb} | {r['verdict'] or '-'} |")

    out += ["", "## 상세", ""]
    for r in records:
        out += [f"### {r['report_id']} — {r['title']}", f"- **생성**: {r['created']}",
                f"- **가설**: {r.get('hypothesis') or '-'}",
                f"- **X**: `{r['x']['domain']}` — {r['x']['desc']}",
                f"- **y**: `{r['y']['source']}` — {r['y']['desc']}",
                f"- **모델+하이퍼**: {r['model']}",
                f"- **방법론**: {r['method']}",
                f"- **목적**: {r['purpose']}",
                f"- **판정지표**: {r['metric']['name']} "
                f"(X={r['metric']['x_domain']}, y={r['metric']['y_source']})"]
        if r["metric"]["warning"]:
            out.append(f"  - ⚠️ {r['metric']['warning']}")
        if r.get("source"):
            out.append(f"- **출처**: `{r['source']['branch']}` @ `{r['source']['commit']}`")
        val_line = json.dumps(r["val"], ensure_ascii=False) if r["val"] else "(미실행)"
        lb_line = "(미제출)"
        if r["lb"]:
            lb_line = f"public {r['lb']['public']} / private {r['lb']['private']}"
        out += [f"- **val**: {val_line}", f"- **LB**: {lb_line}",
                f"- **판정**: {r['verdict'] or '(미정)'}", ""]
    return "\n".join(out) + "\n"
