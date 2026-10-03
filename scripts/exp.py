"""실험 기록소 CLI — sub-agent가 실험 전/후에 호출하는 진입점.

  선보고:  python scripts/exp.py new --title "..." --x-domain sim --x-desc "..." \
             --y-source sim_depth_gt --y-desc "..." --model "..." --method "..." \
             --purpose "..." --metric-name sim_val_rmse --metric-x sim --metric-y sim_depth_gt
  결과:    python scripts/exp.py result EXP-001 --val '{"sim_val_rmse": 2.57}'
           (기본은 기존 val에 **병합** — 두 번째 호출이 첫 호출을 지우지 않는다.
            값이 다른 키를 덮어쓰려면 --replace-key KEY, 통째 교체는 --replace)
  리더보드: python scripts/exp.py lb EXP-001 --public 7.35 --private 7.34
  판정:    python scripts/exp.py verdict EXP-001 "기준선"
  조회:    python scripts/exp.py list | show EXP-001 | render
"""
import argparse
import json

from ai_co_scientist import registry
from ai_co_scientist.config import ensure_utf8_console, load_config, project_root


def main():
    ensure_utf8_console()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    new = sub.add_parser("new", help="선보고 등록 → report_id 발급")
    new.add_argument("--title", required=True)
    new.add_argument("--x-domain", required=True, choices=registry.X_DOMAINS)
    new.add_argument("--x-desc", required=True)
    new.add_argument("--y-source", required=True, choices=registry.Y_SOURCES)
    new.add_argument("--y-desc", required=True)
    new.add_argument("--model", required=True)
    new.add_argument("--method", required=True)
    new.add_argument("--purpose", required=True)
    new.add_argument("--metric-name", required=True)
    new.add_argument("--metric-x", required=True, choices=registry.X_DOMAINS)
    new.add_argument("--metric-y", required=True, choices=registry.Y_SOURCES)
    new.add_argument("--hypothesis", required=True,
                     help="이 실험이 답하는 가설 id (docs/experiment/H<id>-*.md)")
    new.add_argument("--source-branch", default="",
                     help="실험을 낸 코드의 브랜치 — 워커가 자기 워크트리에서 읽은 값")
    new.add_argument("--source-commit", default="",
                     help="실험을 낸 코드의 커밋 — 기록자의 HEAD가 아니라 실행한 트리의 HEAD")

    res = sub.add_parser("result", help="실행 결과 매니페스트 기록 (기본: 기존 val에 병합)")
    res.add_argument("report_id")
    res.add_argument("--val", required=True)
    res.add_argument("--replace-key", action="append", default=[], metavar="KEY",
                     help="값이 다른 이 키만 덮어쓰기 허용 (반복 가능)")
    res.add_argument("--replace", action="store_true",
                     help="기존 매니페스트를 버리고 통째 교체 - 버려진 키를 경고로 출력한다")
    lb = sub.add_parser("lb")
    lb.add_argument("report_id")
    lb.add_argument("--public", type=float, required=True)
    lb.add_argument("--private", type=float, required=True)
    vd = sub.add_parser("verdict")
    vd.add_argument("report_id")
    vd.add_argument("text")
    sub.add_parser("list")
    sh = sub.add_parser("show")
    sh.add_argument("report_id")
    sub.add_parser("render")

    a = ap.parse_args()
    if a.cmd == "new":
        rec = registry.new_report(
            title=a.title, x_domain=a.x_domain, x_desc=a.x_desc,
            y_source=a.y_source, y_desc=a.y_desc, model=a.model, method=a.method,
            purpose=a.purpose, metric_name=a.metric_name,
            metric_x_domain=a.metric_x, metric_y_source=a.metric_y,
            hypothesis=a.hypothesis,
            source_branch=a.source_branch, source_commit=a.source_commit)
        print(rec["report_id"])
        if rec["metric"]["warning"]:
            print("WARNING:", rec["metric"]["warning"])
    elif a.cmd == "result":
        try:
            up = registry.record_result(
                a.report_id, json.loads(a.val),
                replace=a.replace, replace_keys=a.replace_key)
        except ValueError as e:
            raise SystemExit(f"결과 기록 거부: {e}") from e
        print(f"{up.report_id} 결과 기록됨 "
              f"(추가 {len(up.added)} / 유지 {len(up.unchanged)} / 덮어씀 {len(up.overwritten)})")
        for k, old in up.overwritten.items():
            print(f"WARNING: 덮어씀 {k}: {json.dumps(old, ensure_ascii=False)} -> "
                  f"{json.dumps(up.record['val'][k], ensure_ascii=False)}")
        for k, old in up.dropped.items():
            print(f"WARNING: 버려짐 {k}: {json.dumps(old, ensure_ascii=False)}")
    elif a.cmd == "lb":
        registry.record_lb(a.report_id, a.public, a.private)
        print(f"{a.report_id} LB 기록됨: {a.public} / {a.private}")
    elif a.cmd == "verdict":
        registry.set_verdict(a.report_id, a.text)
        print(f"{a.report_id} 판정 기록됨")
    elif a.cmd == "list":
        for r in registry.load_all():
            lb = f"{r['lb']['public']}/{r['lb']['private']}" if r["lb"] else "-"
            print(f"{r['report_id']}  LB={lb:<16} {r['title']}")
    elif a.cmd == "show":
        print(json.dumps(registry.get(a.report_id), ensure_ascii=False, indent=2))
    elif a.cmd == "render":
        out = project_root() / load_config()["paths"]["registry_doc"]
        out.write_text(registry.render_markdown(), encoding="utf-8")
        print(f"rendered → {out}")


if __name__ == "__main__":
    main()
