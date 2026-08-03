"""실험 기록소 CLI — sub-agent가 실험 전/후에 호출하는 진입점.

  선보고:  python scripts/exp.py new --title "..." --x-domain sim --x-desc "..." \
             --y-source sim_depth_gt --y-desc "..." --model "..." --method "..." \
             --purpose "..." --metric-name sim_val_rmse --metric-x sim --metric-y sim_depth_gt
  결과:    python scripts/exp.py result EXP-001 --val '{"sim_val_rmse": 2.57}'
  리더보드: python scripts/exp.py lb EXP-001 --public 6.727 --private 6.772
  판정:    python scripts/exp.py verdict EXP-001 "챔피언 기준선"
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

    res = sub.add_parser("result")
    res.add_argument("report_id")
    res.add_argument("--val", required=True)
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
            metric_x_domain=a.metric_x, metric_y_source=a.metric_y)
        print(rec["report_id"])
        if rec["metric"]["warning"]:
            print("WARNING:", rec["metric"]["warning"])
    elif a.cmd == "result":
        print(registry.record_result(a.report_id, json.loads(a.val))["report_id"], "결과 기록됨")
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
