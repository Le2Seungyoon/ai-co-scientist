"""Lightning Studio GPU CLI — sub-agent가 원격 학습을 돌릴 때 쓰는 진입점.

사용 예:
  python scripts/lightning_studio.py credits
  python scripts/lightning_studio.py up --machine T4
  python scripts/lightning_studio.py push scripts/train_sem_depth.py train_sem_depth.py
  python scripts/lightning_studio.py run "nohup python train_sem_depth.py ... > run.log 2>&1 &"
  python scripts/lightning_studio.py run "tail -5 run.log"
  python scripts/lightning_studio.py pull out/model.pt runtime/ckpt/model.pt
  python scripts/lightning_studio.py down
"""
import argparse

from ai_co_scientist.backends import lightning
from ai_co_scientist.core.config import ensure_utf8_console


def main():
    ensure_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["credits", "up", "push", "run", "pull", "down"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--studio", default=lightning.DEFAULT_STUDIO)
    ap.add_argument("--machine", default=None, help="예: T4, A10G (up 전용)")
    a = ap.parse_args()

    if a.action == "credits":
        print(f"remaining credits: {lightning.get_credits()}")
        return

    st = lightning.studio(a.studio)
    if a.action == "up":
        print(lightning.ensure_running(st, a.machine))
    elif a.action == "push":
        lightning.upload(st, a.args[0], a.args[1])
        print(f"uploaded {a.args[0]} -> {a.args[1]}")
    elif a.action == "run":
        print(lightning.exec_cmd(st, a.args[0]))
    elif a.action == "pull":
        lightning.download(st, a.args[0], a.args[1])
        print(f"downloaded {a.args[0]} -> {a.args[1]}")
    elif a.action == "down":
        print(lightning.stop(st))


if __name__ == "__main__":
    main()
