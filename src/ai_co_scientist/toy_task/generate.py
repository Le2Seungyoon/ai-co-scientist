"""결정적 합성 회귀 데이터 생성 — y = sin(2x) + 0.5x² + ε (M6 toy task)."""
import csv
import sys
from pathlib import Path

import numpy as np


def generate(outdir: str, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, 100)
    y = np.sin(2 * x) + 0.5 * x**2 + rng.normal(0, 0.1, 100)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train.csv": (0, 60), "val.csv": (60, 80), "holdout.csv": (80, 100)}
    for name, (a, b) in splits.items():
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for xi, yi in zip(x[a:b], y[a:b]):
                writer.writerow([f"{xi:.6f}", f"{yi:.6f}"])


if __name__ == "__main__":
    generate(sys.argv[1])
