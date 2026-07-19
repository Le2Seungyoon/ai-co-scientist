import csv
import subprocess
import sys


def test_generate_is_deterministic_and_split(tmp_path):
    out1, out2 = tmp_path / "a", tmp_path / "b"
    for out in (out1, out2):
        subprocess.run([sys.executable, "-m", "ai_co_scientist.toy_task.generate", str(out)],
                       check=True)
    for name, rows in (("train.csv", 60), ("val.csv", 20), ("holdout.csv", 20)):
        with open(out1 / name, encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == rows
    assert (out1 / "train.csv").read_text() == (out2 / "train.csv").read_text()  # 결정적
