"""Check regenerated reports against the committed snapshot: python -m scripts.compare_reports.

CSV and JSON values are compared with a small numeric tolerance, because platforms
can differ in the last floating-point digits. Figures must exist with the same
names; their pixels vary with platform fonts, so their contents are not compared.
"""

import argparse
import io
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd

from src.config import ROOT

REPORT_DIRS = ["reports/metrics", "reports/figures"]
RTOL, ATOL = 1e-6, 1e-9


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def committed_files(revision: str) -> set[str]:
    return set(git("ls-tree", "-r", "--name-only", revision, "--", *REPORT_DIRS).decode().split())


def current_files() -> set[str]:
    return {path.relative_to(ROOT).as_posix() for directory in REPORT_DIRS
            for path in (ROOT / directory).rglob("*") if path.is_file()}


def json_differences(old, new, where: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        if old.keys() != new.keys():
            return [f"{where or '/'}: keys differ"]
        return [d for key in old for d in json_differences(old[key], new[key], f"{where}/{key}")]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{where or '/'}: lengths differ"]
        return [d for i, (a, b) in enumerate(zip(old, new, strict=True))
                for d in json_differences(a, b, f"{where}[{i}]")]
    if isinstance(old, bool) != isinstance(new, bool):  # True == 1 in Python
        return [f"{where or '/'}: {old!r} != {new!r}"]
    numbers = all(isinstance(v, (int, float)) for v in (old, new))
    if old == new or (numbers and math.isclose(old, new, rel_tol=RTOL, abs_tol=ATOL)):
        return []
    return [f"{where or '/'}: {old!r} != {new!r}"]


def csv_differences(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    if list(old.columns) != list(new.columns) or old.shape != new.shape:
        return ["columns or shape differ"]
    problems = []
    for column in old:
        a, b = old[column], new[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=RTOL, atol=ATOL, equal_nan=True):
                problems.append(f"{column}: numeric values differ")
        elif not a.astype(str).equals(b.astype(str)):
            problems.append(f"{column}: values differ")
    return problems


def compare(revision: str = "HEAD") -> list[str]:
    old_files, new_files = committed_files(revision), current_files()
    problems = [f"{name}: missing after training" for name in sorted(old_files - new_files)]
    problems += [f"{name}: not in the committed snapshot" for name in sorted(new_files - old_files)]
    for name in sorted(old_files & new_files):
        old, new = git("show", f"{revision}:{name}"), (ROOT / name).read_bytes()
        if name.endswith(".json"):
            problems += [f"{name}{d}" for d in json_differences(json.loads(old), json.loads(new))]
        elif name.endswith(".csv"):
            problems += [f"{name}: {d}" for d in csv_differences(pd.read_csv(io.BytesIO(old)),
                                                                  pd.read_csv(io.BytesIO(new)))]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD", help="Git revision holding the committed reports")
    problems = compare(parser.parse_args().revision)
    for problem in problems:
        print(problem)
    if problems:
        print("Committed reports differ from a fresh run. Retrain locally and commit the regenerated reports.")
        sys.exit(1)
    print("Regenerated reports match the committed snapshot.")


if __name__ == "__main__":
    main()
