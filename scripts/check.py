"""Full local quality gate with isolated temporary files and combined coverage.

Install requirements-dev.txt and requirements-explain.txt first. This command
trains a fresh run and regenerates tracked reports and executed notebooks.
"""

import os
import subprocess
import sys
from uuid import uuid4

from attrition.config import ROOT


def main() -> None:
    directory = ROOT / ".test-tmp" / str(uuid4())
    directory.mkdir(parents=True)
    env = {**os.environ, "TEMP": str(directory), "TMP": str(directory), "TMPDIR": str(directory),
           "COVERAGE_FILE": str(directory / ".coverage"), "MPLBACKEND": "Agg", "PYTHONUTF8": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    commands = [
        ["ruff", "check", "."],
        ["mypy"],
        ["pytest", "-q", "--cov", "--cov-report=", "--cov-fail-under=0",
         "--basetemp", str(directory / "pytest"), "-o", f"cache_dir={directory / 'cache'}",
         "--junitxml=reports/test-results.xml"],
        ["coverage", "run", "--append", "-m", "attrition.models.train", "--with-shap"],
        ["coverage", "report", "--fail-under=80"],
        ["coverage", "xml", "-o", "reports/coverage.xml"],
        ["scripts.build_notebooks"],
        ["scripts.execute_notebooks"],
        ["scripts.verify_delivery"],
    ]
    for command in commands:
        print(f"Running: {sys.executable} -m {' '.join(command)}", flush=True)
        subprocess.run([sys.executable, "-m", *command], cwd=ROOT, env=env, check=True)
    print(f"Quality checks passed. Temporary evidence: {directory}")


if __name__ == "__main__":
    main()
