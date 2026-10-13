"""Report coverage under an explicit label: python -m scripts.coverage_report pytest-only [--xml PATH].

Two different questions share one coverage database, so every report says which one it answers:

* ``pytest-only``: lines executed while the pytest suite ran. Run it after pytest and before the
  training run is appended. Only these lines are exercised by test code that can fail.
* ``combined``: pytest plus the fresh ``attrition.models.train`` run. Training and plotting lines
  executed there are covered by the run itself, not by assertions, so this figure is higher.

Thresholds are measured floors, not targets. Raise ``PYTEST_ONLY_MIN`` only together with tests that
assert behavior; see docs/development.md for the measured values and how they were chosen.
"""

import argparse
import subprocess
import sys

# Mirrors [tool.coverage.report] fail_under in pyproject.toml, which remains the combined gate.
COMBINED_MIN = 80
# Measured 86% on Windows/Python 3.12 (83% before PR 07); two points of headroom for platform-specific lines.
PYTEST_ONLY_MIN = 84
SCOPES = {
    "pytest-only": (PYTEST_ONLY_MIN, "tests only (training and notebook runs excluded)"),
    "combined": (COMBINED_MIN, "tests plus a fresh training run"),
}


def coverage_command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "coverage", *arguments]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scope", choices=sorted(SCOPES), help="Which measurement the current database holds")
    parser.add_argument("--xml", help="Also write a Cobertura XML report to this path")
    args = parser.parse_args(argv)
    minimum, description = SCOPES[args.scope]
    print(f"== Coverage ({args.scope}): {description}; minimum {minimum}% ==", flush=True)
    status = subprocess.run(coverage_command("report", f"--fail-under={minimum}"), check=False).returncode
    if args.xml:
        # The XML is evidence either way, so it is written even when the gate fails.
        xml_status = subprocess.run(coverage_command("xml", "-o", args.xml), check=False).returncode
        status = status or xml_status
    return status


if __name__ == "__main__":
    sys.exit(main())
