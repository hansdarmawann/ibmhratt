"""The coverage helper gates each labeled measurement and keeps the combined gate aligned with pyproject."""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scripts import coverage_report
from scripts.coverage_report import COMBINED_MIN, PYTEST_ONLY_MIN, SCOPES, main

CHECKOUT = Path(__file__).resolve().parents[1]


def test_combined_gate_matches_pyproject():
    configured = tomllib.loads((CHECKOUT / "pyproject.toml").read_text(encoding="utf-8"))
    assert configured["tool"]["coverage"]["report"]["fail_under"] == COMBINED_MIN


def test_each_scope_gates_on_its_own_threshold():
    assert 0 < PYTEST_ONLY_MIN <= 100
    assert SCOPES["pytest-only"][0] == PYTEST_ONLY_MIN
    assert SCOPES["combined"][0] == COMBINED_MIN


def test_scope_selects_threshold_and_label(monkeypatch, capsys):
    commands = []

    def fake_run(command, check):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(coverage_report.subprocess, "run", fake_run)
    assert main(["pytest-only"]) == 0
    assert commands[-1][1:] == ["-m", "coverage", "report", f"--fail-under={PYTEST_ONLY_MIN}"]
    assert "(pytest-only)" in capsys.readouterr().out
    assert main(["combined", "--xml", "out.xml"]) == 0
    assert [command[3:] for command in commands[-2:]] == [["report", f"--fail-under={COMBINED_MIN}"],
                                                         ["xml", "-o", "out.xml"]]
    assert "(combined)" in capsys.readouterr().out


def test_unknown_scope_is_rejected():
    with pytest.raises(SystemExit):
        main(["everything"])


def test_gate_fails_below_the_minimum_but_still_writes_the_xml(tmp_path, monkeypatch):
    script = tmp_path / "partly.py"
    script.write_text("value = 1\nif value > 1:\n    value = 2\n    value = 3\n", encoding="utf-8")
    monkeypatch.setenv("COVERAGE_FILE", str(tmp_path / ".coverage"))
    monkeypatch.chdir(tmp_path)
    subprocess.run([sys.executable, "-m", "coverage", "run", str(script)], check=True, cwd=tmp_path)
    xml = tmp_path / "coverage.xml"
    monkeypatch.setitem(SCOPES, "pytest-only", (99, "forced"))
    assert main(["pytest-only", "--xml", str(xml)]) != 0
    assert xml.is_file()
    monkeypatch.setitem(SCOPES, "pytest-only", (1, "lenient"))
    assert main(["pytest-only"]) == 0
