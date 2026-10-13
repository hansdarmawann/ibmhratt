"""Command-line entry points: argument handling, output contracts, and failure behavior."""

import functools
import json
import logging
import os
import sys
from pathlib import Path

import pytest

from attrition import config
from attrition.models import predict as predict_module
from attrition.models import prune, train
from attrition.models.artifacts import write_json


def run_main(monkeypatch, module, *arguments):
    monkeypatch.setattr(sys, "argv", [module.__name__, *arguments])
    return module.main()


def test_predict_cli_prints_the_documented_json_shape(monkeypatch, capsys, bundle_factory, employee, tmp_path):
    bundle = bundle_factory(threshold=0.4)
    request = tmp_path / "employee.json"
    request.write_text(json.dumps(employee), encoding="utf-8")
    run_main(monkeypatch, predict_module, str(request), "--model", str(bundle.path))
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"predicted_class", "attrition_probability", "decision_threshold", "run_id"}
    assert result["run_id"] == bundle.run_id
    assert result["decision_threshold"] == 0.4
    assert result["predicted_class"] == ("Yes" if result["attrition_probability"] >= 0.4 else "No")


def test_predict_cli_accepts_the_active_run_pointer(monkeypatch, capsys, bundle_factory, employee, tmp_path):
    bundle = bundle_factory()
    request = tmp_path / "employee.json"
    request.write_text(json.dumps(employee), encoding="utf-8")
    run_main(monkeypatch, predict_module, str(request), "--model", str(tmp_path / "models/current.json"))
    assert json.loads(capsys.readouterr().out)["run_id"] == bundle.run_id


@pytest.mark.parametrize(("content", "error"), [
    ("{not json", json.JSONDecodeError),
    ('{"Age": 30}', ValueError),
    ("[1, 2]", ValueError),
])
def test_predict_cli_rejects_bad_requests(monkeypatch, capsys, bundle_factory, tmp_path, content, error):
    bundle = bundle_factory()
    request = tmp_path / "bad.json"
    request.write_text(content, encoding="utf-8")
    with pytest.raises(error):
        run_main(monkeypatch, predict_module, str(request), "--model", str(bundle.path))
    assert capsys.readouterr().out == ""


def test_predict_cli_requires_an_existing_request_file(monkeypatch, bundle_factory, tmp_path):
    bundle = bundle_factory()
    with pytest.raises(FileNotFoundError):
        run_main(monkeypatch, predict_module, str(tmp_path / "absent.json"), "--model", str(bundle.path))
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, predict_module)
    assert exit_info.value.code == 2


@pytest.fixture
def pruning(monkeypatch, bundle_factory, tmp_path):
    """prune.main uses the checkout pointer; point it at a temporary models directory instead."""
    for index in range(3):
        bundle = bundle_factory(activate=index == 2)
        os.utime(bundle.path, (1_000_000 + index, 1_000_000 + index))
    pointer = tmp_path / "models/current.json"
    monkeypatch.setattr(prune, "prune_runs", functools.partial(prune.prune_runs, pointer=pointer))
    return tmp_path / "models/runs"


def test_prune_cli_dry_run_reports_without_deleting(monkeypatch, caplog, pruning):
    caplog.set_level(logging.INFO)
    run_main(monkeypatch, prune, "--keep", "1", "--dry-run")
    assert len(list(pruning.iterdir())) == 3
    assert "2 run(s) would be removed" in caplog.text


def test_prune_cli_deletes_only_old_inactive_runs(monkeypatch, caplog, pruning):
    caplog.set_level(logging.INFO)
    run_main(monkeypatch, prune, "--keep", "1")
    active = json.loads((pruning.parent / "current.json").read_text(encoding="utf-8"))["run_id"]
    assert {path.name for path in pruning.iterdir()} == {active}
    assert "2 run(s) removed" in caplog.text


def test_prune_cli_refuses_to_keep_zero_runs(monkeypatch, pruning):
    with pytest.raises(ValueError, match="at least one"):
        run_main(monkeypatch, prune, "--keep", "0")
    assert len(list(pruning.iterdir())) == 3


def test_prune_cli_rejects_non_integer_keep(monkeypatch, pruning):
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, prune, "--keep", "many")
    assert exit_info.value.code == 2
    assert len(list(pruning.iterdir())) == 3


@pytest.fixture
def captured_training(monkeypatch):
    calls = []

    def fake_train(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(train, "train", fake_train)
    return calls


def test_training_cli_defaults_do_not_enable_optional_work(monkeypatch, captured_training):
    run_main(monkeypatch, train)
    (args, kwargs), = captured_training
    assert args == (config.DATA_PATH,)
    assert kwargs == {"with_shap": False, "output_root": config.ROOT, "models_dir": None}


def test_training_cli_makes_every_path_absolute(monkeypatch, tmp_path, captured_training):
    monkeypatch.chdir(tmp_path)
    run_main(monkeypatch, train, "--data", "in/data set.csv", "--output-root", "out", "--models-dir", "m",
             "--with-shap")
    (args, kwargs), = captured_training
    assert args == (tmp_path.resolve() / "in/data set.csv",)
    assert kwargs == {"with_shap": True, "output_root": tmp_path.resolve() / "out",
                      "models_dir": tmp_path.resolve() / "m"}
    assert all(isinstance(value, Path) and value.is_absolute() for value in (args[0], kwargs["models_dir"]))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("arguments", [("--data", " "), ("--models-dir", "bad\0path"), ("--unknown",)])
def test_training_cli_rejects_invalid_options_before_training(monkeypatch, captured_training, arguments):
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, train, *arguments)
    assert exit_info.value.code == 2
    assert captured_training == []


def test_prune_cli_creates_nothing_and_stops_on_an_invalid_pointer(monkeypatch, tmp_path):
    pointer = tmp_path / "models/current.json"
    monkeypatch.setattr(prune, "prune_runs", functools.partial(prune.prune_runs, pointer=pointer))
    run_main(monkeypatch, prune, "--keep", "3", "--dry-run")
    assert not pointer.parent.exists()
    write_json({"schema_version": 2, "run_id": "not-a-uuid"}, pointer)
    with pytest.raises(ValueError, match="pointer"):
        run_main(monkeypatch, prune, "--keep", "3")
