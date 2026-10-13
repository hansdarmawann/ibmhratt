"""Artifacts and data used from other locations: relocation, awkward directory names, and line endings."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from attrition import config
from attrition.config import FEATURE_COLUMNS
from attrition.data.load_data import load_data
from attrition.models.artifacts import load_bundle, read_json
from attrition.models.predict import predict
from attrition.monitoring.drift import main as drift_main
from attrition.monitoring.drift import read_records

SOURCE_ROOT = Path(config.__file__).resolve().parents[1]
AWKWARD = "é ü"  # a space and non-ASCII characters, as in real user-profile directories (kept short for MAX_PATH)


def subprocess_env() -> dict[str, str]:
    base = {key: value for key, value in os.environ.items() if key not in (config.ROOT_ENV, config.DATA_ENV,
                                                                            config.MODELS_ENV)}
    return {**base, "PYTHONPATH": str(SOURCE_ROOT), "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}


@pytest.fixture
def relocated(bundle_factory, tmp_path):
    """The same models directory copied to a differently named location."""
    bundle = bundle_factory()
    destination = tmp_path / AWKWARD
    shutil.copytree(tmp_path / "models", destination / "models")
    return bundle, destination / "models"


def test_copied_models_directory_serves_the_same_run(relocated, employee):
    bundle, models = relocated
    moved = load_bundle(models / "current.json")
    assert moved.run_id == bundle.run_id
    assert moved.path == models / "runs" / bundle.run_id
    assert predict(employee, pipeline=moved.pipeline) == predict(employee, pipeline=bundle.pipeline)
    with TestClient(create_app(models / "current.json")) as client:
        assert client.get("/health").json()["run_id"] == bundle.run_id
        assert client.post("/predict", json=employee).json()["run_id"] == bundle.run_id


def test_bundle_directory_can_be_addressed_without_the_pointer(relocated, employee):
    bundle, models = relocated
    direct = models / "runs" / bundle.run_id
    assert load_bundle(str(direct)).run_id == bundle.run_id
    with TestClient(create_app(str(direct))) as client:
        assert client.get("/model").json()["run_id"] == bundle.run_id


def test_manifest_uses_relative_posix_names_on_every_platform(bundle_factory):
    names = read_json(bundle_factory().path / "manifest.json")["files"]
    assert "metrics/model_metrics.json" in names
    for name in names:
        path = PurePosixPath(name)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert "\\" not in name
        assert ":" not in name


def test_prediction_cli_accepts_relative_model_paths_in_awkward_directories(relocated, employee, tmp_path):
    bundle, models = relocated
    working = models.parent
    (working / "employee input.json").write_text(json.dumps(employee), encoding="utf-8")
    command = [sys.executable, "-m", "attrition.models.predict", "employee input.json",
               "--model", str(Path("models") / "runs" / bundle.run_id)]
    result = subprocess.run(command, cwd=working, env=subprocess_env(), capture_output=True, text=True,
                            check=False, timeout=300)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["run_id"] == bundle.run_id


def test_prediction_cli_fails_with_an_actionable_message_for_a_missing_model(tmp_path, employee):
    (tmp_path / "employee.json").write_text(json.dumps(employee), encoding="utf-8")
    command = [sys.executable, "-m", "attrition.models.predict", "employee.json", "--model", "absent/current.json"]
    result = subprocess.run(command, cwd=tmp_path, env=subprocess_env(), capture_output=True, text=True,
                            check=False, timeout=300)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "Run: python -m attrition.models.train" in result.stderr


def test_training_cli_normalizes_relative_paths_and_fails_cleanly(tmp_path):
    command = [sys.executable, "-m", "attrition.models.train", "--data", "missing data.csv",
               "--output-root", "out", "--models-dir", "bundle store"]
    result = subprocess.run(command, cwd=tmp_path, env=subprocess_env(), capture_output=True, text=True,
                            check=False, timeout=300)
    assert result.returncode != 0
    assert f"Dataset not found: {(tmp_path / 'missing data.csv').resolve()}" in result.stderr
    assert not (tmp_path / "bundle store/current.json").exists()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_dataset_loads_identically_with_either_line_ending(raw_data, tmp_path, newline):
    path = tmp_path / f"{AWKWARD}.csv"
    raw_data.to_csv(path, index=False, lineterminator=newline)
    pd.testing.assert_frame_equal(load_data(str(path)), raw_data)


def test_dataset_path_must_be_a_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(tmp_path)


def test_dataset_with_wrong_columns_is_refused(raw_data, tmp_path):
    path = tmp_path / "wrong.csv"
    raw_data.rename(columns={"Age": "Years"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"Invalid columns\. Missing: \['Age'\]; unexpected: \['Years'\]"):
        load_data(path)


@pytest.mark.parametrize("suffix", [".csv", ".CSV"])
def test_drift_records_can_be_csv_in_any_case(partitions, tmp_path, suffix):
    rows = partitions[1][FEATURE_COLUMNS].head(3)
    path = tmp_path / f"records{suffix}"
    rows.to_csv(path, index=False)
    pd.testing.assert_frame_equal(read_records(path), rows.reset_index(drop=True))


def test_drift_records_accept_object_list_and_wrapper_json(partitions, tmp_path):
    rows = partitions[1][FEATURE_COLUMNS].head(3)
    records = json.loads(rows.to_json(orient="records"))
    expected = pd.DataFrame(records)
    for name, payload in {"list.json": records, "wrapped.json": {"employees": records},
                          "single.json": records[0]}.items():
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        frame = read_records(path)
        assert len(frame) == (1 if name == "single.json" else 3)
        pd.testing.assert_frame_equal(frame, expected.head(len(frame)))


def test_drift_cli_runs_against_a_relocated_pointer(relocated, partitions, tmp_path, capsys):
    _, models = relocated
    records = models.parent / "records to check.csv"
    partitions[1][FEATURE_COLUMNS].to_csv(records, index=False)
    assert drift_main([str(records), "--bundle", str(models / "current.json"), "--fail-on-shift"]) == 0
    assert "MonthlyIncome" in capsys.readouterr().out

