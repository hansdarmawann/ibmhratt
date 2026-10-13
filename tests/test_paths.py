"""Operator path overrides: checkout defaults, normalization, and no request-controlled paths."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from attrition import config
from attrition.config import CHECKOUT_ROOT, DATA_RELATIVE, Paths, env_path, normalize_path

OVERRIDES = (config.ROOT_ENV, config.DATA_ENV, config.MODELS_ENV)


def clean_environ(**values: str) -> dict[str, str]:
    """The current environment without path overrides, plus `values`."""
    return {**{key: value for key, value in os.environ.items() if key not in OVERRIDES}, **values}


def test_defaults_match_checkout_layout():
    assert Path(__file__).resolve().parents[1] == CHECKOUT_ROOT
    paths = Paths.from_env({})
    assert paths == Paths(CHECKOUT_ROOT, CHECKOUT_ROOT / DATA_RELATIVE, CHECKOUT_ROOT / "models")
    assert paths.current_run == CHECKOUT_ROOT / "models/current.json"


def test_root_override_moves_every_default(tmp_path):
    paths = Paths.from_env({config.ROOT_ENV: str(tmp_path)})
    assert paths.root == tmp_path.resolve()
    assert paths.data == tmp_path.resolve() / DATA_RELATIVE
    assert paths.current_run == tmp_path.resolve() / "models/current.json"


def test_data_and_models_override_independently(tmp_path):
    paths = Paths.from_env({config.DATA_ENV: str(tmp_path / "in.csv"), config.MODELS_ENV: str(tmp_path / "m")})
    assert paths.root == CHECKOUT_ROOT
    assert paths.data == tmp_path.resolve() / "in.csv"
    assert paths.current_run == tmp_path.resolve() / "m/current.json"


def test_relative_and_home_paths_are_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert env_path(config.MODELS_ENV, {config.MODELS_ENV: " artifacts/models "}) == tmp_path.resolve() / "artifacts/models"
    assert normalize_path("~").is_absolute()
    assert normalize_path("~") == Path.home().resolve()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows drive and separator handling")
def test_windows_separators_normalize():
    assert normalize_path("C:/data\\raw/../x.csv") == Path(r"C:\data\x.csv")


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_override_keeps_default(value):
    assert env_path(config.ROOT_ENV, {config.ROOT_ENV: value}) is None
    assert Paths.from_env({config.ROOT_ENV: value}).root == CHECKOUT_ROOT


def test_invalid_override_names_the_variable():
    with pytest.raises(ValueError, match=config.MODELS_ENV):
        Paths.from_env({config.MODELS_ENV: "models\0evil"})


def test_resolution_never_creates_directories(tmp_path):
    root = tmp_path / "not-created"
    paths = Paths.from_env({config.ROOT_ENV: str(root), config.MODELS_ENV: str(root / "m")})
    assert not root.exists()
    assert not paths.models.exists()


def test_import_reads_overrides_and_cli_serves_external_bundle(bundle_factory, employee, tmp_path):
    """A fresh process picks up the environment; prediction finds the external bundle without --model."""
    bundle = bundle_factory()
    request = tmp_path / "employee.json"
    request.write_text(json.dumps(employee), encoding="utf-8")
    env = clean_environ(**{config.MODELS_ENV: str(tmp_path / "models"),
                           config.DATA_ENV: str(tmp_path / "missing.csv"),
                           "PYTHONPATH": str(Path(config.__file__).parents[1])})
    probe = subprocess.run([sys.executable, "-c", "from attrition import config; "
                            "print(config.CURRENT_RUN); print(config.DATA_PATH)"],
                           env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert probe.stdout.splitlines() == [str((tmp_path / "models/current.json").resolve()),
                                         str((tmp_path / "missing.csv").resolve())]
    result = subprocess.run([sys.executable, "-m", "attrition.models.predict", str(request)],
                            env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["run_id"] == bundle.run_id


def test_training_writes_bundles_to_requested_directories(tmp_path, monkeypatch):
    """Runs land in models_dir, or <output_root>/models by default, never the checkout."""
    from attrition.models.train import train

    def stop(*args):
        raise RuntimeError("stop after creating run directories")

    monkeypatch.setattr("attrition.models.train.load_data", stop)
    with pytest.raises(RuntimeError, match="stop"):
        train(output_root=tmp_path / "out", models_dir=tmp_path / "bundles")
    assert len(list((tmp_path / "bundles/runs").iterdir())) == 1
    assert not (tmp_path / "out").exists()
    with pytest.raises(RuntimeError, match="stop"):
        train(output_root=tmp_path / "out")
    assert len(list((tmp_path / "out/models/runs").iterdir())) == 1


def test_api_requests_cannot_choose_model_paths(bundle_factory, employee, tmp_path):
    bundle_factory()
    other = tmp_path / "elsewhere.joblib"
    with TestClient(create_app(tmp_path / "models/current.json")) as client:
        assert all("{" not in route.path for route in client.app.routes)
        response = client.post("/predict", json={**employee, "model_path": str(other)})
        assert response.status_code == 422
        batch = {"employees": [employee], "model_path": str(other)}
        assert client.post("/predict/batch", json=batch).status_code == 422
        # Query parameters are not part of the contract and cannot redirect loading.
        loaded = client.post("/predict", params={"model_path": str(other)}, json=employee)
        assert loaded.status_code == 200
        assert loaded.json()["run_id"] == client.get("/health").json()["run_id"]
    assert not other.exists()
