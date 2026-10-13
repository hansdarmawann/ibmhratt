"""Optional Optuna search: training-CV only, deterministic, and isolated from the served baseline."""

import json
import subprocess
import sys
import warnings
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from attrition.config import CATEGORIES, CONSTANT_COLUMNS, ID_COLUMN, NUMERIC_BOUNDS, TARGET
from attrition.data.load_data import split_data
from attrition.features.preprocess import build_pipeline
from attrition.models import automl

# Tiny budgets keep every search here to a few seconds.
FOLDS, TRIALS, SEED = 3, 2, 7


@pytest.fixture
def optuna():
    return pytest.importorskip("optuna", reason=f"AutoML tests need Optuna: {automl.INSTALL_HINT}")


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(23)
    count = 120
    frame = pd.DataFrame({
        name: rng.integers(lower, (upper if upper is not None else lower + 20) + 1, count)
        for name, (lower, upper) in NUMERIC_BOUNDS.items()
    })
    for name, categories in CATEGORIES.items():
        frame[name] = rng.choice(categories, count)
    for name, value in CONSTANT_COLUMNS.items():
        frame[name] = value
    frame[ID_COLUMN] = np.arange(1, count + 1)
    # A weak real signal, so trials are not all tied at the prior.
    frame[TARGET] = np.where((frame["OverTime"] == "Yes") & (np.arange(count) % 2 == 0), "Yes", "No")
    return frame


class RecordFit(TransformerMixin, BaseEstimator):
    """First pipeline step; records the rows of every fit, before any learned preprocessing."""

    rows: ClassVar[list[set]] = []

    def fit(self, X, y=None):
        self.rows.append(set(X.index))
        return self

    def transform(self, X):
        return X


def test_importing_automl_needs_no_optuna_and_baseline_never_imports_it():
    code = ("import sys; import attrition.models.automl; print('optuna' in sys.modules); "
            "del sys.modules['attrition.models.automl']; import attrition.models.train, app.api; "
            "print('attrition.models.automl' in sys.modules, 'optuna' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=True)
    assert result.stdout.split() == ["False", "False", "False"]


def test_missing_optuna_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(ModuleNotFoundError, match=r"requirements-automl\.txt"):
        automl.require_optuna()


def test_training_partition_is_the_baseline_training_split(synthetic_data):
    X_train, X_test, y_train, _ = split_data(synthetic_data)
    X, y = automl.training_partition(synthetic_data)
    pd.testing.assert_frame_equal(X, X_train)
    pd.testing.assert_series_equal(y, y_train)
    assert set(X.index).isdisjoint(X_test.index)


def test_parameter_spaces_are_conditional_per_family(optuna):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    families = set()
    for _ in range(40):
        trial = study.ask()
        family, estimator = automl.suggest_estimator(trial, SEED)
        families.add(family)
        assert trial.params["model"] == family
        assert all(name.startswith(f"{family}__") for name in trial.params if name != "model")
        assert estimator.random_state == SEED
        study.tell(trial, 0.0)
    assert families == set(automl.FAMILIES)


BOUNDS = {
    "logistic_regression": [{"C": 0.01, "class_weight": None}, {"C": 10.0, "class_weight": "balanced"}],
    "random_forest": [
        {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": None},
        {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 2, "max_features": 0.5,
         "class_weight": "balanced_subsample"},
    ],
    "hist_gradient_boosting": [
        {"learning_rate": 0.01, "max_iter": 50, "max_leaf_nodes": 4, "min_samples_leaf": 10,
         "l2_regularization": 1e-3, "class_weight": None},
        {"learning_rate": 0.2, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 50,
         "l2_regularization": 10.0, "class_weight": "balanced"},
    ],
}


@pytest.mark.parametrize("family", automl.FAMILIES)
def test_parameter_space_bounds_fit_without_warnings(optuna, synthetic_data, family):
    X, y = automl.training_partition(synthetic_data)
    for bounds in BOUNDS[family]:
        trial = optuna.trial.FixedTrial({"model": family, **{f"{family}__{k}": v for k, v in bounds.items()}})
        name, estimator = automl.suggest_estimator(trial, SEED)
        assert name == family
        assert {key: estimator.get_params()[key] for key in bounds} == bounds
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Invalid or non-converging combinations fail here.
            pipeline = build_pipeline(X, estimator, scale=family == "logistic_regression").fit(X, y)
        assert pipeline.predict_proba(X).shape == (len(X), 2)


def test_search_smoke_is_deterministic_and_fits_only_training_folds(optuna, monkeypatch, synthetic_data):
    X, X_test, y, _ = split_data(synthetic_data)
    allowed = [set(X.iloc[fit].index) for fit, _ in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y)]
    monkeypatch.setattr(RecordFit, "rows", [])

    def recorded_pipeline(*args, **kwargs):
        return Pipeline([("record", RecordFit()), *build_pipeline(*args, **kwargs).steps])

    monkeypatch.setattr(automl, "build_pipeline", recorded_pipeline)
    first, second = (automl.search(X, y, n_trials=TRIALS, timeout=120, seed=SEED, folds=FOLDS) for _ in range(2))
    for result in (first, second):
        result.pop("elapsed_seconds")
    assert first == second
    # Two trials plus the untuned dummy reference, each refitted in every fold, for both runs.
    assert len(RecordFit.rows) == 2 * (TRIALS + 1) * FOLDS
    assert all(rows in allowed for rows in RecordFit.rows), "preprocessing fitted outside a training fold"
    assert all(rows.isdisjoint(X_test.index) for rows in RecordFit.rows)

    best = first["best"]
    assert best["model"] in automl.FAMILIES
    assert set(best["params"]) == set(BOUNDS[best["model"]][0])
    assert len(best["cv_ap"]["folds"]) == FOLDS
    assert best["cv_ap"]["mean"] == pytest.approx(np.mean(best["cv_ap"]["folds"]))
    assert first["cv"] == {"partition": "training", "folds": FOLDS, "stratified": True, "shuffle": True,
                           "random_state": SEED}
    assert first["sampler"] == {"name": "TPESampler", "seed": SEED}
    assert first["trials"] == {"completed": TRIALS, "total": TRIALS}
    assert first["reference"]["dummy_prior"]["tuned"] is False
    assert first["train_rows"] == len(X)
    assert first["holdout_used"] is False and first["served_model_changed"] is False
    json.dumps(first, allow_nan=False)


def test_search_rejects_invalid_budgets(optuna, synthetic_data):
    X, y = automl.training_partition(synthetic_data)
    for kwargs in [{"n_trials": 0}, {"folds": 1}, {"timeout": 0}]:
        with pytest.raises(ValueError, match="at least one trial"):
            automl.search(X, y, **kwargs)


def test_cli_searches_training_rows_and_writes_only_a_new_output(monkeypatch, tmp_path, capsys, synthetic_data):
    source = tmp_path / "data.csv"
    synthetic_data.to_csv(source, index=False)
    X_train, _, _, y_test = split_data(synthetic_data)
    calls = []

    def search(X, y, **kwargs):
        pd.testing.assert_frame_equal(X, X_train)
        assert set(y.index).isdisjoint(y_test.index), "holdout labels reached the search"
        calls.append(kwargs)
        return {"best": {"model": "logistic_regression"}}

    monkeypatch.setattr(automl, "search", search)
    output = tmp_path / "results/search.json"
    args = ["--data", str(source), "--n-trials", "3", "--timeout", "60", "--seed", "11", "--folds", "4"]
    automl.main([*args, "--output", str(output)])
    assert calls == [{"n_trials": 3, "timeout": 60.0, "seed": 11, "folds": 4}]
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == json.loads(capsys.readouterr().out)
    assert len(written["data_sha256"]) == 64
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()) == [
        "data.csv", "results/search.json"]
    with pytest.raises(FileExistsError):
        automl.main([*args, "--output", str(output)])
    assert len(calls) == 1


def test_output_never_goes_inside_the_models_directory(tmp_path):
    models = tmp_path / "models"
    for path in [models / "current.json", models / "runs/search.json"]:
        with pytest.raises(ValueError, match="outside the models directory"):
            automl.check_output(path, models)
    assert automl.check_output(tmp_path / "search.json", models) == tmp_path / "search.json"


@pytest.mark.parametrize("argument", [["--n-trials", "0"], ["--timeout", "0"], ["--timeout", "nan"], ["--folds", "1"]])
def test_cli_rejects_invalid_budgets(argument):
    with pytest.raises(SystemExit):
        automl.main(argument)
