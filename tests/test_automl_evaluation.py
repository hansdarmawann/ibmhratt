"""Nested AutoML evaluation: outer validation rows never reach search, preprocessing, or threshold choice."""

import json
import os
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from attrition.config import (
    CATEGORIES,
    CONSTANT_COLUMNS,
    CV_FOLDS,
    ID_COLUMN,
    NUMERIC_BOUNDS,
    RANDOM_STATE,
    ROOT,
    TARGET,
)
from attrition.data.load_data import split_data
from attrition.features.preprocess import build_pipeline
from attrition.models import automl, automl_evaluation, nested, train
from attrition.models.nested import NESTED_METRICS

# One trial and three outer folds keep each nested run in CI to seconds.
OUTER, TRIALS, SEED = 3, 1, 7


@pytest.fixture
def optuna():
    return pytest.importorskip("optuna", reason=f"AutoML tests need Optuna: {automl.INSTALL_HINT}")


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(29)
    count = 150
    frame = pd.DataFrame({
        name: rng.integers(lower, (upper if upper is not None else lower + 20) + 1, count)
        for name, (lower, upper) in NUMERIC_BOUNDS.items()
    })
    for name, categories in CATEGORIES.items():
        frame[name] = rng.choice(categories, count)
    for name, value in CONSTANT_COLUMNS.items():
        frame[name] = value
    frame[ID_COLUMN] = np.arange(1, count + 1)
    # A weak real signal with enough positives for every inner validation fold.
    frame[TARGET] = np.where((frame["OverTime"] == "Yes") & (np.arange(count) % 3 != 0), "Yes", "No")
    return frame


@pytest.fixture
def logistic_only(monkeypatch):
    """Forests dominate runtime but not data flow: CI searches and compares logistic models through the same code."""
    monkeypatch.setattr(automl, "FAMILIES", ("logistic_regression",))

    def candidates(X):
        return {name: train.build_pipeline(X, LogisticRegression(max_iter=1000, random_state=42, class_weight=weight))
                for name, weight in [("logistic_regression", None), ("logistic_balanced", "balanced")]}

    monkeypatch.setattr(automl_evaluation, "candidate_pipelines", candidates)


class RecordFit(TransformerMixin, BaseEstimator):
    """First pipeline step; logs every fit's rows, before any learned preprocessing, and checks its labels."""

    events: ClassVar[list] = []
    labels: ClassVar[pd.Series] = pd.Series(dtype=int)

    def fit(self, X, y=None):
        # cross_val_predict encodes classification targets as an ndarray.
        np.testing.assert_array_equal(y, self.labels.loc[X.index].to_numpy())
        self.events.append(("fit", frozenset(X.index)))
        return self

    def transform(self, X):
        return X


def rows(frame: pd.DataFrame, positions) -> frozenset:
    return frozenset(frame.iloc[positions].index)


def test_each_procedure_selects_and_thresholds_only_on_outer_training_rows(optuna, logistic_only, monkeypatch,
                                                                          synthetic_data):
    X, X_test, y, _ = split_data(synthetic_data)
    events = []
    monkeypatch.setattr(RecordFit, "events", events)
    monkeypatch.setattr(RecordFit, "labels", y)

    def recorded_pipeline(*args, **kwargs):
        return Pipeline([("record", RecordFit()), *build_pipeline(*args, **kwargs).steps])

    # AutoML trials, the reconstructed candidate, and the baseline candidates all start with RecordFit.
    monkeypatch.setattr(automl, "build_pipeline", recorded_pipeline)
    monkeypatch.setattr(train, "build_pipeline", recorded_pipeline)
    real_search = automl.search

    def search(features, labels, **kwargs):
        pd.testing.assert_series_equal(labels, y.loc[features.index])
        events.append(("search", frozenset(features.index)))
        return real_search(features, labels, **kwargs)

    def recorded_threshold_table(real):
        def threshold_table(labels, probabilities):
            pd.testing.assert_series_equal(labels, y.loc[labels.index])
            assert len(probabilities) == len(labels)
            events.append(("threshold", frozenset(labels.index)))
            return real(labels, probabilities)
        return threshold_table

    real_evaluate = automl_evaluation.evaluate_probabilities

    def evaluate(labels, probabilities, threshold):
        events.append(("score", frozenset(labels.index)))
        return real_evaluate(labels, probabilities, threshold)

    monkeypatch.setattr(automl, "search", search)
    for module in (automl_evaluation, nested):
        monkeypatch.setattr(module, "threshold_table", recorded_threshold_table(module.threshold_table))
    monkeypatch.setattr(automl_evaluation, "evaluate_probabilities", evaluate)

    result = automl_evaluation.nested_folds(X, y, n_trials=TRIALS, timeout=120, seed=SEED, outer_folds=OUTER)

    expected = []
    for fit, validation in StratifiedKFold(OUTER, shuffle=True, random_state=SEED).split(X, y):
        X_fit, y_fit = X.iloc[fit], y.iloc[fit]
        inner = [("fit", rows(X_fit, train_rows))
                 for train_rows, _ in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(X_fit, y_fit)]
        outer_fit, outer_validation = ("fit", rows(X, fit)), ("score", rows(X, validation))
        # AutoML: trials and the dummy reference on inner folds, then OOF predictions of the frozen choice,
        # then the threshold, then the refit on outer training rows, then one score on the outer fold.
        expected += [("search", rows(X, fit)), *inner * (TRIALS + 1), *inner,
                     ("threshold", rows(X, fit)), outer_fit, outer_validation]
        # Baseline: both candidates on the same inner folds, OOF predictions, threshold, refit, score.
        expected += [*inner * 2, *inner, ("threshold", rows(X, fit)), outer_fit, outer_validation]
    assert events == expected
    assert all(rows_seen.isdisjoint(X_test.index) for _, rows_seen in events)
    assert [(row["procedure"], row["outer_fold"]) for row in result] == [
        (procedure, fold) for fold in range(OUTER) for procedure in ("automl", "baseline_fixed")]


def test_nested_evaluation_is_deterministic_and_reports_a_paired_comparison(optuna, logistic_only, synthetic_data):
    X, _, y, _ = split_data(synthetic_data)
    first, second = (automl_evaluation.nested_evaluation(X, y, n_trials=TRIALS, timeout=120, seed=SEED,
                                                         outer_folds=OUTER) for _ in range(2))
    for result in (first, second):
        for search in result["searches"]:
            search["result"].pop("elapsed_seconds")
            for trial in search["result"]["trial_records"]:
                trial.pop("elapsed_seconds")
    assert first == second
    json.dumps(first, allow_nan=False)
    assert first["outer_cv"] == {"partition": "training", "folds": OUTER, "stratified": True, "shuffle": True,
                                 "random_state": SEED}
    assert first["inner_cv"]["folds"] == CV_FOLDS and first["inner_cv"]["random_state"] == SEED
    assert first["automl"]["sampler"] == {"name": "TPESampler", "seed": SEED}
    assert first["automl"]["budget_per_outer_fold"]["n_trials"] == TRIALS
    assert first["holdout_used"] is False and first["served_model_changed"] is False
    assert first["train_rows"] == len(X) and first["train_positives"] == int(y.sum())
    assert set(first["procedures"]) == {"automl", "baseline_fixed"}
    assert len(first["folds"]) == 2 * OUTER
    assert len(first["searches"]) == OUTER
    for fold, search in enumerate(first["searches"]):
        assert search["outer_fold"] == fold
        trial = search["result"]["trial_records"][0]
        row = first["folds"][2 * fold]
        assert trial["state"] == "COMPLETE" and trial["model"] == row["selected_model"]
        assert automl.readable(trial["sampled_params"]) == row["parameters"]
    for row in first["folds"]:
        assert 0 < row["threshold"] < 1
        if row["procedure"] == "automl":
            assert row["selected_model"] in automl.FAMILIES
            assert row["trials_completed"] == TRIALS
            assert 0 <= row["inner_cv_ap"] <= 1
        else:
            assert row["selected_model"] in {"logistic_regression", "logistic_balanced"}
            assert row["parameters"] == {} and row["inner_cv_ap"] is None
    by_fold = {(row["procedure"], row["outer_fold"]): row for row in first["folds"]}
    differences = [by_fold["automl", fold]["pr_auc"] - by_fold["baseline_fixed", fold]["pr_auc"]
                   for fold in range(OUTER)]
    assert first["comparison"]["metrics"]["pr_auc"]["mean"] == pytest.approx(np.mean(differences))
    assert set(first["comparison"]["metrics"]) == set(NESTED_METRICS)


def test_paired_differences_and_selection_optimism():
    folds = [{"procedure": procedure, "outer_fold": fold, "inner_cv_ap": inner,
              **dict.fromkeys(NESTED_METRICS, score)}
             for procedure, fold, inner, score in [
                 ("automl", 0, 0.8, 0.6), ("baseline_fixed", 0, None, 0.5),
                 ("automl", 1, 0.7, 0.4), ("baseline_fixed", 1, None, 0.5),
                 ("automl", 2, 0.9, 0.5), ("baseline_fixed", 2, None, 0.5)]]
    comparison = automl_evaluation.paired_differences(folds)
    assert set(comparison) == set(NESTED_METRICS)
    assert comparison["pr_auc"] == {"mean": pytest.approx(0.0), "std": pytest.approx(np.sqrt(0.02 / 3)),
                                    "min": pytest.approx(-0.1), "max": pytest.approx(0.1),
                                    "automl_higher_folds": 1, "baseline_higher_folds": 1}
    optimism = automl_evaluation.selection_optimism(folds)
    assert optimism == {"mean_inner_best_trial_cv_ap": pytest.approx(0.8), "mean_outer_ap": pytest.approx(0.5),
                        "difference": pytest.approx(0.3)}


def test_cli_evaluates_training_rows_only_and_writes_only_a_new_output(monkeypatch, tmp_path, capsys, synthetic_data):
    source = tmp_path / "data.csv"
    synthetic_data.to_csv(source, index=False)
    X_train, _, _, y_test = split_data(synthetic_data)
    calls = []

    def evaluation(X, y, **kwargs):
        pd.testing.assert_frame_equal(X, X_train)
        assert set(y.index).isdisjoint(y_test.index), "holdout labels reached the nested evaluation"
        calls.append(kwargs)
        return {"holdout_used": False}

    monkeypatch.setattr(automl_evaluation, "nested_evaluation", evaluation)
    output = tmp_path / "results/nested.json"
    args = ["--data", str(source), "--n-trials", "3", "--timeout", "60", "--seed", "11", "--outer-folds", "4"]
    automl_evaluation.main([*args, "--output", str(output)])
    assert calls == [{"n_trials": 3, "timeout": 60.0, "seed": 11, "outer_folds": 4}]
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == json.loads(capsys.readouterr().out)
    assert len(written["data_sha256"]) == 64 and written["elapsed_seconds"] >= 0
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()) == [
        "data.csv", "results/nested.json"]
    with pytest.raises(FileExistsError):
        automl_evaluation.main([*args, "--output", str(output)])
    assert len(calls) == 1


@pytest.mark.parametrize("argument", [["--n-trials", "0"], ["--timeout", "0"], ["--outer-folds", "1"],
                                      ["--outer-folds", "11"]])
def test_cli_rejects_invalid_budgets(argument):
    with pytest.raises(SystemExit):
        automl_evaluation.main(argument)


@pytest.mark.skipif(os.environ.get("ATTRITION_SLOW_TESTS") != "1",
                    reason="opt-in real-data nested AutoML run; set ATTRITION_SLOW_TESTS=1")
def test_real_data_baseline_arm_reproduces_the_training_nested_cv(optuna, partitions):
    X, _, y, _ = partitions
    report = automl_evaluation.nested_evaluation(X, y, n_trials=2, timeout=300, seed=RANDOM_STATE)
    published = pd.read_csv(ROOT / "reports/metrics/nested_cv_folds.csv")
    published = published[published["procedure"] == "fixed"].reset_index(drop=True)
    baseline = pd.DataFrame([row for row in report["folds"] if row["procedure"] == "baseline_fixed"])
    # Same outer folds and the same served procedure as the committed training report: identical rows.
    assert baseline["outer_fold"].tolist() == published["outer_fold"].tolist()
    assert baseline["selected_model"].tolist() == published["selected_model"].tolist()
    np.testing.assert_allclose(baseline[["threshold", *NESTED_METRICS]].to_numpy(dtype=float),
                               published[["threshold", *NESTED_METRICS]].to_numpy(dtype=float), rtol=0, atol=1e-12)
    assert report["train_rows"] == len(X) and report["holdout_used"] is False
