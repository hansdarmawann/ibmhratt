"""Nested CV runs the full selection inside outer training folds and never fits on outer test rows."""

import json
from typing import ClassVar

import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from attrition.features.preprocess import build_pipeline
from attrition.models import nested
from attrition.models.nested import OUTER_FOLDS, nested_cv, nested_summary, run_procedure, tune
from attrition.models.train import select_model

SMALL_GRIDS = {"logistic_regression": {"model__C": [0.1, 1.0]}, "logistic_balanced": {"model__C": [0.1, 1.0]}}


class RecordFitRows(TransformerMixin, BaseEstimator):
    fitted_rows: ClassVar[list[set]] = []  # shared across sklearn clones on purpose

    def fit(self, X, y=None):
        self.fitted_rows.append(set(X.index))
        return self

    def transform(self, X):
        return X


@pytest.fixture
def small(partitions):
    X, _, y, _ = partitions
    indices = pd.concat([y[y == 0].head(80), y[y == 1].head(40)]).index
    return X.loc[indices], y.loc[indices]


def logistic_candidates(X, *, record: bool = False) -> dict:
    def pipeline(**kwargs):
        base = build_pipeline(X, LogisticRegression(max_iter=1000, random_state=42, **kwargs))
        return Pipeline([("record", RecordFitRows()), *base.steps]) if record else base

    return {"logistic_regression": pipeline(), "logistic_balanced": pipeline(class_weight="balanced")}


def test_tune_keeps_first_grid_entry_on_ties(monkeypatch, small):
    X, y = small
    monkeypatch.setattr(nested, "mean_ap", lambda *args: 0.5)
    params, score = tune(logistic_candidates(X)["logistic_regression"], {"model__C": [0.03, 1.0, 3.0]}, X, y, None)
    assert params == {"model__C": 0.03}
    assert score == 0.5


def test_procedure_selects_with_the_prespecified_rule(small):
    X, y = small
    for tuned in (False, True):
        selected, params, threshold, model = run_procedure(logistic_candidates(X), X, y, select_model, tuned=tuned,
                                                          grids=SMALL_GRIDS)
        assert selected in SMALL_GRIDS
        assert bool(params) is tuned
        assert 0 < threshold < 1
        assert model.predict_proba(X).shape == (len(X), 2)


def test_outer_test_rows_are_never_fitted(small):
    X, y = small
    RecordFitRows.fitted_rows.clear()
    folds = nested_cv(logistic_candidates(X, record=True), X, y, select_model, n_jobs=1, grids=SMALL_GRIDS)
    outer_train = [set(X.index[train]) for train, _ in
                   StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=42).split(X, y)]
    assert RecordFitRows.fitted_rows
    assert all(any(rows <= train for train in outer_train) for rows in RecordFitRows.fitted_rows)
    assert list(folds["procedure"]) == ["fixed", "tuned"] * OUTER_FOLDS
    tuned_params = [json.loads(value) for value in folds.loc[folds["procedure"] == "tuned", "parameters"]]
    assert all(set(params) == {"C"} for params in tuned_params)


def test_parallel_and_sequential_runs_match(small):
    X, y = small
    candidates = logistic_candidates(X)
    pd.testing.assert_frame_equal(nested_cv(candidates, X, y, select_model, n_jobs=1, grids=SMALL_GRIDS),
                                  nested_cv(candidates, X, y, select_model, n_jobs=2, grids=SMALL_GRIDS))


def test_summary_reports_means_and_selection_counts():
    folds = pd.DataFrame({"procedure": ["fixed", "tuned", "fixed", "tuned"], "outer_fold": [0, 0, 1, 1],
                          "selected_model": ["a", "a", "b", "a"], "threshold": [0.1, 0.2, 0.3, 0.2],
                          **{metric: [0.5, 0.6, 0.7, 0.6] for metric in nested.NESTED_METRICS}})
    summary = nested_summary(folds)
    assert summary["fixed"]["pr_auc"] == {"mean": pytest.approx(0.6), "std": pytest.approx(0.1)}
    assert summary["fixed"]["selection_counts"] == {"a": 1, "b": 1}
    assert summary["tuned"]["threshold_min"] == summary["tuned"]["threshold_max"] == 0.2
    json.dumps(summary, allow_nan=False)
