"""Test fold isolation, deterministic diagnostics, and capacity arithmetic."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.features.preprocess import build_pipeline
from src.models.diagnostics import (
    calibration_diagnostics,
    capacity_metrics,
    reliability_table,
    stability_diagnostics,
)
from src.models.evaluate import cross_validation_metrics
from src.models.train import select_model


class RecordFitRows(TransformerMixin, BaseEstimator):
    fitted_rows = []

    def fit(self, X, y=None):
        self.fitted_rows.append(set(X.index))
        return self

    def transform(self, X):
        return X


def small_training(partitions):
    X, _, y, _ = partitions
    indices = pd.concat([y[y == 0].head(40), y[y == 1].head(20)]).index
    return X.loc[indices], y.loc[indices]


def test_nested_calibration_fits_only_outer_training_rows(partitions):
    X, y = small_training(partitions)
    base = build_pipeline(X, LogisticRegression(max_iter=1000))
    pipeline = Pipeline([("record", RecordFitRows()), *base.steps])
    RecordFitRows.fitted_rows.clear()
    scores, bins, predictions = calibration_diagnostics(pipeline, X, y)
    # Each outer fold fits the uncalibrated pipeline once and three inner pipelines.
    assert len(RecordFitRows.fitted_rows) == 20
    for fold, (train, validation) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        fits = RecordFitRows.fitted_rows[fold * 4:fold * 4 + 4]
        assert fits[0] == set(X.iloc[train].index)
        for seen in fits:
            assert seen <= set(X.iloc[train].index)
            assert seen.isdisjoint(X.iloc[validation].index)
        assert all(len(seen) < len(train) for seen in fits[1:])
    assert not hasattr(base.named_steps["model"], "classes_")
    assert predictions.row_index.tolist() == X.index.tolist()
    assert bins.groupby("method").n.sum().eq(len(X)).all()
    assert np.isfinite(scores[["brier_score", "log_loss"]].to_numpy()).all()
    again = calibration_diagnostics(pipeline, X, y)
    for actual, expected in zip(again, (scores, bins, predictions), strict=True):
        pd.testing.assert_frame_equal(actual, expected)


def test_reliability_endpoints_and_empty_bins():
    bins = reliability_table([0, 1, 1], [0, 0.55, 1])
    assert len(bins) == 10
    assert bins.iloc[0]["n"] == 1
    assert bins.iloc[-1]["n"] == 1
    assert bins.n.sum() == 3
    assert bins.loc[bins.n == 0, "observed_rate"].isna().all()


def test_stability_is_deterministic_and_reuses_baseline(partitions):
    X, y = small_training(partitions)
    pipelines = {name: build_pipeline(X, LogisticRegression(max_iter=1000, class_weight=weight))
                 for name, weight in [("logistic_regression", None), ("logistic_balanced", "balanced")]}
    first = stability_diagnostics(pipelines, X, y, select_model)
    second = stability_diagnostics(pipelines, X, y, select_model)
    for actual, expected in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(actual, expected)
    assert first[1].seed.tolist() == [42, 43, 44]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    summaries = {name: cross_validation_metrics(model, X, y, cv)[0] for name, model in pipelines.items()}
    selected, _ = select_model(summaries)
    oof = cross_val_predict(pipelines[selected], X, y, cv=cv, method="predict_proba")[:, 1]
    reused = stability_diagnostics(pipelines, X, y, select_model, baseline=(summaries, selected, oof), seeds=(42,))
    pd.testing.assert_frame_equal(reused[1], first[1].iloc[:1])


def test_capacity_rounds_up_and_ties_use_position():
    result = capacity_metrics([0, 1, 1, 0, 0, 0], [0.8, 0.8, 0.9, 0.2, 0.1, 0.0], partition="holdout")
    assert result.selected_count.tolist() == [1, 1, 2]
    assert result.true_positives.tolist() == [1, 1, 1]  # tied row 0 precedes positive row 1
    assert result.iloc[-1].precision == 0.5
    assert result.iloc[-1].recall == 0.5
    assert result.iloc[-1].lift == 1.5
    assert result.partition.eq("holdout").all()
    empty_class = capacity_metrics([0, 0], [0.2, 0.1], partition="training_oof")
    assert empty_class.recall.isna().all()
    with pytest.raises(ValueError):
        capacity_metrics([], [], partition="holdout")


def test_diagnostics_do_not_mutate_serving_pipeline(employee, fitted_pipeline, partitions):
    from src.models.predict import predict

    before = predict(employee, pipeline=fitted_pipeline)
    X, y = small_training(partitions)
    calibration_diagnostics(fitted_pipeline, X, y)
    assert predict(employee, pipeline=fitted_pipeline) == before
