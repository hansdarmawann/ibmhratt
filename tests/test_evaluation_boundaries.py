"""Behavioral leakage checks using synthetic rows and real sklearn selection."""

from dataclasses import FrozenInstanceError
from functools import partial
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from attrition.config import CATEGORIES, CONSTANT_COLUMNS, CV_FOLDS, ID_COLUMN, NUMERIC_BOUNDS, TARGET
from attrition.data.load_data import split_data
from attrition.features.preprocess import build_pipeline
from attrition.models import diagnostics, nested, train
from attrition.models.evaluate import SCORING, bootstrap_intervals, evaluate_probabilities, subgroup_metrics

SMALL_GRIDS = {name: {"model__C": [0.1, 1.0]} for name in ("logistic_regression", "logistic_balanced")}


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(17)
    count = 120
    frame = pd.DataFrame({
        name: rng.integers(lower, min(upper if upper is not None else lower + 20, lower + 20) + 1, count)
        for name, (lower, upper) in NUMERIC_BOUNDS.items()
    })
    for name, categories in CATEGORIES.items():
        frame[name] = rng.choice(categories, count)
    for name, value in CONSTANT_COLUMNS.items():
        frame[name] = value
    frame[ID_COLUMN] = np.arange(1, count + 1)
    frame[TARGET] = np.where(np.arange(count) % 4 == 0, "Yes", "No")
    return frame


def small_candidates(X):
    return {name: build_pipeline(X, LogisticRegression(max_iter=1000, random_state=42, class_weight=weight))
            for name, weight in [("logistic_regression", None), ("logistic_balanced", "balanced")]}


class FitGuard(TransformerMixin, BaseEstimator):
    """Check every sklearn clone's fit, before any learned preprocessing runs."""

    allowed: ClassVar[list[set]] = []
    seen: ClassVar[list[set]] = []
    labels: ClassVar[pd.Series] = pd.Series(dtype=int)

    def fit(self, X, y=None):
        rows = set(X.index)
        assert rows in self.allowed, "preprocessing fitted outside its permitted fold"
        # cross_val_predict encodes classification targets as an ndarray.
        np.testing.assert_array_equal(y, self.labels.loc[X.index].to_numpy())
        self.seen.append(rows)
        return self

    def transform(self, X):
        return X


def test_split_precedes_training_only_exploration(monkeypatch, tmp_path, synthetic_data):
    source = tmp_path / "synthetic.csv"
    synthetic_data.to_csv(source, index=False)
    expected = split_data(synthetic_data)
    calls = []

    def explore(X, y, *_):
        pd.testing.assert_frame_equal(X, expected[0])
        pd.testing.assert_series_equal(y, expected[2])
        assert set(X.index).isdisjoint(expected[1].index)
        calls.append("training exploration")

    monkeypatch.setattr(train, "save_exploration", explore)
    paths = train.create_run_paths(tmp_path, "split")
    parts = train.prepare_data(source, paths)
    assert calls == ["training exploration"]
    pd.testing.assert_frame_equal(parts.X_test, expected[1])
    pd.testing.assert_series_equal(parts.y_test, expected[3])
    manifest = pd.read_csv(paths.processed / "split_manifest.csv")
    assert set(manifest.loc[manifest.partition == "test", "row_index"]) == set(parts.X_test.index)


def test_comparison_ablations_and_oof_fit_only_their_training_folds(monkeypatch, tmp_path, synthetic_data):
    X, X_test, y, _ = split_data(synthetic_data)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42)
    allowed = [set(X.iloc[fit].index) for fit, _ in cv.split(X, y)]
    monkeypatch.setattr(FitGuard, "allowed", allowed)
    monkeypatch.setattr(FitGuard, "seen", [])
    monkeypatch.setattr(FitGuard, "labels", y)

    def guarded_pipeline(*args, **kwargs):
        pipeline = build_pipeline(*args, **kwargs)
        return Pipeline([("guard", FitGuard()), *pipeline.steps])

    monkeypatch.setattr(train, "build_pipeline", guarded_pipeline)
    paths = train.create_run_paths(tmp_path, "folds")
    pipelines, summaries, engineering = train.compare_models(X, y, cv, paths)
    assert len(FitGuard.seen) == 8 * CV_FOLDS  # Five candidates and three ablations.
    assert engineering["adopted"] is False
    expected, _ = train.select_model({name: summaries[name] for name in pipelines})
    # Even unbeatable diagnostic ablations must remain outside model selection.
    for name in summaries.keys() - pipelines.keys():
        summaries[name]["pr_auc"]["mean"] = 1.0
    point = train.choose_operating_point(pipelines, summaries, X, y, cv, paths)
    assert point.selected == expected
    assert len(FitGuard.seen) == 9 * CV_FOLDS
    assert all(rows.isdisjoint(X_test.index) for rows in FitGuard.seen)
    oof = pd.read_csv(paths.processed / "oof_predictions.csv")
    assert oof.row_index.tolist() == X.index.tolist()
    np.testing.assert_array_equal(oof.target, y)
    assert all(not hasattr(p.named_steps["model"], "classes_") for p in pipelines.values())


@pytest.mark.parametrize("holdout_ap", [0.0, 1.0])
def test_train_selectors_cannot_access_holdout_or_reselect_after_scoring(
    monkeypatch, tmp_path, synthetic_data, holdout_ap,
):
    X, X_test, y, y_test = split_data(synthetic_data)
    state = {"evaluated": False}
    stages, threshold_calls, selector_calls = [], [], []

    class GuardedPartitions:
        X_train, y_train = X, y

        @property
        def X_test(self):
            assert state["evaluated"], "holdout features accessed before selection finished"
            return X_test

        @property
        def y_test(self):
            assert state["evaluated"], "holdout labels accessed before selection finished"
            return y_test

    parts = GuardedPartitions()
    monkeypatch.setattr(train, "prepare_data", lambda *_: parts)
    monkeypatch.setattr(train, "candidate_pipelines", small_candidates)
    monkeypatch.setattr(train, "nested_cv", partial(nested.nested_cv, n_jobs=1, grids=SMALL_GRIDS))

    def check_training(features, labels):
        assert not state["evaluated"], "selection ran after holdout evaluation"
        pd.testing.assert_frame_equal(features, X)
        pd.testing.assert_series_equal(labels, y)

    real_compare, real_choose, real_diagnostics = (
        train.compare_models, train.choose_operating_point, train.training_diagnostics,
    )

    def compare(features, labels, *args):
        check_training(features, labels)
        stages.append("compare")
        return real_compare(features, labels, *args)

    def choose(pipelines, summaries, features, labels, *args):
        check_training(features, labels)
        stages.append("choose")
        point = real_choose(pipelines, summaries, features, labels, *args)
        state["point"] = point
        return point

    def diagnose(pipelines, summaries, point, features, labels, *args):
        check_training(features, labels)
        stages.append("diagnostics")
        return real_diagnostics(pipelines, summaries, point, features, labels, *args)

    real_selector, real_threshold_table = train.select_model, train.threshold_table

    def selector(summaries):
        assert not state["evaluated"], "holdout results reached the model selector"
        assert set(summaries) == set(SMALL_GRIDS), "diagnostics reached candidate selection"
        assert all(set(metrics) <= set(SCORING) for metrics in summaries.values())
        selector_calls.append(summaries)
        return real_selector(summaries)

    def threshold_table(labels, probabilities):
        assert not state["evaluated"], "threshold retuned after seeing holdout metrics"
        assert set(labels.index) <= set(y.index)
        assert set(labels.index).isdisjoint(y_test.index)
        pd.testing.assert_series_equal(labels, y.loc[labels.index])
        assert len(probabilities) == len(labels)
        threshold_calls.append(set(labels.index))
        return real_threshold_table(labels, probabilities)

    monkeypatch.setattr(train, "compare_models", compare)
    monkeypatch.setattr(train, "choose_operating_point", choose)
    monkeypatch.setattr(train, "training_diagnostics", diagnose)
    monkeypatch.setattr(train, "select_model", selector)
    for module in (train, diagnostics, nested):
        monkeypatch.setattr(module, "threshold_table", threshold_table)

    def evaluate(pipelines, point, received, paths):
        assert stages == ["compare", "choose", "diagnostics"]
        assert received is parts
        assert point is state["point"]
        with pytest.raises(FrozenInstanceError):
            point.threshold = 0.99
        state["evaluated"] = True
        stages.append("holdout")
        pd.testing.assert_series_equal(received.y_test, y_test)
        metrics = {"pr_auc": holdout_ap, "recall": holdout_ap, "f1": holdout_ap}
        return train.HoldoutResults({}, {}, metrics, pd.DataFrame(), pd.DataFrame())

    # Keep actual selection/diagnostics; stub only holdout scoring and publication.
    monkeypatch.setattr(train, "evaluate_holdout", evaluate)
    monkeypatch.setattr(train, "build_metadata", lambda *_: {})
    monkeypatch.setattr(train, "build_report", lambda metadata, summaries, point, *args: {
        "selected_model": point.selected, "selected_threshold": point.threshold,
    })
    monkeypatch.setattr(train, "write_bundle", lambda *_: ("", {}))
    monkeypatch.setattr(train, "publish_bundle", lambda *_: None)
    monkeypatch.setattr(train, "export_snapshot", lambda *_: None)
    report = train.train(output_root=tmp_path)
    assert stages == ["compare", "choose", "diagnostics", "holdout"]
    assert len(selector_calls) == 13  # Main choice, two repeated seeds, ten nested procedures.
    assert len(threshold_calls) == 14  # Also includes the seed-42 diagnostic reuse.
    assert report == {"selected_model": state["point"].selected, "selected_threshold": state["point"].threshold}


def test_nested_tuner_and_threshold_receive_only_the_current_outer_training_rows(
    monkeypatch, synthetic_data,
):
    X, X_test, y, _ = split_data(synthetic_data)
    candidates = {name: Pipeline([("guard", FitGuard()), *pipeline.steps])
                  for name, pipeline in small_candidates(X).items()}
    monkeypatch.setattr(FitGuard, "seen", [])
    monkeypatch.setattr(FitGuard, "labels", y)
    monkeypatch.setattr(FitGuard, "allowed", [])
    state, procedures, thresholds, scores = {}, [], [], []
    real_procedure, real_mean_ap = nested.run_procedure, nested.mean_ap
    real_threshold_table, real_evaluate = nested.threshold_table, nested.evaluate_probabilities
    outer = list(StratifiedKFold(nested.OUTER_FOLDS, shuffle=True, random_state=42).split(X, y))

    def procedure(pipelines, features, labels, selector, *, tuned, seed, grids):
        fold = len(procedures) // 2
        fit, validation = outer[fold]
        pd.testing.assert_frame_equal(features, X.iloc[fit])
        pd.testing.assert_series_equal(labels, y.iloc[fit])
        state.update(labels=labels, validation=y.iloc[validation])
        inner = StratifiedKFold(nested.TUNING_FOLDS if tuned else CV_FOLDS, shuffle=True, random_state=seed)
        FitGuard.allowed = [set(features.index), *[set(features.iloc[rows].index)
                                                  for rows, _ in inner.split(features, labels)]]
        procedures.append((fold, tuned))
        return real_procedure(pipelines, features, labels, selector, tuned=tuned, seed=seed, grids=grids)

    def mean_ap(pipeline, features, labels, cv):
        pd.testing.assert_series_equal(labels, state["labels"])
        assert features.index.equals(labels.index)
        scores.append(set(labels.index))
        return real_mean_ap(pipeline, features, labels, cv)

    def threshold_table(labels, probabilities):
        pd.testing.assert_series_equal(labels, state["labels"])
        assert set(labels.index).isdisjoint(state["validation"].index)
        thresholds.append(set(labels.index))
        return real_threshold_table(labels, probabilities)

    def selector(summaries):
        assert set(summaries) == set(SMALL_GRIDS)
        assert all(set(metrics) == {"pr_auc"} and set(metrics["pr_auc"]) == {"mean"}
                   for metrics in summaries.values())
        return train.select_model(summaries)

    def evaluate(labels, probabilities, threshold):
        pd.testing.assert_series_equal(labels, state["validation"])
        assert len(thresholds) == len(procedures)
        return real_evaluate(labels, probabilities, threshold)

    monkeypatch.setattr(nested, "run_procedure", procedure)
    monkeypatch.setattr(nested, "mean_ap", mean_ap)
    monkeypatch.setattr(nested, "threshold_table", threshold_table)
    monkeypatch.setattr(nested, "evaluate_probabilities", evaluate)
    result = nested.nested_cv(candidates, X, y, selector, n_jobs=1, grids=SMALL_GRIDS)
    assert len(result) == len(procedures) == len(thresholds) == 2 * nested.OUTER_FOLDS
    assert len(scores) == 6 * nested.OUTER_FOLDS  # Two fixed candidates plus two grids of size two.
    assert FitGuard.seen
    assert all(rows.isdisjoint(X_test.index) for rows in FitGuard.seen)


def test_real_holdout_evaluation_never_calls_selection(monkeypatch, tmp_path, synthetic_data):
    X, X_test, y, y_test = split_data(synthetic_data)
    pipelines = small_candidates(X)
    point = train.OperatingPoint("logistic_regression", "frozen", np.full(len(y), 0.25),
                                 pd.DataFrame(), 0.35)

    def forbidden(*args, **kwargs):
        pytest.fail("holdout evaluation called a selector or tuner")

    for name in ("select_model", "select_threshold", "threshold_table", "compare_models",
                 "choose_operating_point", "training_diagnostics"):
        monkeypatch.setattr(train, name, forbidden)
    monkeypatch.setattr(nested, "tune", forbidden)
    monkeypatch.setattr(train, "save_evaluation_plots", lambda *_: None)
    monkeypatch.setattr(train, "explain_models", lambda *_: None)
    monkeypatch.setattr(train, "bootstrap_intervals", partial(bootstrap_intervals, n_boot=10))
    monkeypatch.setattr(train, "subgroup_metrics", partial(subgroup_metrics, n_boot=10))
    paths = train.create_run_paths(tmp_path, "holdout")
    result = train.evaluate_holdout(pipelines, point, train.Partitions(X, X_test, y, y_test), paths)
    assert set(result.at_default_threshold) == set(pipelines)
    assert all(metrics["threshold"] == 0.5 for metrics in result.at_default_threshold.values())
    assert result.selected_metrics == evaluate_probabilities(y_test, result.probabilities[point.selected], 0.35)
    assert point.selected == "logistic_regression" and point.threshold == 0.35
