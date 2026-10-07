"""Unit tests for selection rules, report rendering, and holdout uncertainty."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import BOOTSTRAP_METRICS, bootstrap_intervals, subgroup_metrics
from src.models.threshold import select_threshold
from src.models.train import RESULTS_END, RESULTS_START, replace_results_block, select_model


def summaries(**ap):
    return {name: {"pr_auc": {"mean": value, "std": 0.0}} for name, value in ap.items()}


@pytest.mark.parametrize("scores,expected", [
    ({"logistic_regression": 0.645, "logistic_balanced": 0.60, "random_forest": 0.65}, "logistic_regression"),
    ({"logistic_regression": 0.60, "logistic_balanced": 0.645, "random_forest": 0.65}, "logistic_balanced"),
    ({"logistic_regression": 0.60, "logistic_balanced": 0.61, "random_forest": 0.65}, "random_forest"),
    ({"logistic_regression": 0.62, "logistic_balanced": 0.61, "random_forest": 0.60}, "logistic_regression"),
])
def test_select_model_prefers_logistic_within_tolerance(scores, expected):
    selected, reason = select_model(summaries(**scores))
    assert selected == expected
    assert f"selected: {expected}" in reason


def test_select_model_ignores_dummy():
    selected, _ = select_model(summaries(dummy=0.99, logistic_regression=0.5,
                                         logistic_balanced=0.4, random_forest=0.45))
    assert selected == "logistic_regression"


def test_select_threshold_tie_breaks_by_precision_then_threshold():
    by_precision = pd.DataFrame({"threshold": [0.2, 0.3], "f2": [0.5, 0.5], "precision": [0.6, 0.4]})
    assert select_threshold(by_precision) == 0.2
    by_threshold = pd.DataFrame({"threshold": [0.2, 0.3], "f2": [0.5, 0.5], "precision": [0.4, 0.4]})
    assert select_threshold(by_threshold) == 0.3
    with pytest.raises(ValueError):
        select_threshold(by_threshold.iloc[:0])


def test_subgroup_metrics_denominators_and_missing_positives():
    X = pd.DataFrame({"Gender": ["Female", "Female", "Male", "Male"],
                      "MaritalStatus": ["Single", "Married", "Single", "Married"],
                      "Age": [25, 35, 45, 55]})
    y = pd.Series([1, 0, 0, 0])
    table = subgroup_metrics(X, y, [0.9, 0.1, 0.6, 0.2], threshold=0.5)
    assert table.groupby("attribute")["n"].sum().eq(len(X)).all()
    male = table.query("attribute == 'Gender' and group == 'Male'").iloc[0]
    assert male["positive_count"] == 0
    assert pd.isna(male["recall"])
    assert male["false_positive_rate"] == 0.5


def test_replace_results_block_only_changes_marked_region():
    contents = f"intro\n{RESULTS_START}\nold\n{RESULTS_END}\noutro\n"
    updated = replace_results_block(contents, "new\n")
    assert updated == f"intro\n{RESULTS_START}\n\nnew\n\n{RESULTS_END}\noutro\n"
    assert replace_results_block("no markers here", "new\n") == "no markers here"


def test_bootstrap_intervals_are_deterministic_and_bracket_estimates():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    probabilities = np.clip(0.2 + 0.4 * y + rng.normal(0, 0.2, len(y)), 0, 1)
    first = bootstrap_intervals(y, probabilities, 0.4, n_boot=200, seed=1)
    second = bootstrap_intervals(y, probabilities, 0.4, n_boot=200, seed=1)
    pd.testing.assert_frame_equal(first, second)
    assert first["metric"].tolist() == BOOTSTRAP_METRICS
    assert (first["lower"] <= first["estimate"]).all()
    assert (first["estimate"] <= first["upper"]).all()
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_intervals(np.zeros(5), np.full(5, 0.1), 0.5, n_boot=10)
