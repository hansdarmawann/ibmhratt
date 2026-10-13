"""Metric inputs, undefined cases, and training-data exclusions fail loudly or are reported explicitly."""

import numpy as np
import pandas as pd
import pytest

from attrition.config import ID_COLUMN
from attrition.features.preprocess import feature_exclusions
from attrition.models.evaluate import bootstrap_intervals, evaluate_probabilities

Y = np.array([0, 0, 1, 1])


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_threshold_must_be_a_probability(threshold):
    with pytest.raises(ValueError, match="Decision threshold"):
        evaluate_probabilities(Y, [0.1, 0.4, 0.3, 0.8], threshold)


@pytest.mark.parametrize("probabilities", [
    [0.1, 0.4, 0.3],
    [0.1, 0.4, 0.3, 0.8, 0.9],
    [0.1, np.nan, 0.3, 0.8],
    [0.1, np.inf, 0.3, 0.8],
    [0.1, 1.0001, 0.3, 0.8],
    [-0.0001, 0.4, 0.3, 0.8],
    [[0.1, 0.4], [0.3, 0.8]],
])
def test_probabilities_must_be_finite_in_range_and_aligned(probabilities):
    with pytest.raises(ValueError, match="one finite probability"):
        evaluate_probabilities(Y, probabilities, 0.5)


@pytest.mark.parametrize(("threshold", "predicted_positive"), [(0.0, 4), (0.3, 3), (0.8, 1), (1.0, 0)])
def test_threshold_is_inclusive_and_endpoints_are_valid(threshold, predicted_positive):
    result = evaluate_probabilities(Y, [0.1, 0.4, 0.3, 0.8], threshold)
    assert result["false_positives"] + result["true_positives"] == predicted_positive
    assert result["n"] == 4
    assert result["positive_count"] == 2
    assert result["threshold"] == threshold


def test_exact_threshold_value_counts_as_positive():
    result = evaluate_probabilities(np.array([1, 0]), [0.5, 0.5], 0.5)
    assert result["confusion_matrix"] == [[0, 1], [0, 1]]


def test_single_class_targets_leave_ranking_metrics_undefined_not_zero():
    result = evaluate_probabilities(np.array([0, 0, 0]), [0.1, 0.2, 0.9], 0.5)
    assert result["roc_auc"] is None and result["pr_auc"] is None
    assert result["recall"] == 0.0 and result["precision"] == 0.0
    assert result["confusion_matrix"] == [[2, 1], [0, 0]]
    assert result["positive_count"] == 0


def test_known_metric_values_for_a_small_example():
    result = evaluate_probabilities(Y, [0.1, 0.4, 0.3, 0.8], 0.35)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["precision"] == 0.5 and result["recall"] == 0.5 and result["f1"] == 0.5
    assert result["roc_auc"] == 0.75
    assert result["pr_auc"] == pytest.approx(0.8333333333)
    assert result["brier_score"] == pytest.approx(np.mean((np.array([0.1, 0.4, 0.3, 0.8]) - Y) ** 2))


@pytest.mark.parametrize("arguments", [{"n_boot": 0}, {"confidence": 0}, {"confidence": 1}, {"confidence": 1.5}])
def test_bootstrap_parameters_are_validated(arguments):
    with pytest.raises(ValueError, match="at least one replicate"):
        bootstrap_intervals(np.array([0, 1] * 5), np.linspace(0.1, 0.9, 10), 0.5, **{"n_boot": 5, **arguments})


def test_bootstrap_seed_changes_the_resamples():
    y = np.array([0] * 30 + [1] * 10)
    probabilities = np.clip(np.linspace(0, 1, 40) + 0.1 * y, 0, 1)
    first = bootstrap_intervals(y, probabilities, 0.5, n_boot=200, seed=1)
    other = bootstrap_intervals(y, probabilities, 0.5, n_boot=200, seed=2)
    assert not first[["lower", "upper"]].equals(other[["lower", "upper"]])
    pd.testing.assert_series_equal(first["estimate"], other["estimate"])


def test_feature_exclusions_confirm_constants_on_the_training_rows(raw_data):
    reasons = feature_exclusions(raw_data)
    assert set(reasons) == {ID_COLUMN, "EmployeeCount", "Over18", "StandardHours"}
    assert all(reason for reason in reasons.values())


@pytest.mark.parametrize(("column", "value"), [("StandardHours", 40), ("Over18", "N"), ("EmployeeCount", 2)])
def test_a_changed_constant_blocks_its_exclusion(raw_data, column, value):
    changed = raw_data.copy()
    changed.loc[changed.index[0], column] = value
    with pytest.raises(ValueError, match=f"Expected {column} to be constant"):
        feature_exclusions(changed)
