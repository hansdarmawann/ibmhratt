"""Tolerant report comparison used by CI's committed-results check."""

import pandas as pd

from scripts.compare_reports import csv_differences, json_differences


def test_json_differences_tolerate_float_noise_only():
    old = {"ap": 0.584, "model": "logistic_regression", "flags": [True, 1], "nested": {"n": 47}}
    assert json_differences(old, {**old, "ap": 0.584 + 1e-12}) == []
    assert json_differences(old, {**old, "ap": 0.59}) == ["/ap: 0.584 != 0.59"]
    assert json_differences(old, {**old, "flags": [1, 1]}) == ["/flags[0]: True != 1"]
    assert json_differences(old, {k: v for k, v in old.items() if k != "nested"}) == ["/: keys differ"]


def test_csv_differences_compare_numbers_with_tolerance():
    old = pd.DataFrame({"metric": ["ap", "f1"], "value": [0.5, None]})
    assert csv_differences(old, old.assign(value=[0.5 + 1e-12, None])) == []
    assert csv_differences(old, old.assign(value=[0.6, None])) == ["value: numeric values differ"]
    assert csv_differences(old, old.assign(metric=["ap", "f2"])) == ["metric: values differ"]
    assert csv_differences(old, old.rename(columns={"value": "v"})) == ["columns or shape differ"]
