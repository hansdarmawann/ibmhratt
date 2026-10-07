"""Structural integrity, failure messages, and reproducible partition checks."""

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PATH, ROOT
from src.data.load_data import load_data
from src.data.validate_data import validate_data


def test_load_and_raw_dataset_integrity(raw_data):
    assert raw_data.shape == (1470, 35)
    assert DATA_PATH.parent == ROOT / "data/raw"
    pd.testing.assert_frame_equal(raw_data, pd.read_csv(DATA_PATH))
    audit = validate_data(raw_data)
    assert audit["duplicate_rows"] == 0
    assert sum(audit["missing_values"].values()) == 0
    assert set(audit["constant_columns"]) == {"EmployeeCount", "Over18", "StandardHours"}


def test_missing_file_and_malformed_csv(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(tmp_path / "missing.csv")
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="Could not parse"):
        load_data(path)


@pytest.mark.parametrize("column,value,message", [
    ("Attrition", "Maybe", "Attrition"),
    ("BusinessTravel", "Never", "Unexpected categorical"),
    ("EmployeeCount", 2, "must be constant"),
    ("Age", -1, "allowed range"),
    ("MonthlyIncome", np.inf, "finite"),
])
def test_invalid_training_values(raw_data, column, value, message):
    broken = raw_data.copy()
    if isinstance(value, float):
        broken[column] = broken[column].astype(float)
    broken.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_data(broken)


def test_missing_target_and_duplicate_rows(raw_data):
    with pytest.raises(ValueError, match="Missing"):
        validate_data(raw_data.drop(columns="Attrition"))
    with pytest.raises(ValueError, match="Duplicate rows"):
        validate_data(pd.concat([raw_data, raw_data.iloc[:1]]))


def test_missing_predictors_are_reported(raw_data):
    frame = raw_data.copy()
    frame.loc[0, "Age"] = np.nan
    with pytest.warns(UserWarning, match="imputation"):
        audit = validate_data(frame)
    assert audit["missing_values"]["Age"] == 1


def test_split_disjoint_stratified_and_reproducible(partitions, raw_data):
    from src.data.load_data import split_data

    X_train, X_test, y_train, y_test = partitions
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.EmployeeNumber).isdisjoint(X_test.EmployeeNumber)
    assert len(X_train) + len(X_test) == len(raw_data)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert "Attrition" not in X_train
    pd.testing.assert_frame_equal(X_train, split_data(raw_data)[0])
