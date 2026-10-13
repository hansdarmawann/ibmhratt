"""Input boundaries for the shared validators and the public prediction function."""

import numpy as np
import pandas as pd
import pytest

from attrition.config import CATEGORIES, FEATURE_COLUMNS, NUMERIC_BOUNDS, TEST_SIZE
from attrition.data.load_data import split_data
from attrition.data.validate_data import validate_data, validate_features
from attrition.models.predict import predict


class FixedProbability:
    """A model stand-in that returns chosen probabilities and records the columns it was given."""

    def __init__(self, probabilities, threshold):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.attrition_metadata_ = {"decision_threshold": threshold, "run_id": "fixed"}
        self.seen_columns = None

    def predict_proba(self, frame):
        self.seen_columns = list(frame.columns)
        return np.column_stack([1 - self.probabilities, self.probabilities])


def one_row(employee, **changes):
    return pd.DataFrame([{**employee, **changes}])


def boundary_cases():
    for name, (lower, upper) in NUMERIC_BOUNDS.items():
        yield pytest.param(name, lower, True, id=f"{name}-min")
        yield pytest.param(name, lower - 1, False, id=f"{name}-below-min")
        if upper is not None:
            yield pytest.param(name, upper, True, id=f"{name}-max")
            yield pytest.param(name, upper + 1, False, id=f"{name}-above-max")


@pytest.mark.parametrize(("field", "value", "accepted"), list(boundary_cases()))
def test_numeric_bounds_are_inclusive(employee, field, value, accepted):
    frame = one_row(employee, **{field: value})
    if accepted:
        validate_features(frame)
    else:
        with pytest.raises(ValueError, match=f"{field} is outside its allowed range"):
            validate_features(frame)


@pytest.mark.parametrize(("value", "message"), [
    (np.nan, None), (None, None), (35.0, None),
    (35.5, "whole numbers"), (np.inf, "finite"), (-np.inf, "finite"),
    ("35", "not strings"), (True, "not strings"), (False, "not strings"),
])
def test_numeric_value_kinds(employee, value, message):
    frame = one_row(employee, Age=value)
    if message is None:
        validate_features(frame)
    else:
        with pytest.raises(ValueError, match=message):
            validate_features(frame)


@pytest.mark.parametrize("value", [5, 1.5, True, "", "   ", "\t", b"bytes"])
def test_categories_must_be_nonempty_strings_or_null(employee, value):
    with pytest.raises(ValueError, match="nonempty strings or null"):
        validate_features(one_row(employee, Gender=value))


def test_novel_categories_depend_on_training_mode(employee):
    novel = one_row(employee, JobRole="Brand new role")
    validate_features(novel)
    with pytest.raises(ValueError, match="Unexpected categorical values for JobRole"):
        validate_features(novel, training=True)
    for field, values in CATEGORIES.items():
        validate_features(one_row(employee, **{field: values[0]}), training=True)


def test_structural_input_errors(employee):
    for bad in (None, [employee], employee, pd.DataFrame(), pd.DataFrame(columns=FEATURE_COLUMNS)):
        with pytest.raises(ValueError, match="nonempty pandas DataFrame"):
            validate_features(bad)
    duplicated = pd.concat([one_row(employee), one_row(employee)[["Age"]]], axis=1)
    with pytest.raises(ValueError, match="Duplicate column"):
        validate_features(duplicated)
    with pytest.raises(ValueError, match="Missing required feature columns: Age, OverTime"):
        validate_features(one_row(employee).drop(columns=["OverTime", "Age"]))


def test_one_bad_row_in_a_batch_is_enough_to_fail(employee):
    frame = pd.DataFrame([employee, {**employee, "Age": 17}, employee])
    with pytest.raises(ValueError, match="Age is outside"):
        validate_features(frame)


def test_target_label_and_constants_are_not_features_for_training_validation(raw_data):
    with pytest.raises(ValueError, match="unexpected"):
        validate_data(raw_data.assign(Extra=1))
    with pytest.raises(ValueError, match="Entirely missing predictor"):
        validate_data(raw_data.assign(Age=np.nan))
    ids = raw_data.copy()
    ids.loc[1, "EmployeeNumber"] = ids.loc[0, "EmployeeNumber"]
    with pytest.raises(ValueError, match="EmployeeNumber"):
        validate_data(ids)
    for bad_id in (0, -4):
        ids = raw_data.copy()
        ids.loc[0, "EmployeeNumber"] = bad_id
        with pytest.raises(ValueError, match="EmployeeNumber"):
            validate_data(ids)


def test_training_data_needs_both_classes(raw_data):
    only_no = raw_data[raw_data["Attrition"] == "No"]
    with pytest.raises(ValueError, match="both Attrition classes"):
        validate_data(only_no)
    with pytest.raises(ValueError, match=r"Yes. and .No."):
        validate_data(raw_data.assign(Attrition=raw_data["Attrition"].where(raw_data.index != 0, None)))


@pytest.mark.parametrize("data", [None, [1, 2], "text", 5, ("Age", 30), pd.Series({"Age": 30})])
def test_predict_accepts_only_dictionaries_and_data_frames(data, fitted_pipeline):
    with pytest.raises(ValueError, match="dictionary or pandas DataFrame"):
        predict(data, pipeline=fitted_pipeline)


def test_predict_rejects_empty_frames_and_unknown_columns(employee, fitted_pipeline):
    with pytest.raises(ValueError, match="nonempty"):
        predict(pd.DataFrame(columns=FEATURE_COLUMNS), pipeline=fitted_pipeline)
    for extra in ("Attrition", "Salary", "attrition"):
        with pytest.raises(ValueError, match="Unexpected prediction fields"):
            predict({**employee, extra: 1}, pipeline=fitted_pipeline)


def test_predict_allows_identifier_and_constant_columns(employee, fitted_pipeline):
    baseline = predict(employee, pipeline=fitted_pipeline)
    extended = {**employee, "EmployeeNumber": 12345, "EmployeeCount": 1, "Over18": "Y", "StandardHours": 80}
    assert predict(extended, pipeline=fitted_pipeline) == baseline


def test_predict_does_not_mutate_its_input(employee, fitted_pipeline):
    record = {**employee, "Age": None}
    frame = pd.DataFrame([record, employee])
    frame_before, record_before = frame.copy(deep=True), dict(record)
    predict(record, pipeline=fitted_pipeline)
    predict(frame, pipeline=fitted_pipeline)
    pd.testing.assert_frame_equal(frame, frame_before)
    assert record == record_before


def test_predict_is_independent_of_column_order_and_index(fitted_pipeline, partitions):
    rows = partitions[1][FEATURE_COLUMNS].head(4)
    shuffled = rows[rows.columns[::-1]].iloc[[2, 0, 3, 1]]
    expected = {index: result["attrition_probability"]
                for index, result in zip(rows.index, predict(rows, pipeline=fitted_pipeline), strict=True)}
    actual = {index: result["attrition_probability"]
              for index, result in zip(shuffled.index, predict(shuffled, pipeline=fitted_pipeline), strict=True)}
    assert actual.keys() == expected.keys()
    for index, probability in expected.items():
        assert actual[index] == pytest.approx(probability)


def test_model_receives_exactly_the_feature_columns_in_training_order(employee):
    model = FixedProbability([0.5], threshold=0.5)
    predict({**employee, "EmployeeNumber": 1, "Over18": "Y"}, pipeline=model)
    assert model.seen_columns == FEATURE_COLUMNS


@pytest.mark.parametrize(("probability", "threshold", "label"), [
    (0.5, 0.5, "Yes"), (0.4999999, 0.5, "No"), (0.5000001, 0.5, "Yes"),
    (0.0, 0.0, "Yes"), (1.0, 1.0, "Yes"), (0.0, 1.0, "No"), (0.999, 1.0, "No"),
])
def test_label_uses_greater_or_equal_comparison_with_the_frozen_threshold(employee, probability, threshold, label):
    result = predict(employee, pipeline=FixedProbability([probability], threshold))
    assert result["predicted_class"] == label
    assert result["decision_threshold"] == threshold
    assert result["attrition_probability"] == probability
    assert result["run_id"] == "fixed"


def test_batch_labels_follow_row_order(employee):
    model = FixedProbability([0.9, 0.1, 0.5, 0.2], threshold=0.5)
    results = predict(pd.DataFrame([employee] * 4), pipeline=model)
    assert [r["predicted_class"] for r in results] == ["Yes", "No", "Yes", "No"]
    assert [r["attrition_probability"] for r in results] == [0.9, 0.1, 0.5, 0.2]


def test_split_sizes_follow_the_configured_fraction(partitions, raw_data):
    X_train, X_test, y_train, y_test = partitions
    assert len(X_test) == round(len(raw_data) * TEST_SIZE) == 294
    assert len(X_train) == len(raw_data) - 294
    assert y_train.sum() + y_test.sum() == (raw_data["Attrition"] == "Yes").sum()
    assert set(y_train) == set(y_test) == {0, 1}
    assert list(X_train.columns) == list(X_test.columns)


@pytest.mark.parametrize(("yes", "no", "accepted"), [(10, 40, True), (9, 41, False), (40, 9, False)])
def test_split_requires_ten_rows_per_class(yes, no, accepted):
    frame = pd.DataFrame({"Attrition": ["Yes"] * yes + ["No"] * no, "Age": range(yes + no)})
    if accepted:
        X_train, X_test, _, y_test = split_data(frame)
        assert len(X_train) + len(X_test) == yes + no
        assert y_test.sum() == round(yes * TEST_SIZE)
    else:
        with pytest.raises(ValueError, match="at least 10 rows"):
            split_data(frame)
