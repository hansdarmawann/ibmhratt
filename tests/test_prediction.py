"""Serialization and the public prediction/threshold contracts."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import evaluate_probabilities
from src.models.predict import load_pipeline, predict
from src.models.threshold import select_threshold, threshold_table


def test_serialization_preserves_probability_and_threshold(employee, fitted_pipeline, artifact):
    loaded = load_pipeline(artifact)
    assert predict(employee, pipeline=fitted_pipeline) == predict(employee, pipeline=loaded)
    result = predict(employee, model_path=artifact)
    assert 0 <= result["attrition_probability"] <= 1
    assert result["decision_threshold"] == 0.35
    assert result["predicted_class"] == ("Yes" if result["attrition_probability"] >= 0.35 else "No")


def test_batch_order_and_nulls(employee, fitted_pipeline):
    second = {**employee, "MonthlyIncome": None, "JobRole": "New role"}
    results = predict(pd.DataFrame([employee, second]), pipeline=fitted_pipeline)
    assert len(results) == 2
    for result, record in zip(results, [employee, second], strict=True):
        single = predict(record, pipeline=fitted_pipeline)
        assert result["predicted_class"] == single["predicted_class"]
        assert result["decision_threshold"] == single["decision_threshold"]
        assert result["attrition_probability"] == pytest.approx(single["attrition_probability"])


def test_stored_operating_threshold_is_used(employee, fitted_pipeline):
    pipeline = copy.deepcopy(fitted_pipeline)
    pipeline.attrition_metadata_["decision_threshold"] = 0.0
    assert predict(employee, pipeline=pipeline)["predicted_class"] == "Yes"
    pipeline.attrition_metadata_["decision_threshold"] = 1.0
    assert predict(employee, pipeline=pipeline)["predicted_class"] == "No"


@pytest.mark.parametrize("change", [{"Age": "35"}, {"Age": True}, {"Age": 35.2},
                                    {"MonthlyIncome": np.inf}, {"OverTime": " "}, {"Attrition": "Yes"}])
def test_invalid_prediction(employee, fitted_pipeline, change):
    with pytest.raises(ValueError):
        predict({**employee, **change}, pipeline=fitted_pipeline)


def test_missing_field_and_model(employee, artifact, tmp_path):
    employee.pop("Age")
    with pytest.raises(ValueError, match="Missing required"):
        predict(employee, model_path=artifact)
    with pytest.raises(FileNotFoundError, match="Run"):
        load_pipeline(tmp_path / "absent.joblib")


def test_threshold_tradeoff_and_known_confusion_counts():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.3, 0.8])
    table = threshold_table(y, probabilities, [0.2, 0.5])
    assert select_threshold(table) == 0.2
    result = evaluate_probabilities(y, probabilities, 0.5)
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["recall"] == 0.5
    assert result["precision"] == 1.0
