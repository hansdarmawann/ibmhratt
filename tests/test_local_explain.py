"""Check true additive outputs for both linear and tree explanations."""

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from attrition.config import FEATURE_COLUMNS
from attrition.features.preprocess import build_pipeline
from attrition.models.local_explain import check_additive, explain_profile, explained_probability
from attrition.models.predict import predict


def test_linear_profile_aggregates_and_reconstructs(employee, fitted_pipeline, partitions):
    background = partitions[0][FEATURE_COLUMNS].head(30)
    result = explain_profile(fitted_pipeline, employee, background)
    assert result["status"] == "generated"
    assert result["units"] == "log-odds of Attrition=Yes"
    assert len(result["contributions"]) == 10
    assert all(row["feature"] in FEATURE_COLUMNS for row in result["contributions"])
    total = sum(row["contribution"] for row in result["contributions"]) + result["other_contribution"]
    np.testing.assert_allclose(total, result["contribution_sum"])
    assert explained_probability(result["base_value"], total, result["units"]) == pytest.approx(
        predict(employee, pipeline=fitted_pipeline)["attrition_probability"])
    changed = {**employee, "Age": None, "JobRole": "Novel role"}
    other = explain_profile(fitted_pipeline, changed, background)
    assert other["status"] == "generated"
    assert other["probability"] == pytest.approx(predict(changed, pipeline=fitted_pipeline)["attrition_probability"])


@pytest.mark.parametrize("estimator", [RandomForestClassifier(n_estimators=8, max_depth=3, random_state=42),
                                      HistGradientBoostingClassifier(max_iter=8, max_leaf_nodes=4, random_state=42)])
def test_tree_explanation_already_uses_probability(estimator, partitions, employee):
    X, _, y, _ = partitions
    pipeline = build_pipeline(X, estimator, scale=False).fit(X, y)
    pipeline.attrition_metadata_ = {"decision_threshold": 0.2}
    result = explain_profile(pipeline, employee, X[FEATURE_COLUMNS].head(30))
    assert result["status"] == "generated"
    assert result["units"] == "probability of Attrition=Yes"
    assert result["base_value"] + result["contribution_sum"] == pytest.approx(result["probability"], abs=1e-6)


def test_optional_failure_does_not_break_prediction(monkeypatch, fitted_pipeline, employee, partitions):
    def unavailable(*args):
        raise ImportError("SHAP absent")

    monkeypatch.setattr("attrition.models.local_explain.shap_values", unavailable)
    result = explain_profile(fitted_pipeline, employee, partitions[0].head(5))
    assert result["status"] == "unavailable"
    assert np.isfinite(predict(employee, pipeline=fitted_pipeline)["attrition_probability"])
    with pytest.raises(ValueError, match="units"):
        explained_probability(0, 0, "unspecified")


def test_additivity_failure_is_a_domain_error():
    check_additive(np.array([0.5]), np.array([0.5 + 1e-8]))
    with pytest.raises(ValueError, match="reconstruct"):
        check_additive(np.array([0.5]), np.array([0.6]))
