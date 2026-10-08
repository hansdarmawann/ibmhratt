"""Engineered features are fixed formulas, robust to zeros and missing inputs, and absent from the served model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from attrition.config import FEATURE_COLUMNS
from attrition.features.engineering import ENGINEERED_FEATURES, add_engineered_features
from attrition.features.preprocess import build_pipeline, normalize_missing
from attrition.models.train import engineering_ablation


def test_formulas_handle_zero_denominators_and_missing_values(employee):
    profiles = pd.DataFrame([
        {**employee, "MonthlyIncome": 6000, "JobLevel": 2, "YearsAtCompany": 0, "TotalWorkingYears": 0,
         "NumCompaniesWorked": 0, "YearsSinceLastPromotion": 0, "EnvironmentSatisfaction": 1,
         "JobSatisfaction": 2, "RelationshipSatisfaction": 3, "WorkLifeBalance": 4},
        {**employee, "JobLevel": None, "JobSatisfaction": None},
    ])
    result = add_engineered_features(normalize_missing(profiles))
    first = result.iloc[0]
    assert first["IncomePerJobLevel"] == 3000
    assert first["CompanyShareOfCareer"] == first["YearsPerPriorEmployer"] == first["PromotionWaitShare"] == 0
    assert first["MeanSatisfaction"] == 2.5
    assert np.isnan(result.iloc[1]["IncomePerJobLevel"])
    assert np.isfinite(result.iloc[1]["MeanSatisfaction"])
    assert list(result.columns) == [*profiles.columns, *ENGINEERED_FEATURES]


def test_engineered_pipeline_adds_features_only_when_requested(partitions):
    X, X_test, y, _ = partitions
    served = build_pipeline(X, LogisticRegression(max_iter=2000)).fit(X, y)
    engineered = build_pipeline(X, LogisticRegression(max_iter=2000), engineered=True).fit(X, y)
    assert "engineer" not in served.named_steps
    names = set(engineered.named_steps["preprocess"].get_feature_names_out())
    assert set(ENGINEERED_FEATURES) <= names
    assert not set(ENGINEERED_FEATURES) & set(served.named_steps["preprocess"].get_feature_names_out())
    assert np.isfinite(engineered.predict_proba(X_test[FEATURE_COLUMNS])).all()


@pytest.mark.parametrize("engineered_ap,meets", [([0.62] * 5, True), ([0.62, 0.62, 0.62, 0.5, 0.5], False),
                                                  ([0.605] * 5, False)])
def test_adoption_rule_needs_gain_and_consistent_folds(engineered_ap, meets):
    base = [0.6] * 5
    folds = pd.DataFrame({"model": ["logistic_regression"] * 5 + ["logistic_engineered_features"] * 5,
                          "pr_auc": base + engineered_ap})
    summaries = {"logistic_regression": {"pr_auc": {"mean": np.mean(base)}},
                 "logistic_engineered_features": {"pr_auc": {"mean": np.mean(engineered_ap)}}}
    result = engineering_ablation(summaries, folds)
    assert result["meets_adoption_rule"] is meets
    assert result["adopted"] is False
    assert result["folds_improved"] == sum(value > 0.6 for value in engineered_ap)
