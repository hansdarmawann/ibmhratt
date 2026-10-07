"""Test the real leakage boundary and inference robustness."""

import numpy as np
import pandas as pd

from src.config import CONSTANT_COLUMNS, ID_COLUMN


def test_excluded_features_and_readable_names(fitted_pipeline):
    names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    for excluded in [*CONSTANT_COLUMNS, ID_COLUMN, "Attrition"]:
        assert not any(name.startswith(excluded) for name in names)
    assert "OverTime_Yes" in names
    assert "MonthlyIncome" in names


def test_imputation_statistics_are_training_only(partitions, fitted_pipeline):
    X_train, X_test, _, _ = partitions
    processor = fitted_pipeline.named_steps["preprocess"]
    numeric_names = processor.transformers_[0][2]
    actual = processor.named_transformers_["numeric"].named_steps["imputer"].statistics_
    np.testing.assert_allclose(actual, X_train[numeric_names].median())
    before = actual.copy()
    adversarial_holdout = X_test.copy()
    adversarial_holdout["MonthlyIncome"] = 1e12
    fitted_pipeline.predict_proba(adversarial_holdout)
    np.testing.assert_array_equal(before, actual)


def test_unknown_categories_missing_values_and_no_mutation(partitions, fitted_pipeline):
    sample = partitions[1].head(2).copy()
    sample.loc[sample.index[0], "JobRole"] = "Unseen Role"
    sample.loc[sample.index[1], "MonthlyIncome"] = np.nan
    sample.loc[sample.index[1], "EducationField"] = None
    before = sample.copy(deep=True)
    proba = fitted_pipeline.predict_proba(sample)
    assert proba.shape == (2, 2)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    pd.testing.assert_frame_equal(sample, before)
