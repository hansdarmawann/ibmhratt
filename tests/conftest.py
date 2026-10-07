"""Small real pipeline fixtures: no pretrained artifact or network required."""

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import FEATURE_COLUMNS
from src.data.load_data import load_data, split_data
from src.features.preprocess import build_pipeline


@pytest.fixture(scope="session")
def raw_data():
    return load_data()


@pytest.fixture(scope="session")
def partitions(raw_data):
    return split_data(raw_data)


@pytest.fixture(scope="session")
def fitted_pipeline(partitions):
    X_train, _, y_train, _ = partitions
    pipeline = build_pipeline(X_train, LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(X_train, y_train)
    pipeline.attrition_metadata_ = {"schema_version": 1, "feature_columns": FEATURE_COLUMNS,
                                     "decision_threshold": 0.35}
    return pipeline


@pytest.fixture(scope="session")
def artifact(tmp_path_factory, fitted_pipeline):
    path = tmp_path_factory.mktemp("artifacts") / "pipeline.joblib"
    joblib.dump(fitted_pipeline, path)
    return path


@pytest.fixture
def employee(partitions):
    return partitions[0][FEATURE_COLUMNS].iloc[0].to_dict()
