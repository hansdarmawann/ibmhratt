"""Small real pipeline fixtures: no pretrained artifact or network required."""

import copy
from uuid import uuid4

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.config import FEATURE_COLUMNS
from src.data.load_data import load_data, split_data
from src.features.preprocess import build_pipeline
from src.models.artifacts import publish_bundle, seal_bundle, write_json
from src.models.evaluate import evaluate_probabilities


@pytest.fixture(scope="session", autouse=True)
def limit_threads():
    with threadpool_limits(limits=1):
        yield


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


@pytest.fixture
def bundle_factory(tmp_path, fitted_pipeline, partitions):
    """Small real bundles; independent of a production artifact or generated reports."""
    def create(*, threshold=0.35, activate=True):
        directory = tmp_path / "models/runs" / str(uuid4())
        directory.mkdir(parents=True)
        pipeline = copy.deepcopy(fitted_pipeline)
        metadata = {**pipeline.attrition_metadata_, "schema_version": 2, "run_id": directory.name,
                    "decision_threshold": threshold, "selected_model": "logistic_regression"}
        pipeline.attrition_metadata_ = metadata
        joblib.dump(pipeline, directory / "pipeline.joblib")
        write_json(metadata, directory / "metadata.json")
        X_train, X_test, _, y_test = partitions
        metrics = evaluate_probabilities(y_test, pipeline.predict_proba(X_test)[:, 1], threshold)
        report = {**metadata, "selected_threshold": threshold, "selected_model_metrics": metrics,
                  "selection_reason": "Test fixture", "shap": {"status": "not_requested"},
                  "stability_diagnostics": {"main_seed": 42}, "sensitive_ablation": {"cv_ap_change": 0.0}}
        write_json(report, directory / "metrics/model_metrics.json")
        write_json(X_train[FEATURE_COLUMNS].iloc[0].to_dict(), directory / "employee.json")
        write_json(X_train[FEATURE_COLUMNS].head(20).to_dict(orient="records"), directory / "background.json")
        bundle = seal_bundle(directory)
        if activate:
            publish_bundle(directory, tmp_path / "models/current.json")
        return bundle
    return create
