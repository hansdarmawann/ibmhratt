"""Exercise real startup, schema validation, and API/model parity."""

import pytest
from fastapi.testclient import TestClient

from app.api import HOLDOUT_KEYS, MAX_BATCH_SIZE, create_app
from attrition import __version__
from attrition.config import FEATURE_COLUMNS
from attrition.models.predict import predict


def test_health_and_prediction(artifact, employee):
    with TestClient(create_app(artifact)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json() == {"status": "ok", "model_available": True, "run_id": None}
        response = client.post("/predict", json=employee)
        assert response.status_code == 200
        expected = predict(employee, model_path=artifact)
        assert response.json() == {"prediction": expected["predicted_class"],
                                   "attrition_probability": expected["attrition_probability"],
                                   "threshold": expected["decision_threshold"], "run_id": None}


def test_api_keeps_one_bundle_until_restart(bundle_factory, tmp_path, employee):
    first = bundle_factory()
    pointer = tmp_path / "models/current.json"
    with TestClient(create_app(pointer)) as client:
        second = bundle_factory(threshold=0.8)
        assert client.get("/health").json()["run_id"] == first.run_id
        result = client.post("/predict", json=employee).json()
        assert result["run_id"] == first.run_id
        assert result["threshold"] == 0.35
    with TestClient(create_app(pointer)) as client:
        result = client.post("/predict", json=employee).json()
        assert result["run_id"] == second.run_id
        assert result["threshold"] == 0.8


@pytest.mark.parametrize("change", [{"Age": -1}, {"Age": "35"}, {"Age": True},
                                    {"Age": 33.1}, {"Gender": " "}, {"Attrition": "Yes"}])
def test_api_rejects_invalid_input(artifact, employee, change):
    with TestClient(create_app(artifact)) as client:
        response = client.post("/predict", json={**employee, **change})
        assert response.status_code == 422
        assert response.json()["detail"]


def test_api_missing_and_nullable_fields(artifact, employee):
    with TestClient(create_app(artifact)) as client:
        assert client.post("/predict", json={"Age": 30}).status_code == 422
        response = client.post("/predict", json={**employee, "Age": None, "JobRole": "Novel role"})
        assert response.status_code == 200


def test_unavailable_model_returns_503(tmp_path, employee):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/health").json()["model_available"] is False
        assert client.post("/predict", json=employee).status_code == 503


def test_model_info_reports_bundle_identity_and_holdout_metrics(bundle_factory, tmp_path):
    bundle = bundle_factory(threshold=0.4)
    with TestClient(create_app(tmp_path / "models/current.json")) as client:
        info = client.get("/model").json()
    assert info["api_version"] == __version__
    assert info["run_id"] == bundle.run_id
    assert info["decision_threshold"] == 0.4
    assert info["feature_count"] == len(FEATURE_COLUMNS)
    assert info["holdout_metrics"] == {key: bundle.report["selected_model_metrics"][key] for key in HOLDOUT_KEYS}


def test_model_info_for_legacy_artifact_has_no_metrics(artifact):
    with TestClient(create_app(artifact)) as client:
        info = client.get("/model").json()
    assert info["run_id"] is None
    assert info["schema_version"] == 1
    assert info["holdout_metrics"] is None


def test_batch_matches_single_predictions_in_order(artifact, employee):
    profiles = [employee, {**employee, "Age": None, "JobRole": "Novel role"}, {**employee, "OverTime": "Yes"}]
    with TestClient(create_app(artifact)) as client:
        batch = client.post("/predict/batch", json={"employees": profiles})
        singles = [client.post("/predict", json=profile).json() for profile in profiles]
    assert batch.status_code == 200
    assert batch.json()["count"] == 3
    # Matrix products over several rows can differ from single rows in the last float digits.
    for result, single in zip(batch.json()["predictions"], singles, strict=True):
        assert result["attrition_probability"] == pytest.approx(single.pop("attrition_probability"))
        assert {key: value for key, value in result.items() if key != "attrition_probability"} == single


@pytest.mark.parametrize("body", [
    {"employees": []},
    {"employees": "not a list"},
    {"employees": [{"Age": 30}]},
    {"employees": [{}], "extra": True},
])
def test_batch_rejects_invalid_requests(artifact, body):
    with TestClient(create_app(artifact)) as client:
        assert client.post("/predict/batch", json=body).status_code == 422


def test_batch_size_limit(artifact, employee):
    with TestClient(create_app(artifact)) as client:
        assert client.post("/predict/batch", json={"employees": [employee] * MAX_BATCH_SIZE}).status_code == 200
        assert client.post("/predict/batch", json={"employees": [employee] * (MAX_BATCH_SIZE + 1)}).status_code == 422


def test_request_id_is_echoed_only_when_safe(artifact):
    with TestClient(create_app(artifact)) as client:
        assert client.get("/health", headers={"X-Request-ID": "trace-123"}).headers["X-Request-ID"] == "trace-123"
        generated = client.get("/health", headers={"X-Request-ID": "bad id\twith spaces"}).headers["X-Request-ID"]
        assert generated != "bad id\twith spaces" and len(generated) == 32
        assert len(client.get("/health").headers["X-Request-ID"]) == 32


def test_optional_api_key_protects_model_endpoints_only(artifact, employee):
    with TestClient(create_app(artifact, api_key="s3cret")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200
        assert client.post("/predict", json=employee).status_code == 401
        assert client.post("/predict", json=employee, headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/model", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert client.post("/predict", json=employee, headers={"X-API-Key": "s3cret"}).status_code == 200
        batch = client.post("/predict/batch", json={"employees": [employee]}, headers={"X-API-Key": "s3cret"})
        assert batch.status_code == 200
