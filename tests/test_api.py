"""Exercise real startup, schema validation, and API/model parity."""

from fastapi.testclient import TestClient
import pytest

from app.api import create_app
from src.models.predict import predict


def test_health_and_prediction(artifact, employee):
    with TestClient(create_app(artifact)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json() == {"status": "ok", "model_available": True}
        response = client.post("/predict", json=employee)
        assert response.status_code == 200
        expected = predict(employee, model_path=artifact)
        assert response.json() == {"prediction": expected["predicted_class"],
                                   "attrition_probability": expected["attrition_probability"],
                                   "threshold": expected["decision_threshold"]}


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
