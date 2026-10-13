"""Observable API behavior: schema and boundary errors, authentication order, and unavailable artifacts."""

import copy
import logging
import shutil

import joblib
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from attrition.config import CATEGORIES, FEATURE_COLUMNS, NUMERIC_BOUNDS
from attrition.models.artifacts import read_json, write_json

API_KEY = "s3cret-key"
PROTECTED = [("get", "/model"), ("post", "/predict"), ("post", "/predict/batch")]


@pytest.fixture(scope="module")
def client(artifact):
    with TestClient(create_app(artifact)) as test_client:
        yield test_client


def numeric_boundary_cases():
    for name, (lower, upper) in NUMERIC_BOUNDS.items():
        yield pytest.param(name, lower, True, id=f"{name}-min")
        yield pytest.param(name, lower - 1, False, id=f"{name}-below-min")
        if upper is not None:
            yield pytest.param(name, upper, True, id=f"{name}-max")
            yield pytest.param(name, upper + 1, False, id=f"{name}-above-max")


@pytest.mark.parametrize(("field", "value", "accepted"), list(numeric_boundary_cases()))
def test_numeric_bounds_are_inclusive(client, employee, field, value, accepted):
    response = client.post("/predict", json={**employee, field: value})
    assert response.status_code == (200 if accepted else 422)
    if not accepted:
        assert response.json()["detail"][0]["loc"] == ["body", field]


@pytest.mark.parametrize("field", FEATURE_COLUMNS)
def test_every_feature_is_required_but_nullable(client, employee, field):
    missing = {key: value for key, value in employee.items() if key != field}
    response = client.post("/predict", json=missing)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]
    assert client.post("/predict", json={**employee, field: None}).status_code == 200


@pytest.mark.parametrize("field", FEATURE_COLUMNS)
def test_wrong_json_types_are_rejected(client, employee, field):
    wrong = "text" if field in NUMERIC_BOUNDS else 5
    response = client.post("/predict", json={**employee, field: wrong})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


def test_category_length_limits(client, employee):
    assert client.post("/predict", json={**employee, "JobRole": "x" * 100}).status_code == 200
    assert client.post("/predict", json={**employee, "JobRole": "x" * 101}).status_code == 422
    assert client.post("/predict", json={**employee, "JobRole": ""}).status_code == 422
    assert client.post("/predict", json={**employee, "JobRole": "\t \n"}).status_code == 422


def test_known_and_novel_categories_are_accepted(client, employee):
    for field, values in CATEGORIES.items():
        for value in (*values, f"Novel {field}"):
            assert client.post("/predict", json={**employee, field: value}).status_code == 200, (field, value)


@pytest.mark.parametrize("content", [b"{not json", b"", b"[1, 2]", b"null", b'"text"', b"42"])
def test_malformed_bodies_get_422(client, content):
    response = client.post("/predict", content=content, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_unknown_field_and_target_label_are_rejected(client, employee):
    for extra in ({"Unexpected": 1}, {"Attrition": "Yes"}, {"EmployeeNumber": 7}):
        response = client.post("/predict", json={**employee, **extra})
        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_unknown_routes_and_methods_do_not_reach_the_model(client):
    assert client.get("/predict").status_code == 405
    assert client.delete("/model").status_code == 405
    assert client.get("/missing").status_code == 404


def test_one_invalid_batch_row_rejects_the_whole_batch(client, employee):
    rows = [employee, employee, {**employee, "Age": 17}, employee]
    response = client.post("/predict/batch", json={"employees": rows})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "employees", 2, "Age"]
    assert "predictions" not in response.json()


def test_single_row_batch_and_response_shape(client, employee):
    response = client.post("/predict/batch", json={"employees": [employee]})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"count", "predictions"}
    assert body["count"] == 1
    assert set(body["predictions"][0]) == {"prediction", "attrition_probability", "threshold", "run_id"}
    assert body["predictions"][0]["prediction"] in {"Yes", "No"}


def test_request_id_boundaries(client):
    accepted = "a" * 64
    assert client.get("/health", headers={"X-Request-ID": accepted}).headers["X-Request-ID"] == accepted
    for rejected in ("a" * 65, "has/slash", "semi;colon", "ünïcode".encode()):
        # Non-ASCII values must go over the wire as bytes; the server must still not echo them.
        echoed = client.get("/health", headers={"X-Request-ID": rejected}).headers["X-Request-ID"]
        assert echoed != rejected
        assert len(echoed) == 32


def test_request_id_is_present_on_errors(client, artifact):
    assert len(client.get("/missing").headers["X-Request-ID"]) == 32
    assert len(client.post("/predict", json={}).headers["X-Request-ID"]) == 32
    with TestClient(create_app(artifact, api_key=API_KEY)) as protected:
        rejected = protected.post("/predict", json={}, headers={"X-Request-ID": "trace-1"})
    assert rejected.status_code == 401
    assert rejected.headers["X-Request-ID"] == "trace-1"


def request(client, method, path, employee, headers=None):
    body = {"employees": [employee]} if path.endswith("batch") else employee
    return client.request(method, path, json=body if method == "post" else None, headers=headers)


@pytest.mark.parametrize("supplied", [None, "", " ", "S3CRET-KEY", f"{API_KEY} ", API_KEY[:-1], API_KEY + "x"])
@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_wrong_or_missing_keys_are_rejected_everywhere(artifact, employee, method, path, supplied):
    headers = None if supplied is None else {"X-API-Key": supplied}
    with TestClient(create_app(artifact, api_key=API_KEY)) as client:
        response = request(client, method, path, employee, headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing or invalid API key."}
    assert API_KEY not in response.text


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_correct_key_is_accepted_and_header_name_is_case_insensitive(artifact, employee, method, path):
    with TestClient(create_app(artifact, api_key=API_KEY)) as client:
        assert request(client, method, path, employee, {"X-API-Key": API_KEY}).status_code == 200
        assert request(client, method, path, employee, {"x-api-key": API_KEY}).status_code == 200


def test_key_is_ignored_when_none_is_configured(artifact, employee):
    with TestClient(create_app(artifact)) as client:
        assert client.post("/predict", json=employee, headers={"X-API-Key": "anything"}).status_code == 200


def test_authentication_precedes_validation_and_availability(tmp_path, artifact, employee):
    with TestClient(create_app(artifact, api_key=API_KEY)) as client:
        invalid = client.post("/predict", json={"Age": "not a number", "Secret": "value-to-hide"})
        assert invalid.status_code == 401
        assert "value-to-hide" not in invalid.text
        authenticated = client.post("/predict", json={"Age": 5}, headers={"X-API-Key": API_KEY})
        assert authenticated.status_code == 422
    with TestClient(create_app(tmp_path / "missing.joblib", api_key=API_KEY)) as client:
        assert client.get("/model").status_code == 401
        assert client.get("/model", headers={"X-API-Key": API_KEY}).status_code == 503


def test_open_endpoints_never_require_a_key(artifact):
    with TestClient(create_app(artifact, api_key=API_KEY)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200


def test_logs_do_not_contain_keys_or_request_values(artifact, employee, caplog):
    caplog.set_level(logging.INFO)
    marker = "DISTINCTIVE-ROLE-VALUE"
    with TestClient(create_app(artifact, api_key=API_KEY)) as client:
        client.post("/predict", json={**employee, "JobRole": marker}, headers={"X-API-Key": API_KEY})
        client.post("/predict", json={**employee, "JobRole": marker}, headers={"X-API-Key": "wrong-key-value"})
    assert "request_id=" in caplog.text
    for secret in (API_KEY, "wrong-key-value", marker, str(employee["MonthlyIncome"]) + ","):
        assert secret not in caplog.text


def unavailable_artifacts():
    """Each builder returns the path an operator might hand to the API; none can be served."""
    def missing_pointer(tmp_path, bundle_factory, fitted_pipeline):
        return tmp_path / "models/current.json"

    def malformed_pointer(tmp_path, bundle_factory, fitted_pipeline):
        pointer = tmp_path / "models/current.json"
        pointer.parent.mkdir(parents=True)
        pointer.write_text("{not json", encoding="utf-8")
        return pointer

    def tampered_run(tmp_path, bundle_factory, fitted_pipeline):
        bundle = bundle_factory()
        target = bundle.path / "pipeline.joblib"
        target.write_bytes(target.read_bytes() + b"corrupt")
        return tmp_path / "models/current.json"

    def pruned_run(tmp_path, bundle_factory, fitted_pipeline):
        bundle = bundle_factory()
        shutil.rmtree(bundle.path)
        return tmp_path / "models/current.json"

    def truncated_legacy(tmp_path, bundle_factory, fitted_pipeline):
        path = tmp_path / "truncated.joblib"
        path.write_bytes(b"\x80\x04\x95")
        return path

    def incompatible_legacy(tmp_path, bundle_factory, fitted_pipeline):
        pipeline = copy.deepcopy(fitted_pipeline)
        pipeline.attrition_metadata_["decision_threshold"] = 1.5
        path = tmp_path / "incompatible.joblib"
        joblib.dump(pipeline, path)
        return path

    def schema_two_as_legacy(tmp_path, bundle_factory, fitted_pipeline):
        pipeline = copy.deepcopy(fitted_pipeline)
        pipeline.attrition_metadata_["schema_version"] = 2
        path = tmp_path / "schema2.joblib"
        joblib.dump(pipeline, path)
        return path

    for builder in (missing_pointer, malformed_pointer, tampered_run, pruned_run, truncated_legacy,
                    incompatible_legacy, schema_two_as_legacy):
        yield pytest.param(builder, id=builder.__name__)


@pytest.mark.parametrize("builder", list(unavailable_artifacts()))
def test_unusable_artifacts_yield_safe_503s(builder, tmp_path, bundle_factory, fitted_pipeline, employee):
    path = builder(tmp_path, bundle_factory, fitted_pipeline)
    headers = {"X-API-Key": API_KEY}
    with TestClient(create_app(path, api_key=API_KEY)) as client:
        responses = [client.get("/health"), client.get("/model", headers=headers),
                     client.post("/predict", json=employee, headers=headers),
                     client.post("/predict/batch", json={"employees": [employee]}, headers=headers)]
    assert [response.status_code for response in responses] == [503] * 4
    assert responses[0].json() == {"status": "unavailable", "model_available": False, "run_id": None}
    for response in responses:
        assert str(tmp_path) not in response.text
        assert "Traceback" not in response.text
        assert "joblib" not in response.text.lower()


def test_unusable_artifact_does_not_poison_later_valid_startup(tmp_path, bundle_factory, employee):
    bundle = bundle_factory()
    pointer = tmp_path / "models/current.json"
    good = read_json(pointer)
    write_json({**good, "manifest_sha256": "0" * 64}, pointer)
    with TestClient(create_app(pointer)) as client:
        assert client.get("/health").status_code == 503
    write_json(good, pointer)
    with TestClient(create_app(pointer)) as client:
        assert client.get("/health").json()["run_id"] == bundle.run_id
        assert client.post("/predict", json=employee).status_code == 200


def test_value_errors_from_prediction_become_422_not_500(bundle_factory, tmp_path, employee, monkeypatch):
    bundle_factory()

    def reject(*args, **kwargs):
        raise ValueError("Rejected by the prediction layer")

    monkeypatch.setattr("app.api.predict", reject)
    with TestClient(create_app(tmp_path / "models/current.json")) as client:
        single = client.post("/predict", json=employee)
        batch = client.post("/predict/batch", json={"employees": [employee]})
    for response in (single, batch):
        assert response.status_code == 422
        assert response.json() == {"detail": "Rejected by the prediction layer"}


def test_unexpected_failures_return_a_generic_500_without_internals(bundle_factory, tmp_path, employee, monkeypatch):
    bundle_factory()

    def explode(*args, **kwargs):
        raise RuntimeError(f"secret internals at {tmp_path}")

    monkeypatch.setattr("app.api.predict", explode)
    with TestClient(create_app(tmp_path / "models/current.json"), raise_server_exceptions=False) as client:
        response = client.post("/predict", json=employee)
    assert response.status_code == 500
    assert "secret internals" not in response.text
    assert str(tmp_path) not in response.text
    assert len(response.headers.get("X-Request-ID", "")) in (0, 32)
