"""Defensive API behavior: authentication order, request limits, safe errors, and logs."""

import json
import logging

import anyio
import pytest
from fastapi.testclient import TestClient

from app.api import MAX_BATCH_SIZE, MAX_BODY_BYTES, MAX_REPORTED_ERRORS, create_app
from attrition.config import CATEGORIES

KEY = "test-key-7c1f"
UNAUTHORIZED = {"detail": "Missing or invalid API key."}
MODEL_ENDPOINTS = [("GET", "/model"), ("POST", "/predict"), ("POST", "/predict/batch")]


def body_for(path, employee):
    return {"employees": [employee]} if path.endswith("batch") else employee


@pytest.mark.parametrize(("method", "path"), MODEL_ENDPOINTS)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": ""}, {"X-API-Key": "wrong"},
                                     {"X-API-Key": KEY.upper()}, {"X-API-Key": KEY + "x"}, {"X-API-Key": KEY[:-1]}])
def test_missing_or_incorrect_key_is_rejected(artifact, employee, method, path, headers):
    with TestClient(create_app(artifact, api_key=KEY)) as client:
        kwargs = {"json": body_for(path, employee)} if method == "POST" else {}
        response = client.request(method, path, headers=headers, **kwargs)
    assert response.status_code == 401
    assert response.json() == UNAUTHORIZED


@pytest.mark.parametrize("content", [b"{not json", b"[]", b'{"Age": "x"}', b""])
def test_unauthenticated_body_is_rejected_before_parsing(artifact, content):
    """Without the key, malformed or invalid JSON gets 401, never validation details."""
    with TestClient(create_app(artifact, api_key=KEY)) as client:
        for path in ("/predict", "/predict/batch"):
            response = client.post(path, content=content, headers={"Content-Type": "application/json"})
            assert response.status_code == 401
            assert response.json() == UNAUTHORIZED


def test_unauthenticated_streamed_body_is_not_read(artifact):
    """A chunked body over the size limit still gets 401: the key is checked before any read."""
    consumed = []

    def chunks():
        consumed.append(True)
        yield b"x" * 4096

    with TestClient(create_app(artifact, api_key=KEY, max_body_bytes=1024)) as client:
        response = client.post("/predict", content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 401
    assert not consumed


def test_open_endpoints_and_docs_stay_available_with_a_key(artifact):
    with TestClient(create_app(artifact, api_key=KEY)) as client:
        for path in ("/", "/health", "/docs", "/openapi.json"):
            assert client.get(path).status_code == 200


def test_openapi_declares_the_key_on_model_endpoints_only(artifact):
    with TestClient(create_app(artifact, api_key=KEY)) as client:
        schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["APIKeyHeader"] == {
        "type": "apiKey", "in": "header", "name": "X-API-Key"}
    for method, path in MODEL_ENDPOINTS:
        assert schema["paths"][path][method.lower()]["security"] == [{"APIKeyHeader": []}]
    for path in ("/", "/health"):
        assert "security" not in schema["paths"][path]["get"]


@pytest.mark.parametrize("path", ["/predict", "/predict/batch"])
def test_declared_oversized_body_gets_413(artifact, employee, path):
    with TestClient(create_app(artifact, max_body_bytes=1024)) as client:
        assert client.post(path, json=body_for(path, employee)).status_code == 200
        oversized = json.dumps({**body_for(path, employee), "pad": "x" * 2048}).encode()
        response = client.post(path, content=oversized, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds 1024 bytes."}
    assert response.headers["X-Request-ID"]


def test_undeclared_oversized_body_gets_413(artifact, employee):
    """A chunked body has no Content-Length, so the limit applies while it is read."""
    payload = json.dumps(employee).encode()

    def chunks():
        yield from [payload] * 10

    with TestClient(create_app(artifact, api_key=KEY, max_body_bytes=4 * len(payload))) as client:
        response = client.post("/predict", content=chunks(),
                               headers={"Content-Type": "application/json", "X-API-Key": KEY})
    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert response.json() == {"detail": f"Request body exceeds {4 * len(payload)} bytes."}


def test_streamed_body_reading_stops_at_the_limit(artifact):
    """Driven directly over ASGI: the TestClient would join all chunks into one message."""
    delivered, sent = [], []

    async def receive():
        delivered.append(400)
        return {"type": "http.request", "body": b" " * 400, "more_body": len(delivered) < 50}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/predict", "raw_path": b"/predict", "root_path": "", "query_string": b"",
             "headers": [(b"content-type", b"application/json"), (b"transfer-encoding", b"chunked")],
             "client": ("127.0.0.1", 1), "server": ("testserver", 80)}
    anyio.run(create_app(artifact, max_body_bytes=1000), scope, receive, send)
    assert sent[0]["status"] == 413
    assert len(delivered) == 3  # 1,200 bytes is the first total over 1,000.


def test_default_limit_admits_a_full_batch_of_long_values(artifact, employee):
    """The body limit must never reject a schema-valid maximum batch in a plausible encoding."""
    record = {**employee, **dict.fromkeys(CATEGORIES, "é" * 100)}
    # Indented, ASCII-escaped JSON is a large but common client encoding (Python's json default).
    content = json.dumps({"employees": [record] * MAX_BATCH_SIZE}, indent=2).encode()
    assert MAX_BODY_BYTES / 2 < len(content) < MAX_BODY_BYTES
    with TestClient(create_app(artifact)) as client:
        response = client.post("/predict/batch", content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    assert response.json()["count"] == MAX_BATCH_SIZE


@pytest.mark.parametrize(("path", "change"), [
    ("/predict", {"Age": "SECRET-7319"}),
    ("/predict", {"Notes": "SECRET-7319"}),
    ("/predict", {"Attrition": "SECRET-7319"}),
    ("/predict/batch", {"EmployeeNumber": "SECRET-7319"}),
    ("/predict/batch", {"JobRole": "SECRET-7319" * 20}),
])
def test_validation_errors_do_not_echo_submitted_values(artifact, employee, path, change):
    with TestClient(create_app(artifact)) as client:
        response = client.post(path, json=body_for(path, {**employee, **change}))
    assert response.status_code == 422
    assert "SECRET" not in response.text
    detail = response.json()["detail"]
    assert detail and all(set(error) == {"type", "loc", "msg"} for error in detail)
    assert next(iter(change)) in detail[0]["loc"]


def test_unexpected_nested_fields_are_rejected_with_their_location(artifact, employee):
    """Identifiers and labels cannot be smuggled into a batch record."""
    with TestClient(create_app(artifact)) as client:
        response = client.post("/predict/batch", json={"employees": [employee, {**employee, "Attrition": "No"}]})
    assert response.status_code == 422
    assert response.json()["detail"] == [{"type": "extra_forbidden", "loc": ["body", "employees", 1, "Attrition"],
                                          "msg": "Extra inputs are not permitted"}]


def test_malformed_json_reports_position_without_input(artifact):
    with TestClient(create_app(artifact)) as client:
        response = client.post("/predict", content=b'{"Age": SECRET', headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json() == {"detail": [{"type": "json_invalid", "loc": ["body", 8], "msg": "JSON decode error"}]}


def test_validation_error_list_is_bounded(artifact):
    """A maximum batch of empty records has 30,000 errors; the response lists a bounded number."""
    with TestClient(create_app(artifact)) as client:
        response = client.post("/predict/batch", json={"employees": [{}] * MAX_BATCH_SIZE})
    assert response.status_code == 422
    assert len(response.json()["detail"]) == MAX_REPORTED_ERRORS
    assert len(response.content) < 20_000


def test_internal_errors_return_a_generic_500(artifact, employee):
    def fail(_):
        raise RuntimeError("SECRET internal path /srv/models/pipeline.joblib")

    with TestClient(create_app(artifact), raise_server_exceptions=False) as client:
        client.app.state.pipeline.predict_proba = fail
        response = client.post("/predict", json=employee)
    assert response.status_code == 500
    assert response.text == "Internal Server Error"


def test_tampered_bundle_is_not_served_and_details_stay_internal(bundle_factory, tmp_path, employee):
    bundle = bundle_factory()
    with (bundle.path / "pipeline.joblib").open("ab") as file:
        file.write(b"tampered")
    with TestClient(create_app(tmp_path / "models/current.json")) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=employee)
    assert health.status_code == 503
    assert health.json() == {"status": "unavailable", "model_available": False, "run_id": None}
    assert prediction.status_code == 503
    for response in (health, prediction):
        assert "checksum" not in response.text.lower()
        assert str(tmp_path) not in response.text and bundle.run_id not in response.text


def test_responses_carry_security_headers(artifact, employee):
    with TestClient(create_app(artifact, api_key=KEY)) as client:
        responses = [client.get("/health"), client.get("/missing"), client.post("/predict", json=employee),
                     client.post("/predict", json={}, headers={"X-API-Key": KEY}),
                     client.post("/predict", json=employee, headers={"X-API-Key": KEY})]
    assert [response.status_code for response in responses] == [200, 404, 401, 422, 200]
    for response in responses:
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Request-ID"]


def test_logs_exclude_keys_and_payload_values(artifact, employee, caplog):
    marker = "Marker Role 7319"
    with caplog.at_level(logging.DEBUG), TestClient(create_app(artifact, api_key=KEY)) as client:
        assert client.post("/predict", json={**employee, "JobRole": marker},
                           headers={"X-API-Key": KEY}).status_code == 200
        assert client.post("/predict", json={**employee, "Age": marker},
                           headers={"X-API-Key": KEY}).status_code == 422
        assert client.post("/predict", json=employee, headers={"X-API-Key": "wrong-" + KEY}).status_code == 401
    assert "path=/predict status=200" in caplog.text
    assert KEY not in caplog.text
    assert marker not in caplog.text


def test_request_log_encodes_the_path(artifact, caplog):
    """Decoded control characters and spaces cannot forge extra log lines or fields."""
    with caplog.at_level(logging.INFO, logger="app.api"), TestClient(create_app(artifact)) as client:
        client.get("/x%0Ainjected%20status=200")
    [message] = [record.getMessage() for record in caplog.records if "injected" in record.getMessage()]
    assert "\n" not in message and " " not in message.split("path=")[1].split(" status=")[0]
    assert message.count("status=") == 1
    assert "path=/x%0Ainjected%20status%3D200 status=404" in message
