"""FastAPI interface: uvicorn app.api:app --host 127.0.0.1 --port 8000.

Set ATTRITION_API_KEY to require an X-API-Key header on the model endpoints.
The key is a demo guard; public exposure needs the gateway controls in docs/serving.md.
"""

import logging
import os
import re
import secrets
import time
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import quote
from uuid import uuid4

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, create_model
from sklearn.pipeline import Pipeline
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from attrition import __version__
from attrition.config import CATEGORIES, CURRENT_RUN, DISCLAIMER, NUMERIC_BOUNDS
from attrition.models.artifacts import load_bundle, load_legacy_pipeline
from attrition.models.predict import predict

LOGGER = logging.getLogger(__name__)
MAX_BATCH_SIZE = 1000
# 8 KiB per record: a full batch of realistic records (about 0.7 KB each) fits with ample room.
MAX_BODY_BYTES = MAX_BATCH_SIZE * 8 * 1024
# Bounds the 422 response: one invalid batch can otherwise produce thousands of entries.
MAX_REPORTED_ERRORS = 100
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
HOLDOUT_KEYS = ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier_score",
                "threshold", "n", "positive_count")
API_KEY_NAME = "X-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


class BodySizeLimit:
    """Reject a request body over max_bytes with HTTP 413, whether declared or streamed in chunks."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        detail = f"Request body exceeds {self.max_bytes} bytes."
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdecimal() and int(declared) > self.max_bytes:
            await JSONResponse(status_code=413, content={"detail": detail})(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                # FastAPI re-raises HTTPException from body reading, so this becomes a 413 response.
                raise HTTPException(status_code=413, detail=detail)
            return message

        await self.app(scope, limited_receive, send)


def employee_schema() -> type[BaseModel]:
    """Generate an explicit required-field OpenAPI schema from shared bounds."""
    fields: dict[str, Any] = {}
    for name, (lower, upper) in NUMERIC_BOUNDS.items():
        fields[name] = (Annotated[StrictInt | None, Field(ge=lower, le=upper)], ...)
    for name, values in CATEGORIES.items():
        fields[name] = (Annotated[StrictStr | None, Field(min_length=1, max_length=100,
                                                        description=f"Training categories: {', '.join(values)}. "
                                                        "Novel nonempty categories are permitted.")], ...)
    return create_model("EmployeeInput", __config__=ConfigDict(extra="forbid"), **fields)


if TYPE_CHECKING:  # Type checkers cannot follow create_model; the runtime schema is generated.
    EmployeeInput = BaseModel
else:
    EmployeeInput = employee_schema()


class PredictionResponse(BaseModel):
    """A probabilistic signal and the threshold used to derive its label."""

    prediction: Literal["Yes", "No"]
    attrition_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    run_id: str | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employees: list[EmployeeInput] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    """Predictions in request order."""

    count: int
    predictions: list[PredictionResponse]


class ModelInfo(BaseModel):
    """The loaded artifact's identity, operating point, and frozen holdout metrics."""

    api_version: str
    run_id: str | None = None
    schema_version: int
    selected_model: str | None = None
    decision_threshold: float
    feature_count: int
    train_rows: int | None = None
    data_sha256: str | None = None
    package_versions: dict[str, str] | None = None
    holdout_metrics: dict[str, float | int | None] | None = None


def load_model(path: str | Path) -> tuple[Pipeline, dict | None]:
    """Load a verified bundle with its holdout metrics, or an explicit legacy file without them."""
    path = Path(path)
    if path.suffix == ".joblib":
        return load_legacy_pipeline(path), None
    bundle = load_bundle(path)
    metrics = bundle.report["selected_model_metrics"]
    return bundle.pipeline, {key: metrics[key] for key in HOLDOUT_KEYS}


def as_response(result: dict) -> dict:
    return {"prediction": result["predicted_class"], "attrition_probability": result["attrition_probability"],
            "threshold": result["decision_threshold"], "run_id": result["run_id"]}


def loaded_pipeline(request: Request) -> Pipeline:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model unavailable. Run training and restart the service.")
    return pipeline


def create_app(model_path: str | Path = CURRENT_RUN, *, api_key: str | None = None,
               max_body_bytes: int = MAX_BODY_BYTES) -> FastAPI:
    """Build an app with a once-per-process loaded, injectable model artifact.

    With an api_key, model endpoints require a matching X-API-Key header; / and /health stay open.
    Request bodies over max_body_bytes are rejected with HTTP 413.
    """
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.pipeline, application.state.holdout_metrics = load_model(model_path)
            LOGGER.info("Inference model loaded")
        except Exception:
            # Keep /health available to report an unready service; do not expose internals.
            LOGGER.exception("Model load failed; train a compatible model before serving")
            application.state.pipeline, application.state.holdout_metrics = None, None
        yield
        application.state.pipeline = None

    async def require_api_key(supplied: str | None = Security(API_KEY_HEADER)) -> None:
        if api_key and not (supplied and secrets.compare_digest(supplied.encode(), api_key.encode())):
            raise HTTPException(status_code=401, detail="Missing or invalid API key.")

    class KeyCheckedRoute(APIRoute):
        """Routes that depend on require_api_key check it before FastAPI reads or parses the body.

        Unauthenticated callers cannot make the service parse JSON or learn validation details.
        The dependency still runs afterwards and declares the scheme in OpenAPI.
        """

        def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handler = super().get_route_handler()
            if not any(dependency.dependency is require_api_key for dependency in self.dependencies):
                return handler

            async def checked(request: Request) -> Response:
                await require_api_key(request.headers.get(API_KEY_NAME))
                return await handler(request)

            return checked

    protected = [Depends(require_api_key)]
    application = FastAPI(title="Employee Attrition Analysis", version=__version__,
                          description=DISCLAIMER, lifespan=lifespan)
    application.router.route_class = KeyCheckedRoute
    application.state.pipeline = None
    # Added before request_context, so it runs inside it: rejected requests still get IDs and logs.
    application.add_middleware(BodySizeLimit, max_bytes=max_body_bytes)

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        """Tag every response with a request ID; log status and timing, never request bodies."""
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Predictions describe individual profiles; shared caches and browsers must not keep them.
        response.headers["Cache-Control"] = "no-store"
        # Percent-encode the decoded path, as uvicorn's access log does, so it cannot forge log fields.
        LOGGER.info("request_id=%s method=%s path=%s status=%d duration_ms=%.1f", request_id,
                    request.method, quote(request.scope["path"]), response.status_code,
                    (time.perf_counter() - started) * 1000)
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_failed(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Report where and why validation failed without echoing submitted values back."""
        errors = [{"type": error["type"], "loc": list(error["loc"]), "msg": error["msg"]}
                  for error in exc.errors()[:MAX_REPORTED_ERRORS]]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.get("/")
    def root() -> dict:
        return {"service": "Employee Attrition Analysis", "version": __version__, "docs": "/docs",
                "disclaimer": DISCLAIMER}

    @application.get("/health")
    def health(request: Request):
        available = request.app.state.pipeline is not None
        run_id = request.app.state.pipeline.attrition_metadata_.get("run_id") if available else None
        return JSONResponse(status_code=200 if available else 503,
                            content={"status": "ok" if available else "unavailable",
                                     "model_available": available, "run_id": run_id})

    @application.get("/model", response_model=ModelInfo, dependencies=protected)
    def model_info(request: Request) -> dict:
        metadata = loaded_pipeline(request).attrition_metadata_
        return {"api_version": __version__, "run_id": metadata.get("run_id"),
                "schema_version": metadata["schema_version"], "selected_model": metadata.get("selected_model"),
                "decision_threshold": metadata["decision_threshold"],
                "feature_count": len(metadata["feature_columns"]), "train_rows": metadata.get("train_rows"),
                "data_sha256": metadata.get("data_sha256"), "package_versions": metadata.get("package_versions"),
                "holdout_metrics": request.app.state.holdout_metrics}

    @application.post("/predict", response_model=PredictionResponse, dependencies=protected)
    def predict_employee(employee: EmployeeInput, request: Request) -> dict:
        pipeline = loaded_pipeline(request)
        try:
            return as_response(predict(employee.model_dump(), pipeline=pipeline))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/predict/batch", response_model=BatchResponse, dependencies=protected)
    def predict_employees(batch: BatchRequest, request: Request) -> dict:
        pipeline = loaded_pipeline(request)
        frame = pd.DataFrame([employee.model_dump() for employee in batch.employees])
        try:
            results = predict(frame, pipeline=pipeline)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"count": len(results), "predictions": [as_response(result) for result in results]}

    return application


app = create_app(api_key=os.environ.get("ATTRITION_API_KEY") or None)
