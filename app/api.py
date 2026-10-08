"""FastAPI interface: uvicorn app.api:app --host 127.0.0.1 --port 8000.

Set ATTRITION_API_KEY to require an X-API-Key header on the model endpoints.
"""

import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, create_model
from sklearn.pipeline import Pipeline

from attrition import __version__
from attrition.config import CATEGORIES, CURRENT_RUN, DISCLAIMER, NUMERIC_BOUNDS
from attrition.models.artifacts import load_bundle, load_legacy_pipeline
from attrition.models.predict import predict

LOGGER = logging.getLogger(__name__)
MAX_BATCH_SIZE = 1000
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
HOLDOUT_KEYS = ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier_score",
                "threshold", "n", "positive_count")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


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


def create_app(model_path: str | Path = CURRENT_RUN, *, api_key: str | None = None) -> FastAPI:
    """Build an app with a once-per-process loaded, injectable model artifact.

    With an api_key, model endpoints require a matching X-API-Key header; / and /health stay open.
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

    protected = [Depends(require_api_key)]
    application = FastAPI(title="Employee Attrition Analysis", version=__version__,
                          description=DISCLAIMER, lifespan=lifespan)
    application.state.pipeline = None

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        """Tag every response with a request ID; log status and timing, never request bodies."""
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        LOGGER.info("request_id=%s method=%s path=%s status=%d duration_ms=%.1f", request_id,
                    request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
        return response

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
