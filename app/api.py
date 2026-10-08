"""FastAPI interface: uvicorn app.api:app --host 127.0.0.1 --port 8000."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, create_model

from src.config import CATEGORIES, CURRENT_RUN, DISCLAIMER, NUMERIC_BOUNDS
from src.models.predict import load_pipeline, predict

LOGGER = logging.getLogger(__name__)


def employee_schema() -> type[BaseModel]:
    """Generate an explicit required-field OpenAPI schema from shared bounds."""
    fields = {}
    for name, (lower, upper) in NUMERIC_BOUNDS.items():
        fields[name] = (Annotated[StrictInt | None, Field(ge=lower, le=upper)], ...)
    for name, values in CATEGORIES.items():
        fields[name] = (Annotated[StrictStr | None, Field(min_length=1, max_length=100,
                                                        description=f"Training categories: {', '.join(values)}. "
                                                        "Novel nonempty categories are permitted.")], ...)
    return create_model("EmployeeInput", __config__=ConfigDict(extra="forbid"), **fields)


EmployeeInput = employee_schema()


class PredictionResponse(BaseModel):
    """A probabilistic signal and the threshold used to derive its label."""

    prediction: Literal["Yes", "No"]
    attrition_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    run_id: str | None = None


def create_app(model_path: str | Path = CURRENT_RUN) -> FastAPI:
    """Build an app with a once-per-process loaded, injectable model artifact."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.pipeline = load_pipeline(model_path)
            LOGGER.info("Inference model loaded")
        except Exception:
            # Keep /health available to report an unready service; do not expose internals.
            LOGGER.exception("Model load failed; train a compatible model before serving")
            application.state.pipeline = None
        yield
        application.state.pipeline = None

    application = FastAPI(title="Employee Attrition Analysis", version="1.0.0",
                          description=DISCLAIMER, lifespan=lifespan)
    application.state.pipeline = None

    @application.get("/")
    def root() -> dict:
        return {"service": "Employee Attrition Analysis", "docs": "/docs", "disclaimer": DISCLAIMER}

    @application.get("/health")
    def health(request: Request):
        available = request.app.state.pipeline is not None
        run_id = request.app.state.pipeline.attrition_metadata_.get("run_id") if available else None
        return JSONResponse(status_code=200 if available else 503,
                            content={"status": "ok" if available else "unavailable",
                                     "model_available": available, "run_id": run_id})

    @application.post("/predict", response_model=PredictionResponse)
    def predict_employee(employee: EmployeeInput, request: Request) -> dict:
        pipeline = request.app.state.pipeline
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model unavailable. Run training and restart the service.")
        try:
            result = predict(employee.model_dump(), pipeline=pipeline)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"prediction": result["predicted_class"],
                "attrition_probability": result["attrition_probability"],
                "threshold": result["decision_threshold"], "run_id": result["run_id"]}

    return application


app = create_app()
