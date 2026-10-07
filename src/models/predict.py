"""Validated inference from a dictionary or DataFrame and a complete artifact."""

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from src.config import CONSTANT_COLUMNS, FEATURE_COLUMNS, ID_COLUMN, MODEL_PATH
from src.data.validate_data import validate_features

LOGGER = logging.getLogger(__name__)


def load_pipeline(path: str | Path = MODEL_PATH) -> Pipeline:
    """Load only a trusted local joblib file; pickle formats can execute code."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Model is unavailable. Run: python -m src.models.train")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "attrition_metadata_"):
        raise ValueError("Invalid model artifact: expected a complete pipeline and metadata.")
    metadata = pipeline.attrition_metadata_
    threshold = metadata.get("decision_threshold")
    if (metadata.get("schema_version") != 1 or metadata.get("feature_columns") != FEATURE_COLUMNS
            or not isinstance(threshold, (int, float)) or not np.isfinite(threshold)
            or not 0 <= threshold <= 1 or list(pipeline.classes_) != [0, 1]):
        raise ValueError("Invalid model schema, classes, or decision threshold; retrain the model.")
    return pipeline


def predict(data: dict | pd.DataFrame, *, pipeline: Pipeline | None = None,
            model_path: str | Path = MODEL_PATH) -> dict | list[dict]:
    """Use the artifact's frozen threshold; dictionary in gives dictionary out.

    A DataFrame returns a list in the original row order. Null values are imputed;
    missing keys are rejected. Novel categorical values are accepted by the
    encoder but may reduce reliability. Labels and unknown extra fields fail.
    """
    if not isinstance(data, (dict, pd.DataFrame)):
        raise ValueError("Prediction input must be a dictionary or pandas DataFrame.")
    single = isinstance(data, dict)
    frame = pd.DataFrame([data]) if single else data.copy()
    extra = set(frame.columns) - set(FEATURE_COLUMNS) - set(CONSTANT_COLUMNS) - {ID_COLUMN}
    if extra:
        raise ValueError(f"Unexpected prediction fields: {sorted(extra)}")
    validate_features(frame)
    if pipeline is None:
        pipeline = load_pipeline(model_path)
    threshold = float(pipeline.attrition_metadata_["decision_threshold"])
    LOGGER.info("Prediction request received: %d record(s)", len(frame))
    probabilities = pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    outputs = [{"predicted_class": "Yes" if probability >= threshold else "No",
                "attrition_probability": float(probability), "decision_threshold": threshold}
               for probability in probabilities]
    return outputs[0] if single else outputs


def main() -> None:
    """Read one JSON employee record from a file and emit JSON to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to a JSON object of employee features")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(predict(data, model_path=args.model), indent=2))


if __name__ == "__main__":
    main()
