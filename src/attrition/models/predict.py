"""Validated inference from a dictionary or DataFrame and a complete artifact."""

import argparse
import json
import logging
from pathlib import Path
from typing import overload

import pandas as pd
from sklearn.pipeline import Pipeline

from attrition.config import CONSTANT_COLUMNS, CURRENT_RUN, FEATURE_COLUMNS, ID_COLUMN
from attrition.data.validate_data import validate_features
from attrition.models.artifacts import load_bundle, load_legacy_pipeline

LOGGER = logging.getLogger(__name__)


def load_pipeline(path: str | Path = CURRENT_RUN) -> Pipeline:
    """Load the active verified bundle, or an explicit legacy schema-1 file."""
    path = Path(path)
    if path.suffix == ".joblib":
        return load_legacy_pipeline(path)
    return load_bundle(path).pipeline


@overload
def predict(data: dict, *, pipeline: Pipeline | None = None, model_path: str | Path = CURRENT_RUN) -> dict: ...
@overload
def predict(data: pd.DataFrame, *, pipeline: Pipeline | None = None,
            model_path: str | Path = CURRENT_RUN) -> list[dict]: ...
def predict(data: dict | pd.DataFrame, *, pipeline: Pipeline | None = None,
            model_path: str | Path = CURRENT_RUN) -> dict | list[dict]:
    """Use the artifact's frozen threshold; dictionary in gives dictionary out.

    A DataFrame returns a list in the original row order. Null values are imputed;
    missing keys are rejected. Novel categorical values are accepted by the
    encoder but may reduce reliability. Labels and unknown extra fields fail.
    """
    if not isinstance(data, (dict, pd.DataFrame)):
        raise ValueError("Prediction input must be a dictionary or pandas DataFrame.")
    single = isinstance(data, dict)
    frame = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
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
                "attrition_probability": float(probability), "decision_threshold": threshold,
                "run_id": pipeline.attrition_metadata_.get("run_id")}
               for probability in probabilities]
    return outputs[0] if single else outputs


def main() -> None:
    """Read one JSON employee record from a file and emit JSON to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to a JSON object of employee features")
    parser.add_argument("--model", type=Path, default=CURRENT_RUN)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(predict(data, model_path=args.model), indent=2))


if __name__ == "__main__":
    main()
