"""Versioned, verified experiment bundles with an atomic active-run pointer.

Checksums detect incomplete or mixed runs; they do not make untrusted pickle safe.
Only load bundles produced by a trusted training process.
"""

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from src.config import CURRENT_RUN, FEATURE_COLUMNS


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pipeline(pipeline: Pipeline) -> dict:
    if not isinstance(pipeline, Pipeline) or not isinstance(getattr(pipeline, "attrition_metadata_", None), dict):
        raise ValueError("Invalid model artifact: expected a complete pipeline and metadata.")
    metadata = pipeline.attrition_metadata_
    threshold = metadata.get("decision_threshold")
    if (metadata.get("schema_version") not in (1, 2) or metadata.get("feature_columns") != FEATURE_COLUMNS
            or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold) or not 0 <= threshold <= 1
            or list(getattr(pipeline, "classes_", [])) != [0, 1]):
        raise ValueError("Invalid model schema, classes, or decision threshold; retrain the model.")
    return metadata


def load_legacy_pipeline(path: Path) -> Pipeline:
    """Explicit path support for the previous, standalone schema-1 artifact."""
    if not path.is_file():
        raise FileNotFoundError("Model is unavailable. Run: python -m src.models.train")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        pipeline = joblib.load(path)
    metadata = validate_pipeline(pipeline)
    if metadata["schema_version"] != 1:
        raise ValueError("Schema-2 models must be loaded through their verified bundle.")
    return pipeline


@dataclass(frozen=True)
class Bundle:
    path: Path
    run_id: str
    pipeline: Pipeline
    report: dict

    @property
    def metrics_dir(self) -> Path:
        return self.path / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.path / "figures"


def active_run_path(pointer: Path = CURRENT_RUN) -> Path:
    """Resolve a single pointer snapshot; callers keep this path for the request."""
    if not pointer.is_file():
        raise FileNotFoundError("Model is unavailable. Run: python -m src.models.train")
    document = read_json(pointer)
    run_id = document.get("run_id")
    try:
        if str(UUID(run_id)) != run_id or document.get("schema_version") != 2:
            raise ValueError
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid active-run pointer.") from exc
    path = pointer.parent / "runs" / run_id
    if sha256(path / "manifest.json") != document.get("manifest_sha256"):
        raise ValueError("Active bundle manifest checksum mismatch.")
    return path


def load_bundle(path: str | Path = CURRENT_RUN) -> Bundle:
    """Verify every file before loading the model and matching its report."""
    path = Path(path)
    if not path.is_dir():
        path = active_run_path(path)
    manifest = read_json(path / "manifest.json")
    run_id = manifest.get("run_id")
    if manifest.get("schema_version") != 2 or run_id != path.name:
        raise ValueError("Invalid bundle version or run ID.")
    try:
        if str(UUID(run_id)) != run_id:
            raise ValueError
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid bundle run ID.") from exc
    files = manifest.get("files", {})
    required = {"pipeline.joblib", "metadata.json", "metrics/model_metrics.json", "employee.json", "background.json"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("Incomplete bundle manifest.")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} - {"manifest.json"}
    if actual != set(files):
        raise ValueError("Bundle files do not match the manifest.")
    for name, checksum in files.items():
        relative = PurePosixPath(name)
        target = path / name
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or not target.resolve().is_relative_to(path.resolve()) or target.is_symlink()):
            raise ValueError("Invalid bundle file path.")
        if sha256(target) != checksum:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        pipeline = joblib.load(path / "pipeline.joblib")
    metadata = validate_pipeline(pipeline)
    report = read_json(path / "metrics/model_metrics.json")
    if (metadata.get("schema_version") != 2 or metadata.get("run_id") != run_id
            or metadata != read_json(path / "metadata.json")
            or any(report.get(key) != value for key, value in metadata.items())
            or report.get("selected_threshold") != metadata["decision_threshold"]):
        raise ValueError("Bundle model, metadata, and report do not match.")
    # These inputs are part of the serving bundle, not optional external files.
    import pandas as pd

    from src.data.validate_data import validate_features

    validate_features(pd.DataFrame([read_json(path / "employee.json")]))
    validate_features(pd.DataFrame(read_json(path / "background.json")))
    return Bundle(path, run_id, pipeline, report)


def seal_bundle(path: Path) -> Bundle:
    """Write the manifest only once all training outputs exist, then verify it."""
    files = {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    write_json({"schema_version": 2, "run_id": path.name, "files": files}, path / "manifest.json")
    return load_bundle(path)


def publish_bundle(path: Path, pointer: Path = CURRENT_RUN) -> Bundle:
    """A failed validation or pointer write leaves the previous run active."""
    bundle = load_bundle(path)
    if path.resolve() != (pointer.parent / "runs" / bundle.run_id).resolve():
        raise ValueError("Bundle must be inside the pointer's runs directory.")
    temporary = pointer.with_name(f".current-{uuid4()}.tmp")
    try:
        write_json({"schema_version": 2, "run_id": bundle.run_id,
                    "manifest_sha256": sha256(path / "manifest.json")}, temporary)
        temporary.replace(pointer)
    finally:
        temporary.unlink(missing_ok=True)
    return bundle
