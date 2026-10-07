"""Validate the IBM schema without fitting any preprocessing statistics."""

import warnings

import numpy as np
import pandas as pd

from src.config import (
    CATEGORIES, CONSTANT_COLUMNS, FEATURE_COLUMNS, ID_COLUMN, NUMERIC_BOUNDS,
    REQUIRED_COLUMNS, TARGET, TARGET_MAPPING,
)


def data_audit(frame: pd.DataFrame) -> dict:
    """Return JSON-serializable structural diagnostics without changing data."""
    return {
        "shape": list(frame.shape),
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "numeric_columns": frame.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": frame.select_dtypes(exclude="number").columns.tolist(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": frame.isna().sum().astype(int).to_dict(),
        "constant_columns": [c for c in frame if frame[c].nunique(dropna=False) == 1],
    }


def validate_features(frame: pd.DataFrame, *, training: bool = False) -> None:
    """Reject malformed values; permit null predictors for pipeline imputation.

    Known-category checks are strict during training. At inference, nonempty
    novel categories are allowed and encoded with handle_unknown='ignore'.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Input must be a nonempty pandas DataFrame.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    for col, (lower, upper) in NUMERIC_BOUNDS.items():
        values = frame[col].dropna()
        if values.empty:
            continue
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError(f"{col} must contain numbers or null, not strings/booleans.")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{col} must contain finite numbers.")
        if (values % 1 != 0).any():
            raise ValueError(f"{col} must contain whole numbers for this dataset schema.")
        if (values < lower).any() or (upper is not None and (values > upper).any()):
            raise ValueError(f"{col} is outside its allowed range [{lower}, {upper}].")
    for col, allowed in CATEGORIES.items():
        values = frame[col].dropna()
        if not values.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"{col} must contain nonempty strings or null.")
        unexpected = sorted(set(values) - set(allowed))
        if unexpected and training:
            raise ValueError(f"Unexpected categorical values for {col}: {unexpected}")


def validate_data(frame: pd.DataFrame) -> dict:
    """Validate a raw training dataset and return its audit report.

    Missing predictors are reported and imputed within CV folds. Missing targets,
    duplicate observations/IDs, schema errors, and unrecognized labels fail.
    Constant-column assumptions are verified again on the training partition.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
    if missing or extra:
        raise ValueError(f"Invalid columns. Missing: {missing}; unexpected: {extra}")
    validate_features(frame, training=True)
    if frame[TARGET].isna().any() or not frame[TARGET].isin(TARGET_MAPPING).all():
        raise ValueError("Attrition must contain only 'Yes' and 'No', without nulls.")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Training requires both Attrition classes: Yes and No.")
    if frame.duplicated().any():
        raise ValueError("Duplicate rows found; resolve them before splitting.")
    ids = frame[ID_COLUMN]
    if (not pd.api.types.is_numeric_dtype(ids) or ids.isna().any()
            or not np.isfinite(ids.to_numpy(dtype=float)).all()
            or (ids <= 0).any() or (ids % 1 != 0).any() or ids.duplicated().any()):
        raise ValueError("EmployeeNumber must contain unique positive integer identifiers.")
    for col, expected in CONSTANT_COLUMNS.items():
        if frame[col].isna().any() or not frame[col].eq(expected).all():
            raise ValueError(f"{col} must be constant {expected!r}; revisit feature exclusions.")
    empty = [c for c in FEATURE_COLUMNS if frame[c].isna().all()]
    if empty:
        raise ValueError(f"Entirely missing predictor columns: {empty}")
    audit = data_audit(frame)
    if sum(audit["missing_values"].values()):
        warnings.warn("Missing predictors detected; imputation is fitted on training folds only.",
                      UserWarning, stacklevel=2)
    return audit
