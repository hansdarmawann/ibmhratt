"""Input drift check: python -m attrition.monitoring.drift records.csv [--output report.csv].

Training stores a reference profile of the training rows in every bundle. The
Population Stability Index (PSI) compares each feature's binned distribution in
new records with that profile. Drift means inputs differ from training data; it
does not show that predictions are wrong or that attrition itself changed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from attrition.config import CATEGORIES, CURRENT_RUN, FEATURE_COLUMNS, NUMERIC_BOUNDS
from attrition.data.validate_data import validate_features
from attrition.models.artifacts import load_bundle

LOGGER = logging.getLogger(__name__)
PROFILE_FILE = "reference_profile.json"
PSI_MODERATE, PSI_SHIFT = 0.1, 0.25
# Empty bins would make PSI infinite; a small floor keeps it finite and comparable.
EPSILON = 1e-4
# PSI on few rows mostly measures sampling noise.
MIN_RELIABLE_ROWS = 100


def bin_shares(values: pd.Series, cuts: list[float]) -> list[float] | None:
    """Shares in bins (-inf, c0], (c0, c1], ..., (c_last, inf); None without values."""
    if values.empty:
        return None
    counts = np.bincount(np.searchsorted(cuts, values.to_numpy(dtype=float), side="left"), minlength=len(cuts) + 1)
    return (counts / counts.sum()).tolist()


def reference_profile(X: pd.DataFrame) -> dict:
    """Training deciles for numeric features and category shares; fitted on training rows only."""
    numeric = {}
    for column in NUMERIC_BOUNDS:
        values = X[column].dropna().astype(float)
        cuts = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9))).tolist()
        numeric[column] = {"cuts": cuts, "shares": bin_shares(values, cuts),
                           "missing_rate": float(X[column].isna().mean())}
    categorical = {}
    for column in CATEGORIES:
        shares = X[column].dropna().astype(str).value_counts(normalize=True).sort_index()
        categorical[column] = {"shares": {name: float(share) for name, share in shares.items()},
                               "missing_rate": float(X[column].isna().mean())}
    return {"schema_version": 1, "rows": len(X), "numeric": numeric, "categorical": categorical}


def psi(expected: list[float], actual: list[float]) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    a = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float | None) -> str:
    if value is None:
        return "no data"
    if value >= PSI_SHIFT:
        return "shift"
    return "moderate" if value >= PSI_MODERATE else "stable"


def drift_report(frame: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """One row per feature, largest PSI first. Unseen categories form their own bin."""
    rows = []
    for column, reference in profile["numeric"].items():
        current = bin_shares(frame[column].dropna().astype(float), reference["cuts"])
        value = psi(reference["shares"], current) if current is not None else None
        rows.append({"feature": column, "kind": "numeric", "psi": value, "status": status(value),
                     "reference_missing_rate": reference["missing_rate"],
                     "current_missing_rate": float(frame[column].isna().mean()), "unseen_share": None})
    for column, reference in profile["categorical"].items():
        known = list(reference["shares"])
        observed = frame[column].dropna().astype(str).value_counts(normalize=True)
        value, unseen = None, None
        if not observed.empty:
            unseen = float(observed[~observed.index.isin(known)].sum())
            current = [float(observed.get(name, 0.0)) for name in known] + [unseen]
            value = psi([*reference["shares"].values(), 0.0], current)
        rows.append({"feature": column, "kind": "categorical", "psi": value, "status": status(value),
                     "reference_missing_rate": reference["missing_rate"],
                     "current_missing_rate": float(frame[column].isna().mean()), "unseen_share": unseen})
    report = pd.DataFrame(rows)
    return report.sort_values("psi", ascending=False, na_position="last", kind="stable", ignore_index=True)


def read_records(path: Path) -> pd.DataFrame:
    """CSV, or JSON as one object, a list of objects, or {"employees": [...]} like /predict/batch."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("employees", [data])
    return pd.DataFrame(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", type=Path, help="CSV or JSON employee records")
    parser.add_argument("--bundle", type=Path, default=CURRENT_RUN, help="Bundle directory or active-run pointer")
    parser.add_argument("--output", type=Path, help="Optional CSV path for the full report")
    parser.add_argument("--fail-on-shift", action="store_true", help="Exit with status 1 when any feature shifts")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bundle = load_bundle(args.bundle)
    profile_path = bundle.path / PROFILE_FILE
    if not profile_path.is_file():
        parser.error(f"Run {bundle.run_id} has no {PROFILE_FILE}; retrain to create one.")
    frame = read_records(args.records)
    validate_features(frame)
    if len(frame) < MIN_RELIABLE_ROWS:
        LOGGER.warning("Only %d records; PSI below %d rows mostly reflects sampling noise.", len(frame),
                       MIN_RELIABLE_ROWS)
    report = drift_report(frame[FEATURE_COLUMNS], json.loads(profile_path.read_text(encoding="utf-8")))
    print(report.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    counts = report["status"].value_counts()
    LOGGER.info("Run %s, %d records: %s", bundle.run_id, len(frame),
                ", ".join(f"{counts.get(name, 0)} {name}" for name in ["shift", "moderate", "stable", "no data"]))
    if args.output:
        report.to_csv(args.output, index=False)
    return 1 if args.fail_on_shift and counts.get("shift", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
