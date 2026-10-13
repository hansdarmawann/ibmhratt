"""Portable AutoML summaries; no pickle loading, training, or baseline report mutation."""

import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
from pathlib import Path, PureWindowsPath
from uuid import uuid4

import pandas as pd
from threadpoolctl import threadpool_info

from attrition.config import CHECKOUT_ROOT, MODELS_DIR, ROOT

REPORT_SCHEMA_VERSION = 1
PACKAGES = ("ibmhratt", "optuna", "scikit-learn", "numpy", "pandas", "scipy", "threadpoolctl")
TRIAL_COLUMNS = ("outer_fold", "number", "state", "model", "sampled_params", "estimator_params", "cv_ap",
                 "value", "failure", "elapsed_seconds")
COMPARISON_COLUMNS = ("metric", "automl_mean", "automl_std", "baseline_mean", "baseline_std", "paired_difference")


def json_text(value) -> str:
    """Canonical ordering and strict numbers; runtime measurements remain variable."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def provenance() -> dict:
    """Read-only Git provenance where source is a checkout; absent Git is valid for a wheel."""
    revision, dirty = None, None
    if CHECKOUT_ROOT is not None:
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CHECKOUT_ROOT, capture_output=True,
                                      text=True, check=True, timeout=5).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=CHECKOUT_ROOT, capture_output=True,
                                        text=True, check=True, timeout=5).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            revision, dirty = None, None
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"git_revision": revision, "git_dirty": dirty, "package_versions": versions,
            "python": platform.python_version(), "platform": platform.platform(),
            "threadpools": threadpool_info()}


def training_metadata(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fingerprint only training inputs, including row order/index; no raw rows are serialized."""
    digest = hashlib.sha256()
    for value in (X, y):
        digest.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
    columns = [{"name": str(name), "dtype": str(dtype)} for name, dtype in X.dtypes.items()]
    digest.update(json_text(columns).encode("utf-8"))
    return {"columns": columns, "sha256": digest.hexdigest(), "target_mapping": {"No": 0, "Yes": 1},
            "preprocessing": "attrition.features.preprocess.build_pipeline", "engineered_features": False}


def check_destination(path: Path, models_dir: Path = MODELS_DIR) -> Path:
    """Normalize operator paths and protect serving state and canonical report/data directories."""
    path = path.expanduser().resolve()
    roots = {ROOT.resolve()}
    if CHECKOUT_ROOT is not None:
        roots.add(CHECKOUT_ROOT.resolve())
    protected = [models_dir.resolve(), *((root / folder).resolve() for root in roots
                  for folder in ("models", "reports/metrics", "reports/figures", "data"))]
    if any(path.is_relative_to(folder) for folder in protected):
        raise ValueError("Write results outside the models directory and canonical baseline report/data directories.")
    if any(path == (root / name).resolve() for root in roots for name in ("README.md", "reports/results.md")):
        raise ValueError("Refusing to write a canonical baseline report.")
    return path


def run_path(directory: Path, run_id: str | None = None) -> Path:
    """Preflight without creating files; the exclusive mkdir during writing resolves races."""
    run_id = str(uuid4()) if run_id is None else run_id
    if (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id) is None
            or run_id.endswith(".") or PureWindowsPath(run_id).is_reserved()):
        raise ValueError("Run ID must be 1-64 portable letters/digits/dots/underscores/hyphens, starting with a "
                         "letter/digit; reserved names and trailing dots are refused.")
    path = check_destination(directory / run_id)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite run {path}; choose a new run ID.")
    return path


def csv_text(rows: list[dict], columns: tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(row[key], sort_keys=True, allow_nan=False)
                         if isinstance(row.get(key), (dict, list)) else row.get(key) for key in columns})
    return stream.getvalue()


def trial_rows(result: dict) -> list[dict]:
    if "searches" in result:
        return [{"outer_fold": search["outer_fold"], **trial}
                for search in sorted(result["searches"], key=lambda item: item["outer_fold"])
                for trial in sorted(search["result"]["trial_records"], key=lambda item: item["number"])]
    return sorted(result["trial_records"], key=lambda item: item["number"])


def comparison_rows(result: dict) -> list[dict]:
    """Aligned nested development estimates only; never read baseline holdout artifacts."""
    if "procedures" not in result:
        return []
    automl, baseline = result["procedures"]["automl"], result["procedures"]["baseline_fixed"]
    return [{"metric": metric, "automl_mean": automl[metric]["mean"], "automl_std": automl[metric]["std"],
             "baseline_mean": baseline[metric]["mean"], "baseline_std": baseline[metric]["std"],
             "paired_difference": result["comparison"]["metrics"][metric]["mean"]}
            for metric in sorted(result["comparison"]["metrics"])]


def report_files(result: dict) -> dict[str, str]:
    """Pure rendering: the same metadata produces byte-identical UTF-8 JSON/CSV/Markdown."""
    comparisons = comparison_rows(result)
    lines = ["# AutoML development report", "", result.get("interpretation", result.get("note", "")), ""]
    if comparisons:
        lines += ["AP is average precision (`pr_auc`); SD is population fold SD, not a confidence interval.", "",
                  "| Metric | AutoML mean ± SD | Baseline mean ± SD | Paired AutoML − baseline |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {row['metric']} | {row['automl_mean']:.6f} ± {row['automl_std']:.6f} | "
                  f"{row['baseline_mean']:.6f} ± {row['baseline_std']:.6f} | {row['paired_difference']:+.6f} |"
                  for row in comparisons]
    elif result.get("status") == "incomplete":
        lines += ["Nested evaluation incomplete: no procedure comparison estimates are reported. "
                  "Inspect the failed outer fold and its trial records in result.json and trials.csv."]
    else:
        lines += ["Search-only run: no baseline procedure comparison was performed. Use nested evaluation "
                  "to compare aligned development-CV estimates."]
    lines += ["", "No holdout scores, model bundles, or serving changes are produced.", ""]
    return {"result.json": json_text({**result, "report_schema_version": REPORT_SCHEMA_VERSION}),
            "trials.csv": csv_text(trial_rows(result), TRIAL_COLUMNS),
            "comparison.csv": csv_text(comparisons, COMPARISON_COLUMNS),
            "report.md": "\n".join(lines)}


def write_report(result: dict, path: Path) -> None:
    """Create one new run directory; never reuse it, even after an interrupted write."""
    files = report_files(result)  # Validate serialization before creating anything.
    path = check_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    for name, contents in files.items():
        with (path / name).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
