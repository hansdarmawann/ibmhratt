"""Optional Optuna search on training CV: python -m attrition.models.automl --n-trials 30 --timeout 900 --seed 42.

Install requirements-automl.txt first. The search proposes a candidate model
family and hyperparameters, scored only by stratified CV average precision on the
training partition. It never reads the holdout, fits a final model, selects a
threshold, writes a bundle, or changes models/current.json; the served baseline
from attrition.models.train is unchanged. Nothing runs at import time.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from threadpoolctl import threadpool_limits

from attrition.config import CV_FOLDS, DATA_ENV, DATA_PATH, DATA_RELATIVE, MODELS_DIR, RANDOM_STATE, normalize_path
from attrition.data.load_data import load_data, split_data
from attrition.features.preprocess import build_pipeline

if TYPE_CHECKING:
    import optuna

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
OBJECTIVE = "average_precision"
FAMILIES = ("logistic_regression", "random_forest", "hist_gradient_boosting")
# Suggested budget for manual experiments; tests and CI pass much smaller ones.
DEFAULT_TRIALS, DEFAULT_TIMEOUT = 30, 900.0
INSTALL_HINT = "python -m pip install -r requirements-automl.txt -c requirements-lock.txt"


def require_optuna():
    """Import Optuna on demand so the baseline package never needs it."""
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"AutoML needs the optional Optuna dependency: {INSTALL_HINT}") from exc
    return optuna


def training_partition(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """The baseline's training rows only; the holdout split is discarded before any search."""
    X_train, _, y_train, _ = split_data(data)
    return X_train, y_train


def suggest_estimator(trial: "optuna.Trial", seed: int) -> tuple[str, Any]:
    """Sample a family, then only that family's parameters, in ranges sized for about 1,000 rows.

    Parameter names carry the family prefix so conditional spaces never share a name.
    Logistic regression keeps the default L2/lbfgs solver, valid on every supported
    scikit-learn version, and tunes regularization strength and class weighting.
    """
    family = trial.suggest_categorical("model", FAMILIES)
    if family == "logistic_regression":
        return family, LogisticRegression(
            C=trial.suggest_float("logistic_regression__C", 0.01, 10.0, log=True),
            class_weight=trial.suggest_categorical("logistic_regression__class_weight", [None, "balanced"]),
            max_iter=2000, random_state=seed)
    if family == "random_forest":
        return family, RandomForestClassifier(
            n_estimators=trial.suggest_int("random_forest__n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("random_forest__max_depth", 3, 10),
            min_samples_leaf=trial.suggest_int("random_forest__min_samples_leaf", 2, 20),
            max_features=trial.suggest_categorical("random_forest__max_features", ["sqrt", 0.3, 0.5]),
            class_weight=trial.suggest_categorical("random_forest__class_weight", [None, "balanced_subsample"]),
            n_jobs=1, random_state=seed)
    return family, HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("hist_gradient_boosting__learning_rate", 0.01, 0.2, log=True),
        max_iter=trial.suggest_int("hist_gradient_boosting__max_iter", 50, 300, step=25),
        max_leaf_nodes=trial.suggest_int("hist_gradient_boosting__max_leaf_nodes", 4, 31),
        min_samples_leaf=trial.suggest_int("hist_gradient_boosting__min_samples_leaf", 10, 50),
        l2_regularization=trial.suggest_float("hist_gradient_boosting__l2_regularization", 1e-3, 10.0, log=True),
        class_weight=trial.suggest_categorical("hist_gradient_boosting__class_weight", [None, "balanced"]),
        early_stopping=False, random_state=seed)


def cv_scores(pipeline, X: pd.DataFrame, y: pd.Series, cv) -> list[float]:
    """Per-fold AP; the whole pipeline, including preprocessing, is refitted inside each training fold."""
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=OBJECTIVE, n_jobs=1, error_score="raise")
    # Rounding keeps scores, and so the sampler's history, identical across platforms.
    return [round(float(score), 10) for score in scores]


def summarize(scores: list[float]) -> dict:
    return {"mean": round(float(np.mean(scores)), 10), "std": round(float(np.std(scores)), 10), "folds": scores}


def readable(params: dict) -> dict:
    """Drop the family prefix, so parameters read like the estimator's own arguments."""
    return {key.split("__", 1)[1]: value for key, value in params.items() if "__" in key}


def search(X: pd.DataFrame, y: pd.Series, *, n_trials: int = DEFAULT_TRIALS,
           timeout: float | None = DEFAULT_TIMEOUT, seed: int = RANDOM_STATE, folds: int = CV_FOLDS) -> dict:
    """Run a seeded, serial TPE search on the given training rows and return the best candidate.

    Every trial is scored by the same seeded stratified folds of (X, y). A fixed
    trial count reproduces the same result; a timeout that stops the search early
    makes the number of completed trials depend on machine speed.
    """
    optuna = require_optuna()
    if n_trials < 1 or folds < 2 or (timeout is not None and not timeout > 0):
        raise ValueError("Use at least one trial, two folds, and a positive timeout.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    def objective(trial) -> float:
        family, estimator = suggest_estimator(trial, seed)
        summary = summarize(cv_scores(build_pipeline(X, estimator, scale=family == "logistic_regression"), X, y, cv))
        trial.set_user_attr("cv_ap", summary)
        return summary["mean"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
    elapsed = time.perf_counter() - started
    # Optuna always finishes at least one trial; a failing fit raises instead of being skipped.
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial
    reference = cv_scores(build_pipeline(X, DummyClassifier(strategy="prior")), X, y, cv)
    LOGGER.info("Best trial %d: %s, training CV AP %.3f", best.number, best.params["model"], best.value)
    return {
        "schema_version": SCHEMA_VERSION,
        "objective": OBJECTIVE, "direction": "maximize",
        "cv": {"partition": "training", "folds": folds, "stratified": True, "shuffle": True, "random_state": seed},
        "sampler": {"name": "TPESampler", "seed": seed},
        "budget": {"n_trials": n_trials, "timeout_seconds": timeout, "n_jobs": 1},
        "trials": {"completed": len(completed), "total": len(study.trials)},
        "elapsed_seconds": round(elapsed, 3),
        "best": {"trial": best.number, "model": best.params["model"], "params": readable(best.params),
                 "cv_ap": best.user_attrs["cv_ap"]},
        "reference": {"dummy_prior": {"tuned": False, "cv_ap": summarize(reference)}},
        "train_rows": len(X), "train_positives": int(y.sum()),
        "package_versions": {name: importlib.metadata.version(name)
                             for name in ["optuna", "scikit-learn", "numpy", "pandas"]},
        "holdout_used": False,
        "served_model_changed": False,
        "note": "Training-CV development estimate of a candidate configuration; the chosen trial's CV AP is "
                "optimistic because the same folds selected it. No final model, threshold, or holdout score.",
    }


def check_output(path: Path, models_dir: Path = MODELS_DIR) -> Path:
    """Write only new files outside the models directory, so a search can never touch bundles or current.json."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new output file.")
    if path.is_relative_to(models_dir.resolve()):
        raise ValueError(f"Write search results outside the models directory ({models_dir}).")
    return path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=positive_int, default=DEFAULT_TRIALS,
                        help=f"Trial budget (default: {DEFAULT_TRIALS})")
    parser.add_argument("--timeout", type=positive_float, default=DEFAULT_TIMEOUT,
                        help=f"Stop starting new trials after this many seconds (default: {DEFAULT_TIMEOUT:g})")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help=f"Seed for the sampler, folds, and estimators (default: {RANDOM_STATE})")
    parser.add_argument("--folds", type=int, default=CV_FOLDS, choices=range(2, 11), metavar="{2..10}",
                        help=f"Stratified training CV folds (default: {CV_FOLDS})")
    parser.add_argument("--data", type=normalize_path, default=DATA_PATH,
                        help=f"Raw CSV (default: ${DATA_ENV} or <root>/{DATA_RELATIVE.as_posix()})")
    parser.add_argument("--output", type=normalize_path, default=None,
                        help="Also write the JSON result to this new file (never inside the models directory)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output = check_output(args.output) if args.output else None
    data = load_data(args.data)
    X_train, y_train = training_partition(data)
    # Single-threaded math, as in training, keeps trials reproducible and avoids oversubscription.
    with threadpool_limits(limits=1):
        result = search(X_train, y_train, n_trials=args.n_trials, timeout=args.timeout, seed=args.seed,
                        folds=args.folds)
    result["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text)
        LOGGER.info("Wrote %s", output)
    print(text, end="")


if __name__ == "__main__":
    main()
