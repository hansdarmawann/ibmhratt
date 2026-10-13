"""Nested CV of the optional AutoML procedure: python -m attrition.models.automl_evaluation --n-trials 30 --seed 42.

Install requirements-automl.txt first. Seeded stratified outer folds of the
training partition assess; each outer training fold runs the complete AutoML
procedure on its own inner folds. The Optuna search chooses a family and
hyperparameters, out-of-fold probabilities of that frozen configuration choose
the F2 threshold, and the refitted pipeline is scored once on the outer
validation rows. The served baseline procedure is scored on the same outer and
inner folds, so the comparison is paired. Results are training-only development
estimates: the holdout is never read, and no model, bundle, report, or
models/current.json is written. Nothing runs at import time.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from attrition.config import CV_FOLDS, DATA_ENV, DATA_PATH, DATA_RELATIVE, RANDOM_STATE, normalize_path
from attrition.data.load_data import load_data
from attrition.models import automl
from attrition.models.evaluate import evaluate_probabilities
from attrition.models.nested import NESTED_METRICS, OUTER_FOLDS, nested_summary, run_procedure
from attrition.models.threshold import select_threshold, threshold_table
from attrition.models.train import candidate_pipelines, select_model

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
AUTOML, BASELINE = "automl", "baseline_fixed"
THRESHOLD_RULE = ("Maximize F2 on inner out-of-fold probabilities of the frozen configuration, using outer "
                  "training rows only; ties: precision, then the higher threshold")
INTERPRETATION = (
    "Training-only development estimate, not a final test. Both procedures are scored on the same outer folds, "
    "so per-fold differences are paired, but the folds share most of their training rows and are not independent: "
    "a standard error or t-test computed from them would overstate certainty, so none is reported. On this small "
    "fictional dataset, a small mean difference, or most folds favoring one procedure, is not evidence that it "
    "generalizes better. The published holdout was already examined and must not settle the comparison; "
    "confirming a difference needs new independent data.")


@dataclass(frozen=True)
class Choice:
    """A procedure's choices on outer training rows, frozen before its outer fold is scored."""

    model: str
    params: dict
    threshold: float
    pipeline: Pipeline  # Fitted on the outer training rows only.
    inner_cv_ap: float | None = None
    trials_completed: int | None = None


def automl_procedure(X: pd.DataFrame, y: pd.Series, *, n_trials: int, timeout: float | None, seed: int) -> Choice:
    """The complete AutoML selection procedure, applied to (X, y) only.

    The search scores trials on seeded stratified inner folds and keeps the best
    mean AP. Only then do out-of-fold probabilities of that frozen configuration,
    on the same inner folds, choose the threshold, as in baseline training.
    Finally the configuration is refitted on all of (X, y).
    """
    result = automl.search(X, y, n_trials=n_trials, timeout=timeout, seed=seed, folds=CV_FOLDS)
    best = result["best"]
    pipeline = automl.candidate_pipeline(X, best["model"], best["params"], seed)
    inner = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    oof = cross_val_predict(clone(pipeline), X, y, cv=inner, method="predict_proba", n_jobs=1)[:, 1]
    threshold = select_threshold(threshold_table(y, oof))
    return Choice(best["model"], best["params"], threshold, pipeline.fit(X, y),
                  inner_cv_ap=best["cv_ap"]["mean"], trials_completed=result["trials"]["completed"])


def baseline_procedure(pipelines: dict, X: pd.DataFrame, y: pd.Series, seed: int) -> Choice:
    """The served procedure (fixed candidates, prefer-logistic AP rule, OOF F2 threshold) on (X, y) only."""
    selected, params, threshold, model = run_procedure(pipelines, X, y, select_model, tuned=False, seed=seed)
    return Choice(selected, params, threshold, model)


def fold_row(procedure: str, fold: int, choice: Choice, X: pd.DataFrame, y: pd.Series) -> dict:
    """Score a frozen choice once on its outer validation rows."""
    metrics = evaluate_probabilities(y, choice.pipeline.predict_proba(X)[:, 1], choice.threshold)
    return {"procedure": procedure, "outer_fold": fold, "selected_model": choice.model,
            "parameters": choice.params, "threshold": choice.threshold, "inner_cv_ap": choice.inner_cv_ap,
            "trials_completed": choice.trials_completed, **{metric: metrics[metric] for metric in NESTED_METRICS}}


def nested_folds(X: pd.DataFrame, y: pd.Series, *, n_trials: int, timeout: float | None, seed: int = RANDOM_STATE,
                 outer_folds: int = OUTER_FOLDS) -> list[dict]:
    """One row per procedure and outer fold; each procedure sees only that fold's outer training rows."""
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    # Built from column types only, as in baseline training's nested CV; every fit happens inside a fold.
    pipelines = candidate_pipelines(X)
    rows = []
    for fold, (fit, validation) in enumerate(outer.split(X, y)):
        LOGGER.info("Outer fold %d of %d: %d training rows", fold + 1, outer_folds, len(fit))
        X_fit, y_fit = X.iloc[fit], y.iloc[fit]
        X_validation, y_validation = X.iloc[validation], y.iloc[validation]
        choice = automl_procedure(X_fit, y_fit, n_trials=n_trials, timeout=timeout, seed=seed)
        rows.append(fold_row(AUTOML, fold, choice, X_validation, y_validation))
        rows.append(fold_row(BASELINE, fold, baseline_procedure(pipelines, X_fit, y_fit, seed),
                             X_validation, y_validation))
    return rows


def paired_differences(rows: list[dict]) -> dict:
    """AutoML minus baseline on each outer fold, per metric; positive values favor AutoML."""
    frame = pd.DataFrame(rows)
    result = {}
    for metric in NESTED_METRICS:
        table = frame.pivot(index="outer_fold", columns="procedure", values=metric).astype(float)
        differences = table[AUTOML] - table[BASELINE]
        result[metric] = {"mean": float(differences.mean()), "std": float(differences.std(ddof=0)),
                          "min": float(differences.min()), "max": float(differences.max()),
                          "automl_higher_folds": int((differences > 0).sum()),
                          "baseline_higher_folds": int((differences < 0).sum())}
    return result


def selection_optimism(rows: list[dict]) -> dict:
    """How far the best trial's inner CV AP overstated the AP the procedure achieved on outer folds."""
    automl_rows = [row for row in rows if row["procedure"] == AUTOML]
    inner = float(np.mean([row["inner_cv_ap"] for row in automl_rows]))
    outer = float(np.mean([row["pr_auc"] for row in automl_rows]))
    return {"mean_inner_best_trial_cv_ap": inner, "mean_outer_ap": outer, "difference": inner - outer}


def nested_evaluation(X: pd.DataFrame, y: pd.Series, *, n_trials: int = automl.DEFAULT_TRIALS,
                      timeout: float | None = automl.DEFAULT_TIMEOUT, seed: int = RANDOM_STATE,
                      outer_folds: int = OUTER_FOLDS) -> dict:
    """Nested CV of both procedures on the given training rows, with a paired comparison."""
    rows = nested_folds(X, y, n_trials=n_trials, timeout=timeout, seed=seed, outer_folds=outer_folds)
    return {
        "schema_version": SCHEMA_VERSION,
        "objective": automl.OBJECTIVE,
        "method": "Nested CV on the training partition: each outer training fold runs the complete procedure on "
                  "its own inner folds; the outer validation rows score the frozen result once.",
        "outer_cv": {"partition": "training", "folds": outer_folds, "stratified": True, "shuffle": True,
                     "random_state": seed},
        "inner_cv": {"partition": "outer training fold", "folds": CV_FOLDS, "stratified": True, "shuffle": True,
                     "random_state": seed, "shared_by": [AUTOML, BASELINE]},
        "automl": {"families": list(automl.FAMILIES), "sampler": {"name": "TPESampler", "seed": seed},
                   "budget_per_outer_fold": {"n_trials": n_trials, "timeout_seconds": timeout, "n_jobs": 1}},
        "baseline": "The served procedure: fixed candidates, highest inner CV AP preferring logistic regression "
                    "within 0.01, then the inner OOF F2 threshold",
        "threshold_rule": THRESHOLD_RULE,
        "procedures": nested_summary(pd.DataFrame(rows)),
        "comparison": {"difference": f"{AUTOML} minus {BASELINE} on the same outer fold",
                       "metrics": paired_differences(rows)},
        "automl_selection_optimism": selection_optimism(rows),
        "folds": rows,
        "train_rows": len(X), "train_positives": int(y.sum()),
        "package_versions": {name: importlib.metadata.version(name)
                             for name in ["optuna", "scikit-learn", "numpy", "pandas"]},
        "holdout_used": False,
        "served_model_changed": False,
        "interpretation": INTERPRETATION,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=automl.positive_int, default=automl.DEFAULT_TRIALS,
                        help=f"Trial budget of each outer fold's search (default: {automl.DEFAULT_TRIALS})")
    parser.add_argument("--timeout", type=automl.positive_float, default=automl.DEFAULT_TIMEOUT,
                        help="Seconds after which each outer fold's search starts no new trial "
                             f"(default: {automl.DEFAULT_TIMEOUT:g})")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help=f"Seed for outer and inner folds, the sampler, and estimators (default: {RANDOM_STATE})")
    parser.add_argument("--outer-folds", type=int, default=OUTER_FOLDS, choices=range(2, 11), metavar="{2..10}",
                        help=f"Stratified outer folds of the training partition (default: {OUTER_FOLDS})")
    parser.add_argument("--data", type=normalize_path, default=DATA_PATH,
                        help=f"Raw CSV (default: ${DATA_ENV} or <root>/{DATA_RELATIVE.as_posix()})")
    parser.add_argument("--output", type=normalize_path, default=None,
                        help="Also write the JSON result to this new file (never inside the models directory)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output = automl.check_output(args.output) if args.output else None
    data = load_data(args.data)
    X_train, y_train = automl.training_partition(data)
    started = time.perf_counter()
    # Single-threaded math, as in training, keeps folds and trials reproducible.
    with threadpool_limits(limits=1):
        result = nested_evaluation(X_train, y_train, n_trials=args.n_trials, timeout=args.timeout, seed=args.seed,
                                   outer_folds=args.outer_folds)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
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
