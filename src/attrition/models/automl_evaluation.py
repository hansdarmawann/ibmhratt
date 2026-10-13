"""Nested CV of the optional AutoML procedure: python -m attrition.models.automl_evaluation --n-trials 30 --seed 42.

Install requirements-automl.txt first. Seeded stratified outer folds of the
training partition assess; each outer training fold runs the complete AutoML
procedure on its own inner folds. The Optuna search chooses a family and
hyperparameters, out-of-fold probabilities of that frozen configuration choose
the F2 threshold, and the refitted pipeline is scored once on the outer
validation rows. The served baseline procedure is scored on the same outer and
inner folds, so the comparison is paired. Results are training-only development
estimates: the holdout is never read, and no model, bundle, canonical baseline
report, or models/current.json is written. Optional experiment reports are isolated. Nothing runs at import time.
"""

import argparse
import hashlib
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
from attrition.models import automl_reporting as reporting
from attrition.models.evaluate import evaluate_probabilities
from attrition.models.nested import NESTED_METRICS, OUTER_FOLDS, nested_summary, run_procedure
from attrition.models.threshold import select_threshold, threshold_table
from attrition.models.train import candidate_pipelines, select_model

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 2
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


class NoSuccessfulSearch(RuntimeError):
    """Carry inspectable failure evidence out of a nested fold without fitting a candidate."""

    def __init__(self, result: dict):
        super().__init__("No successful AutoML trials in an outer fold; no nested estimate can be computed.")
        self.result = result
        self.outer_fold: int | None = None
        self.completed_folds: list[dict] = []


@dataclass(frozen=True)
class Choice:
    """A procedure's choices on outer training rows, frozen before its outer fold is scored."""

    model: str
    params: dict
    threshold: float
    pipeline: Pipeline  # Fitted on the outer training rows only.
    inner_cv_ap: float | None = None
    trials_completed: int | None = None
    search_result: dict | None = None


def automl_procedure(X: pd.DataFrame, y: pd.Series, *, n_trials: int, timeout: float | None, seed: int,
                     search=None) -> Choice:
    """The complete AutoML selection procedure, applied to (X, y) only.

    The search scores trials on seeded stratified inner folds and keeps the best
    mean AP. Only then do out-of-fold probabilities of that frozen configuration,
    on the same inner folds, choose the threshold, as in baseline training.
    Finally the configuration is refitted on all of (X, y). `search` defaults to
    the Optuna search; any replacement must return the same result shape.
    """
    search = automl.search if search is None else search
    result = search(X, y, n_trials=n_trials, timeout=timeout, seed=seed, folds=CV_FOLDS)
    best = result["best"]
    if best is None:
        raise NoSuccessfulSearch(result)
    pipeline = automl.candidate_pipeline(X, best["model"], best["params"], seed)
    inner = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    oof = cross_val_predict(clone(pipeline), X, y, cv=inner, method="predict_proba", n_jobs=1)[:, 1]
    threshold = select_threshold(threshold_table(y, oof))
    return Choice(best["model"], best["params"], threshold, pipeline.fit(X, y),
                  inner_cv_ap=best["cv_ap"]["mean"], trials_completed=result["trials"]["completed"], search_result=result)


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
                 outer_folds: int = OUTER_FOLDS, search_history: list[dict] | None = None) -> list[dict]:
    """One row per procedure and outer fold; each procedure sees only that fold's outer training rows."""
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    # Built from column types only, as in baseline training's nested CV; every fit happens inside a fold.
    pipelines = candidate_pipelines(X)
    rows: list[dict] = []
    for fold, (fit, validation) in enumerate(outer.split(X, y)):
        LOGGER.info("Outer fold %d of %d: %d training rows", fold + 1, outer_folds, len(fit))
        X_fit, y_fit = X.iloc[fit], y.iloc[fit]
        X_validation, y_validation = X.iloc[validation], y.iloc[validation]
        try:
            choice = automl_procedure(X_fit, y_fit, n_trials=n_trials, timeout=timeout, seed=seed)
        except NoSuccessfulSearch as exc:
            if search_history is not None:
                search_history.append({"outer_fold": fold, "result": exc.result})
            exc.outer_fold, exc.completed_folds = fold, rows
            raise
        if search_history is not None:
            search_history.append({"outer_fold": fold, "result": choice.search_result})
        rows.append(fold_row(AUTOML, fold, choice, X_validation, y_validation))
        rows.append(fold_row(BASELINE, fold, baseline_procedure(pipelines, X_fit, y_fit, seed),
                             X_validation, y_validation))
    return rows


def paired_differences(rows: list[dict], first: str = AUTOML, second: str = BASELINE) -> dict:
    """First minus second procedure on each outer fold, per metric; positive values favor the first."""
    frame = pd.DataFrame(rows)
    result = {}
    for metric in NESTED_METRICS:
        table = frame.pivot(index="outer_fold", columns="procedure", values=metric).astype(float)
        differences = table[first] - table[second]
        result[metric] = {"mean": float(differences.mean()), "std": float(differences.std(ddof=0)),
                          "min": float(differences.min()), "max": float(differences.max()),
                          f"{first}_higher_folds": int((differences > 0).sum()),
                          f"{second.removesuffix('_fixed')}_higher_folds": int((differences < 0).sum())}
    return result


def selection_optimism(rows: list[dict], procedure: str = AUTOML) -> dict:
    """How far the best trial's inner CV AP overstated the AP the procedure achieved on outer folds."""
    automl_rows = [row for row in rows if row["procedure"] == procedure]
    inner = float(np.mean([row["inner_cv_ap"] for row in automl_rows]))
    outer = float(np.mean([row["pr_auc"] for row in automl_rows]))
    return {"mean_inner_best_trial_cv_ap": inner, "mean_outer_ap": outer, "difference": inner - outer}


def nested_evaluation(X: pd.DataFrame, y: pd.Series, *, n_trials: int = automl.DEFAULT_TRIALS,
                      timeout: float | None = automl.DEFAULT_TIMEOUT, seed: int = RANDOM_STATE,
                      outer_folds: int = OUTER_FOLDS) -> dict:
    """Nested CV of both procedures on the given training rows, with a paired comparison."""
    searches: list[dict] = []
    failure = None
    try:
        rows = nested_folds(X, y, n_trials=n_trials, timeout=timeout, seed=seed, outer_folds=outer_folds,
                            search_history=searches)
    except NoSuccessfulSearch as exc:
        rows = exc.completed_folds
        failure = {"outer_fold": exc.outer_fold, "message": str(exc)}
    environment = reporting.provenance()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if failure is None else "incomplete",
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
        **({"procedures": nested_summary(pd.DataFrame(rows)),
            "comparison": {"difference": f"{AUTOML} minus {BASELINE} on the same outer fold",
                           "metrics": paired_differences(rows)},
            "automl_selection_optimism": selection_optimism(rows)} if failure is None else {"failure": failure}),
        "folds": rows, "searches": searches,
        "train_rows": len(X), "train_positives": int(y.sum()),
        "provenance": environment, "package_versions": environment["package_versions"],
        "training_data": reporting.training_metadata(X, y),
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
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--report-dir", type=normalize_path,
                         help="Write trial history and paired development comparison into a new run directory")
    parser.add_argument("--run-id", help="New report run ID (default: UUID); requires --report-dir")
    outputs.add_argument("--output", type=normalize_path, default=None,
                        help="Also write the JSON result to this new file (never inside the models directory)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.run_id is not None and args.report_dir is None:
        parser.error("--run-id requires --report-dir")
    report_path = reporting.run_path(args.report_dir, args.run_id) if args.report_dir else None
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
    if report_path:
        result["run_id"] = report_path.name
    text = reporting.json_text(result)
    if report_path:
        reporting.write_report(result, report_path)
        LOGGER.info("Wrote report run %s", report_path)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text)
        LOGGER.info("Wrote %s", output)
    print(text, end="")
    if result.get("status") == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
