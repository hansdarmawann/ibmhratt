"""Nested cross-validation of the whole selection procedure; diagnostics only.

The CV AP reported for the selected model is optimistic: the same folds chose the
model and estimated it. Nested CV repeats the complete procedure (candidate
comparison, the prefer-logistic rule, and the F2 threshold) inside each outer
training fold, then scores it once on the untouched outer fold. A second,
"tuned" procedure also searches small prespecified grids in the inner folds.
Neither result changes the served model, whose selection rule was fixed before
holdout evaluation. Everything here uses the training partition only.
"""

import json
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from attrition.config import CV_FOLDS, RANDOM_STATE
from attrition.models.evaluate import evaluate_probabilities
from attrition.models.threshold import select_threshold, threshold_table

LOGGER = logging.getLogger(__name__)
OUTER_FOLDS = 5
# Fewer inner folds keep the grid search affordable inside every outer fold.
TUNING_FOLDS = 3
# Small grids around the fixed settings, chosen before any tuning results were seen.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logistic_regression": {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    "logistic_balanced": {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    "random_forest": {"model__max_depth": [5, 7], "model__min_samples_leaf": [3, 5, 10]},
    "hist_gradient_boosting": {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [7, 15]},
}
NESTED_METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1"]


def mean_ap(pipeline, X: pd.DataFrame, y: pd.Series, cv) -> float:
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="average_precision", n_jobs=1, error_score="raise")
    # Rounding makes ties, resolved by grid order, identical across platforms.
    return round(float(scores.mean()), 10)


def tune(pipeline, grid: dict, X: pd.DataFrame, y: pd.Series, cv) -> tuple[dict, float]:
    """Best parameters by mean AP; ties keep the earlier grid entry."""
    best_params, best_score = {}, -np.inf
    for params in ParameterGrid(grid):
        score = mean_ap(clone(pipeline).set_params(**params), X, y, cv)
        if score > best_score:
            best_params, best_score = params, score
    return best_params, best_score


def run_procedure(pipelines: dict, X: pd.DataFrame, y: pd.Series, selector, *, tuned: bool,
                  seed: int = RANDOM_STATE, grids: dict | None = None) -> tuple[str, dict, float, Pipeline]:
    """Apply the selection procedure to (X, y) only; return model name, parameters, threshold, fitted model."""
    grids = PARAM_GRIDS if grids is None else grids
    cv = StratifiedKFold(TUNING_FOLDS if tuned else CV_FOLDS, shuffle=True, random_state=seed)
    summaries, candidates = {}, {}
    for name, pipeline in pipelines.items():
        if name == "dummy":  # The selection rule never chooses the baseline.
            continue
        params, score = tune(pipeline, grids[name], X, y, cv) if tuned else ({}, mean_ap(pipeline, X, y, cv))
        summaries[name] = {"pr_auc": {"mean": score}}
        candidates[name] = (clone(pipeline).set_params(**params), params)
    selected, _ = selector(summaries)
    model, params = candidates[selected]
    oof = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold = select_threshold(threshold_table(y, oof))
    return selected, params, threshold, clone(model).fit(X, y)


def outer_fold(pipelines: dict, X: pd.DataFrame, y: pd.Series, selector, fold: int, fit, test,
               seed: int, grids: dict) -> list[dict]:
    """Both procedures for one outer fold; runs in a worker process with single-threaded math."""
    rows = []
    with threadpool_limits(limits=1):
        for procedure, tuned in [("fixed", False), ("tuned", True)]:
            selected, params, threshold, model = run_procedure(pipelines, X.iloc[fit], y.iloc[fit], selector,
                                                               tuned=tuned, seed=seed, grids=grids)
            metrics = evaluate_probabilities(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1], threshold)
            readable = {key.removeprefix("model__"): value for key, value in params.items()}
            rows.append({"procedure": procedure, "outer_fold": fold, "selected_model": selected,
                         "parameters": json.dumps(readable, sort_keys=True) if readable else "",
                         "threshold": threshold, **{metric: metrics[metric] for metric in NESTED_METRICS}})
    return rows


def nested_cv(pipelines: dict, X: pd.DataFrame, y: pd.Series, selector, *, seed: int = RANDOM_STATE,
              n_jobs: int = OUTER_FOLDS, grids: dict | None = None) -> pd.DataFrame:
    """One row per procedure and outer fold, scored on that outer fold at its own threshold.

    Outer folds are independent and seeded, so running them in parallel gives the same rows.
    Grids are passed explicitly because worker processes import this module afresh.
    """
    grids = PARAM_GRIDS if grids is None else grids
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=seed)
    LOGGER.info("Nested CV: %d outer folds, fixed and tuned procedures", OUTER_FOLDS)
    results = Parallel(n_jobs=n_jobs)(delayed(outer_fold)(pipelines, X, y, selector, fold, fit, test, seed, grids)
                                      for fold, (fit, test) in enumerate(outer.split(X, y)))
    return pd.DataFrame([row for rows in results for row in rows])


def nested_summary(folds: pd.DataFrame) -> dict:
    """Mean and population SD per procedure, plus how often each model and threshold was chosen."""
    summary = {}
    for procedure, group in folds.groupby("procedure", sort=False):
        summary[str(procedure)] = {
            **{metric: {"mean": float(group[metric].mean()), "std": float(group[metric].std(ddof=0))}
               for metric in NESTED_METRICS},
            "selection_counts": {str(name): int(count)
                                 for name, count in group["selected_model"].value_counts().sort_index().items()},
            "threshold_min": float(group["threshold"].min()),
            "threshold_max": float(group["threshold"].max()),
        }
    return summary
