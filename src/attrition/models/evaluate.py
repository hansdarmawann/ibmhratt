"""Positive-class evaluation and explicit average-precision reporting."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

from attrition.config import DEFAULT_THRESHOLD, RANDOM_STATE

SCORING = {
    "roc_auc": "roc_auc", "pr_auc": "average_precision", "precision": "precision",
    "recall": "recall", "f1": "f1",
}
BOOTSTRAP_METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1"]


def evaluate_probabilities(y_true, probabilities,
                           threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Report AP (named pr_auc), threshold metrics, and counts; Yes is positive."""
    probabilities = np.asarray(probabilities, dtype=float)
    if not 0 <= threshold <= 1 or not np.isfinite(threshold):
        raise ValueError("Decision threshold must be finite and between 0 and 1.")
    if (probabilities.ndim != 1 or len(probabilities) != len(y_true)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Provide one finite probability in [0, 1] for every target.")
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, probabilities)) if both_classes else None,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "n": len(predicted), "positive_count": int(np.asarray(y_true).sum()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def cross_validation_metrics(pipeline, X, y, cv) -> tuple[dict, pd.DataFrame]:
    """Clone/refit the entire pipeline independently in each stratified fold."""
    # zero_division is explicit, including for the all-retained dummy baseline.
    scoring = {**SCORING, "precision": make_scorer(precision_score, zero_division=0)}
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                             n_jobs=1, error_score="raise")
    folds = pd.DataFrame({m: results[f"test_{m}"] for m in SCORING})
    folds.index.name = "fold"
    summary = {m: {"mean": float(folds[m].mean()), "std": float(folds[m].std(ddof=0))}
               for m in SCORING}
    return summary, folds


def bootstrap_intervals(y_true, probabilities, threshold: float, *, n_boot: int = 1000,
                        confidence: float = 0.95, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Stratified percentile bootstrap for one frozen model and threshold.

    Resampling within each class keeps holdout prevalence fixed, so every replicate
    contains both classes. Intervals describe holdout sampling variability only;
    they do not cover split, model-selection, or threshold-selection variability.
    """
    if n_boot < 1 or not 0 < confidence < 1:
        raise ValueError("Use at least one replicate and a confidence level in (0, 1).")
    y = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    point = evaluate_probabilities(y, probabilities, threshold)
    by_class = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(len(rows) == 0 for rows in by_class):
        raise ValueError("Bootstrap intervals require both classes in the holdout.")
    rng = np.random.default_rng(seed)
    replicates = {metric: np.empty(n_boot) for metric in BOOTSTRAP_METRICS}
    for i in range(n_boot):
        rows = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in by_class])
        metrics = evaluate_probabilities(y[rows], probabilities[rows], threshold)
        for metric in BOOTSTRAP_METRICS:
            replicates[metric][i] = metrics[metric]
    alpha = (1 - confidence) / 2
    return pd.DataFrame([
        {"metric": metric, "estimate": point[metric],
         "lower": float(np.quantile(replicates[metric], alpha)),
         "upper": float(np.quantile(replicates[metric], 1 - alpha)),
         "confidence": confidence, "replicates": n_boot, "threshold": float(threshold)}
        for metric in BOOTSTRAP_METRICS
    ])


def group_rate_intervals(y, flagged, rng, *, n_boot: int, confidence: float) -> dict:
    """Stratified percentile bootstrap of recall, false-positive rate, and selection rate in one group.

    Positives and negatives are resampled separately, so their counts stay fixed;
    a rate without any rows of its class has no interval.
    """
    y, flagged = np.asarray(y).astype(bool), np.asarray(flagged).astype(float)
    positives, negatives = flagged[y], flagged[~y]

    def resampled_rates(values):
        if not len(values):
            return None
        return values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)

    recall, fpr = resampled_rates(positives), resampled_rates(negatives)
    selected = sum(rates * len(values) for rates, values in [(recall, positives), (fpr, negatives)]
                   if rates is not None) / len(y)
    alpha = (1 - confidence) / 2
    result = {}
    for name, replicates in [("recall", recall), ("false_positive_rate", fpr), ("selection_rate", selected)]:
        bounds = (None, None) if replicates is None else np.quantile(replicates, [alpha, 1 - alpha])
        result[f"{name}_lower"], result[f"{name}_upper"] = (None if bound is None else float(bound) for bound in bounds)
    return result


def subgroup_metrics(X: pd.DataFrame, y: pd.Series, probabilities, threshold: float, *, n_boot: int = 1000,
                     confidence: float = 0.95, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Descriptive holdout slices with denominators and bootstrap intervals; not a fairness certification.

    Intervals cover sampling variability within each group for the frozen model and threshold only.
    """
    rng = np.random.default_rng(seed)
    groups = X[["Gender", "MaritalStatus"]].copy()
    groups["AgeBand"] = pd.cut(X["Age"], [17, 29, 39, 49, 100],
                               labels=["18-29", "30-39", "40-49", "50+"])
    rows = []
    for column in groups:
        for group in groups[column].dropna().unique():
            mask = groups[column].eq(group).to_numpy()
            metrics = evaluate_probabilities(y.to_numpy()[mask], np.asarray(probabilities)[mask], threshold)
            negatives = metrics["true_negatives"] + metrics["false_positives"]
            rows.append({
                "attribute": column, "group": str(group), "n": metrics["n"],
                "positive_count": metrics["positive_count"],
                "recall": metrics["recall"] if metrics["positive_count"] else None,
                "false_positive_rate": metrics["false_positives"] / negatives if negatives else None,
                "selection_rate": float((np.asarray(probabilities)[mask] >= threshold).mean()),
                "precision": metrics["precision"],
                **group_rate_intervals(y.to_numpy()[mask], np.asarray(probabilities)[mask] >= threshold, rng,
                                       n_boot=n_boot, confidence=confidence),
            })
    return pd.DataFrame(rows)
