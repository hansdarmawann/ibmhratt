"""Positive-class evaluation and explicit average-precision reporting."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_validate

from src.config import DEFAULT_THRESHOLD

SCORING = {
    "roc_auc": "roc_auc", "pr_auc": "average_precision", "precision": "precision",
    "recall": "recall", "f1": "f1",
}


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
    from sklearn.metrics import make_scorer

    scoring = {**SCORING, "precision": make_scorer(precision_score, zero_division=0)}
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                             n_jobs=1, error_score="raise")
    folds = pd.DataFrame({m: results[f"test_{m}"] for m in SCORING})
    folds.index.name = "fold"
    summary = {m: {"mean": float(folds[m].mean()), "std": float(folds[m].std(ddof=0))}
               for m in SCORING}
    return summary, folds


def subgroup_metrics(X: pd.DataFrame, y: pd.Series, probabilities, threshold: float) -> pd.DataFrame:
    """Descriptive holdout slices with denominators; not a fairness certification."""
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
            })
    return pd.DataFrame(rows)
