"""Choose an operating threshold using training out-of-fold probabilities only."""

import numpy as np
import pandas as pd

from attrition.models.evaluate import evaluate_probabilities


def threshold_table(y, probabilities, thresholds=None) -> pd.DataFrame:
    """Measure the recall/precision tradeoff without fitting an estimator."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for threshold in thresholds:
        metrics = evaluate_probabilities(y, probabilities, float(threshold))
        precision, recall = metrics["precision"], metrics["recall"]
        metrics["f2"] = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        metrics.pop("confusion_matrix")
        rows.append(metrics)
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> float:
    """Maximize OOF F2; break ties by precision then the higher threshold.

    F2 is an educational preference for recall, not an inferred business cost.
    A real operating point requires agreed intervention costs and validation.
    """
    if table.empty:
        raise ValueError("Threshold table cannot be empty.")
    return float(table.sort_values(["f2", "precision", "threshold"],
                                    ascending=False).iloc[0]["threshold"])
