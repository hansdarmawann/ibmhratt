"""Training-only diagnostics; none of these results change the serving model."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from attrition.models.evaluate import cross_validation_metrics
from attrition.models.threshold import select_threshold, threshold_table


def reliability_table(y, probabilities, *, bins: int = 10) -> pd.DataFrame:
    """Equal-width bins, including empty bins and the endpoints zero and one."""
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        mask = indices == index
        rows.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins,
                     "n": int(mask.sum()),
                     "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
                     "observed_rate": float(y[mask].mean()) if mask.any() else None})
    return pd.DataFrame(rows)


def calibration_diagnostics(pipeline, X, y, *, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Five outer folds; sigmoid calibration uses three inner folds only.

    The candidate identity was selected on training CV, so these are diagnostics
    conditional on that choice, not an unbiased estimate of model selection.
    """
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    probabilities = {name: np.empty(len(y)) for name in ("uncalibrated", "sigmoid")}
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (fit, validation) in enumerate(outer.split(X, y)):
        base = clone(pipeline).fit(X.iloc[fit], y.iloc[fit])
        inner = StratifiedKFold(3, shuffle=True, random_state=seed)
        calibrated = CalibratedClassifierCV(clone(pipeline), method="sigmoid", cv=inner, ensemble=True, n_jobs=1)
        calibrated.fit(X.iloc[fit], y.iloc[fit])
        probabilities["uncalibrated"][validation] = base.predict_proba(X.iloc[validation])[:, 1]
        probabilities["sigmoid"][validation] = calibrated.predict_proba(X.iloc[validation])[:, 1]
        fold_ids[validation] = fold
    scores = pd.DataFrame([{"method": name, "brier_score": brier_score_loss(y, values),
                            "log_loss": log_loss(y, values, labels=[0, 1])}
                           for name, values in probabilities.items()])
    bins = pd.concat([reliability_table(y, values).assign(method=name)
                      for name, values in probabilities.items()], ignore_index=True)
    predictions = pd.DataFrame({"row_index": X.index, "fold": fold_ids, "target": y.to_numpy(), **probabilities})
    return scores, bins, predictions


def stability_diagnostics(pipelines, X, y, selector, *, baseline=None,
                          seeds=(42, 43, 44)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the same selection rule; the baseline can reuse existing CV/OOF."""
    comparisons, choices = [], []
    for seed in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        if seed == 42 and baseline is not None:
            summaries, selected, oof = baseline
        else:
            summaries = {name: cross_validation_metrics(model, X, y, cv)[0]
                         for name, model in pipelines.items()}
            selected, _ = selector(summaries)
            oof = cross_val_predict(pipelines[selected], X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        threshold = select_threshold(threshold_table(y, oof))
        for name, summary in summaries.items():
            comparisons.append({"seed": seed, "model": name, "mean_ap": summary["pr_auc"]["mean"],
                                "std_ap": summary["pr_auc"]["std"]})
        choices.append({"seed": seed, "selected_model": selected, "threshold": threshold,
                        "selected_cv_ap": summaries[selected]["pr_auc"]["mean"]})
    return pd.DataFrame(comparisons), pd.DataFrame(choices)


def capacity_metrics(y, probabilities, *, partition: str, fractions=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Rank by descending score, breaking ties by the original row position."""
    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    if (len(y) == 0 or probabilities.shape != y.shape or not np.isin(y, [0, 1]).all()
            or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any()
            or any(not 0 < fraction <= 1 for fraction in fractions)):
        raise ValueError("Capacity metrics require binary targets, probabilities, and fractions in (0, 1].")
    order = np.argsort(-probabilities, kind="stable")
    positives = int(y.sum())
    prevalence = float(y.mean())
    rows = []
    for fraction in fractions:
        count = int(np.ceil(fraction * len(y)))
        found = int(y[order[:count]].sum())
        precision = found / count
        rows.append({"partition": partition, "capacity_fraction": fraction, "selected_count": count,
                     "positive_count": positives, "true_positives": found, "precision": precision,
                     "recall": found / positives if positives else None,
                     "lift": precision / prevalence if prevalence else None})
    return pd.DataFrame(rows)
