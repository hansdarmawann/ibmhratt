"""Readable coefficients, permutation importance, and optional SHAP outputs."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import FEATURE_COLUMNS, RANDOM_STATE
from src.models.local_explain import shap_values
from src.visualization.plots import importance_plot, save_figure

LOGGER = logging.getLogger(__name__)


def explain_models(pipelines: dict, selected: str, X_test, y_test,
                   metrics_dir: Path, figures_dir: Path) -> None:
    """Explain already frozen models; results never influence model selection."""
    for name, pipeline in pipelines.items():
        estimator = pipeline.named_steps["model"]
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        if hasattr(estimator, "coef_"):
            table = pd.DataFrame({"feature": names, "coefficient": estimator.coef_[0]})
            table.sort_values("coefficient", ascending=False).to_csv(
                metrics_dir / f"{name}_coefficients.csv", index=False)
            if name == "logistic_regression":
                importance_plot(table, "coefficient", figures_dir / "logistic_coefficients.png",
                                "Logistic coefficients: conditional log-odds associations")
        if hasattr(estimator, "feature_importances_"):
            pd.DataFrame({"feature": names, "importance": estimator.feature_importances_}).sort_values(
                "importance", ascending=False).to_csv(metrics_dir / f"{name}_importance.csv", index=False)
    # Permute original columns for understandable aggregate, rather than dummy indices.
    results = permutation_importance(pipelines[selected], X_test[FEATURE_COLUMNS], y_test,
                                     scoring="average_precision", n_repeats=10,
                                     random_state=RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS,
                               "mean_ap_decrease": results.importances_mean,
                               "std_ap_decrease": results.importances_std})
    importance.sort_values("mean_ap_decrease", ascending=False).to_csv(
        metrics_dir / "permutation_importance.csv", index=False)
    importance_plot(importance, "mean_ap_decrease", figures_dir / "feature_importance.png",
                    "Holdout permutation importance: AP decrease (10 repeats)")


def explain_shap(pipeline, X_train, X_test, metrics_dir: Path, figures_dir: Path) -> dict:
    """Optionally explain up to 60 holdout observations; no SHAP core dependency.

    Linear explanations are in log-odds. Tree explanations explicitly use
    positive-class probability; both paths check additivity in their own units.
    """
    try:
        import shap

        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        background = X_train.sample(min(100, len(X_train)), random_state=RANDOM_STATE)
        explanation, units, samples = shap_values(pipeline, background, X_test.iloc[:60])
        importance = pd.DataFrame({"feature": names,
                                   "mean_absolute_shap": np.abs(explanation.values).mean(axis=0)})
        importance.sort_values("mean_absolute_shap", ascending=False).to_csv(
            metrics_dir / "shap_importance.csv", index=False)
        # beeswarm jitters with NumPy's global state; seed it locally for a reproducible figure.
        state = np.random.get_state()
        np.random.seed(RANDOM_STATE)
        try:
            shap.plots.beeswarm(explanation, max_display=15, show=False)
        finally:
            np.random.set_state(state)
        save_figure(plt.gcf(), figures_dir / "shap_summary.png")
        shap.plots.waterfall(explanation[0], max_display=12, show=False)
        save_figure(plt.gcf(), figures_dir / "shap_individual.png")
        pd.DataFrame({"feature": names, "transformed_value": samples[0],
                      "shap_value": explanation.values[0]}).to_csv(
            metrics_dir / "shap_individual.csv", index=False)
        return {"status": "generated", "units": units, "rows": len(samples),
                "individual_holdout_position": 0,
                "individual_base_value": float(np.asarray(explanation.base_values[0])),
                "individual_model_probability": float(pipeline.predict_proba(X_test.iloc[:1])[0, 1])}
    except Exception as exc:
        # Only the optional explainer is isolated. Model training errors still fail.
        LOGGER.warning("Optional SHAP skipped: %s: %s", type(exc).__name__, exc)
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
