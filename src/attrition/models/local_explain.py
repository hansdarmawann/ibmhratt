"""Optional per-profile explanations; core serving does not depend on SHAP."""

import logging

import numpy as np
import pandas as pd

from attrition.models.predict import predict

LOGGER = logging.getLogger(__name__)


def check_additive(reconstructed, expected) -> None:
    """Base value plus contributions must reproduce the model output."""
    if not np.allclose(reconstructed, expected, rtol=1e-5, atol=1e-6):
        raise ValueError("Explanation contributions do not reconstruct the model output.")


def shap_values(pipeline, background: pd.DataFrame, profiles: pd.DataFrame):
    """Return additive SHAP values with explicit positive-class output units."""
    import shap

    processor = pipeline[:-1]
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    transformed_background = processor.transform(background)
    samples = processor.transform(profiles)
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, transformed_background, feature_names=names)
        units = "log-odds of Attrition=Yes"
        expected = pipeline.decision_function(profiles)
    else:
        explainer = shap.TreeExplainer(model, data=transformed_background, feature_names=names,
                                       feature_perturbation="interventional", model_output="probability")
        units = "probability of Attrition=Yes"
        expected = pipeline.predict_proba(profiles)[:, 1]
    explanation = explainer(samples)
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    totals = np.asarray(explanation.base_values).reshape(-1) + explanation.values.sum(axis=1)
    check_additive(totals, expected)
    return explanation, units, samples


def explained_probability(base: float, contributions: float, units: str) -> float:
    """Convert additive output to probability only when the output is log-odds."""
    total = base + contributions
    if units == "log-odds of Attrition=Yes":
        # Stable sigmoid, including very large negative logits.
        return float(np.exp(-np.logaddexp(0, -total)))
    if units == "probability of Attrition=Yes":
        return float(total)
    raise ValueError(f"Unsupported explanation units: {units}")


def explain_profile(pipeline, profile: dict, background: pd.DataFrame) -> dict:
    """Aggregate one-hot contributions to original fields; isolate optional errors."""
    try:
        result = predict(profile, pipeline=pipeline)
        explanation, units, _ = shap_values(pipeline, background, pd.DataFrame([profile]))
        preprocessor = pipeline.named_steps["preprocess"]
        numeric = list(preprocessor.transformers_[0][2])
        categorical = list(preprocessor.transformers_[1][2])
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        originals = numeric + [column for column, values in zip(categorical, encoder.categories_, strict=True)
                               for _ in values]
        contributions = pd.DataFrame({"feature": originals, "contribution": explanation.values[0]})
        grouped = contributions.groupby("feature", sort=False).contribution.sum()
        grouped = grouped.reindex(grouped.abs().sort_values(ascending=False, kind="stable").index)
        base = float(np.asarray(explanation.base_values).reshape(-1)[0])
        probability = explained_probability(base, float(grouped.sum()), units)
        check_additive(probability, result["attrition_probability"])
        return {"status": "generated", "units": units, "base_value": base,
                "probability": result["attrition_probability"], "run_id": result["run_id"],
                "contribution_sum": float(grouped.sum()), "other_contribution": float(grouped.iloc[10:].sum()),
                "contributions": [{"feature": name, "contribution": float(value),
                                   "direction": "Higher" if value > 0 else "Lower" if value < 0 else "Neutral"}
                                  for name, value in grouped.iloc[:10].items()]}
    except Exception as exc:
        LOGGER.warning("Profile explanation unavailable: %s", type(exc).__name__)
        return {"status": "unavailable", "reason": "An optional explanation is unavailable for this model/environment."}
