"""Interactive educational dashboard using the same saved inference pipeline."""

import json
import sys
from pathlib import Path

# Streamlit executes this file with app/ on sys.path; keep imports portable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import CATEGORIES, DISCLAIMER, FIGURES_DIR, METRICS_DIR, MODEL_PATH, NUMERIC_BOUNDS
from src.models.predict import load_pipeline, predict

st.set_page_config(page_title="Employee Attrition | Model Lab", page_icon="📊", layout="wide")
st.title("Employee Attrition · Model Lab")
st.caption("An educational study of predictive associations in the IBM HR sample dataset")
st.info(DISCLAIMER)


@st.cache_resource
def cached_pipeline(artifact_path: str, modified_ns: int):
    """Refresh the resource when a new artifact is trained."""
    return load_pipeline(artifact_path)


def show_figure(name: str, caption: str) -> None:
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info("Run training to generate this figure.")


def show_table(name: str) -> None:
    path = METRICS_DIR / name
    if path.exists():
        st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


def main() -> None:
    """Render report sections and validated, probability-first inference."""
    metrics_path = METRICS_DIR / "model_metrics.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    tabs = st.tabs(["Project Overview", "Dataset Overview", "Exploratory Analysis", "Model Performance",
                    "Prediction Demo", "Feature Importance", "Model Explanation", "Responsible ML"])
    with tabs[0]:
        st.header("Can the sample data distinguish attrition from retention?")
        st.write("Compare a trivial baseline, logistic regression, a random forest, and gradient boosting. "
                 "The project prioritizes average precision and recall of observed attrition cases.")
        st.markdown("**Workflow:** validate → stratified split → training-only EDA → fold-fitted preprocessing "
                    "→ model comparison → out-of-fold threshold selection → final holdout evaluation.")
        st.write("The probability is a model estimate. It has not been calibrated or validated for real employees.")
    with tabs[1]:
        audit_path = METRICS_DIR / "data_audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            a, b, c = st.columns(3)
            a.metric("Rows", audit["shape"][0])
            b.metric("Columns", audit["shape"][1])
            c.metric("Missing values", sum(audit["missing_values"].values()))
            st.write("Constant columns:", ", ".join(audit["constant_columns"]))
            st.write("Duplicate rows:", audit["duplicate_rows"])
            st.write("EmployeeNumber is excluded as an identifier. Verified constants carry no variation.")
        else:
            st.info("Run `python -m src.models.train` to generate dataset and model reports.")
        show_figure("attrition_distribution.png", "Training partition only; Yes is the minority class.")
    with tabs[2]:
        st.write("All exploratory plots use the training partition. Associations do not establish causality.")
        show_figure("categorical_attrition.png", "Observed group rates; sample sizes differ.")
        show_figure("numeric_distributions.png", "Class-normalized densities make different group sizes comparable.")
        show_figure("numeric_correlations.png", "Related career measures are investigated without automatic deletion.")
        show_table("eda_category_target_rates.csv")
    with tabs[3]:
        if report:
            st.write(report["selection_reason"])
            m = report["selected_model_metrics"]
            columns = st.columns(4)
            for column, key, label in zip(columns, ["pr_auc", "recall", "precision", "f1"],
                                           ["Average precision", "Recall", "Precision", "F1"], strict=True):
                column.metric(label, f"{m[key]:.3f}")
            st.caption(f"Final holdout at threshold {report['selected_threshold']:.2f}. "
                       "AP summarizes ranking across thresholds. The table below uses threshold 0.50.")
            show_table("model_comparison.csv")
            if (METRICS_DIR / "holdout_bootstrap_ci.csv").exists():
                st.caption("Stratified bootstrap intervals for the selected model at its frozen threshold. "
                           "They reflect holdout sampling variability only.")
                show_table("holdout_bootstrap_ci.csv")
            show_figure("precision_recall_curve.png", "Holdout precision-recall comparison")
            show_figure("roc_curve.png", "Holdout ROC comparison")
            show_figure("threshold_analysis.png", "Threshold chosen on training out-of-fold F2, before examining holdout labels")
            show_figure("confusion_matrix.png", "Rows = observed class; columns = predicted class")
    with tabs[4]:
        st.subheader("Explore one illustrative profile")
        st.write("Change the sample values to explore model behavior. Enter only fictional or sample data.")
        if not MODEL_PATH.exists():
            st.info("Run `python -m src.models.train` before trying predictions.")
        else:
            try:
                pipeline = cached_pipeline(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
            except Exception:
                st.error("The model could not be loaded. Retrain it in the current environment.")
                return
            sample_path = PROJECT_ROOT / "examples/employee.json"
            sample = json.loads(sample_path.read_text(encoding="utf-8")) if sample_path.exists() else {}
            with st.form("employee"):
                columns = st.columns(3)
                inputs = {}
                for i, (name, (lower, upper)) in enumerate(NUMERIC_BOUNDS.items()):
                    inputs[name] = columns[i % 3].number_input(name, min_value=int(lower),
                                                              max_value=int(upper) if upper is not None else None,
                                                              value=int(sample.get(name, lower)), step=1)
                for i, (name, choices) in enumerate(CATEGORIES.items()):
                    default = choices.index(sample[name]) if sample.get(name) in choices else 0
                    inputs[name] = columns[i % 3].selectbox(name, choices, index=default)
                submitted = st.form_submit_button("Estimate attrition probability")
            if submitted:
                result = predict(inputs, pipeline=pipeline)
                a, b, c = st.columns(3)
                a.metric("Attrition probability", f"{result['attrition_probability']:.1%}")
                b.metric("Threshold-derived class", result["predicted_class"])
                c.metric("Selected threshold", f"{result['decision_threshold']:.2f}")
                st.caption("Yes indicates a higher predicted attrition probability than the selected threshold. "
                           "It is not a statement that someone will resign.")
    with tabs[5]:
        show_figure("feature_importance.png", "Decrease in holdout average precision when a feature is shuffled")
        show_table("permutation_importance.csv")
        show_figure("logistic_coefficients.png", "Numeric coefficients are per training-standard-deviation change; categories are encoded indicators.")
        st.write("Correlated features can share or mask importance. Coefficients are conditional associations, "
                 "and neither coefficients nor permutation scores measure causal effects.")
    with tabs[6]:
        if report and report["shap"]["status"] == "generated":
            st.write("SHAP units:", report["shap"]["units"])
            show_figure("shap_summary.png", "Global summary for up to 60 held-out sample records")
            show_figure("shap_individual.png", "Local explanation of the first holdout record; this is separate from the profile form.")
        else:
            st.info("Optional SHAP: install requirements-explain.txt and train with --with-shap. "
                    "Coefficients and permutation importance remain available without SHAP.")
    with tabs[7]:
        st.warning(DISCLAIMER)
        st.write("Gender, Age, and MaritalStatus are included only for this comparative educational analysis. "
                 "Their suitability for any HR application requires separate governance. Removing them does not "
                 "remove proxies in job role, income, tenure, or other variables.")
        if report:
            st.write(f"Training CV AP change without the three sensitive attributes: "
                     f"{report['sensitive_ablation']['cv_ap_change']:+.3f}.")
            show_table("subgroup_metrics.csv")
        st.write("Subgroup estimates have small denominators and substantial uncertainty. These descriptive slices "
                 "do not establish fairness. Require human oversight, privacy controls, calibration, temporal "
                 "validation, and a defined support-oriented purpose before considering any real-world use.")


if __name__ == "__main__":
    main()
