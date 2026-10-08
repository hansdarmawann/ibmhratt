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

from src.config import CATEGORIES, CURRENT_RUN, DISCLAIMER, NUMERIC_BOUNDS
from src.models.artifacts import active_run_path, load_bundle
from src.models.local_explain import explain_profile
from src.models.predict import predict

st.set_page_config(page_title="Employee Attrition | Model Lab", page_icon="📊", layout="wide")
st.title("Employee Attrition · Model Lab")
st.caption("An educational study of predictive associations in the IBM HR sample dataset")
st.info(DISCLAIMER)


@st.cache_resource
def cached_bundle(run_path: str, run_id: str):
    """Each immutable run has a separate verified resource."""
    return load_bundle(run_path)


def show_figure(directory: Path, name: str, caption: str) -> None:
    path = directory / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info("Run training to generate this figure.")


def show_table(directory: Path, name: str) -> None:
    path = directory / name
    if path.exists():
        st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


def main() -> None:
    """Render report sections and validated, probability-first inference."""
    try:
        run_path = active_run_path(CURRENT_RUN)
        bundle = cached_bundle(str(run_path), run_path.name)
    except FileNotFoundError:
        st.info("Run `python -m src.models.train` to create a complete model/report bundle.")
        return
    except Exception:
        st.error("The active experiment bundle could not be verified. Retrain or restore a valid bundle.")
        return
    report = bundle.report
    metrics_dir, figures_dir = bundle.metrics_dir, bundle.figures_dir
    st.caption(f"Experiment run: {bundle.run_id}")
    tabs = st.tabs(["Project Overview", "Dataset Overview", "Exploratory Analysis", "Model Performance",
                    "Prediction Demo", "Feature Importance", "Model Explanation", "Responsible ML"])
    with tabs[0]:
        st.header("Can the sample data distinguish attrition from retention?")
        st.write("Compare a trivial baseline, logistic regression, a random forest, and gradient boosting. "
                 "The project prioritizes average precision and recall of observed attrition cases.")
        st.markdown("**Workflow:** validate → stratified split → training-only EDA → fold-fitted preprocessing "
                    "→ model comparison → out-of-fold threshold selection → final holdout evaluation.")
        st.write("Serving probabilities remain uncalibrated estimates. Calibration is evaluated separately on training OOF predictions; no real-employee validation is claimed.")
    with tabs[1]:
        audit_path = metrics_dir / "data_audit.json"
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
        show_figure(figures_dir, "attrition_distribution.png", "Training partition only; Yes is the minority class.")
    with tabs[2]:
        st.write("All exploratory plots use the training partition. Associations do not establish causality.")
        show_figure(figures_dir, "categorical_attrition.png", "Observed group rates; sample sizes differ.")
        show_figure(figures_dir, "numeric_distributions.png", "Class-normalized densities make different group sizes comparable.")
        show_figure(figures_dir, "numeric_correlations.png", "Related career measures are investigated without automatic deletion.")
        show_table(metrics_dir, "eda_category_target_rates.csv")
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
            show_table(metrics_dir, "model_comparison.csv")
            if (metrics_dir / "holdout_bootstrap_ci.csv").exists():
                st.caption("Stratified bootstrap intervals for the selected model at its frozen threshold. "
                           "They reflect holdout sampling variability only.")
                show_table(metrics_dir, "holdout_bootstrap_ci.csv")
            show_figure(figures_dir, "precision_recall_curve.png", "Holdout precision-recall comparison")
            show_figure(figures_dir, "roc_curve.png", "Holdout ROC comparison")
            show_figure(figures_dir, "threshold_analysis.png", "Threshold chosen on training out-of-fold F2, before examining holdout labels")
            show_figure(figures_dir, "confusion_matrix.png", "Rows = observed class; columns = predicted class")
            st.subheader("Calibration diagnostics")
            st.caption("Five outer training folds and three inner calibration folds; sigmoid calibration is not served. "
                       "Lower Brier score and log loss are better. The selected model identity is fixed from training CV.")
            show_table(metrics_dir, "calibration_metrics.csv")
            show_figure(figures_dir, "calibration_reliability.png", "Ten equal-width bins; bin counts are available in the table.")
            show_table(metrics_dir, "calibration_bins.csv")
            st.subheader("Training CV stability")
            st.write(report["stability_diagnostics"])
            show_table(metrics_dir, "stability_selections.csv")
            show_table(metrics_dir, "stability_comparison.csv")
            st.subheader("Capacity diagnostics")
            st.caption("Top 5%, 10%, and 20% by score, rounding counts up. Ties preserve row order. "
                       "Training OOF and final holdout are separate; these are not intervention recommendations.")
            show_table(metrics_dir, "capacity_metrics.csv")
    with tabs[4]:
        st.subheader("Explore one illustrative profile")
        st.write("Change the sample values to explore model behavior. Enter only fictional or sample data.")
        pipeline = bundle.pipeline
        sample = json.loads((bundle.path / "employee.json").read_text(encoding="utf-8"))
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
            st.subheader("Why this profile received this estimate")
            background = pd.DataFrame(json.loads((bundle.path / "background.json").read_text(encoding="utf-8")))
            explanation = explain_profile(pipeline, inputs, background)
            if explanation["status"] == "generated":
                st.caption(f"Contributions are in {explanation['units']}. Higher/lower refers to model output, "
                           "not a causal effect. One-hot categories are combined into their original fields.")
                st.dataframe(pd.DataFrame(explanation["contributions"]), hide_index=True, width="stretch")
                st.caption(f"Base: {explanation['base_value']:.4f}; contributions from remaining features: "
                           f"{explanation['other_contribution']:+.4f}. "
                           "The base plus ALL contributions reconstructs the model output.")
            else:
                st.info("Prediction is available. Optional SHAP explanation is unavailable; "
                        "install requirements-explain.txt in a compatible environment to enable it.")
    with tabs[5]:
        show_figure(figures_dir, "feature_importance.png", "Decrease in holdout average precision when a feature is shuffled")
        show_table(metrics_dir, "permutation_importance.csv")
        show_figure(figures_dir, "logistic_coefficients.png", "Numeric coefficients are per training-standard-deviation change; categories are encoded indicators.")
        st.write("Correlated features can share or mask importance. Coefficients are conditional associations, "
                 "and neither coefficients nor permutation scores measure causal effects.")
    with tabs[6]:
        if report and report["shap"]["status"] == "generated":
            st.write("SHAP units:", report["shap"]["units"])
            show_figure(figures_dir, "shap_summary.png", "Global summary for up to 60 held-out sample records")
            show_figure(figures_dir, "shap_individual.png", "Local explanation of the first holdout record; this is separate from the profile form.")
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
            show_table(metrics_dir, "subgroup_metrics.csv")
        st.write("Subgroup estimates have small denominators and substantial uncertainty. These descriptive slices "
                 "do not establish fairness. Require human oversight, privacy controls, calibration, temporal "
                 "validation, and a defined support-oriented purpose before considering any real-world use.")


if __name__ == "__main__":
    main()
