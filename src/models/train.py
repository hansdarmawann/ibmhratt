"""Reproducible training CLI: python -m src.models.train [--with-shap]."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from threadpoolctl import threadpool_limits

from src.config import (
    CV_FOLDS,
    DATA_PATH,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REDUNDANT_COLUMNS,
    ROOT,
    SENSITIVE_COLUMNS,
    TARGET_MAPPING,
)
from src.data.explore import save_exploration
from src.data.load_data import load_data, split_data
from src.data.validate_data import data_audit
from src.features.preprocess import build_pipeline, feature_exclusions
from src.models.evaluate import (
    bootstrap_intervals,
    cross_validation_metrics,
    evaluate_probabilities,
    subgroup_metrics,
)
from src.models.explain import explain_models, explain_shap
from src.models.threshold import select_threshold, threshold_table
from src.visualization.plots import save_evaluation_plots

LOGGER = logging.getLogger(__name__)
RESULTS_START, RESULTS_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
CI_LABELS = {"pr_auc": "AP", "roc_auc": "ROC-AUC", "precision": "precision", "recall": "recall", "f1": "F1"}


def save_json(value: dict, path: Path) -> None:
    """Write strict JSON with portable UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def candidate_pipelines(X_train: pd.DataFrame) -> dict:
    """Use modest, prespecified models suited to a small imbalanced dataset."""
    estimators = {
        "dummy": DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        "logistic_balanced": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                                random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=7,
                                                min_samples_leaf=5, class_weight="balanced_subsample",
                                                n_jobs=1, random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                                                 max_leaf_nodes=7, min_samples_leaf=20,
                                                                 l2_regularization=2.0,
                                                                 early_stopping=False,
                                                                 random_state=RANDOM_STATE),
    }
    return {name: build_pipeline(X_train, model, scale=name.startswith("logistic"))
            for name, model in estimators.items()}


def select_model(summaries: dict) -> tuple[str, str]:
    """Use training CV AP; prefer logistic simplicity when within 0.01 of best."""
    eligible = {name: metrics for name, metrics in summaries.items() if name != "dummy"}
    best = max(eligible, key=lambda name: eligible[name]["pr_auc"]["mean"])
    selected = best
    for simpler in ["logistic_regression", "logistic_balanced"]:
        if eligible[best]["pr_auc"]["mean"] - eligible[simpler]["pr_auc"]["mean"] <= 0.01:
            selected = simpler
            break
    reason = ("Selection uses only five-fold training average precision. Prefer unweighted, then "
              "balanced logistic regression when within 0.01 AP of the best non-dummy model, "
              "for interpretability and simpler operation. Otherwise use the highest mean AP. "
              f"Best AP candidate: {best}; selected: {selected}. This rule was fixed before holdout evaluation.")
    return selected, reason


def markdown_results(report: dict) -> str:
    """Render real computed metrics for README and a stand-alone experiment report."""
    rows = ["| Model | CV AP (mean ± SD) | Test AP | Test ROC-AUC | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metric in report["holdout_at_0_5"].items():
        cv = report["cross_validation"][name]["pr_auc"]
        rows.append(f"| {name} | {cv['mean']:.3f} ± {cv['std']:.3f} | {metric['pr_auc']:.3f} | "
                    f"{metric['roc_auc']:.3f} | {metric['precision']:.3f} | {metric['recall']:.3f} | {metric['f1']:.3f} |")
    m = report["selected_model_metrics"]
    rows += ["", "Table classification metrics use threshold 0.50. AP is average precision, not trapezoidal PR area.",
             "", f"Selected **{report['selected_model']}** at threshold **{report['selected_threshold']:.2f}**, "
             "chosen by maximum F2 on training out-of-fold predictions.", "",
             f"At the frozen operating threshold, holdout AP = **{m['pr_auc']:.3f}**, "
             f"ROC-AUC = **{m['roc_auc']:.3f}**, precision = **{m['precision']:.3f}**, "
             f"recall = **{m['recall']:.3f}**, F1 = **{m['f1']:.3f}**.", ""]
    ci = report.get("selected_model_ci")
    if ci:
        bounds = ", ".join(f"{label} [{ci['intervals'][key]['lower']:.3f}, {ci['intervals'][key]['upper']:.3f}]"
                           for key, label in CI_LABELS.items())
        rows += [f"{ci['confidence']:.0%} stratified bootstrap intervals ({ci['replicates']:,} holdout resamples, "
                 f"frozen model and threshold): {bounds}. They reflect holdout sampling variability only, "
                 "not split or model-selection variability.", ""]
    rows += [f"- True positives: {m['true_positives']} observed attrition cases flagged.",
             f"- True negatives: {m['true_negatives']} observed retention cases not flagged.",
             f"- False positives: {m['false_positives']} observed retention cases flagged.",
             f"- False negatives: {m['false_negatives']} observed attrition cases missed.", "",
             report["selection_reason"], "",
             "OOF threshold scores reuse the training folds used for model comparison and are selection diagnostics, "
             "not an unbiased performance estimate. The holdout is evaluated after choices are frozen. "
             f"There are {m['positive_count']} positive holdout examples, so small count changes materially affect recall."]
    return "\n".join(rows) + "\n"


def replace_results_block(contents: str, result_text: str) -> str:
    """Swap only the generated block between README markers; other text is untouched."""
    if RESULTS_START not in contents or RESULTS_END not in contents:
        return contents
    before = contents.split(RESULTS_START, 1)[0]
    after = contents.split(RESULTS_END, 1)[1]
    return before + RESULTS_START + "\n\n" + result_text + "\n" + RESULTS_END + after


def train(data_path: Path = DATA_PATH, *, with_shap: bool = False) -> dict:
    """Run the complete experiment and save a full preprocessing/model pipeline."""
    for directory in [METRICS_DIR, FIGURES_DIR, MODEL_PATH.parent, ROOT / "data/processed"]:
        directory.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Model training started")
    data = load_data(data_path)
    save_json(data_audit(data), METRICS_DIR / "data_audit.json")
    X_train, X_test, y_train, y_test = split_data(data)
    exclusions = feature_exclusions(X_train)
    save_json(exclusions, METRICS_DIR / "feature_exclusions.json")
    manifest = pd.DataFrame({"row_index": data.index, "partition": "train"})
    manifest.loc[X_test.index, "partition"] = "test"
    manifest.to_csv(ROOT / "data/processed/split_manifest.csv", index=False)
    save_exploration(X_train, y_train, METRICS_DIR, FIGURES_DIR)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipelines = candidate_pipelines(X_train)
    summaries = {}
    fold_tables = []
    for name, pipeline in pipelines.items():
        LOGGER.info("Cross-validating %s", name)
        summaries[name], folds = cross_validation_metrics(pipeline, X_train, y_train, cv)
        fold_tables.append(folds.assign(model=name))
    # Fixed diagnostic experiments, not extra candidates for final selection.
    for name, excluded in [("logistic_without_sensitive", SENSITIVE_COLUMNS),
                            ("logistic_reduced_correlations", REDUNDANT_COLUMNS)]:
        estimator = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
        pipeline = build_pipeline(X_train, estimator, exclude=excluded)
        summaries[name], folds = cross_validation_metrics(pipeline, X_train, y_train, cv)
        fold_tables.append(folds.assign(model=name))
    pd.concat(fold_tables).to_csv(METRICS_DIR / "cv_fold_metrics.csv")
    selected, reason = select_model({name: summaries[name] for name in pipelines})
    LOGGER.info("Selected %s using training CV", selected)
    oof = cross_val_predict(pipelines[selected], X_train, y_train, cv=cv,
                           method="predict_proba", n_jobs=1)[:, 1]
    thresholds = threshold_table(y_train, oof)
    threshold = select_threshold(thresholds)
    thresholds.to_csv(METRICS_DIR / "threshold_analysis.csv", index=False)
    pd.DataFrame({"row_index": X_train.index, "target": y_train.to_numpy(),
                  "oof_probability": oof}).to_csv(ROOT / "data/processed/oof_predictions.csv", index=False)
    LOGGER.info("Threshold frozen at %.2f; starting final holdout evaluation", threshold)
    probabilities, holdout = {}, {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        probabilities[name] = pipeline.predict_proba(X_test)[:, 1]
        holdout[name] = evaluate_probabilities(y_test, probabilities[name], DEFAULT_THRESHOLD)
    selected_metrics = evaluate_probabilities(y_test, probabilities[selected], threshold)
    pd.DataFrame(holdout).T.drop(columns="confusion_matrix").to_csv(METRICS_DIR / "model_comparison.csv")
    save_evaluation_plots(y_test, probabilities, selected, selected_metrics, thresholds, FIGURES_DIR)
    explain_models(pipelines, selected, X_test, y_test, METRICS_DIR, FIGURES_DIR)
    subgroup_metrics(X_test, y_test, probabilities[selected], threshold).to_csv(
        METRICS_DIR / "subgroup_metrics.csv", index=False)
    # Uncertainty of the frozen choice only; intervals never feed back into selection.
    intervals = bootstrap_intervals(y_test, probabilities[selected], threshold)
    intervals.to_csv(METRICS_DIR / "holdout_bootstrap_ci.csv", index=False)
    shap_status = (explain_shap(pipelines[selected], X_train, X_test, METRICS_DIR, FIGURES_DIR)
                   if with_shap else {"status": "not_requested", "enable": "python -m src.models.train --with-shap"})
    versions = {name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scikit-learn", "joblib"]}
    metadata = {
        "schema_version": 1, "selected_model": selected, "decision_threshold": threshold,
        "feature_columns": FEATURE_COLUMNS, "target_mapping": TARGET_MAPPING,
        "random_state": RANDOM_STATE, "python_version": platform.python_version(),
        "package_versions": versions, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "trained_on": "training partition only; holdout not used for refitting",
    }
    final_pipeline = pipelines[selected]
    final_pipeline.attrition_metadata_ = metadata
    # Atomic replacement avoids exposing a partially written artifact to inference.
    temporary = MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(final_pipeline, temporary)
    temporary.replace(MODEL_PATH)
    save_json(metadata, MODEL_PATH.with_suffix(".json"))
    LOGGER.info("Model saved to %s", MODEL_PATH)
    report = {
        **metadata, "metric_definition": "pr_auc is sklearn average_precision_score",
        "cross_validation": summaries, "holdout_at_0_5": holdout,
        "selected_threshold": threshold, "selected_model_metrics": selected_metrics,
        "selected_model_ci": {
            "method": "stratified percentile bootstrap of the holdout; model and threshold frozen",
            "confidence": float(intervals["confidence"].iloc[0]),
            "replicates": int(intervals["replicates"].iloc[0]),
            "intervals": {row.metric: {"lower": float(row.lower), "upper": float(row.upper)}
                          for row in intervals.itertuples()},
        },
        "selection_reason": reason, "threshold_rule": "Maximize training OOF F2, ties: precision then threshold",
        "shap": shap_status,
        "sensitive_ablation": {"excluded": SENSITIVE_COLUMNS,
                               "cv_ap_change": summaries["logistic_without_sensitive"]["pr_auc"]["mean"]
                               - summaries["logistic_regression"]["pr_auc"]["mean"]},
        "correlation_ablation": {"excluded": REDUNDANT_COLUMNS,
                                 "cv_ap_change": summaries["logistic_reduced_correlations"]["pr_auc"]["mean"]
                                 - summaries["logistic_regression"]["pr_auc"]["mean"]},
    }
    save_json(report, METRICS_DIR / "model_metrics.json")
    result_text = markdown_results(report)
    (ROOT / "reports/results.md").write_text("# Executed experiment results\n\n" + result_text, encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        contents = readme.read_text(encoding="utf-8")
        updated = replace_results_block(contents, result_text)
        if updated != contents:
            readme.write_text(updated, encoding="utf-8")
    # One sample is enough for a reproducible local request; no raw request logging.
    example = X_train[FEATURE_COLUMNS].iloc[0].to_dict()
    save_json(example, ROOT / "examples/employee.json")
    LOGGER.info("Model training completed: AP %.3f; recall %.3f; F1 %.3f",
                selected_metrics["pr_auc"], selected_metrics["recall"], selected_metrics["f1"])
    return report


def main() -> None:
    """Parse CLI options and run training with informative logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--with-shap", action="store_true", help="Generate optional SHAP explanations")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Small datasets are faster and more reproducible without thread oversubscription.
    with threadpool_limits(limits=1):
        train(args.data, with_shap=args.with_shap)


if __name__ == "__main__":
    main()
