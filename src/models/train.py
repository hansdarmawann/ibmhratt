"""Reproducible training CLI: python -m src.models.train [--with-shap]."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
from pathlib import Path
from uuid import uuid4

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
from src.models.artifacts import publish_bundle, seal_bundle
from src.models.diagnostics import calibration_diagnostics, capacity_metrics, stability_diagnostics
from src.models.evaluate import (
    bootstrap_intervals,
    cross_validation_metrics,
    evaluate_probabilities,
    subgroup_metrics,
)
from src.models.explain import explain_models, explain_shap
from src.models.threshold import select_threshold, threshold_table
from src.visualization.plots import save_calibration_plot, save_evaluation_plots

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
    if "calibration_diagnostics" in report:
        scores = report["calibration_diagnostics"]
        raw, sigmoid = scores[0], scores[1]
        stability = report["stability_diagnostics"]
        capacity = next(row for row in report["capacity_metrics"]
                        if row["partition"] == "holdout" and row["capacity_fraction"] == 0.1)
        rows += ["", "### Three additional findings", "",
                 f"1. Training outer-OOF Brier score: uncalibrated **{raw['brier_score']:.3f}**, "
                 f"sigmoid **{sigmoid['brier_score']:.3f}**; log loss "
                 f"**{raw['log_loss']:.3f}** versus **{sigmoid['log_loss']:.3f}**. "
                 "Calibration is diagnostic only; serving probabilities are unchanged.",
                 f"2. Across training CV seeds 42, 43, and 44, selections were "
                 f"**{stability['selection_counts']}**, with thresholds from "
                 f"**{stability['threshold_min']:.2f}** to **{stability['threshold_max']:.2f}**. "
                 "Seed 42 remains the main experiment.",
                 f"3. The highest-scored 10% of holdout profiles ({capacity['selected_count']} rows) "
                 f"have precision **{capacity['precision']:.3f}**, recall **{capacity['recall']:.3f}**, "
                 f"and lift **{capacity['lift']:.2f}**. This is a capacity diagnostic, not an intervention policy."]
    return "\n".join(rows) + "\n"


def replace_results_block(contents: str, result_text: str) -> str:
    """Swap only the generated block between README markers; other text is untouched."""
    if RESULTS_START not in contents or RESULTS_END not in contents:
        return contents
    before = contents.split(RESULTS_START, 1)[0]
    after = contents.split(RESULTS_END, 1)[1]
    return before + RESULTS_START + "\n\n" + result_text + "\n" + RESULTS_END + after


def train(data_path: Path = DATA_PATH, *, with_shap: bool = False, output_root: Path = ROOT) -> dict:
    """Run the complete experiment and save a full preprocessing/model pipeline."""
    run_id = str(uuid4())
    run_dir = output_root / "models/runs" / run_id
    metrics_dir, figures_dir, processed_dir = (run_dir / name for name in ("metrics", "figures", "processed"))
    for directory in [metrics_dir, figures_dir, processed_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Model training started")
    data = load_data(data_path)
    save_json(data_audit(data), metrics_dir / "data_audit.json")
    X_train, X_test, y_train, y_test = split_data(data)
    exclusions = feature_exclusions(X_train)
    save_json(exclusions, metrics_dir / "feature_exclusions.json")
    manifest = pd.DataFrame({"row_index": data.index, "partition": "train"})
    manifest.loc[X_test.index, "partition"] = "test"
    manifest.to_csv(processed_dir / "split_manifest.csv", index=False)
    save_exploration(X_train, y_train, metrics_dir, figures_dir)
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
    pd.concat(fold_tables).to_csv(metrics_dir / "cv_fold_metrics.csv")
    selected, reason = select_model({name: summaries[name] for name in pipelines})
    LOGGER.info("Selected %s using training CV", selected)
    oof = cross_val_predict(pipelines[selected], X_train, y_train, cv=cv,
                           method="predict_proba", n_jobs=1)[:, 1]
    thresholds = threshold_table(y_train, oof)
    threshold = select_threshold(thresholds)
    thresholds.to_csv(metrics_dir / "threshold_analysis.csv", index=False)
    pd.DataFrame({"row_index": X_train.index, "target": y_train.to_numpy(),
                  "oof_probability": oof}).to_csv(processed_dir / "oof_predictions.csv", index=False)
    LOGGER.info("Running training-only calibration and stability diagnostics")
    calibration_scores, calibration_bins, calibration_oof = calibration_diagnostics(pipelines[selected], X_train, y_train)
    calibration_scores.to_csv(metrics_dir / "calibration_metrics.csv", index=False)
    calibration_bins.to_csv(metrics_dir / "calibration_bins.csv", index=False)
    calibration_oof.to_csv(processed_dir / "calibration_oof.csv", index=False)
    save_calibration_plot(calibration_bins, figures_dir)
    repeat_scores, repeat_choices = stability_diagnostics(
        pipelines, X_train, y_train, select_model,
        baseline=({name: summaries[name] for name in pipelines}, selected, oof))
    repeat_scores.to_csv(metrics_dir / "stability_comparison.csv", index=False)
    repeat_choices.to_csv(metrics_dir / "stability_selections.csv", index=False)
    LOGGER.info("Threshold frozen at %.2f; starting final holdout evaluation", threshold)
    probabilities, holdout = {}, {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        probabilities[name] = pipeline.predict_proba(X_test)[:, 1]
        holdout[name] = evaluate_probabilities(y_test, probabilities[name], DEFAULT_THRESHOLD)
    selected_metrics = evaluate_probabilities(y_test, probabilities[selected], threshold)
    capacity = pd.concat([capacity_metrics(y_train, oof, partition="training_oof"),
                          capacity_metrics(y_test, probabilities[selected], partition="holdout")], ignore_index=True)
    capacity.to_csv(metrics_dir / "capacity_metrics.csv", index=False)
    pd.DataFrame(holdout).T.drop(columns="confusion_matrix").to_csv(metrics_dir / "model_comparison.csv")
    save_evaluation_plots(y_test, probabilities, selected, selected_metrics, thresholds, figures_dir)
    explain_models(pipelines, selected, X_test, y_test, metrics_dir, figures_dir)
    subgroup_metrics(X_test, y_test, probabilities[selected], threshold).to_csv(
        metrics_dir / "subgroup_metrics.csv", index=False)
    # Uncertainty of the frozen choice only; intervals never feed back into selection.
    intervals = bootstrap_intervals(y_test, probabilities[selected], threshold)
    intervals.to_csv(metrics_dir / "holdout_bootstrap_ci.csv", index=False)
    shap_status = (explain_shap(pipelines[selected], X_train, X_test, metrics_dir, figures_dir)
                   if with_shap else {"status": "not_requested", "enable": "python -m src.models.train --with-shap"})
    versions = {name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scikit-learn", "joblib"]}
    metadata = {
        "schema_version": 2, "run_id": run_id, "selected_model": selected, "decision_threshold": threshold,
        "feature_columns": FEATURE_COLUMNS, "target_mapping": TARGET_MAPPING,
        "random_state": RANDOM_STATE, "python_version": platform.python_version(),
        "package_versions": versions, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "trained_on": "training partition only; holdout not used for refitting",
    }
    final_pipeline = pipelines[selected]
    final_pipeline.attrition_metadata_ = metadata
    joblib.dump(final_pipeline, run_dir / "pipeline.joblib")
    save_json(metadata, run_dir / "metadata.json")
    report = {
        **metadata, "metric_definition": "pr_auc is sklearn average_precision_score",
        "cross_validation": summaries, "holdout_at_0_5": holdout,
        "selected_threshold": threshold, "selected_model_metrics": selected_metrics,
        "calibration_diagnostics": calibration_scores.to_dict(orient="records"),
        "stability_diagnostics": {
            "selection_counts": {name: int(count) for name, count in repeat_choices.selected_model.value_counts().items()},
            "threshold_min": float(repeat_choices.threshold.min()),
            "threshold_max": float(repeat_choices.threshold.max()),
            "threshold_std": float(repeat_choices.threshold.std(ddof=0)),
            "main_seed": RANDOM_STATE,
        },
        "capacity_metrics": capacity.to_dict(orient="records"),
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
    save_json(report, metrics_dir / "model_metrics.json")
    result_text = markdown_results(report)
    (run_dir / "results.md").write_text("# Executed experiment results\n\n" + result_text, encoding="utf-8")
    readme = output_root / "README.md"
    if readme.exists():
        contents = readme.read_text(encoding="utf-8")
        updated = replace_results_block(contents, result_text)
        if updated != contents:
            readme.write_text(updated, encoding="utf-8")
    # One sample is enough for a reproducible local request; no raw request logging.
    example = X_train[FEATURE_COLUMNS].iloc[0].to_dict()
    save_json(example, run_dir / "employee.json")
    background = X_train[FEATURE_COLUMNS].sample(min(100, len(X_train)), random_state=RANDOM_STATE)
    save_json(json.loads(background.to_json(orient="records")), run_dir / "background.json")
    seal_bundle(run_dir)
    # Git exports are snapshots only. Serving always reads the immutable bundle.
    for source, destination in [(metrics_dir, output_root / "reports/metrics"),
                                (figures_dir, output_root / "reports/figures"),
                                (processed_dir, output_root / "data/processed")]:
        shutil.copytree(source, destination, dirs_exist_ok=True)
    export_report = {key: value for key, value in report.items() if key != "run_id"}
    save_json(export_report, output_root / "reports/metrics/model_metrics.json")
    shutil.copyfile(run_dir / "results.md", output_root / "reports/results.md")
    save_json(example, output_root / "examples/employee.json")
    publish_bundle(run_dir, output_root / "models/current.json")
    LOGGER.info("Validated run %s is now active", run_id)
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
