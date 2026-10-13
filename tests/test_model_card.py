"""The model card's figures must match the committed artifacts, and its links must resolve."""

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs" / "model-card.md"
METRICS = ROOT / "reports" / "metrics"

pytestmark = pytest.mark.skipif(not CARD.exists(), reason="checkout without docs")


@pytest.fixture(scope="module")
def card():
    return CARD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def report():
    return json.loads((METRICS / "model_metrics.json").read_text(encoding="utf-8"))


def rows(name):
    with (METRICS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value, digits=3):
    return f"{float(value):.{digits}f}"


def test_identity_and_data_provenance_match_metadata(card, report):
    assert report["selected_model"] == "logistic_regression"
    assert f"`{report['selected_model']}`" in card
    assert f"**{report['decision_threshold']:.2f}**" in card
    assert report["data_sha256"] in card
    assert f"{report['train_rows']:,} training rows" in card
    for name, version in {"Python": report["python_version"], **report["package_versions"]}.items():
        pattern = {"scikit-learn": "scikit-learn", "numpy": "NumPy", "pandas": "pandas", "joblib": "joblib"}.get(name, name)
        assert f"{pattern} {version}" in card


def test_holdout_figures_match_selected_model_metrics(card, report):
    metrics = report["selected_model_metrics"]
    expected = (
        f"AP {f(metrics['pr_auc'])}, ROC-AUC {f(metrics['roc_auc'])}, precision {f(metrics['precision'])}, "
        f"recall {f(metrics['recall'])}, F1 {f(metrics['f1'])} (TP {metrics['true_positives']}, "
        f"FP {metrics['false_positives']}, FN {metrics['false_negatives']}, TN {metrics['true_negatives']})"
    )
    assert expected in card
    assert metrics["n"] == 294 and metrics["positive_count"] == 47
    flagged = metrics["true_positives"] + metrics["false_positives"]
    assert f"flags {flagged} of {metrics['n']}" in card


def test_bootstrap_intervals_match_artifact(card):
    for row in rows("holdout_bootstrap_ci.csv"):
        label = {"pr_auc": "AP", "roc_auc": "ROC-AUC", "precision": "precision", "recall": "recall", "f1": "F1"}[row["metric"]]
        assert f"{label} [{f(row['lower'])}, {f(row['upper'])}]" in card


def test_cv_nested_and_oof_figures_match_artifacts(card, report):
    cv = report["cross_validation"]["logistic_regression"]
    assert f"AP {f(cv['pr_auc']['mean'])} ± {f(cv['pr_auc']['std'])}; ROC-AUC {f(cv['roc_auc']['mean'])} ± {f(cv['roc_auc']['std'])}" in card
    procedures = report["nested_cv"]["procedures"]
    for name in ("fixed", "tuned"):
        assert f"AP {f(procedures[name]['pr_auc']['mean'])} ± {f(procedures[name]['pr_auc']['std'])}" in card
    oof = next(row for row in rows("threshold_analysis.csv") if abs(float(row["threshold"]) - report["decision_threshold"]) < 1e-9)
    assert (
        f"precision {f(oof['precision'])}, recall {f(oof['recall'])}, F1 {f(oof['f1'])}, F2 {f(oof['f2'])} "
        f"(TP {oof['true_positives']}, FP {oof['false_positives']}, FN {oof['false_negatives']}, TN {oof['true_negatives']})"
    ) in card


def test_seed_stability_calibration_and_capacity_match_artifacts(card, report):
    selections = rows("stability_selections.csv")
    assert ", ".join(f(row["selected_cv_ap"]) for row in selections[:2]) in card
    assert f(selections[2]["selected_cv_ap"]) in card
    assert all(row["selected_model"] == "logistic_regression" for row in selections)
    calibration = {row["method"]: row for row in rows("calibration_metrics.csv")}
    assert f"Brier score {f(calibration['uncalibrated']['brier_score'])} uncalibrated versus {f(calibration['sigmoid']['brier_score'])} calibrated" in card
    assert f"log loss {f(calibration['uncalibrated']['log_loss'])} versus {f(calibration['sigmoid']['log_loss'])}" in card
    top = next(r for r in rows("capacity_metrics.csv") if r["partition"] == "holdout" and float(r["capacity_fraction"]) == 0.1)
    assert f"precision {f(top['precision'])}, recall {f(top['recall'])}, and lift {f(top['lift'], 2)}" in card


def test_ablation_figures_match_artifacts(card, report):
    assert f"{report['sensitive_ablation']['cv_ap_change']:+.3f}" in card
    assert f"{report['correlation_ablation']['cv_ap_change']:+.3f}" in card
    engineered = report["feature_engineering_ablation"]
    assert f"{engineered['cv_ap_change']:+.3f}" in card
    assert engineered["folds_improved"] == 4 and engineered["meets_adoption_rule"] and not engineered["adopted"]


def test_subgroup_examples_match_artifact(card):
    groups = {(row["attribute"], row["group"]): row for row in rows("subgroup_metrics.csv")}
    divorced = groups[("MaritalStatus", "Divorced")]
    assert (divorced["n"], divorced["positive_count"]) == ("64", "4")
    assert f"recall of {f(divorced['recall'], 2)}" in card
    older = groups[("AgeBand", "50+")]
    assert older["positive_count"] == "8"
    assert f"recall {f(older['recall'], 2)}, interval [{f(older['recall_lower'], 2)}, {f(older['recall_upper'], 2)}]" in card
    assert f"{f(groups[('MaritalStatus', 'Single')]['selection_rate'], 2)} for Single versus {f(divorced['selection_rate'], 2)} for Divorced" in card


def test_explanation_figures_match_artifacts(card):
    permutation = rows("permutation_importance.csv")[:3]
    assert f"`{permutation[0]['feature']}` is largest at {f(permutation[0]['mean_ap_decrease'])} ± {f(permutation[0]['std_ap_decrease'])}" in card
    for row in permutation[1:]:
        assert f"`{row['feature']}` ({f(row['mean_ap_decrease'])})" in card
    coefficients = sorted(rows("logistic_regression_coefficients.csv"), key=lambda row: -float(row["coefficient"]))[:3]
    for row in coefficients:
        assert f"`{row['feature']}` ({f(row['coefficient'], 2)})" in card


def test_automl_estimate_is_labeled_and_matches_the_recorded_run(card):
    assert "training-only development estimates" in card
    assert "not committed" in card
    assert not list((ROOT / "reports").glob("automl*")), "an AutoML artifact now exists: cite it instead"
    note = ROOT / "docs" / "pr-progress" / "PR-05.md"
    if not note.exists():  # the PR 05 note arrives when this branch is brought up to date with main
        pytest.skip("docs/pr-progress/PR-05.md is not in this checkout")
    recorded = note.read_text(encoding="utf-8")
    for figure in ("0.6529 ± 0.0553", "0.6514 ± 0.0611", "mean +0.0014, SD 0.0074", "3 of 5 folds", "0.6357", "0.6529"):
        assert figure in recorded
    for figure in ("0.653 ± 0.055", "0.651 ± 0.061", "mean +0.001, SD 0.007", "3 of 5 folds", "0.636", "0.653"):
        assert figure in card


def test_baseline_arm_of_the_automl_estimate_matches_committed_nested_cv(card, report):
    fixed = report["nested_cv"]["procedures"]["fixed"]["pr_auc"]
    assert f"| Served baseline procedure | {f(fixed['mean'])} ± {f(fixed['std'])} |" in card


def test_card_is_linked_and_relative_links_resolve(card):
    assert "docs/model-card.md" in (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)\s]+)\)", card):
        assert (CARD.parent / target.split("#")[0]).exists(), target
    assert card.count("```") % 2 == 0
