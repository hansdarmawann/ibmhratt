"""Create the four narrative exploration notebooks from maintained cell sources."""

import nbformat as nbf

from src.config import ROOT

SETUP = """from pathlib import Path
import sys
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Image
from src.config import FIGURES_DIR, METRICS_DIR, FEATURE_COLUMNS
from src.data.load_data import load_data, split_data
from src.data.explore import exploration_tables
data = load_data()
X_train, X_test, y_train, y_test = split_data(data)
"""


def write_notebook(filename: str, cells: list[tuple[str, str]]) -> None:
    """Write a valid portable notebook; execution is a separate explicit step."""
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python (ibmhratt)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook.cells = [nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
                      for kind, text in cells]
    directory = ROOT / "notebooks"
    directory.mkdir(exist_ok=True)
    nbf.write(notebook, directory / filename)


def main() -> None:
    """Keep exploratory narratives separate from reusable src implementations."""
    write_notebook("01_data_understanding.ipynb", [
        ("md", "# 01 · Data understanding\n\nCan employee attributes distinguish observed attrition from retention? "
         "This educational analysis uses the fictional IBM HR sample. We inspect schema and data quality first, "
         "then reserve a holdout before studying feature relationships. Predictive associations are not causal evidence."),
        ("code", SETUP),
        ("md", "## Structural audit\n\nSchema checks do not estimate preprocessing statistics. Missing predictors can "
         "be imputed later; missing labels, duplicate rows, invalid types, and unexpected training categories fail clearly."),
        ("code", "from src.data.validate_data import data_audit\naudit = data_audit(data)\n"
         "display(Markdown(f'**Shape:** {data.shape[0]:,} rows × {data.shape[1]} columns; '"
         "f'**duplicate rows:** {audit[\"duplicate_rows\"]}.'))\n"
         "display(pd.DataFrame({'dtype': data.dtypes.astype(str), 'missing': data.isna().sum(), 'unique': data.nunique()}))"),
        ("md", "## Split before learning\n\nThe same stratified 80/20 split (seed 42) is reused throughout. "
         "`Yes → 1` and `No → 0` remain explicit. Test outcomes are not used for model or threshold selection."),
        ("code", "display(pd.DataFrame({'partition': ['train', 'test'], 'rows': [len(X_train), len(X_test)]}))\n"
         "display(y_train.value_counts().rename(index={0: 'No', 1: 'Yes'}).to_frame('training_count'))\n"
         "assert set(X_train.index).isdisjoint(X_test.index)\n"
         "display(Image(filename=str(FIGURES_DIR / 'attrition_distribution.png')))"),
        ("md", "## Investigate exclusions\n\nA unique ID does not have a portable predictive meaning. "
         "Constant columns cannot distinguish outcomes; confirm their values on training rows before removal."),
        ("code", "from src.features.preprocess import feature_exclusions\n"
         "display(pd.Series(feature_exclusions(X_train), name='reason').to_frame())\n"
         "display(X_train[['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']].nunique().to_frame('unique'))\n"
         "display(X_train[FEATURE_COLUMNS].describe(include='all').T)"),
        ("md", "## Key Findings\n\nThe audit above establishes the shape, types, missingness, and duplicates from the "
         "actual file. Three verified constants and the employee identifier are excluded. Attrition is the minority "
         "class, so an all-retained classifier can have misleadingly high accuracy.\n\n## Next Steps\n\n"
         "Explore training-only distributions and subgroup sizes, then compare against an explicit dummy baseline."),
    ])
    write_notebook("02_exploratory_data_analysis.ipynb", [
        ("md", "# 02 · Exploratory data analysis\n\nAll relationships below use the training partition. We investigate "
         "which attributes are associated with observed attrition, without asserting that changing an attribute "
         "would change an employee's outcome."),
        ("code", SETUP + "\ntables = exploration_tables(X_train, y_train)"),
        ("md", "## Distributions and observed group rates\n\nInspect every categorical/ordinal predictor with group "
         "counts alongside rates. Small groups can have unstable rates; absolute counts alone obscure differences in group size."),
        ("code", "display(tables['category_target_rates'])\n"
         "display(Image(filename=str(FIGURES_DIR / 'categorical_attrition.png')))\n"
         "overtime = tables['category_target_rates'].query(\"feature == 'OverTime'\")\n"
         "display(Markdown('Training overtime groups: ' + '; '.join(\n"
         "    f'{row.value}: {row.attrition_rate:.1%} observed attrition (n={row.count})'\n"
         "    for row in overtime.itertuples()) + '. These are associations, not treatment effects.'))"),
        ("md", "## Numeric distributions and feature-versus-target analysis\n\nDensity histograms normalize the unequal "
         "class sizes. The tables cover all numeric predictors, including daily/hourly/monthly rates, satisfaction "
         "ratings, training, salary hikes, and tenure variables."),
        ("code", "display(tables['summary_statistics'])\ndisplay(tables['numeric_by_target'].T)\n"
         "display(Image(filename=str(FIGURES_DIR / 'numeric_distributions.png')))"),
        ("md", "## Outlier inspection\n\nAn IQR flag is an inspection prompt, not evidence of an error. "
         "High income may reflect seniority, and bounded ordinal scales are not continuous measurements. "
         "We retain observations and use regularization and shallow trees rather than arbitrary trimming."),
        ("code", "display(tables['iqr_outlier_counts'].sort_values('outlier_count', ascending=False))\n"
         "display(X_train.nlargest(8, 'MonthlyIncome')[['MonthlyIncome', 'JobLevel', 'JobRole', 'TotalWorkingYears']])"),
        ("md", "## Related compensation and tenure measurements\n\nExamine correlation before removing information. "
         "JobLevel and MonthlyIncome describe related aspects of seniority; career tenure fields also overlap. "
         "The modeling notebook reports a prespecified reduced-feature CV experiment."),
        ("code", "display(tables['numeric_correlations'])\n"
         "display(Image(filename=str(FIGURES_DIR / 'numeric_correlations.png')))\n"
         "corr = tables['numeric_correlations']\n"
         "display(Markdown(f'JobLevel–MonthlyIncome training correlation: **{corr.loc[\"JobLevel\", \"MonthlyIncome\"]:.3f}**. '"
         "'Correlation can make individual importance estimates unstable; it does not alone justify dropping either field.'))"),
        ("md", "## Key Findings\n\nGroup rates, normalized distributions, and summary tables reveal predictive "
         "associations and unequal sample sizes. Outlier flags and related career variables warrant investigation, "
         "not automatic deletion. Age, Gender, and MaritalStatus require a responsible-use review.\n\n"
         "## Next Steps\n\nKeep preprocessing inside cross-validation and compare models using average precision, "
         "recall, and F1. Preserve the reserved holdout until the operating policy is fixed."),
    ])
    write_notebook("03_feature_engineering.ipynb", [
        ("md", "# 03 · Feature engineering and leakage prevention\n\nUse an explicit schema with automatically "
         "separated numeric/categorical predictors. Ordinal ratings remain numeric for a transparent baseline; "
         "this assumes equal spacing and is a documented limitation. No target-derived feature or oversampling is used."),
        ("code", SETUP),
        ("md", "## Build, then fit on training rows only\n\nNumeric medians, scaling statistics, category modes, "
         "and one-hot vocabulary are learned by the pipeline. Each CV clone will learn them independently. "
         "Unseen inference categories produce all-zero indicators for that field."),
        ("code", "from sklearn.linear_model import LogisticRegression\n"
         "from src.features.preprocess import build_pipeline, feature_exclusions\n"
         "pipeline = build_pipeline(X_train, LogisticRegression(max_iter=2000, random_state=42))\n"
         "pipeline.fit(X_train, y_train)\n"
         "processor = pipeline.named_steps['preprocess']\n"
         "display(pd.Series(feature_exclusions(X_train), name='reason'))\n"
         "display(pd.Series(processor.get_feature_names_out(), name='transformed_feature'))"),
        ("md", "## Inspect fitted statistics and missing/unseen inputs\n\nInference cannot update the fitted "
         "imputer or encoder. Required fields must be present, but explicit nulls are imputed. The next example "
         "modifies a training example solely to verify that the transformation remains usable."),
        ("code", "names = processor.transformers_[0][2]\n"
         "medians = processor.named_transformers_['numeric'].named_steps['imputer'].statistics_\n"
         "np.testing.assert_allclose(medians, X_train[names].median())\n"
         "display(pd.DataFrame({'feature': names, 'training_median': medians}))\n"
         "example = X_train.head(1).copy()\n"
         "example['JobRole'] = 'Unseen demonstration role'\nexample['MonthlyIncome'] = np.nan\n"
         "display(pd.DataFrame(pipeline.predict_proba(example), columns=['No', 'Yes']))"),
        ("md", "## Visual check of the engineered representation\n\nScaling makes numeric coefficients comparable "
         "per training standard deviation, while indicator coefficients remain conditional contrasts. Full one-hot "
         "encoding with regularization avoids claiming a unique causal interpretation for each coefficient."),
        ("code", "import matplotlib.pyplot as plt\n"
         "scaled = processor.named_transformers_['numeric'].transform(X_train[names])\n"
         "fig, ax = plt.subplots(figsize=(8, 3))\n"
         "ax.hist(scaled[:, names.index('MonthlyIncome')], bins=25, color='#357b89')\n"
         "ax.set(title='Standardized training MonthlyIncome', xlabel='Training standard deviations', ylabel='Count')\n"
         "display(fig)\nplt.close(fig)"),
        ("md", "## Key Findings\n\nThe complete pipeline handles missing values and unseen categories, drops "
         "documented nonpredictive fields, and preserves human-readable feature names. Its learned medians match "
         "training rows exactly. Scaling does not remove income skew or make ordinal scales continuous.\n\n"
         "## Next Steps\n\nCompare candidate models with fold-fitted preprocessing and inspect a sensitive-attribute "
         "ablation alongside the correlated-feature experiment."),
    ])
    write_notebook("04_model_experiments.ipynb", [
        ("md", "# 04 · Model experiments and interpretation\n\nRun `python -m src.models.train --with-shap` first. "
         "This notebook reads executed experiment artifacts; it does not tune models on test outcomes. "
         "Average precision is reported as PR-AUC/AP throughout, not trapezoidal PR area."),
        ("code", SETUP + "\nreport = json.loads((METRICS_DIR / 'model_metrics.json').read_text(encoding='utf-8'))\n"
         "display(Markdown((ROOT / 'reports/results.md').read_text(encoding='utf-8')))"),
        ("md", "## Stratified five-fold comparison\n\nCompare mean and population SD for AP, ROC-AUC, precision, "
         "recall, and F1. Models were fixed in advance. The dummy prior predicts retention at 0.50; balanced "
         "logistic regression changes the tradeoff and may distort uncalibrated probabilities."),
        ("code", "cv = pd.DataFrame({name: {f'{metric}_{stat}': value for metric, values in metrics.items()\n"
         "    for stat, value in values.items()} for name, metrics in report['cross_validation'].items()}).T\n"
         "display(cv)\ndisplay(Markdown(report['selection_reason']))\n"
         "display(Image(filename=str(FIGURES_DIR / 'precision_recall_curve.png')))\n"
         "display(Image(filename=str(FIGURES_DIR / 'roc_curve.png')))"),
        ("md", "## Threshold optimization and mistakes in business terms\n\nThe operating point maximizes F2 on "
         "out-of-fold training predictions, giving recall more weight than precision. This is an illustrative "
         "preference, not an estimate of business costs. Missing a potential support opportunity and contacting "
         "someone who would remain may have different costs; real deployment would need an agreed policy."),
        ("code", "display(pd.read_csv(METRICS_DIR / 'threshold_analysis.csv'))\n"
         "display(Image(filename=str(FIGURES_DIR / 'threshold_analysis.png')))\n"
         "display(Image(filename=str(FIGURES_DIR / 'confusion_matrix.png')))"),
        ("md", "A true positive flags an observed attrition case; a false positive flags an observed retention case. "
         "A true negative leaves a retention case unflagged; a false negative misses an attrition case. "
         "The final holdout scores are the evaluation evidence. OOF scores reused in selection are optimistic diagnostics."),
        ("md", "## Interpretable associations\n\nInspect positive and negative logistic coefficients plus original-feature "
         "permutation importance. Correlated predictors can divide or mask importance. Neither method identifies causes."),
        ("code", "coefficients = pd.read_csv(METRICS_DIR / 'logistic_regression_coefficients.csv')\n"
         "display(coefficients.head(10))\ndisplay(coefficients.tail(10))\n"
         "display(Image(filename=str(FIGURES_DIR / 'feature_importance.png')))\n"
         "display(report['shap'])\n"
         "if report['shap']['status'] == 'generated':\n"
         "    display(Image(filename=str(FIGURES_DIR / 'shap_summary.png')))\n"
         "    display(Image(filename=str(FIGURES_DIR / 'shap_individual.png')))"),
        ("md", "## Calibration, stability, and capacity\n\nCalibration compares unchanged predictions with sigmoid "
         "calibration using five outer training folds and three inner folds, including preprocessing. The selected "
         "model identity comes from training CV, so these remain conditional diagnostics. Serving probabilities "
         "and the F2 threshold are unchanged. Reliability bins have equal width; empty bins remain in the CSV."),
        ("code", "display(pd.read_csv(METRICS_DIR / 'calibration_metrics.csv'))\n"
         "display(Image(filename=str(FIGURES_DIR / 'calibration_reliability.png')))\n"
         "display(pd.read_csv(METRICS_DIR / 'calibration_bins.csv'))\n"
         "display(pd.read_csv(METRICS_DIR / 'stability_comparison.csv'))\n"
         "display(pd.read_csv(METRICS_DIR / 'stability_selections.csv'))\n"
         "display(report['stability_diagnostics'])\n"
         "display(pd.read_csv(METRICS_DIR / 'capacity_metrics.csv'))"),
        ("md", "Seeds 42, 43, and 44 vary CV folds within the same training partition; they do not measure "
         "holdout-split uncertainty. Seed 42 remains the main experiment. Capacity tables select the top 5%, "
         "10%, and 20% by score, rounding counts up and preserving row order for ties. Training OOF and "
         "holdout rows are separate. Lift compares selected-group precision with partition prevalence."),
        ("md", "## Responsible ML and ablations\n\nCompare the same unweighted logistic model with/without "
         "Gender, Age, and MaritalStatus, and separately without three related career variables. These prespecified "
         "experiments are training-CV diagnostics, not a fairness guarantee or an extra test-selected model. "
         "The final model retains the sensitive attributes for this educational comparison."),
        ("code", "display(report['sensitive_ablation'])\ndisplay(report['correlation_ablation'])\n"
         "display(pd.read_csv(METRICS_DIR / 'subgroup_metrics.csv'))\n"
         "display(Markdown(f\"CV AP change after removing sensitive attributes: **{report['sensitive_ablation']['cv_ap_change']:+.3f}**. \"\n"
         "    f\"CV AP change after removing the correlated subset: **{report['correlation_ablation']['cv_ap_change']:+.3f}**. \"\n"
         "    'Small differences must be considered alongside fold variability.'))"),
        ("md", "## Key Findings\n\nThe executed results quantify how much each model improves over the dummy baseline. "
         "Threshold selection increases recall at the expense of false positives. Explanations describe model "
         "associations, and subgroup metrics remain uncertain because denominators are small. The fictional, "
         "cross-sectional sample does not validate a real employment decision system.\n\n"
         "## Next Steps\n\nValidate the calibration findings independently, investigate temporal generalization, "
         "formal fairness assessment, and support-oriented human oversight. This project must not be used as an "
         "automated system for firing, promotion, hiring, disciplinary action, or other high-impact employment decisions."),
    ])


if __name__ == "__main__":
    main()
