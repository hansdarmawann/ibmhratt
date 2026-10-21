# Model card: employee attrition classifier

[← Back to README](../README.md)

A reviewer-oriented summary of what the served model is, how it was built and evaluated, what the evidence does and does not show, and how to reproduce it. Every number below comes from a committed artifact named next to it, with one labeled exception (the optional AutoML development estimate, whose output is not committed; see [Measured AutoML development estimate](#measured-automl-development-estimate)). If a number here disagrees with its artifact, the artifact is authoritative and this page needs a fix. The training CLI does **not** regenerate this page (only the README results block and `reports/results.md`), so [tests/test_model_card.py](../tests/test_model_card.py) checks the figures against the artifacts.

## At a glance

| Item | Value |
|---|---|
| Purpose | Educational demonstration of a leakage-aware tabular workflow. **Not** a decision tool. |
| Served model | `logistic_regression` (L2, `C=1.0`), `schema_version` 2 bundle |
| Output | Estimated attrition probability, plus `Yes`/`No` from the frozen threshold **0.20** |
| Input | 30 predictors from one employee record ([examples/employee.json](../examples/employee.json)) |
| Training data | Fictional IBM HR sample, 1,470 rows; 1,176 training rows |
| Final evaluation | 294 reserved holdout rows with 47 observed attrition cases, already examined (see [Evaluation](#evaluation-and-uncertainty)) |
| Seed | 42 (split, folds, estimators) |
| Source of truth | [model_metrics.json](../reports/metrics/model_metrics.json) |

## Intended and prohibited use

**Intended:** learning and demonstrating leakage-safe evaluation, calibration and subgroup diagnostics, explanation tooling, and model serving on public fictional data.

**Prohibited:** hiring, firing, promotion, compensation, discipline, or any other decision about an individual's employment; ranking or surveilling real employees; entering real personnel data into the API, dashboard, or notebooks. Outputs are associations in a small fictional sample, not causal evidence, and not forecasts for a person. The probability that a profile receives says nothing about why someone leaves or what would change that. See [Responsible AI](responsible-ai.md) for what any real use would additionally need (a support-oriented purpose, consent and privacy controls, governance review, human oversight, fairness definitions, independent validation, and monitoring).

## Data

- **Source:** `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`, the IBM HR sample as distributed on [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), described by its publisher as fictional data created by IBM data scientists. Its SHA-256 is recorded in the model metadata (`data_sha256` in [model_metrics.json](../reports/metrics/model_metrics.json)): `e9f55fbf0a5c058306225d131311e135379d82ad0c94c33738ec75b9a179db9c`.
- **Rights:** redistribution terms of the dataset and the source-code license are unverified and are open owner decisions (see [Licensing status](../README.md#licensing-status)). This card makes no claim about either.
- **Shape and target:** 1,470 rows, 35 columns ([data_audit.json](../reports/metrics/data_audit.json)). `Attrition` Yes = 1, No = 0; 237 Yes (16.1%). No missing values or duplicate rows.
- **Predictors:** 30. Four columns are excluded after checking the training rows: `EmployeeCount`, `Over18`, `StandardHours` (constant) and `EmployeeNumber` (identifier) ([feature_exclusions.json](../reports/metrics/feature_exclusions.json)).
- **Sensitive attributes retained:** `Age`, `Gender`, `MaritalStatus`. See [Fairness and subgroup limits](#fairness-and-subgroup-limits).
- **Not in the data:** time. It is cross-sectional with no observation window or prediction horizon, so the data cannot establish that a feature was known before someone left.

## Splits and procedure

| Stage | Data | Notes |
|---|---|---|
| Holdout split | Seeded stratified 80/20 split: 1,176 training rows (190 attrition cases) and 294 holdout rows (47) | Created before exploration; `TEST_SIZE=0.20`, `RANDOM_STATE=42` in `src/attrition/config.py` |
| EDA | Training rows only | |
| Candidate comparison | Five-fold stratified CV inside training | Preprocessing (imputation, scaling, one-hot encoding) is fitted inside each fold with the estimator |
| Model choice | Mean CV average precision, plus a prespecified preference for unweighted then balanced logistic regression when within 0.01 AP of the best non-dummy model | Fixed before holdout evaluation |
| Threshold | Maximum F2 on training out-of-fold (OOF) predictions; ties broken by precision, then the higher threshold | Selected 0.20 |
| Final evaluation | Holdout, after model and threshold are frozen | The served pipeline is fitted on training rows only |

Candidates: a prior-based dummy classifier, logistic regression, class-balanced logistic regression, a constrained balanced random forest, and a regularized histogram gradient boosting model. [Data and methodology](methodology.md#evaluation-boundaries) lists exactly what each stage may see and which tests ([test_evaluation_boundaries.py](../tests/test_evaluation_boundaries.py)) enforce it. Those tests check the current workflow's data flow; they are not a sandbox against arbitrary code.

## Baseline versus AutoML

These are two different things and must not be conflated.

- **Baseline (served).** `python -m attrition.models.train` compares the five fixed candidates above, applies the selection rule, picks the threshold, evaluates the holdout, and writes the bundle behind `models/current.json`. Everything in this card's results sections describes this baseline.
- **AutoML search (optional, development only).** `python -m attrition.models.automl` runs an opt-in, seeded Optuna search over logistic regression, random forest, and histogram gradient boosting, scored by mean stratified CV AP on the training partition ([usage](development.md#optional-automl-search-optuna)). It never reads holdout rows, fits a final model, chooses a threshold, writes a bundle, or changes the served model. The best trial's CV AP is optimistic because the same folds picked it.
- **Nested AutoML evaluation (optional, development only).** `python -m attrition.models.automl_evaluation` repeats the whole AutoML procedure (search, then an OOF F2 threshold from the frozen configuration, then a refit) inside each outer training fold and scores it once on that fold's outer validation rows. The served baseline procedure runs on the same outer and inner folds, so the comparison is paired ([method](development.md#nested-evaluation-of-the-automl-procedure)). It uses the training partition only, never computes a holdout score, and does not change the served model.

### Measured AutoML development estimate

Seed 42, 30 trials per outer fold, 5 outer folds on the 1,176 training rows. These are **training-only development estimates**, not a final test and not comparable to the holdout table above. The evaluation's JSON output is written only where the operator asks and is **not committed**, so this table is not backed by a tracked machine-readable artifact. It comes from the run recorded in `docs/pr-progress/PR-05.md` and was independently reproduced for this card by rerunning the command below (same figures, data SHA-256 `e9f55fbf0a5c…`, Optuna 5.0.0, scikit-learn 1.8.0). Its baseline arm matches the `fixed` rows of the committed [nested_cv_folds.csv](../reports/metrics/nested_cv_folds.csv).

| Procedure | Outer AP (mean ± SD) | ROC-AUC | Precision | Recall | F1 | Selections | Thresholds |
|---|---:|---:|---:|---:|---:|---|---|
| AutoML | 0.653 ± 0.055 | 0.838 ± 0.032 | 0.382 ± 0.044 | 0.737 ± 0.062 | 0.500 ± 0.037 | logistic regression in 5 of 5 folds | 0.15 to 0.20 |
| Served baseline procedure | 0.651 ± 0.061 | 0.839 ± 0.029 | 0.369 ± 0.047 | 0.758 ± 0.065 | 0.492 ± 0.034 | logistic regression in 5 of 5 folds | 0.10 to 0.20 |

Paired per-fold AP difference (AutoML minus baseline): mean +0.001, SD 0.007, range -0.006 to +0.014, AutoML higher in 3 of 5 folds. The mean inner best-trial AP was 0.636 versus a mean outer AP of 0.653, so the optimism expected from selection did not show up on these small folds; that reflects sampling noise, not an absence of selection bias in general.

**Interpretation.** The two procedures are practically indistinguishable here, and AutoML selected the same model family as the baseline in every fold. The outer folds share most of their training rows, so the per-fold differences are not independent and no significance test is reported. A gain, had there been one, on this small fictional dataset would not be evidence of generalization, and the examined holdout must not be used to choose between the procedures. The served model was not changed.

## Metrics

- **Average precision (AP)**, labeled `pr_auc` in the JSON: scikit-learn's `average_precision_score`, a weighted mean of precision over recall steps. It is not trapezoidal area under the precision-recall curve. A no-skill classifier scores about the positive rate (0.16 here: the dummy's CV AP is 0.162 and holdout AP is 0.160).
- **ROC-AUC:** probability that a random attrition case is scored above a random retention case.
- **Precision, recall, F1, F2:** computed for the positive class (`Attrition = Yes`) at a stated threshold. F2 weights recall more than precision. It is an educational preference, not an estimated cost, and acting on it would require an agreed intervention policy.
- **Brier score and log loss:** probability error; lower is better.
- **Capacity lift:** precision among the top-scored share of rows divided by the base rate.
- Accuracy is not used: always predicting "No" would score about 84%.

## Evaluation and uncertainty

The selected model's numbers, with their sources, keeping the different evaluation kinds apart:

| Kind | Result | What it is and is not | Artifact |
|---|---|---|---|
| Ordinary 5-fold CV (training, threshold 0.50 for P/R/F1) | AP 0.651 ± 0.061; ROC-AUC 0.839 ± 0.029 | Used to choose the model, so it is optimistic for the chosen model. ± is the population SD across folds. | `cross_validation` in [model_metrics.json](../reports/metrics/model_metrics.json), [cv_fold_metrics.csv](../reports/metrics/cv_fold_metrics.csv) |
| Nested CV, current procedure | AP 0.651 ± 0.061 | Repeats the candidate comparison, selection rule, and threshold choice inside each outer training fold and scores the outer fold once. A diagnostic, not an independent test. | `nested_cv` in the JSON, [nested_cv_folds.csv](../reports/metrics/nested_cv_folds.csv) |
| Nested CV, with small prespecified grids | AP 0.646 ± 0.048 | Tuning did not help on these folds. | same |
| Nested CV of the AutoML procedure | AP 0.653 ± 0.055 (served-procedure arm on the same folds: 0.651 ± 0.061) | Development estimate for the optional AutoML tool; see [Baseline versus AutoML](#baseline-versus-automl). | no committed artifact; `docs/pr-progress/PR-05.md` |
| OOF at threshold 0.20 (training) | precision 0.428, recall 0.721, F1 0.537, F2 0.634 (TP 137, FP 183, FN 53, TN 803) | Selection diagnostic: the same OOF predictions chose the threshold. | [threshold_analysis.csv](../reports/metrics/threshold_analysis.csv) |
| **Final holdout** at frozen threshold 0.20 | AP 0.584, ROC-AUC 0.812, precision 0.423, recall 0.638, F1 0.508 (TP 30, FP 41, FN 17, TN 206) | Computed after choices were frozen; it reflects sampling noise from only 47 positives. | `selected_model_metrics` in the JSON, [results.md](../reports/results.md) |

**Holdout uncertainty.** 95% stratified bootstrap intervals (1,000 resamples, model and threshold fixed): AP [0.460, 0.711], ROC-AUC [0.734, 0.883], precision [0.333, 0.517], recall [0.489, 0.766], F1 [0.404, 0.607] ([holdout_bootstrap_ci.csv](../reports/metrics/holdout_bootstrap_ci.csv)). They capture holdout sampling variability only. They do not capture the variability from the choice of split, model, or threshold.

**Seed sensitivity** (training CV only; [stability_comparison.csv](../reports/metrics/stability_comparison.csv), [stability_selections.csv](../reports/metrics/stability_selections.csv)). Across CV seeds 42, 43, and 44 the selected model was logistic regression each time, with CV AP 0.651, 0.670, and 0.656 and thresholds 0.20, 0.20, and 0.15. This does not measure sensitivity to the holdout split.

**The holdout is not a fresh benchmark.**

- It is not evaluated strictly once. The holdout is scored for all five candidates (the other four are *post-selection descriptive diagnostics* that never fed back into the choice), for the selected model at its frozen threshold, and again for bootstrap, subgroup, capacity, and permutation-importance summaries.
- The published holdout has already been examined. Repeated model development against it would turn it into a tuning set. Confirming generalization after further development needs new independent data.
- With 294 rows and 47 positives, one or two predictions move recall by several points.

## Calibration

Reported probabilities are uncalibrated estimates. A diagnostic comparison on training rows (five outer folds, sigmoid calibration fitted with three inner folds) gave Brier score 0.090 uncalibrated versus 0.093 calibrated and log loss 0.321 versus 0.321 ([calibration_metrics.csv](../reports/metrics/calibration_metrics.csv)). Calibration therefore did not help on these folds and the served probabilities are not calibrated. This diagnostic was run conditional on the already chosen model class, so it does not validate calibration on new data. Class-weighted candidates (not served) shift scores upward by design, so their raw probabilities should not be read as frequencies. Do not read a score of 0.30 as "a 30% chance"; the bin-level view is in [calibration_bins.csv](../reports/metrics/calibration_bins.csv) and `calibration_reliability.png`.

## Operating point and capacity

At threshold 0.20, the holdout flags 71 of 294 profiles (30 true positives and 41 false positives) and misses 17 of the 47 observed cases. Flagging only the top 10% of holdout profiles (30 rows) gives precision 0.633, recall 0.404, and lift 3.96 ([capacity_metrics.csv](../reports/metrics/capacity_metrics.csv)). These describe ranking quality on this sample; they are not a policy. Lowering the threshold flags more people, and the right trade-off depends on an intervention that this project does not define.

## Fairness and subgroup limits

Subgroup recall, false-positive rate, precision, and selection rate on the holdout, with 95% bootstrap intervals, are in [subgroup_metrics.csv](../reports/metrics/subgroup_metrics.csv). The groups are small. For example, the Divorced group has 64 rows with 4 observed attrition cases, so its recall of 1.00 has an interval of [1.00, 1.00] that only reflects those four cases, and the 50+ age band has 8 positives (recall 0.25, interval [0.00, 0.50]). Selection rates differ across groups (for instance 0.40 for Single versus 0.11 for Divorced), which can reflect differences in the data as well as in the model.

- No formal fairness definition has been chosen or certified, and none is claimed.
- A training CV ablation that drops `Gender`, `Age`, and `MaritalStatus` changes logistic AP by -0.019 (`sensitive_ablation`); that is a diagnostic and does not show these attributes are acceptable to use in real HR work.
- Removing sensitive columns would not remove proxies such as job role, compensation, or tenure.
- Subgroup results use one holdout split and are descriptive only.

## Explanation limits

- **Logistic coefficients** ([logistic_regression_coefficients.csv](../reports/metrics/logistic_regression_coefficients.csv)): numeric features act per one training standard deviation and categories act on one-hot indicators. Full one-hot encoding and correlated predictors make single coefficients unreliable to interpret in isolation. The largest positive coefficients are `JobRole_Laboratory Technician` (1.03), `OverTime_Yes` (0.92), and `BusinessTravel_Travel_Frequently` (0.90).
- **Permutation importance** (holdout AP decrease over ten shuffles, [permutation_importance.csv](../reports/metrics/permutation_importance.csv)): `OverTime` is largest at 0.233 ± 0.025, followed by `JobSatisfaction` (0.066) and `NumCompaniesWorked` (0.059). Correlated features can share or mask importance.
- **SHAP** is optional and not used for any choice. Logistic contributions are in log-odds, not probability points. The dashboard's per-profile explanation uses the active bundle's training background. The default container omits SHAP.
- Every explanation describes how this model uses its inputs. None demonstrates a cause. For example, a higher overtime association does not show that overtime makes anyone leave.

## Feature engineering result

Five prespecified formulas added to the logistic model improved training CV AP by +0.014 and were higher in 4 of 5 folds, which meets the rule written beforehand (at least +0.01 and at least four folds). They are **not adopted**: the rule was evaluated on training CV used for these choices, and adoption would need validation on new data (`feature_engineering_ablation` in the JSON). A dropped-correlated-features ablation (`JobLevel`, `YearsInCurrentRole`, `YearsWithCurrManager`) changed AP by -0.005 and is also only a diagnostic.

## Known limitations

- Small, fictional, cross-sectional sample; no temporal, external, or population validation.
- One holdout split, already examined; wide intervals.
- Probabilities are not calibrated. Ordinal ratings are treated as numbers with equal spacing.
- Broad schema bounds are modeling assumptions, not fitted ranges. Unseen categories are accepted and encoded as unknown, which can reduce reliability.
- Structural leakage safeguards do not rule out real-world look-ahead bias (features that would be unavailable before an event).
- No monitoring of live inputs beyond the optional PSI drift check.
- The AutoML development estimate is training-only, from one seed and one trial budget, and not backed by a committed artifact; it should be treated as indicative of nothing more than "no visible difference on these folds".
- Associations only; nothing here supports causal conclusions or individual decisions.

## Reproducibility

Recorded environment in the metadata: Python 3.12.15, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.8.0, joblib 1.6.0, seed 42. Reproduce with the pinned constraints:

```bash
python -m pip install -r requirements-dev.txt -c requirements-lock.txt
python -m attrition.models.train           # regenerates reports/, README results block, and a new model bundle
python -m pytest -q                        # includes tests/test_model_card.py
python -m scripts.check                    # full gate: lint, types, tests, fresh training, notebooks, live services
```

CI checks that the committed README results block and `reports/results.md` match a fresh run and compares every committed metrics file. Other platforms may install different transitive packages, so tiny numeric differences are possible; the comparison tolerates these. Training rewrites tracked reports and bundles, so review `git status` before committing regenerated files. The optional AutoML search is separate and opt-in:

```bash
python -m pip install -r requirements-automl.txt -c requirements-lock.txt
python -m attrition.models.automl --n-trials 30 --timeout 900 --seed 42 --output ~/attrition-automl/search-seed42.json
python -m attrition.models.automl_evaluation --n-trials 30 --timeout 900 --seed 42 --output ~/attrition-automl/nested-seed42.json   # about 5 minutes
```

## Maintenance

If training is rerun and any artifact changes, update this card (and `tests/test_model_card.py`) in the same change. If the AutoML development estimate is refreshed, cite its output file, seed, trial budget, data hash, and package versions, and keep it labeled a development estimate unless it was scored on genuinely new data. Do not add AutoML numbers to the holdout comparison.
