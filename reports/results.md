# Executed experiment results

| Model | CV AP (mean ± SD) | Test AP | Test ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| dummy | 0.162 ± 0.000 | 0.160 | 0.500 | 0.000 | 0.000 | 0.000 |
| logistic_regression | 0.651 ± 0.061 | 0.584 | 0.812 | 0.615 | 0.340 | 0.438 |
| logistic_balanced | 0.606 ± 0.082 | 0.561 | 0.803 | 0.349 | 0.638 | 0.451 |
| random_forest | 0.552 ± 0.058 | 0.436 | 0.789 | 0.524 | 0.468 | 0.494 |
| hist_gradient_boosting | 0.604 ± 0.045 | 0.542 | 0.796 | 0.786 | 0.234 | 0.361 |

Table classification metrics use threshold 0.50. AP is average precision, not trapezoidal PR area.

Selected **logistic_regression** at threshold **0.20**, chosen by maximum F2 on training out-of-fold predictions.

At the frozen operating threshold, holdout AP = **0.584**, ROC-AUC = **0.812**, precision = **0.423**, recall = **0.638**, F1 = **0.508**.

95% stratified bootstrap intervals (1,000 holdout resamples, frozen model and threshold): AP [0.460, 0.711], ROC-AUC [0.734, 0.883], precision [0.333, 0.517], recall [0.489, 0.766], F1 [0.404, 0.607]. They reflect holdout sampling variability only, not split or model-selection variability.

- True positives: 30 observed attrition cases flagged.
- True negatives: 206 observed retention cases not flagged.
- False positives: 41 observed retention cases flagged.
- False negatives: 17 observed attrition cases missed.

Selection uses only five-fold training average precision. Prefer unweighted, then balanced logistic regression when within 0.01 AP of the best non-dummy model, for interpretability and simpler operation. Otherwise use the highest mean AP. Best AP candidate: logistic_regression; selected: logistic_regression. This rule was fixed before holdout evaluation.

OOF threshold scores reuse the training folds used for model comparison and are selection diagnostics, not an unbiased performance estimate. The holdout is evaluated after choices are frozen. There are 47 positive holdout examples, so small count changes materially affect recall.
