# FLAML benchmark

[← Development](development.md#optional-flaml-benchmark)

An optional experiment comparing [FLAML](https://github.com/microsoft/FLAML)'s CFO hyperparameter search with the Optuna AutoML search and the served baseline procedure. It is a development benchmark only: it does not train, select, promote, or serve a model, and nothing imports it implicitly. Use only the bundled fictional dataset or synthetic data; results never authorize individual employment decisions.

## Install and run

FLAML is a separate optional dependency. The serving image, `requirements.txt`, `requirements-serving.txt`, and `requirements-automl.txt` do not include it.

```bash
python -m pip install -r requirements-flaml.txt -c requirements-lock.txt   # includes requirements-automl.txt
python -m attrition.models.flaml_benchmark --n-trials 30 --timeout 900 --seed 42 --report-dir ~/attrition-experiments --run-id flaml-seed42-v1
```

`make flaml-benchmark` runs the same install and benchmark without writing a report directory. Each outer fold runs two searches, so the default five outer folds run ten searches of up to 30 trials each. That is a manual budget, not a CI job. For a bounded smoke run use `--n-trials 1 --outer-folds 2`.

| Option | Default | Meaning |
|---|---|---|
| `--n-trials` | 30 | Trial budget of each search (FLAML and Optuna alike), per outer fold |
| `--timeout` | 900 | Seconds after which each search starts no new trial |
| `--seed` | 42 | Seeds the outer and inner folds, both searchers, and every estimator |
| `--outer-folds` | 5 | Stratified outer folds of the training partition (2–10) |
| `--data` | `ATTRITION_DATA_PATH` or the checkout CSV | Raw CSV, validated like training |
| `--report-dir`, `--run-id` | none | Write `result.json`, `trials.csv`, `comparison.csv`, and `report.md` into a new run directory |
| `--output` | none | Also write the JSON to this **new** file |

Output paths follow the [AutoML experiment rules](automl-experiments.md#new-run-directories): new run directories and files only, and the models directory, canonical `reports/metrics/`, `reports/figures/`, `data/`, `README.md`, and `reports/results.md` are refused. If the FLAML or Optuna dependency is missing, the command fails with the install command before loading data.

## Protocol

The benchmark reuses the nested protocol of [the AutoML evaluation](development.md#nested-evaluation-of-the-automl-procedure), so every procedure faces the same training-only rules:

- The CSV is validated and split exactly like training. Only the training partition is used; holdout rows, labels, and published holdout metrics never reach any search, threshold, or comparison.
- Seeded stratified outer folds split the training partition. Inside each outer training fold, and on those rows only, three procedures run on the same seeded inner folds:
  1. **FLAML**: the CFO search picks a family and hyperparameters by mean inner-CV AP.
  2. **Optuna AutoML**: the existing TPE search does the same.
  3. **Baseline**: the served procedure (fixed candidates, prefer-logistic AP rule).
- For both searches, the out-of-fold probabilities of the frozen choice then pick the F2 threshold. This is the same shared code and the same grid and tie-breaks as baseline training. Each choice is refitted on the outer training rows and scored once on the outer validation rows.
- Every fit, including preprocessing, happens inside the relevant training fold. Outer validation labels are used only for the final score.

With the default seed and folds, the baseline rows reproduce the `fixed` rows of `reports/metrics/nested_cv_folds.csv`.

## How FLAML is used

The benchmark drives `flaml.tune` with FLAML's CFO (cost-frugal optimization) searcher. It does not use `flaml.AutoML`, for three reasons:

- `flaml.AutoML` fits its own preprocessing on all rows it receives before its internal CV, which would break this repository's rule that learned transforms are fitted inside each training fold.
- `flaml.AutoML` searches its own learners (LightGBM, XGBoost, and others) in its own ranges. A difference in results would then mix the search algorithm with the model families and ranges.
- `flaml[automl]` adds LightGBM and XGBoost. The `flaml[blendsearch]` extra requires `optuna<=3.6.1`, which conflicts with the repository's Optuna 5 pin.

As a result:

- **Same search space.** `SPACE` in `attrition.models.flaml_benchmark` declares the Optuna space: same families, parameters, inclusive integer bounds and steps, log scales, and options. `tests/test_flaml_benchmark.py` checks it against the distributions Optuna samples, including FLAML's exclusive `randint` upper bound. Categorical options are passed to FLAML as JSON labels, because its local search fails on mixed-type or `None` choices.
- **Same trial evaluation.** Each trial is rebuilt by `automl.candidate_pipeline`, the constructor the Optuna trials use, and scored by the same seeded stratified CV AP with the shared preprocessing pipeline.
- **Search start.** CFO first evaluates one start point per family, in family order, at the smallest ensemble sizes (`n_estimators=100` for random forest, `max_iter=50` for gradient boosting; other values drawn from the seeded searcher). It then searches locally from the best. FLAML ignores a starting-family index for this kind of choice, so explicit start points avoid favoring a family. With fewer trials than families, only the first families are tried.
- **Seeded and time-independent.** CFO gets the run's seed and `cost_attr=None`, so measured trial times never steer its proposals.
- **Failures.** A recoverable estimator error (`ValueError`/`RuntimeError`, including non-finite scores) marks the trial `FAIL`. CFO is told the trial scored below any attainable AP, and a failed trial is never selected. Unexpected exceptions abort. If a search has no successful trial, the benchmark stops with `status: incomplete`, keeps the failed search's evidence, writes its report, and exits 1.
- **Same budget.** Both searches get the same trial count and timeout. CFO is designed for time-budgeted, cost-aware search, so equal trial counts compare the algorithms at a fixed number of evaluations, not at FLAML's intended operating point.

## Output

Results share the AutoML conventions: sorted strict JSON, ordered CSV, UTF-8 with LF newlines, provenance (Git revision and dirty status, Python, platform, threadpools, and package versions including `flaml`), a training-data fingerprint, the raw CSV SHA-256, and `holdout_used`/`served_model_changed` always `false`.

| File | Contents |
|---|---|
| `result.json` | Complete result: fold rows per procedure, per-procedure means and population SDs, paired comparisons (`flaml_minus_baseline`, `flaml_minus_automl`, `automl_minus_baseline`), selection optimism per search, every search's full trial history, and the run ID |
| `trials.csv` | One row per trial, ordered by outer fold, framework (FLAML, then Optuna), and trial number |
| `comparison.csv` | Per metric: each procedure's mean and SD and the three paired mean differences; header only for an incomplete run |
| `report.md` | Comparison table and interpretation, or an explanation that no comparison is available |

A FLAML candidate can be rebuilt as an unfitted pipeline with `automl.candidate_pipeline(X, best["model"], best["params"], seed)`, as described for [Optuna candidates](automl-experiments.md#recorded-metadata-and-candidate-reconstruction). No joblib or pickle file is written or loaded.

## Interpreting results

These are training-only development estimates, not a final test. All three procedures are scored on the same outer folds, so per-fold differences are paired. The folds share most of their training rows and are not independent, so no standard error, interval, or significance test is reported. A small mean difference, or most folds favoring one procedure, does not show better generalization on this small fictional dataset. Another budget, seed, or search space can change the ranking. The published holdout has already been examined and must not settle the comparison; confirming a difference needs new independent data.

A benchmark result is not a reason to change the served model. Any promotion is a separate, owner-approved change (plan PR 11) with its own review gate.

## Installation and CI cost

Measured on 2026-10-13 on one Windows machine (Python 3.12.15, NumPy 2.5.3 already installed):

- `pip install flaml==2.7.0` took 4.2 s, added about 2.8 MB (203 files), and installed no other package: base FLAML needs only NumPy.
- `requirements-dev.txt` includes `requirements-flaml.txt`, so CI installs FLAML and runs `tests/test_flaml_benchmark.py`. These tests use synthetic data, one trial (at most four in the seeded search test), and logistic-only nested runs. The real-data test is opt-in with `ATTRITION_SLOW_TESTS=1`. The PR records the measured test runtime.
- `requirements-lock.txt` pins `flaml==2.7.0`, so the existing `pip-audit` step scans it. The serving image is unchanged.

## Reproducibility limits

The [AutoML reproducibility limits](automl-experiments.md#reproducibility-limits-and-verification) apply. FLAML's proposals also depend on its version: pin and record it, and compare runs only under matching FLAML, Optuna, scikit-learn, NumPy and pandas versions. A timeout that stops a search early makes the number of completed trials machine-dependent; use a trial count that finishes within the timeout.
