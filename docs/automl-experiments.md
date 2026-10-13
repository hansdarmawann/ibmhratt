# AutoML experiment tracking

[← Development](development.md#optional-automl-search-optuna)

Use only the bundled fictional dataset or synthetic data. These commands record development experiments; they do not promote a model or authorize individual employment decisions.

## New run directories

Install the optional dependency as described in [Development](development.md). Run from your assigned worktree with its own Python environment:

```bash
python -m attrition.models.automl --n-trials 30 --timeout 900 --seed 42 --report-dir reports/automl --run-id search-seed42-v1
python -m attrition.models.automl_evaluation --n-trials 30 --timeout 900 --seed 42 --report-dir reports/automl --run-id nested-seed42-v1
```

These are manual experiment budgets, not mandatory CI jobs. The nested command repeats the search per outer fold. For a bounded smoke run, use `--n-trials 1 --outer-folds 2` (or `--folds 2` for search).

Both CLIs accept `--report-dir` and an optional `--run-id`. Without a run ID, a UUID is generated. The directory is explicitly chosen by the operator; no reports are written by default. Run IDs contain 1–64 letters, digits, dots, underscores or hyphens and start with a letter or digit. Traversal, absolute paths, separators, Windows reserved names and trailing dots are refused. The new run directory is created exclusively, so concurrent/repeated runs cannot reuse an existing ID. An interrupted write leaves that directory reserved; use a new ID after reviewing the partial run.

The existing `--output path.json` interface continues to write a new JSON file with the same legacy fields plus schema-2 metadata. It is mutually exclusive with `--report-dir`; `--run-id` requires `--report-dir`. Both output interfaces refuse existing destinations, serving directories (including an externally configured models directory), canonical baseline metrics/figures/data directories, `README.md`, and `reports/results.md`. Paths are resolved before checking, including relative paths and symlink targets.

Each report directory contains:

| File | Contents |
|---|---|
| `result.json` | Complete result, search/evaluation schema version 2, report schema version 1, and run ID |
| `trials.csv` | One row per trial, ordered by outer fold and trial number; structured cells contain JSON |
| `comparison.csv` | Paired outer-CV metric means, population SDs and AutoML-minus-baseline differences for completed nested evaluations; header only for search or incomplete evaluation |
| `report.md` | Comparison table and interpretation, or an explanation that no comparison is available |

These files stay under the chosen run directory. The baseline auto-generated README block, `reports/results.md`, `reports/metrics/`, `reports/figures/`, model bundles and `models/current.json` are unchanged. No mutable SQLite database is introduced. JSON/CSV reports can be manually reviewed and committed as historical evidence; never commit a mutable study database or real personnel data.

## Recorded metadata and candidate reconstruction

Search records include the objective (`average_precision`), maximize direction, training-only fold configuration and seed, TPE sampler seed, requested trial/time budget and serial execution, training row/positive counts, per-fold AP, mean and population SD, trial state, sampled family-prefixed parameters, full estimator parameters (including fixed defaults), estimator class name, trial/search elapsed seconds, and failure type/message. The untuned dummy reference remains separate from search candidates. Nested results retain the full search result for every attempted outer fold, in addition to thresholds and paired outer-validation metrics.

Provenance includes Python and platform, installed package versions (including Optuna, scikit-learn, NumPy, pandas, SciPy and threadpoolctl), threadpool information, Git revision and working-tree dirty status when available. Git absence, a wheel install, or unavailable package metadata produces null fields. Git calls are read-only. A dirty revision does not identify the exact uncommitted source: preserve the reviewed diff or manually commit the experiment code before a reproducibility run.

Training metadata records column order/types, the target mapping, the shared preprocessing builder, and a SHA-256 fingerprint of training features/labels including row order and index. CLI results also include the raw CSV SHA-256. The baseline's original stratified 80/20 split uses seed 42; the AutoML `--seed` changes search/evaluation seeds, not that reserved split. Input fingerprints depend on pandas hashing and should be compared under recorded versions. No raw rows or labels are serialized.

Rebuild a search candidate as an **unfitted** pipeline using the recorded family, sampled parameters and seed. Full estimator parameters are recorded so changes in estimator defaults can be detected. Use the same source revision, dependency versions and fictional/synthetic input:

```python
import json
from pathlib import Path

from attrition.data.load_data import load_data
from attrition.models import automl, automl_reporting

result = json.loads(Path("reports/automl/search-seed42-v1/result.json").read_text(encoding="utf-8"))
X, y = automl.training_partition(load_data())
assert automl_reporting.training_metadata(X, y) == result["training_data"]
best = result["best"]
if best is None:
    raise ValueError("This run has no successful candidate; inspect trial_records.")
pipeline = automl.candidate_pipeline(X, best["model"], best["params"], result["sampler"]["seed"])
assert pipeline.named_steps["model"].get_params(deep=False) == best["estimator_params"]
```

The estimator class string is descriptive metadata, not dynamically imported executable input. Reconstruction uses the existing fixed family constructors and shared preprocessing. No joblib/pickle is written or loaded, no final pipeline is fitted by search, and the API/dashboard gain no artifact-path interface. Reconstruction and promotion of a served model remain separate activities; promotion requires owner approval and PR 11.

## Failures and comparison interpretation

Recoverable estimator errors (`ValueError`/`RuntimeError`, including non-finite objective scores) mark the trial `FAIL`, record its error and elapsed time, and allow subsequent trials. Failed trials cannot become the best candidate. Programming errors and interrupts still abort rather than being silently treated as ordinary estimator failures.

If all search trials fail, `best` is null and `status` is `no_successful_trials`. The search CLI writes the failure evidence, prints JSON, then exits 1. If an outer fold has no successful trials, nested evaluation stops, retains attempted searches and already completed fold rows, reports `status: incomplete`, omits procedure/comparison summaries, writes evidence, and exits 1. A malformed input, failure of the dummy reference or unexpected exception may abort before a report is produced; inspect the command's stderr. Error messages are diagnostic text and may include operator paths; review them before sharing.

A search-only run has no baseline comparison; its winning CV score is optimistic because those same folds selected it. The nested report compares the complete AutoML and served baseline procedures using the same training-only outer partitions and metrics, with inner preprocessing/search/OOF threshold choice confined to each outer training fold. It does not read baseline holdout metrics or use outer-validation labels for selection. Historical baseline reports are never used as optimizer inputs.

AP is average precision (`pr_auc` in nested JSON), not trapezoidal PR area. Fold SDs and paired differences are descriptive development estimates; overlapping training folds are not independent, so these reports provide no confidence interval or significance test. Apparent gains on this small fictional dataset do not establish generalization. The published holdout has already been examined and must not settle an iterative comparison; confirmation requires new independent data.

## Reproducibility limits and verification

JSON keys and CSV rows/cells have stable ordering, strict finite numeric serialization, UTF-8 encoding and LF newlines. Rendering the same result is byte-repeatable. Independent searches are not byte-identical: elapsed time, run ID, Git dirty status, platform and threadpool metadata can differ.

Use a fixed trial count that completes before the timeout, fixed seeds, serial jobs and one math thread, as the CLIs do. Time-limited budgets depend on machine speed and cannot guarantee the same completed trial count. Versions, OS/CPU and BLAS/OpenMP implementations can change numerical results; decimal rounding does not guarantee platform equivalence. Stochastic estimators and sampler behavior depend on their implementations as well as seeds. Matching metadata helps audit a run but does not make the small dataset representative or provide an independent test set.

`tests/test_automl_reporting.py` verifies metadata, candidate reconstruction, fingerprints, failures, strict/ordered serialization, baseline preservation, report directory collisions, path protections and CLI behavior. Existing AutoML tests continue to instrument preprocessing, search and threshold boundaries. Large experiments remain opt-in.
