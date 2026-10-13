# Manual AutoML candidate promotion

[Development](development.md) · [Experiment evidence](automl-experiments.md) · [Serving](serving.md)

Promotion is an optional, operator-only workflow for the **fictional educational demo**.
It does not authorize hiring, firing, promotion, discipline, or other decisions about real people.
No candidate is selected for serving by its CV score automatically. The baseline training CLI,
AutoML search/evaluation CLIs, and app startup never invoke this workflow.

## Prepare an inactive candidate

Use the Python 3.12 development environment with `requirements-automl.txt` and
`requirements-lock.txt`. Generate a complete search report and a complete nested evaluation
report using the same fictional CSV, dependency versions, seed, trial/time budget and five inner folds, as in
[experiment tracking](automl-experiments.md). Nested evaluation assesses the search procedure;
it does not validate a particular fitted candidate. Review the baseline comparison, uncertainty,
failed trials, parameter choices, and costs before selecting a trial number. A nonwinning
completed trial can be selected explicitly. An apparent AP gain is not sufficient approval.

```bash
python -m attrition.models.promotion prepare --search reports/automl/search-seed42-v1/result.json --evaluation reports/automl/nested-seed42-v1/result.json --trial 0
```

`--data path.csv` overrides the input dataset. `--models-dir path` is a global option and must
appear **before** the subcommand; it defaults to `ATTRITION_MODELS_DIR` or the usual models
directory. All paths come from trusted operators, never HTTP requests. Only bundled fictional
or synthetic data may be used. Do not feed arbitrary downloaded joblib files to this workflow.

Preparation validates report schema/status, raw CSV checksum, the original training partition's
feature/label fingerprint, row counts, dependency versions, source revision presence, aligned
training-only CV configuration, and the explicit trial's completed state, scores and full estimator
parameters. It reconstructs through fixed estimator constructors; class strings are never imported.
It fits fold preprocessing and selects the F2 threshold using training OOF predictions, then fits
the frozen configuration on the original training partition. The original stratified split stays
at seed 42. Only split creation sees the raw target vector; fitting and threshold selection receive
training rows exclusively. No new search, holdout scoring, or baseline report regeneration occurs.

The command prints a new UUID and bundle path under `models/runs/`. The schema-2 bundle includes
the pipeline, threshold metadata, development summaries, source report snapshots, original report
checksums, preparation provenance, a training sample/background, drift reference, and a verified
manifest. Copied JSON evidence is normalized; the original input checksums refer to the input
bytes, while the manifest hashes the stored snapshots. Preserve the original reports too.
`selected_model_metrics` is null: there are no candidate holdout results. Existing bundles,
`current.json`, the generated README results and canonical reports are untouched. Failed writes
may leave an unsealed, inactive UUID directory; inspect it manually, never force reuse.

Fingerprints and report assertions detect mismatched inputs; they cannot prove that a report's
author respected the protocol. Source and data-separation review remains a human responsibility.
Git revision/dirty state is retained, but a dirty revision alone does not identify source changes.
Preserve and review the exact source/diff used by both reports and preparation. Version matching
does not guarantee numerical equivalence across platforms or BLAS implementations.

## Verify the exact candidate before approval

Use a maintenance window: stop training, pruning, and other artifact writers. Retain the current
baseline and candidate bundles and back up the models directory. Promotion commands serialize
each other with `.promotion.lock`; baseline training and pruning do not acquire that lock.
This is a local single-operator workflow, not a distributed compare-and-swap service. The models
directory must be on a filesystem supporting atomic replacement within the directory.

Create an **isolated staging models directory**, copy the candidate into its `runs/<UUID>` folder,
and publish only that staging pointer for smoke testing:

```python
from pathlib import Path
import shutil
from attrition.models.artifacts import load_bundle, publish_bundle

candidate = Path("models/runs/<candidate-UUID>")  # Substitute the actual UUID.
staging = Path(".test-tmp/promotion-smoke-models")  # Choose a fresh directory.
staged = staging / "runs" / candidate.name
shutil.copytree(candidate, staged)  # Refuses an existing destination.
assert load_bundle(staged).run_id == candidate.name
publish_bundle(staged, staging / "current.json")
```

Point the API and dashboard processes at staging with `ATTRITION_MODELS_DIR`; do not change
the live pointer for testing. Use distinct local ports. Start the API with an API key and verify
`/health`, authentication rejection, `/model`, `/predict`, and `/predict/batch` using the
bundle's `employee.json`. Confirm the expected UUID, frozen threshold, finite probabilities and
`holdout_metrics: null`. Use the [serving commands](serving.md), including PowerShell environment
syntax. Start Streamlit against the same staging directory, inspect every tab, and submit the
fictional profile. It must show development estimates, no claimed candidate holdout metrics,
and working predictions even when optional SHAP is unavailable. Record commands and outcomes.

Automated regression coverage:

```bash
python -m pytest -q tests/test_promotion.py tests/test_artifacts.py tests/test_api.py tests/test_dashboard.py
```

These synthetic tests exercise schema validation, rejected reviews, training boundaries, the API,
Streamlit, and exact rollback. They supplement smoke testing of the actual candidate in its target
runtime. Passing synthetic tests alone is not a candidate approval.

## Review and activate

A healthy existing `current.json` is required so every activation has a verified rollback target.
Generate a draft without modifying the candidate bundle:

```bash
python -m attrition.models.promotion review --run-id <candidate-UUID> > approval.json
```

Save/edit the JSON as UTF-8 (PowerShell 5 redirection can use UTF-16; use
`... | Set-Content -Encoding utf8 approval.json` there). The draft is **pending**, with empty
evidence and every check false. It binds the candidate UUID/manifest and the exact current pointer.
The owner must fill in:

- `decision`: `approve`; `reviewer`: their identity; `reason`: substantive candidate justification.
- `reviewed_at`: an ISO timestamp with timezone, for example `2026-10-13T14:00:00+07:00`.
- `evidence.source_review`: reviewed source revision/diff, report locations, data separation and limitations.
- `evidence.api_smoke` and `evidence.dashboard_smoke`: actual commands, runtime, results and candidate UUID.
- Each `checks` entry: true only after reviewing provenance, data separation, development evaluation,
  artifact schema, API smoke, dashboard smoke, rollback readiness, and responsible use.

Keep approval outside the sealed bundle. Changed files invalidate its manifest and approval.
Approval JSON is an auditable local attestation, **not authentication, a digital signature, or an
access-control boundary**. Restrict write access to models, receipts, reports and approvals.

```bash
python -m attrition.models.promotion activate --approval approval.json
```

The command verifies the candidate and rollback target again, rejects pending/incomplete reviews
and stale current pointers, writes a new recovery receipt under `models/promotions/<UUID>.json`,
then atomically replaces `current.json`. It never edits a prior bundle or baseline report.
Retain the printed receipt path. Restart the API: existing processes intentionally retain the model
loaded at startup. Streamlit resolves the current bundle on rerun. Confirm the served UUID.

## Roll back and recover

```bash
python -m attrition.models.promotion rollback --promotion-id <receipt-UUID> --expected-current <candidate-UUID> --reviewer "Owner name" --reason "Reason for rollback"
```

Rollback verifies the original activation receipt, exact previous manifest, and the expected current
pointer, writes its own receipt, and atomically restores the previous pointer. A missing/corrupt prior
bundle, changed manifest, or newer active run stops rollback without changing serving state. Restart
the API and rerun the dashboard, then verify the restored UUID and predictions. No files are deleted.

A receipt is written **before** the pointer switch so an interrupted command retains recovery evidence.
Its presence alone is not proof of successful activation. Compare `current.json` with its `before` and
`after` values: `before` means the switch did not complete; `after` means it completed; another value
means a later writer intervened and needs investigation. A stale lock after process termination must
be investigated and removed by the operator only when no writer remains. No force-unlock command is
provided. Retain/back up receipts and rollback bundles: the existing prune command does not protect
all receipt-referenced runs, so do not prune them during the rollback retention period.

The package/API version, schema-2 pointer/manifest, prediction fields, and baseline training behavior
are unchanged. API model info already permits null holdout metrics. The dashboard adds a separate
development-only presentation for AutoML bundles; baseline metrics keep their existing presentation.
Optuna remains optional for serving. No deployment, Git publication, or active-model change occurs
merely by installing this code.

Deploy these serving changes before activating a candidate: older dashboard/API code assumes every
schema-2 report contains holdout scores. The existing `scripts.verify_delivery` is a baseline-only
reproducibility check; use the candidate smoke procedure above for AutoML bundles.
