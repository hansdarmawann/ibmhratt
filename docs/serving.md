# Serving and operations

[← Back to README](../README.md)

Running the API, the dashboard, and the container, and operating the versioned model bundles.

## API usage

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/docs` for the complete Pydantic/OpenAPI schema. `GET /` describes the service; `GET /health` returns status and model availability, with HTTP 503 if the artifact cannot load. `POST /predict` accepts a single employee object with all 30 predictor keys. Numeric fields are strict whole numbers with bounds; categories are nonempty strings. Fields can explicitly be null for imputation, but missing or extra keys fail with HTTP 422. Novel categories are accepted and encoded as unseen values; they may reduce reliability.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @examples/employee.json
```

PowerShell equivalent:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body (Get-Content examples/employee.json -Raw)
```

Responses contain `prediction` (Yes/No), `attrition_probability`, `threshold`, and nullable `run_id`, populated from the actual model. Health responses also identify the loaded run; explicit legacy schema-1 artifacts return null. No mock probabilities are served.

`POST /predict/batch` accepts `{"employees": [...]}` with 1 to 1,000 employee objects using the same per-record schema, and returns `count` plus `predictions` in request order. `GET /model` reports the API version, run ID, selected model, decision threshold, feature count, training rows, dataset hash, pinned package versions, and the frozen holdout metrics of the loaded bundle (null metrics for an explicit legacy file).

Every response carries an `X-Request-ID` header: a safe caller-supplied value (letters, digits, `.`, `_`, `-`, at most 64 characters) is echoed, otherwise one is generated. Responses also send `X-Content-Type-Options: nosniff` and `Cache-Control: no-store`. The `app.api` logger records request ID, method, percent-encoded path, status, and duration only, never request bodies or headers. Set `ATTRITION_API_KEY` before starting the server to require a matching `X-API-Key` header on `/model`, `/predict`, and `/predict/batch`; `/`, `/health`, and the OpenAPI docs stay open. The key is checked before the request body is read, so a request without it gets HTTP 401 whatever its payload. Bodies over 8,192,000 bytes get HTTP 413. The key is a demo guard only; see [Deployment boundaries](#deployment-boundaries). The Python module accepts a dictionary or DataFrame and returns `predicted_class`, `attrition_probability`, `decision_threshold`, and nullable `run_id`; a DataFrame yields a list in row order.

```bash
python -m attrition.models.predict examples/employee.json
```

The model loads once during API startup; restart after retraining. Requests log event/count information without raw profiles. Joblib artifacts must come from a trusted source because pickle-based loading can execute code. The optional shared API key is a basic demo guard, not production identity, authorization, rate limiting, or TLS; [Deployment boundaries](#deployment-boundaries) lists what any wider exposure requires.

### Example responses

Captured from a local run of the current model; `run_id` changes with every training run.

`POST /predict` with [`examples/employee.json`](../examples/employee.json):

```json
{"prediction": "No", "attrition_probability": 0.004045697721405301, "threshold": 0.2, "run_id": "60d37a55-5c15-47fa-a961-d99776beef18"}
```

`POST /predict/batch` with that profile and a copy changed to `"OverTime": "Yes", "JobSatisfaction": 1`, sent with `X-Request-ID: docs-example-1` (echoed in the response header):

```json
{
  "count": 2,
  "predictions": [
    {"prediction": "No", "attrition_probability": 0.0040456977214052975, "threshold": 0.2, "run_id": "60d37a55-5c15-47fa-a961-d99776beef18"},
    {"prediction": "No", "attrition_probability": 0.03593143779864496, "threshold": 0.2, "run_id": "60d37a55-5c15-47fa-a961-d99776beef18"}
  ]
}
```

Single and batch probabilities can differ in the last floating-point digits because rows are scored together.

`GET /model`:

```json
{
  "api_version": "1.1.0",
  "run_id": "60d37a55-5c15-47fa-a961-d99776beef18",
  "schema_version": 2,
  "selected_model": "logistic_regression",
  "decision_threshold": 0.2,
  "feature_count": 30,
  "train_rows": 1176,
  "data_sha256": "e9f55fbf0a5c058306225d131311e135379d82ad0c94c33738ec75b9a179db9c",
  "package_versions": {"numpy": "2.5.3", "pandas": "3.0.6", "scikit-learn": "1.8.0", "joblib": "1.6.0"},
  "holdout_metrics": {
    "pr_auc": 0.5836342531326222, "roc_auc": 0.8115255405289, "precision": 0.4225352112676056,
    "recall": 0.6382978723404256, "f1": 0.5084745762711864, "brier_score": 0.09918620635299684,
    "threshold": 0.2, "n": 294, "positive_count": 47
  }
}
```

`POST /predict` with only `{"Age": 30}` returns HTTP 422 with one entry per missing field (29 here), for example:

```json
{"type": "missing", "loc": ["body", "DailyRate"], "msg": "Field required"}
```

Validation errors give the type, location, and message of each problem but never echo submitted values. At most 100 entries are listed, so one invalid batch cannot produce an oversized response.

## Deployment boundaries

The API is built for local demonstration on a loopback address: `--host 127.0.0.1`, or `docker run -p 127.0.0.1:8000:8000`. Any wider exposure (a LAN, a cloud VM, or a public URL) requires the controls below. They live outside this repository, and the optional API key replaces none of them.

| Control | Required before wider exposure | What the service itself does |
|---|---|---|
| Authentication and authorization | An identity-aware gateway or reverse proxy (for example OIDC, mutual TLS, or a managed API gateway) with per-client credentials, revocation, and audit logs | Optional single shared `X-API-Key`. There are no users, roles, expiry, or lockout, and rotating the key needs a restart. |
| TLS | Terminate HTTPS at the gateway; never send the key over plain HTTP | Plain HTTP only |
| Rate and concurrency limits | Per-client limits at the gateway | None |
| Body limits | A gateway limit at or below the service's own | HTTP 413 above 8,192,000 bytes, whether declared or streamed in chunks; HTTP 422 above 1,000 batch records |
| Trusted artifact provisioning | Build images from reviewed commits; mount external bundles read-only from storage that only trusted operators can write | Loads only the operator-configured path and verifies checksums, schema, and metadata at startup. Checksums do not make an untrusted pickle safe. |
| Monitoring | Collect logs; alert on 401, 413, 422, and 5xx rates, latency, `/health` failures, and [input drift](#input-drift-monitoring) | Request ID, method, encoded path, status, and duration per request; generic 500 and 503 bodies, with details in server logs only |

The service also rejects unknown fields, labels, and identifiers in every record, and returns no stack traces or artifact paths to callers. Requests from callers without the key are rejected before their body is parsed. Uvicorn's own access log includes query strings, so never put secrets in URLs.

**API key handling.** Generate a long random key, for example `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Supply it at runtime (`docker run -e ATTRITION_API_KEY`, or a secret manager), never in the image, a build argument, a committed file, or a URL. It is one static secret shared by every client: anyone who holds it has full access, and nothing limits guessing. API-key authentication alone is not production security.

**Container.** The image runs as the unprivileged `appuser`, and CI's container smoke test asserts a non-root UID and serving-only contents. The API only reads model files. To serve bundles from outside the image, mount them read-only and point `ATTRITION_MODELS_DIR` at the mount, for example `-v "$PWD/models:/models:ro" -e ATTRITION_MODELS_DIR=/models`. The files must be readable by the container user; `docker run --rm --entrypoint id employee-attrition-ml` prints its UID. CI does not exercise external mounts. Further restrictions such as `--read-only`, `--cap-drop ALL`, and `--security-opt no-new-privileges` are reasonable defaults but are also untested here, so verify them in your environment before relying on them.

Whatever the deployment, predictions remain educational signals and must not drive decisions about individual employees; see [Responsible AI](responsible-ai.md).

## Streamlit usage

```bash
python -m streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. Tabs cover project/dataset overview, training-only EDA, model performance, an editable prediction form, feature importance, model explanation, and responsible ML. The form displays estimated probability first, its threshold-derived class, and the frozen operating threshold. Resources refresh when the active run changes. Model, report, plots, sample input, and explanation background are read from that same verified run.

### Screenshots

Project overview, with the disclaimer and the active run ID:

![Dashboard overview tab](images/dashboard-overview.png)

Model performance: selection rule, frozen holdout metrics at the operating threshold, the model comparison at 0.50, and bootstrap intervals:

![Dashboard model performance tab](images/dashboard-performance.png)

Prediction demo: the estimated probability comes first, then the threshold-derived class and the per-feature contributions for the submitted profile:

![Dashboard prediction demo with explanation](images/dashboard-prediction.png)

## Docker usage

```bash
docker build -t employee-attrition-ml .
python -m scripts.verify_container --image employee-attrition-ml
docker run --rm -p 127.0.0.1:8000:8000 employee-attrition-ml
```

The multi-stage Python 3.12 slim image, pinned by digest, trains and verifies a bundle in its builder stage. The final stage installs constrained serving dependencies and copies the verified bundle and serving code. It contains no raw training dataset or training CLI and runs as an unprivileged user with a health check. Building requires a running Linux-container Docker engine and dependency-download access. SHAP is optional and omitted from the default image build.

## Versioned experiment bundles

Training writes a new `models/runs/<UUID>/` directory containing `pipeline.joblib`, `metadata.json`, metrics, figures, processed diagnostics, `employee.json`, `background.json`, `reference_profile.json`, and a checksum manifest. The pipeline and report use schema version 2 and the same `run_id`. All files and metadata are verified before atomically replacing `models/current.json`. Failed training or publication leaves the previous active run available.

API startup resolves and loads one run for its process lifetime; restart it after retraining. Each dashboard rerun resolves one active run, and caches that immutable bundle by its UUID. It never combines a model from one run with reports from another. Checksums detect accidental corruption and mixing; they do not make untrusted joblib/pickle safe.

To keep bundles outside the checkout, set `ATTRITION_MODELS_DIR` (or `ATTRITION_ROOT`) for training, the API, the dashboard, and the prune/drift commands alike. Only operators set these variables; requests cannot. See [Paths and wheel installs](development.md#paths-and-wheel-installs).

Each training run adds a bundle. `python -m attrition.models.prune --keep 5` removes all but the five most recently written runs and never removes the active run; add `--dry-run` to list what would be removed. Only UUID-named run folders are considered, and an unverifiable `current.json` stops pruning without deleting anything. A running API keeps serving the bundle it loaded at startup.

The existing `reports/`, `data/processed/`, and `examples/` paths remain reproducible exports. Git reports omit runtime UUIDs. Schema-1 files can still be loaded explicitly with `python -m attrition.models.predict examples/employee.json --model <file>.joblib`. Pickles record module paths, so artifacts trained before the package moved from `src` to `attrition` fail with a clear "retrain" error; run `python -m attrition.models.train` to replace them.

## Manual AutoML promotion

Approved AutoML candidates use a separate [manual promotion and rollback workflow](automl-promotion.md).
Their bundles contain development evidence without candidate holdout scores: `/model` returns
`holdout_metrics: null`, and the dashboard labels the estimates accordingly. Baseline report exports
remain historical baseline evidence. Restart the API after activation or rollback; retain prior bundles
and receipts throughout the rollback period before considering pruning.

## Input drift monitoring

Each bundle stores `reference_profile.json`: training-row deciles for numeric features, category shares, and missing rates. The drift check compares new records with that profile using the Population Stability Index (PSI) per feature:

```bash
python -m attrition.monitoring.drift new_records.csv --output reports/drift.csv
python -m attrition.monitoring.drift batch.json --fail-on-shift   # exit status 1 if any feature shifts
```

Input can be CSV or JSON (one object, a list, or the `{"employees": [...]}` body used by `/predict/batch`) and must pass the same schema validation as prediction. Status follows common PSI conventions: below 0.10 stable, 0.10 to 0.25 moderate, 0.25 or more shift. Unseen categories form their own bin and are reported as `unseen_share`; missing values are excluded from PSI and reported as missing rates. Below 100 records, PSI mostly reflects sampling noise and the command warns. Drift shows that inputs differ from training data; it does not show that predictions are wrong or that attrition rates changed, which requires labeled outcomes.
