# Development

[← Back to README](../README.md)

Setting up the environment, running the checks, and how CI/CD validates and delivers the project.

## Installation: Conda `ibmhratt`

Run commands from the repository root. Python **3.12** is the verified runtime.

```bash
# Only if the environment does not already exist:
conda create -n ibmhratt python=3.12 pip -y
conda activate ibmhratt
python -m pip install -r requirements-dev.txt -c requirements-lock.txt
python -m attrition.models.train
python -m pytest -q
```

Alternatively, `conda env create -f environment.yml` creates the named environment using the same package constraints. `requirements-serving.txt` declares API/dashboard dependencies; `requirements.txt` includes those plus training plots. The Docker runtime installs only serving dependencies; `requirements-dev.txt` adds testing, linting, and notebook tools; the lock file pins the packages used for the recorded results. Use the constraints above when reproducing them. Platforms may install different platform-specific transitive dependencies.

If activation is inconvenient, use `conda run --no-capture-output -n ibmhratt python -m attrition.models.train`. In VS Code, select **Python (ibmhratt)** as the interpreter and notebook kernel. All project paths are derived from `src/attrition/config.py`, without machine-specific absolute paths. `requirements-dev.txt` installs the `attrition` package in editable mode, so `python -m attrition...` commands work after installation and paths default to the checkout. To use data or artifacts elsewhere, see [Paths and wheel installs](#paths-and-wheel-installs).

Optional SHAP and executed notebooks:

```bash
python -m pip install -r requirements-explain.txt -c requirements-lock.txt
python -m attrition.models.train --with-shap
python -m scripts.execute_notebooks
```

The four notebooks cover data understanding, EDA, feature engineering, and model experiments. Training must run first because notebooks display its generated figures and metrics. Reusable logic lives in `src/attrition/`; `scripts/build_notebooks.py` regenerates notebook cell sources and clears their outputs.

## Paths and wheel installs

Without overrides, paths resolve relative to the checkout exactly as before. Trusted operators can relocate them with environment variables; no code needs editing:

| Variable | Default | Used for |
|---|---|---|
| `ATTRITION_ROOT` | the checkout (for an installed wheel: the working directory) | base for the defaults below and for exported reports |
| `ATTRITION_DATA_PATH` | `<root>/data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` | training data |
| `ATTRITION_MODELS_DIR` | `<root>/models` | run bundles and the active `current.json` pointer used by the API, dashboard, prediction, drift, and pruning |

Values are read once at import, so restart services after changing them. Blank values are ignored, `~` expands, relative values resolve against the working directory, and both `/` and `\` separators work on Windows. Resolving a path never creates it: readers report a missing file, and only training writes (and `prune` deletes old runs). Joblib bundles can execute code when loaded, so point `ATTRITION_MODELS_DIR` only at directories you trust. API requests can never choose a model or output path: the request schemas forbid extra fields and no endpoint takes a path.

The training CLI also accepts explicit paths, which take precedence over the variables:

```bash
python -m attrition.models.train --data /srv/hr/attrition.csv --output-root /srv/attrition --models-dir /srv/attrition/bundles
```

`--output-root` receives `reports/`, `data/processed/`, `examples/employee.json`, and the README results block (only if a `README.md` exists there). Bundles go to `--models-dir`; without it, to `<output-root>/models` when `--output-root` is given, otherwise to `ATTRITION_MODELS_DIR` or `<root>/models`.

A regular (non-editable) wheel works outside the checkout:

```bash
python -m pip wheel . --no-deps --wheel-dir dist        # or `python -m build` if the build package is installed
python -m pip install dist/ibmhratt-1.1.0-py3-none-any.whl -c requirements-lock.txt
python -m attrition.models.predict employee.json --model /srv/attrition/bundles/current.json
```

Packaging limits:

- The wheel's dependencies come from `requirements-serving.txt` only, so the library and the `predict`, `prune`, and drift CLIs work out of the box. Training also needs matplotlib (`python -m pip install matplotlib -c requirements-lock.txt`), and `--with-shap` needs `requirements-explain.txt`. The requirement files remain the single source; no extras duplicate them.
- The wheel contains only the `attrition` package. The FastAPI and Streamlit apps (`app/`), raw data, notebooks, and `scripts/` stay in the checkout or container. Set `ATTRITION_MODELS_DIR` before starting the apps to serve external bundles.
- The repository scripts (`scripts.check`, `compare_reports`, `execute_notebooks`, `verify_delivery`, `build_notebooks`) and the test suite are checkout tooling that assume the default layout. Run them without path overrides.
- The package declares no license metadata; licensing is an open owner decision.

`tests/test_paths.py` covers the overrides, and `tests/test_packaging.py` builds the wheel from a minimal source copy, installs it into a separate directory, and runs prediction from outside the checkout.

## Testing and reproducibility

```bash
python -m ruff check .
python -m mypy
python -m pytest -q --cov --cov-report= --cov-fail-under=0
python -m scripts.coverage_report pytest-only
python -m scripts.execute_notebooks
python -m scripts.verify_delivery
```

### Two coverage figures

Coverage answers different questions depending on what ran, so every report is labeled:

| Label | What is measured | Gate | Where |
|---|---|---|---|
| `pytest-only` | Lines executed while the test suite ran. Training, plotting, and SHAP lines count only if a test reaches them. | `PYTEST_ONLY_MIN` in `scripts/coverage_report.py` (currently **84%**) | CI step before training; `make coverage`; `python -m scripts.check` |
| `combined` | The test suite plus a fresh `python -m attrition.models.train --with-shap`. Lines run by the training command are "covered" without any assertion. | `fail_under = 80` in `pyproject.toml` (unchanged) | CI and `python -m scripts.check` after training |

How the pytest-only floor was chosen: before the behavioral tests of PR 07 the suite measured **83%** pytest-only (126 tests); with them it measures **86%** (550 passed, 7 strict `xfail`) on Windows with Python 3.12. The floor is two points below the measurement to absorb platform-specific lines, not a target. Lowering it needs a stated reason; raising it should come with tests that assert behavior, not with smoke tests that only execute lines. The remaining uncovered code is mostly plotting (`visualization/plots.py`), SHAP (`models/explain.py`), data exploration (`data/explore.py`), the training orchestration branches in `models/train.py` (covered by the separate training run), and Streamlit presentation branches; a line count there says little about correctness, so the tests concentrate on observable behavior instead: schema and boundary errors, authentication order, unusable or incompatible artifacts, external paths, and command-line contracts. `reports/coverage-pytest.xml` is a generated, gitignored report.

Known gaps are recorded as strict `xfail` tests rather than hidden: corrupt legacy `.joblib` files raise raw `struct.error`/`KeyError`/`EOFError` instead of a clear `ValueError`, and an active-run pointer that is valid JSON but not an object raises `AttributeError` instead of "Invalid active-run pointer." Both fail safely (nothing is loaded or pruned; the API answers 503). When the source is fixed the strict `xfail` fails and should be converted to a normal test.

The pytest suite includes unit tests and integration tests. Tests cover loading/schema failures against the canonical `data/raw/` CSV, disjoint stratified partitions, train-only imputation statistics, unknown categories/nulls, dictionary/batch prediction, serialization parity, stored threshold use, known confusion counts, API validation and unavailable-model behavior, model-selection and threshold tie-break rules, subgroup denominators, README result injection, deterministic bootstrap intervals, and an automated Streamlit form submission. Behavioral regression tests (PR 07) add inclusive numeric bounds for every field through the API and validators, required/nullable/wrongly typed fields, authentication rejection and ordering for every protected endpoint, safe 503/500 bodies and logs, incompatible schema/threshold/class metadata, mixed or tampered bundles, relocated models directories with spaces and non-ASCII names, line-ending independence, and the `predict`, `prune`, `train`, and drift command-line contracts. Test fixtures train a small real pipeline; tests do not require a pretrained production artifact or manual interaction.

After training and notebook execution, `scripts.verify_delivery` also checks the delivered artifact against its recorded metrics and bootstrap point estimates, OOF coverage, SHAP additivity, executed notebook cells, and actual loopback HTTP startup/prediction for both services. Its temporary servers are stopped automatically.

`make install`, `make train`, `make explain`, `make lint`, `make typecheck`, `make test`, `make coverage`, `make check`, `make api`, `make app`, `make notebooks`, and `make prune` wrap the documented Python commands when Make is available. Activate `ibmhratt` first; PowerShell users can use the Python commands directly.

Optional Git hooks run Ruff, mypy, and basic file checks before each commit, using the activated environment's pinned tools: `python -m pip install pre-commit`, then `pre-commit install`. Generated reports and notebooks are excluded; CI compares them with a fresh run instead.

Model binaries and generated split/OOF CSVs are gitignored because they are reproducible outputs. Source, raw data, executed notebooks, metrics, and figures are intended for Git. `models/.gitkeep` and `data/processed/.gitkeep` preserve their folders. Model metadata records the raw CSV SHA-256, package/Python versions, seed, features, row counts, and threshold. No Git commit or remote deployment is required to run locally.

For the complete local quality gate:

```bash
python -m pip install -r requirements-dev.txt -r requirements-explain.txt -c requirements-lock.txt
python -m scripts.check
```

This runs lint, type checking, all tests, a labeled pytest-only coverage gate (see [Two coverage figures](#two-coverage-figures)), fresh training, combined coverage (minimum 80%), notebook regeneration/execution, and live service verification. Tests alone can report lower coverage because training and plotting are exercised by the separate integration run. Temporary files and the coverage database use a unique directory under `.test-tmp/`; reports and notebooks are regenerated. In a restricted Windows sandbox, pytest's private-directory ACLs may require running this command with the sandbox's approved execution access. No assertions are skipped to work around permissions.

## CI/CD: GitHub Actions and GHCR

[`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml) runs on pull requests, pushes to `main`, version tags matching `v*`, manual dispatch, and weekly on Mondays so newly published advisories surface even without code changes. Scheduled runs never publish.

- **CI:** Python 3.12 on Ubuntu and Windows, constrained dependency installation, `pip check`, Ruff lint, mypy type checking, all pytest tests with coverage, a pytest-only coverage gate evaluated before training, fresh training with SHAP measured in the same coverage database, a combined 80% coverage gate, a check that the committed README and `reports/results.md` match the fresh run, a tolerant comparison of every committed metrics file and figure name (`python -m scripts.compare_reports`), notebook execution, and delivery verification including live API/dashboard startup. Test results and generated model/report artifacts are retained for 14 days.
- **Dependency audit:** `pip-audit` checks every package pinned in `requirements-lock.txt` against known vulnerabilities and fails on any finding.
- **Container validation:** after both CI jobs and the audit pass, build the Docker image, test its real `/health` and `/predict` endpoints, and scan it with Trivy, failing on fixable high or critical vulnerabilities in OS or Python packages. Pull requests, manual runs, and scheduled runs build and test without publishing.
- **CD (continuous delivery):** after container validation passes on a push to `main` or a `v*` tag, publish that same tested image to `ghcr.io/<owner>/<repository>`. Main publishes `latest` and a commit SHA tag; version tags publish the Git tag and a commit SHA tag. This delivers a container image; running it on a server is a separate deployment step.

Publishing uses the workflow's `GITHUB_TOKEN` with job-scoped `packages: write`; no personal access token or registry password secret is required. Enable GitHub Actions in the repository and allow the workflow to create packages. Existing GHCR packages must grant this repository Actions access. Forks validate containers but skip publishing.

Supply-chain pinning: every third-party action is pinned to a full commit SHA (with its version in a comment), the Python base image and the Trivy image are pinned by digest, and `pip-audit` is pinned by version. Dependabot proposes updates to all of them; GitHub Actions updates arrive as one grouped pull request. See [GitHub's Docker publishing documentation](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images).

For this repository, the main-branch image can be run with:

```bash
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/hansdarmawann/ibmhratt:latest
```

GHCR packages are private by default. Authenticate with `docker login ghcr.io` to pull a private package, or explicitly make the educational demo package public in its GitHub package settings.
