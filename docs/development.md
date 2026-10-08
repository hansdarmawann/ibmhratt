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

If activation is inconvenient, use `conda run --no-capture-output -n ibmhratt python -m attrition.models.train`. In VS Code, select **Python (ibmhratt)** as the interpreter and notebook kernel. All project paths are derived from `src/attrition/config.py`, without machine-specific absolute paths. `requirements-dev.txt` installs the `attrition` package in editable mode, so `python -m attrition...` commands work after installation; paths resolve relative to the checkout, so use an editable install or `PYTHONPATH=src` rather than a regular `pip install .`.

Optional SHAP and executed notebooks:

```bash
python -m pip install -r requirements-explain.txt -c requirements-lock.txt
python -m attrition.models.train --with-shap
python -m scripts.execute_notebooks
```

The four notebooks cover data understanding, EDA, feature engineering, and model experiments. Training must run first because notebooks display its generated figures and metrics. Reusable logic lives in `src/attrition/`; `scripts/build_notebooks.py` regenerates notebook cell sources and clears their outputs.

## Testing and reproducibility

```bash
python -m ruff check .
python -m mypy
python -m pytest -q --cov --cov-report=term --cov-fail-under=0
python -m scripts.execute_notebooks
python -m scripts.verify_delivery
```

The pytest suite includes unit tests and integration tests. Tests cover loading/schema failures against the canonical `data/raw/` CSV, disjoint stratified partitions, train-only imputation statistics, unknown categories/nulls, dictionary/batch prediction, serialization parity, stored threshold use, known confusion counts, API validation and unavailable-model behavior, model-selection and threshold tie-break rules, subgroup denominators, README result injection, deterministic bootstrap intervals, and an automated Streamlit form submission. Test fixtures train a small real pipeline; tests do not require a pretrained production artifact or manual interaction.

After training and notebook execution, `scripts.verify_delivery` also checks the delivered artifact against its recorded metrics and bootstrap point estimates, OOF coverage, SHAP additivity, executed notebook cells, and actual loopback HTTP startup/prediction for both services. Its temporary servers are stopped automatically.

`make install`, `make train`, `make explain`, `make lint`, `make typecheck`, `make test`, `make coverage`, `make check`, `make api`, `make app`, `make notebooks`, and `make prune` wrap the documented Python commands when Make is available. Activate `ibmhratt` first; PowerShell users can use the Python commands directly.

Optional Git hooks run Ruff, mypy, and basic file checks before each commit, using the activated environment's pinned tools: `python -m pip install pre-commit`, then `pre-commit install`. Generated reports and notebooks are excluded; CI compares them with a fresh run instead.

Model binaries and generated split/OOF CSVs are gitignored because they are reproducible outputs. Source, raw data, executed notebooks, metrics, and figures are intended for Git. `models/.gitkeep` and `data/processed/.gitkeep` preserve their folders. Model metadata records the raw CSV SHA-256, package/Python versions, seed, features, row counts, and threshold. No Git commit or remote deployment is required to run locally.

For the complete local quality gate:

```bash
python -m pip install -r requirements-dev.txt -r requirements-explain.txt -c requirements-lock.txt
python -m scripts.check
```

This runs lint, type checking, all tests, fresh training, combined coverage (minimum 80%), notebook regeneration/execution, and live service verification. Tests alone can report lower coverage because training and plotting are exercised by the separate integration run. Temporary files and the coverage database use a unique directory under `.test-tmp/`; reports and notebooks are regenerated. In a restricted Windows sandbox, pytest's private-directory ACLs may require running this command with the sandbox's approved execution access. No assertions are skipped to work around permissions.

## CI/CD: GitHub Actions and GHCR

[`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml) runs on pull requests, pushes to `main`, version tags matching `v*`, manual dispatch, and weekly on Mondays so newly published advisories surface even without code changes. Scheduled runs never publish.

- **CI:** Python 3.12 on Ubuntu and Windows, constrained dependency installation, `pip check`, Ruff lint, mypy type checking, all pytest tests with coverage, fresh training with SHAP measured in the same coverage database, a combined 80% coverage gate, a check that the committed README and `reports/results.md` match the fresh run, a tolerant comparison of every committed metrics file and figure name (`python -m scripts.compare_reports`), notebook execution, and delivery verification including live API/dashboard startup. Test results and generated model/report artifacts are retained for 14 days.
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
