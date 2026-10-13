# Contributing

Thanks for helping improve this educational project. Read the [responsible-use notice](README.md) first: the project must not be turned into a tool for hiring, firing, promotion, discipline, or other decisions about individual employees.

## Ground rules

- **No real personnel data.** Use only the bundled fictional dataset or synthetic data. Never put real employee records, credentials, or personal data in code, tests, fixtures, issues, or pull requests.
- **No high-impact use cases.** Do not add features that score, rank, or flag real individuals for employment decisions.
- Keep changes small and scoped. One topic per pull request; avoid unrelated refactors.
- Do not weaken existing security checks, loosen CI, or remove tests to make a change pass.

## Development setup

Python **3.12** is the verified runtime. From the repository root:

```bash
conda create -n ibmhratt python=3.12 pip -y   # skip if the environment exists
conda activate ibmhratt
python -m pip install -r requirements-dev.txt -c requirements-lock.txt
python -m attrition.models.train              # generates the model bundle that tests, API and dashboard use
```

Details, optional SHAP, notebooks, Docker and CI are in [docs/development.md](docs/development.md).

## Checks to run before opening a pull request

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
python -m scripts.check      # full gate: lint, types, tests, fresh training, 80% combined coverage, notebooks, live services
```

`make lint`, `make typecheck`, `make test`, and `make check` wrap the same commands. Full training and `scripts.check` regenerate files under `reports/` and `notebooks/`; run `git status` afterwards and keep regenerated outputs out of your pull request unless the change is meant to alter them. CI checks that committed reports match a fresh run, so never hand-edit generated sections of `README.md` or `reports/`.

## Machine-learning evaluation rules

Pull requests that touch data, features, models, or evaluation must keep these boundaries:

1. The stratified holdout is reserved before exploration. **Never use holdout data, labels, or metrics to select models, hyperparameters, features, or the decision threshold.** The published holdout has already been examined, so it is not a fresh benchmark for repeated experiments.
2. Every learned transform (imputation, scaling, encoding, feature selection) is fitted inside training folds only, through a `Pipeline`.
3. Model choice, tuning, and threshold choice use training data only (cross-validation or out-of-fold predictions).
4. Distinguish ordinary CV, nested CV, out-of-fold threshold diagnostics, and the final held-out evaluation in code and prose. Report uncertainty; do not present small differences on this small dataset as evidence of generalization.
5. Do not change the served model, `models/current.json`, the public inference contract, or committed baseline reports without an explicit, reviewed reason.
6. Add or update tests for behavior you change, with deterministic seeds.

## Pull requests

- Branch from the latest `main` and name the branch by type, for example `docs/...`, `fix/...`, `feat/...`, `test/...`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages, for example `docs: add contribution and security guidance`.
- Fill in the pull request template, including the leakage checklist, and list the commands you actually ran with their results.
- Update documentation alongside behavior changes, and check that links and commands in it still work.

## Security and licensing

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md), not in public issues. The repository does not yet declare a source-code license or dataset redistribution terms; see [Licensing status](README.md#licensing-status).
