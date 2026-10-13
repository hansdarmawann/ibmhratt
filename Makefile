PYTHON ?= python

.PHONY: install train explain lint typecheck test coverage check api app notebooks prune automl automl-evaluate flaml-benchmark
install:
	$(PYTHON) -m pip install -r requirements-dev.txt -c requirements-lock.txt
train:
	$(PYTHON) -m attrition.models.train
explain:
	$(PYTHON) -m pip install -r requirements-explain.txt -c requirements-lock.txt
	$(PYTHON) -m attrition.models.train --with-shap
lint:
	$(PYTHON) -m ruff check .
typecheck:
	$(PYTHON) -m mypy
test:
	$(PYTHON) -m pytest -q
# Pytest-only coverage: the training run is not included (`make check` runs the combined gate).
coverage:
	$(PYTHON) -m pytest -q --cov --cov-report= --cov-fail-under=0
	$(PYTHON) -m scripts.coverage_report pytest-only
check:
	$(PYTHON) -m scripts.check
api:
	$(PYTHON) -m uvicorn app.api:app --host 127.0.0.1 --port 8000
app:
	$(PYTHON) -m streamlit run app/streamlit_app.py
notebooks:
	$(PYTHON) -m scripts.execute_notebooks
prune:
	$(PYTHON) -m attrition.models.prune --keep 5
# Opt-in manual search; never part of train, test, or check.
automl:
	$(PYTHON) -m pip install -r requirements-automl.txt -c requirements-lock.txt
	$(PYTHON) -m attrition.models.automl --n-trials 30 --timeout 900 --seed 42
# Opt-in nested evaluation of that search against the served procedure; about five searches long.
automl-evaluate:
	$(PYTHON) -m pip install -r requirements-automl.txt -c requirements-lock.txt
	$(PYTHON) -m attrition.models.automl_evaluation --n-trials 30 --timeout 900 --seed 42
# Opt-in FLAML, Optuna, and baseline procedures on the same nested folds; about ten searches long.
flaml-benchmark:
	$(PYTHON) -m pip install -r requirements-flaml.txt -c requirements-lock.txt
	$(PYTHON) -m attrition.models.flaml_benchmark --n-trials 30 --timeout 900 --seed 42
