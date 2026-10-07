PYTHON ?= python

.PHONY: install train explain test api app notebooks
install:
	$(PYTHON) -m pip install -r requirements.txt -c requirements-lock.txt
train:
	$(PYTHON) -m src.models.train
explain:
	$(PYTHON) -m pip install -r requirements-explain.txt -c requirements-lock.txt
	$(PYTHON) -m src.models.train --with-shap
test:
	$(PYTHON) -m pytest -q
api:
	$(PYTHON) -m uvicorn app.api:app --host 127.0.0.1 --port 8000
app:
	$(PYTHON) -m streamlit run app/streamlit_app.py
notebooks:
	$(PYTHON) -m scripts.execute_notebooks
