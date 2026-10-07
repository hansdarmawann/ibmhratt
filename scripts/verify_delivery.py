"""Verify the trained artifact, executed notebooks, and real local HTTP services."""

import hashlib
import json
import logging
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import nbformat
import numpy as np
import pandas as pd

from src.config import DATA_PATH, METRICS_DIR, ROOT
from src.data.load_data import load_data, split_data
from src.models.evaluate import evaluate_probabilities
from src.models.predict import load_pipeline, predict

LOGGER = logging.getLogger(__name__)


@contextmanager
def local_service(arguments: list[str], health_path: str):
    """Start a temporary loopback-only service and always terminate its process."""
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    command = [part.replace("{port}", str(port)) for part in arguments]
    log_path = ROOT / "reports/service-smoke.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-m", *command], cwd=ROOT,
            stdout=log_file, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(f"Service exited; inspect {log_path.name}.")
                try:
                    with urllib.request.urlopen(base_url + health_path, timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                raise TimeoutError(f"Service did not become ready; inspect {log_path.name}.")
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> None:
    """Check actual generated outputs without tuning or modifying the model."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = json.loads((METRICS_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    pipeline = load_pipeline()
    assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() == report["data_sha256"]
    assert pipeline.attrition_metadata_["decision_threshold"] == report["selected_threshold"]
    X_train, X_test, _, y_test = split_data(load_data())
    actual = evaluate_probabilities(y_test, pipeline.predict_proba(X_test)[:, 1],
                                    report["selected_threshold"])
    for name in ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier_score"]:
        np.testing.assert_allclose(actual[name], report["selected_model_metrics"][name])
    assert actual["confusion_matrix"] == report["selected_model_metrics"]["confusion_matrix"]
    oof = pd.read_csv(ROOT / "data/processed/oof_predictions.csv")
    assert set(oof.row_index) == set(X_train.index)
    assert not oof.row_index.duplicated().any()
    if report["shap"]["status"] == "generated":
        contributions = pd.read_csv(METRICS_DIR / "shap_individual.csv").shap_value.sum()
        log_odds = contributions + report["shap"]["individual_base_value"]
        np.testing.assert_allclose(1 / (1 + np.exp(-log_odds)),
                                   report["shap"]["individual_model_probability"])
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 4
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None, f"Unexecuted cell in {path.name}"
                assert all(output.output_type != "error" for output in cell.outputs)
    LOGGER.info("Saved pipeline metrics, OOF coverage, SHAP additivity, and four notebooks verified")
    example = json.loads((ROOT / "examples/employee.json").read_text(encoding="utf-8"))
    with local_service(["uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "{port}"],
                       "/health") as base_url:
        request = urllib.request.Request(base_url + "/predict", data=json.dumps(example).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=10) as response:
            prediction = json.load(response)
        expected = predict(example, pipeline=pipeline)
        np.testing.assert_allclose(prediction["attrition_probability"], expected["attrition_probability"])
        assert prediction["prediction"] == expected["predicted_class"]
        assert prediction["threshold"] == expected["decision_threshold"]
    LOGGER.info("Live FastAPI /health and /predict verified against saved pipeline")
    with local_service(["streamlit", "run", "app/streamlit_app.py", "--server.address=127.0.0.1",
                        "--server.port={port}", "--server.headless=true", "--browser.gatherUsageStats=false"],
                       "/_stcore/health") as base_url:
        with urllib.request.urlopen(base_url, timeout=10) as response:
            assert response.status == 200
    LOGGER.info("Live Streamlit HTTP startup verified; services stopped")


if __name__ == "__main__":
    main()
