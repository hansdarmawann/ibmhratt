"""The manual gate binds reviewed artifacts, isolates training data, and permits exact rollback."""

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from app.api import create_app
from attrition.config import CATEGORIES, CONSTANT_COLUMNS, ID_COLUMN, NUMERIC_BOUNDS, ROOT, TARGET
from attrition.data.load_data import load_data, split_data
from attrition.features.preprocess import build_pipeline
from attrition.models import automl, automl_evaluation, promotion
from attrition.models.artifacts import load_bundle, read_json, sha256, write_json


@pytest.fixture(scope="module")
def source_files(tmp_path_factory):
    pytest.importorskip("optuna")
    directory = tmp_path_factory.mktemp("promotion-evidence")
    count, rng = 120, np.random.default_rng(73)
    frame = pd.DataFrame({name: rng.integers(lower, (upper if upper is not None else lower + 20) + 1, count)
                          for name, (lower, upper) in NUMERIC_BOUNDS.items()})
    for name, categories in CATEGORIES.items():
        frame[name] = rng.choice(categories, count)
    for name, value in CONSTANT_COLUMNS.items():
        frame[name] = value
    frame[ID_COLUMN] = np.arange(1, count + 1)
    frame[TARGET] = np.where(np.arange(count) % 3 == 0, "Yes", "No")
    data = directory / "synthetic.csv"
    frame.to_csv(data, index=False)
    X, y = automl.training_partition(load_data(data))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(automl, "FAMILIES", ("logistic_regression",))
        patch.setattr(automl_evaluation, "candidate_pipelines", lambda X: {
            "logistic_regression": build_pipeline(X, LogisticRegression(max_iter=1000, random_state=42))})
        search = automl.search(X, y, n_trials=2, timeout=120, seed=7, folds=5)
        evaluation = automl_evaluation.nested_evaluation(X, y, n_trials=2, timeout=120, seed=7, outer_folds=2)
    for name, result in (("search", search), ("evaluation", evaluation)):
        result["data_sha256"] = sha256(data)
        write_json(result, directory / f"{name}.json")
    return directory


@pytest.fixture
def sources(source_files, tmp_path):
    for name in ("search.json", "evaluation.json", "synthetic.csv"):
        shutil.copyfile(source_files / name, tmp_path / name)
    return tmp_path


def prepare(sources, **kwargs):
    return promotion.prepare(sources / "search.json", sources / "evaluation.json", trial_number=0,
                             data_path=sources / "synthetic.csv", models_dir=sources / "models", **kwargs)


@pytest.fixture
def prepared(sources, bundle_factory):
    baseline = bundle_factory()
    return baseline, prepare(sources)


def approved(candidate, sources):
    review = promotion.review(candidate.run_id, sources / "models")
    review.update(decision="approve", reviewer="Synthetic test reviewer", reason="Synthetic gate test",
                  reviewed_at=datetime.now(UTC).isoformat(), checks=dict.fromkeys(promotion.CHECKS, True),
                  evidence=dict.fromkeys(promotion.EVIDENCE, "Synthetic test evidence"))
    path = sources / "approval.json"
    write_json(review, path)
    return path


def snapshot(path):
    return {file.relative_to(path).as_posix(): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def test_prepare_never_activates_or_overwrites_baseline(sources, bundle_factory):
    baseline = bundle_factory()
    prior = snapshot(baseline.path)
    pointer = sources / "models/current.json"
    before = pointer.read_bytes()
    canonical = {name: snapshot(ROOT / name) for name in ("reports", "data", "examples")}
    candidate = prepare(sources)
    assert pointer.read_bytes() == before
    assert snapshot(baseline.path) == prior
    assert {name: snapshot(ROOT / name) for name in canonical} == canonical
    assert candidate.report["selected_model_metrics"] is None
    assert candidate.report["promotion"]["holdout_used"] is False
    assert candidate.report["promotion"]["selected_trial"] == 0
    assert candidate.report["promotion"]["search_sha256"] == sha256(sources / "search.json")
    assert load_bundle(candidate.path).run_id == candidate.run_id
    second = prepare(sources)
    assert second.run_id != candidate.run_id
    assert second.report["decision_threshold"] == candidate.report["decision_threshold"]
    np.testing.assert_array_equal(second.pipeline.predict_proba(load_data(sources / "synthetic.csv")),
                                  candidate.pipeline.predict_proba(load_data(sources / "synthetic.csv")))


def test_all_fits_and_threshold_selection_receive_only_training_rows(sources, monkeypatch):
    X, holdout, y, _ = split_data(load_data(sources / "synthetic.csv"))
    fit_rows, threshold_rows = [], []
    original_fit, original_table = Pipeline.fit, promotion.threshold_table

    def fit(self, frame, labels=None, **kwargs):
        if isinstance(frame, pd.DataFrame) and ID_COLUMN in frame:
            fit_rows.append(set(frame.index))
            assert set(frame.index) <= set(X.index)
            assert set(frame.index).isdisjoint(holdout.index)
            np.testing.assert_array_equal(labels, y.loc[frame.index])
        return original_fit(self, frame, labels, **kwargs)

    def table(labels, scores):
        pd.testing.assert_series_equal(labels, y)
        threshold_rows.append(set(labels.index))
        return original_table(labels, scores)

    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must not search again or evaluate the holdout")

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(promotion, "threshold_table", table)
    monkeypatch.setattr(automl, "search", forbidden)
    monkeypatch.setattr("attrition.models.train.evaluate_holdout", forbidden)
    prepare(sources)
    assert len(fit_rows) == 6  # Five OOF fits plus one final training-only fit.
    assert all(len(rows) < len(X) for rows in fit_rows[:-1])
    assert fit_rows[-1] == set(X.index) and threshold_rows == [set(X.index)]


def test_explicit_nonwinning_trial_is_used(sources):
    search = read_json(sources / "search.json")
    number = next(trial["number"] for trial in search["trial_records"] if trial["number"] != search["best"]["trial"])
    candidate = promotion.prepare(sources / "search.json", sources / "evaluation.json", trial_number=number,
                                  data_path=sources / "synthetic.csv", models_dir=sources / "models")
    assert candidate.report["promotion"]["selected_trial"] == number
    assert candidate.pipeline.named_steps["model"].get_params() == search["trial_records"][number]["estimator_params"]


@pytest.mark.parametrize(("filename", "key", "value"), [
    ("search", "holdout_used", True), ("evaluation", "holdout_used", True),
    ("search", "served_model_changed", True), ("evaluation", "status", "incomplete"),
    ("search", "schema_version", 1), ("search", "data_sha256", "wrong"),
    ("evaluation", "training_data", {}), ("search", "package_versions", {}),
    ("evaluation", "provenance", {}), ("evaluation", "folds", []),
    ("search", "trial_records", []), ("evaluation", "inner_cv", {}),
    ("evaluation", "automl", {}),
])
def test_invalid_evidence_refused_before_fitting(sources, monkeypatch, filename, key, value):
    path = sources / f"{filename}.json"
    result = read_json(path)
    result[key] = value
    write_json(result, path)
    monkeypatch.setattr(Pipeline, "fit", lambda *a, **kw: pytest.fail("Invalid evidence reached fitting"))
    with pytest.raises(ValueError):
        prepare(sources)
    assert not (sources / "models").exists()


def test_estimator_drift_refused(sources):
    path = sources / "search.json"
    result = read_json(path)
    result["trial_records"][0]["estimator_params"]["C"] = 999
    write_json(result, path)
    with pytest.raises(ValueError, match="Reconstructed"):
        prepare(sources)


def test_missing_development_metric_refused(sources):
    path = sources / "evaluation.json"
    result = read_json(path)
    del result["procedures"]["automl"]["pr_auc"]
    write_json(result, path)
    with pytest.raises(ValueError, match="finite AP"):
        prepare(sources)


def test_json_must_be_a_finite_object(tmp_path):
    path = tmp_path / "input.json"
    for value in ('[]', '{"score": NaN}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            promotion.document(path)


def test_pending_review_is_not_approval(prepared, sources):
    baseline, candidate = prepared
    review = promotion.review(candidate.run_id, sources / "models")
    assert review["decision"] == "pending" and not any(review["checks"].values())
    assert review["expected_current"]["run_id"] == baseline.run_id
    write_json(review, sources / "draft.json")
    with pytest.raises(ValueError, match="Explicit manual"):
        promotion.activate(sources / "draft.json", sources / "models")
    assert load_bundle(sources / "models/current.json").run_id == baseline.run_id


@pytest.mark.parametrize("check", promotion.CHECKS)
def test_every_manual_check_is_required(prepared, sources, check):
    baseline, candidate = prepared
    path = approved(candidate, sources)
    review = read_json(path)
    review["checks"][check] = False
    write_json(review, path)
    with pytest.raises(ValueError, match="Every manual"):
        promotion.activate(path, sources / "models")
    assert load_bundle(sources / "models/current.json").run_id == baseline.run_id


@pytest.mark.parametrize(("field", "value"), [("manifest_sha256", "changed"), ("reviewer", ""),
                                               ("reason", " "), ("evidence", {}),
                                               ("reviewed_at", "2026-10-13T10:00:00")])
def test_approval_requires_bound_identity_and_review_evidence(prepared, sources, field, value):
    baseline, candidate = prepared
    path = approved(candidate, sources)
    review = read_json(path)
    review[field] = value
    write_json(review, path)
    with pytest.raises(ValueError):
        promotion.activate(path, sources / "models")
    assert load_bundle(sources / "models/current.json").run_id == baseline.run_id


def test_stale_approval_never_replaces_newer_run(prepared, sources, bundle_factory):
    _, candidate = prepared
    approval = approved(candidate, sources)
    newer = bundle_factory()
    with pytest.raises(ValueError, match="Current pointer changed"):
        promotion.activate(approval, sources / "models")
    assert load_bundle(sources / "models/current.json").run_id == newer.run_id


def test_activate_and_exact_rollback_keep_prior_bundles(prepared, sources):
    baseline, candidate = prepared
    models = sources / "models"
    before, prior_bundles = (models / "current.json").read_bytes(), snapshot(models / "runs")
    receipt_path = promotion.activate(approved(candidate, sources), models)
    receipt = read_json(receipt_path)
    assert receipt["before"]["run_id"] == baseline.run_id
    assert receipt["after"]["run_id"] == candidate.run_id
    assert receipt["authorization"]["review"]["reviewer"] == "Synthetic test reviewer"
    assert load_bundle(models / "current.json").run_id == candidate.run_id
    promotion.rollback(receipt_path.stem, expected_current=candidate.run_id, reviewer="Owner", reason="Smoke rollback",
                       models_dir=models)
    assert (models / "current.json").read_bytes() == before
    assert snapshot(models / "runs") == prior_bundles
    assert len(list((models / "promotions").glob("*.json"))) == 2


def test_corruption_never_replaces_active_run(prepared, sources):
    baseline, candidate = prepared
    path = approved(candidate, sources)
    (candidate.path / "pipeline.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        promotion.activate(path, sources / "models")
    assert load_bundle(sources / "models/current.json").run_id == baseline.run_id


def test_failed_pointer_replace_keeps_previous_run_and_recovery_receipt(prepared, sources, monkeypatch):
    baseline, candidate = prepared
    approval = approved(candidate, sources)
    models = sources / "models"

    def fail(*args):
        raise OSError("simulated pointer write failure")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        promotion.activate(approval, models)
    assert load_bundle(models / "current.json").run_id == baseline.run_id
    assert len(list((models / "promotions").glob("*.json"))) == 1
    assert not (models / ".promotion.lock").exists()
    assert not list(models.glob(".current-*.tmp"))


def test_concurrent_promotion_refused(prepared, sources):
    baseline, candidate = prepared
    lock = sources / "models/.promotion.lock"
    lock.write_text("another operator")
    with pytest.raises(FileExistsError):
        promotion.activate(approved(candidate, sources), sources / "models")
    assert lock.read_text() == "another operator"
    assert load_bundle(sources / "models/current.json").run_id == baseline.run_id


def test_rollback_refuses_stale_pointer_and_corrupt_previous_bundle(prepared, sources, bundle_factory):
    baseline, candidate = prepared
    models = sources / "models"
    receipt = promotion.activate(approved(candidate, sources), models)
    with pytest.raises(ValueError, match="expected-current"):
        promotion.rollback(receipt.stem, expected_current=baseline.run_id, reviewer="Owner", reason="Test",
                           models_dir=models)
    (baseline.path / "employee.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        promotion.rollback(receipt.stem, expected_current=candidate.run_id, reviewer="Owner", reason="Test",
                           models_dir=models)
    # A separate activation receipt provides a healthy rollback target for the stale-pointer case.
    newer = bundle_factory()
    receipt = promotion.activate(approved(candidate, sources), models)
    latest = bundle_factory()
    with pytest.raises(ValueError, match="Current pointer changed"):
        promotion.rollback(receipt.stem, expected_current=candidate.run_id, reviewer="Owner", reason="Test",
                           models_dir=models)
    assert load_bundle(models / "current.json").run_id == latest.run_id
    assert load_bundle(newer.path).run_id == newer.run_id


@pytest.mark.parametrize("name", ["../outside", "C:\\outside", "not-a-uuid", str(uuid4()).upper()])
def test_run_ids_cannot_escape_models_directory(tmp_path, name):
    with pytest.raises(ValueError):
        promotion.review(name, tmp_path)


@pytest.mark.parametrize("name", ["reports/metrics", "data", "examples"])
def test_models_cannot_be_canonical_data_or_report_paths(name):
    with pytest.raises(ValueError, match="canonical"):
        promotion.models_root(ROOT / name)


def test_models_directory_cannot_mutate_an_existing_bundle(prepared):
    baseline, _ = prepared
    before = snapshot(baseline.path)
    with pytest.raises(ValueError, match="inside an existing bundle"):
        promotion.models_root(baseline.path / "nested-models")
    assert snapshot(baseline.path) == before


def test_candidate_serving_needs_no_optuna(prepared, monkeypatch):
    _, candidate = prepared
    monkeypatch.setitem(sys.modules, "optuna", None)
    with TestClient(create_app(candidate.path)) as client:
        result = client.post("/predict", json=read_json(candidate.path / "employee.json"))
        assert result.status_code == 200 and result.json()["run_id"] == candidate.run_id


def test_api_and_dashboard_candidate_smoke_then_rollback(prepared, sources, monkeypatch):
    baseline, candidate = prepared
    models, pointer = sources / "models", sources / "models/current.json"
    employee = read_json(candidate.path / "employee.json")
    receipt = promotion.activate(approved(candidate, sources), models)
    with TestClient(create_app(pointer, api_key="smoke-key")) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json=employee).status_code == 401
        headers = {"X-API-Key": "smoke-key"}
        info_response = client.get("/model", headers=headers)
        assert info_response.status_code == 200
        info = info_response.json()
        assert info["run_id"] == candidate.run_id and info["holdout_metrics"] is None
        output = client.post("/predict", json=employee, headers=headers)
        assert output.status_code == 200 and output.json()["run_id"] == candidate.run_id
        assert output.json()["threshold"] == candidate.report["decision_threshold"]
        assert 0 <= output.json()["attrition_probability"] <= 1
        batch = client.post("/predict/batch", json={"employees": [employee, employee]}, headers=headers)
        assert batch.status_code == 200 and batch.json()["count"] == 2
        monkeypatch.setattr("attrition.config.CURRENT_RUN", pointer)
        monkeypatch.setattr("attrition.models.local_explain.explain_profile", lambda *args: {"status": "unavailable"})
        app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
        assert not app.exception
        assert any("no final holdout metrics" in info.value for info in app.info)
        assert any("Nested CV AP mean" in table.value.columns for table in app.dataframe)
        assert not any("Final holdout at threshold" in caption.value for caption in app.caption)
        app.button[0].click().run()
        assert not app.exception and any(m.label == "Attrition probability" for m in app.metric)
        promotion.rollback(receipt.stem, expected_current=candidate.run_id, reviewer="Owner", reason="Smoke complete",
                           models_dir=models)
        # API deliberately keeps the once-loaded run until restart; dashboard resolves on rerun.
        assert client.get("/model", headers=headers).json()["run_id"] == candidate.run_id
        app.run()
        assert not app.exception and any(baseline.run_id in caption.value for caption in app.caption)
    with TestClient(create_app(pointer)) as restarted:
        info = restarted.get("/model").json()
        assert info["run_id"] == baseline.run_id and info["holdout_metrics"] is not None


def test_cli_prepare_review_activate_rollback(sources, bundle_factory, capsys):
    baseline = bundle_factory()
    common = ["--models-dir", str(sources / "models")]
    promotion.main([*common, "prepare", "--search", str(sources / "search.json"),
                    "--evaluation", str(sources / "evaluation.json"), "--trial", "0",
                    "--data", str(sources / "synthetic.csv")])
    result = json.loads(capsys.readouterr().out)
    assert result["activated"] is False
    promotion.main([*common, "review", "--run-id", result["run_id"]])
    assert json.loads(capsys.readouterr().out)["decision"] == "pending"
    candidate = load_bundle(result["path"])
    promotion.main([*common, "activate", "--approval", str(approved(candidate, sources))])
    receipt = Path(json.loads(capsys.readouterr().out)["receipt"])
    promotion.main([*common, "rollback", "--promotion-id", receipt.stem, "--expected-current", candidate.run_id,
                    "--reviewer", "Owner", "--reason", "CLI test"])
    assert read_json(Path(json.loads(capsys.readouterr().out)["receipt"]))["action"] == "rollback"
    assert load_bundle(sources / "models/current.json").run_id == baseline.run_id


def test_promotion_import_and_help_do_not_load_optuna_or_publish():
    code = "import sys; import attrition.models.promotion; print('optuna' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60)
    assert result.stdout.strip() == "False"
    result = subprocess.run([sys.executable, "-m", "attrition.models.promotion", "--help"],
                            capture_output=True, text=True, check=True, timeout=60)
    assert "prepare" in result.stdout and "rollback" in result.stdout
