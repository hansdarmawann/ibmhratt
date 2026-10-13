"""Experiment evidence is complete, repeatable, and isolated from baseline reports and serving."""

import csv
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from attrition.config import ROOT
from attrition.models import automl, automl_evaluation
from attrition.models import automl_reporting as reporting


@pytest.fixture
def optuna():
    return pytest.importorskip("optuna")


@pytest.fixture
def training_rows(partitions, monkeypatch):
    X = partitions[0].head(90).copy()
    y = pd.Series((np.arange(len(X)) % 3 == 0).astype(int), index=X.index)
    monkeypatch.setattr(automl, "FAMILIES", ("logistic_regression",))
    return X, y


def test_metadata_schema_and_reconstructing_the_best_trial(optuna, training_rows):
    X, y = training_rows
    result = automl.search(X, y, n_trials=2, timeout=120, seed=7, folds=3)
    assert result["schema_version"] == 2 and result["status"] == "complete"
    assert result["objective"] == "average_precision"
    assert len(result["trial_records"]) == 2
    for number, trial in enumerate(result["trial_records"]):
        assert trial["number"] == number and trial["state"] == "COMPLETE"
        assert trial["model"] == trial["sampled_params"]["model"] == "logistic_regression"
        assert len(trial["cv_ap"]["folds"]) == 3
        assert trial["value"] == trial["cv_ap"]["mean"]
        assert trial["failure"] is None and trial["elapsed_seconds"] >= 0
        assert trial["estimator_params"]["random_state"] == 7
    best = result["best"]
    pipeline = automl.candidate_pipeline(X, best["model"], best["params"], result["sampler"]["seed"])
    assert pipeline.named_steps["model"].get_params(deep=False) == best["estimator_params"]
    assert best["estimator_class"].endswith(".LogisticRegression")
    assert automl.cv_scores(pipeline, X, y, StratifiedKFold(3, shuffle=True, random_state=7)) == best["cv_ap"]["folds"]
    assert result["provenance"]["package_versions"] == result["package_versions"]
    assert all(result["package_versions"][name] for name in ("optuna", "scikit-learn", "numpy", "pandas"))
    assert result["provenance"]["python"].startswith("3.")
    assert len(result["provenance"]["git_revision"]) == 40
    assert isinstance(result["provenance"]["git_dirty"], bool)
    assert result["training_data"]["columns"] == [
        {"name": str(name), "dtype": str(dtype)} for name, dtype in X.dtypes.items()]
    json.dumps(result, allow_nan=False)


def test_training_fingerprint_changes_with_labels_order_and_schema(training_rows):
    X, y = training_rows
    expected = reporting.training_metadata(X, y)
    assert expected == reporting.training_metadata(X.copy(), y.copy())
    assert len(expected["sha256"]) == 64
    assert expected != reporting.training_metadata(X.iloc[::-1], y.iloc[::-1])
    assert expected != reporting.training_metadata(X, 1 - y)
    assert expected != reporting.training_metadata(X[X.columns[::-1]], y)


@pytest.mark.parametrize("all_fail", [False, True])
def test_trial_failure_is_recorded_and_never_selected(optuna, training_rows, monkeypatch, all_fail):
    X, y = training_rows
    real = automl.cv_scores
    calls = []

    def scores(pipeline, *args):
        is_dummy = type(pipeline.named_steps["model"]).__name__ == "DummyClassifier"
        calls.append(is_dummy)
        if not is_dummy and (all_fail or len(calls) == 1):
            raise ValueError("synthetic estimator failure")
        return real(pipeline, *args)

    monkeypatch.setattr(automl, "cv_scores", scores)
    result = automl.search(X, y, n_trials=2, timeout=120, seed=7, folds=3)
    assert result["trials"] == {"total": 2, "completed": 0 if all_fail else 1}
    failed = result["trial_records"][0]
    assert failed["state"] == "FAIL" and failed["value"] is None and failed["cv_ap"] is None
    assert failed["failure"] == {"type": "ValueError", "message": "synthetic estimator failure"}
    assert failed["elapsed_seconds"] >= 0 and failed["sampled_params"]["model"] == "logistic_regression"
    assert (result["best"] is None) is all_fail
    if not all_fail:
        assert result["best"]["trial"] == 1
    files = reporting.report_files(result)
    trials = list(csv.DictReader(io.StringIO(files["trials.csv"])))
    assert json.loads(trials[0]["failure"]) == failed["failure"]
    assert json.loads(files["result.json"])["best"] == result["best"]
    assert "no baseline procedure comparison" in files["report.md"]


def test_unexpected_trial_error_still_aborts(optuna, training_rows, monkeypatch):
    X, y = training_rows

    def broken(*args):
        raise TypeError("programming error")

    monkeypatch.setattr(automl, "cv_scores", broken)
    with pytest.raises(TypeError, match="programming error"):
        automl.search(X, y, n_trials=1, timeout=120, seed=7, folds=3)


def test_provenance_without_git_or_optional_package(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(reporting.subprocess, "run", missing)
    real = reporting.importlib.metadata.version

    def version(name):
        if name == "optuna":
            raise reporting.importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(reporting.importlib.metadata, "version", version)
    result = reporting.provenance()
    assert result["git_revision"] is None and result["git_dirty"] is None
    assert result["package_versions"]["optuna"] is None
    assert result["package_versions"]["numpy"]


@pytest.mark.parametrize("exception", [subprocess.CalledProcessError(1, "git"), subprocess.TimeoutExpired("git", 5)])
def test_provenance_handles_git_failure(monkeypatch, exception):
    def failure(*args, **kwargs):
        raise exception
    monkeypatch.setattr(reporting.subprocess, "run", failure)
    assert reporting.provenance()["git_revision"] is None


def comparison_result():
    return {"searches": [
        {"outer_fold": fold, "result": {"trial_records": [{"number": 0, "state": "COMPLETE", "model": "model",
                                                            "sampled_params": {"b": 2, "a": 1}}]}}
        for fold in [1, 0]],
        "procedures": {"automl": {"pr_auc": {"mean": 0.6, "std": 0.1}},
                       "baseline_fixed": {"pr_auc": {"mean": 0.5, "std": 0.2}}},
        "comparison": {"metrics": {"pr_auc": {"mean": 0.1}}},
        "interpretation": "Aligned training-only development estimates, not a final test."}


def test_repeatable_summaries_and_aligned_comparison_table(tmp_path):
    result = comparison_result()
    first = reporting.report_files(result)
    # Dictionary insertion order and original search order do not affect summaries.
    reordered = {key: result[key] for key in reversed(result)}
    second = reporting.report_files(reordered)
    assert first == second
    rows = list(csv.DictReader(io.StringIO(first["trials.csv"])))
    assert [row["outer_fold"] for row in rows] == ["0", "1"]
    assert json.loads(rows[0]["sampled_params"]) == {"a": 1, "b": 2}
    comparison = next(csv.DictReader(io.StringIO(first["comparison.csv"])))
    assert comparison == {"metric": "pr_auc", "automl_mean": "0.6", "automl_std": "0.1",
                          "baseline_mean": "0.5", "baseline_std": "0.2", "paired_difference": "0.1"}
    assert "| pr_auc | 0.600000 ± 0.100000 | 0.500000 ± 0.200000 | +0.100000 |" in first["report.md"]
    assert "not a confidence interval" in first["report.md"]
    path = reporting.run_path(tmp_path, "test-run")
    reporting.write_report(result, path)
    for name, content in first.items():
        assert (path / name).read_bytes() == content.encode("utf-8")
    before = {name: (path / name).read_bytes() for name in first}
    with pytest.raises(FileExistsError):
        reporting.write_report(result, path)
    assert before == {name: (path / name).read_bytes() for name in first}
    with pytest.raises(FileExistsError):
        reporting.run_path(tmp_path, "test-run")
    assert reporting.run_path(tmp_path) != reporting.run_path(tmp_path)


def test_nonfinite_metadata_creates_no_report(tmp_path):
    result = {"trial_records": [], "bad": float("nan")}
    path = reporting.run_path(tmp_path, "invalid")
    with pytest.raises(ValueError):
        reporting.write_report(result, path)
    assert not path.exists()


@pytest.mark.parametrize("run_id", ["", "CON", "NUL", "COM1", "run.", "../escape", "..", ".", "/absolute", "a/b", "a\\b", "x" * 65])
def test_run_id_cannot_escape_directory(tmp_path, run_id):
    with pytest.raises(ValueError, match="Run ID"):
        reporting.run_path(tmp_path, run_id)


@pytest.mark.parametrize("relative", ["reports/metrics/new.json", "reports/figures/new.json", "reports/results.md",
                                      "README.md", "data/processed/new.json", "models/current.json"])
def test_output_rejects_baseline_and_serving_paths(relative):
    with pytest.raises(ValueError):
        automl.check_output(ROOT / relative)
    with pytest.raises(ValueError):
        reporting.check_destination(ROOT / relative)


def test_relative_and_resolved_output_paths_are_protected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    with pytest.raises(ValueError):
        automl.check_output(Path("models/current.json"), models)
    assert automl.check_output(Path("legacy/search.json"), models) == tmp_path / "legacy/search.json"


@pytest.mark.parametrize("module", [automl, automl_evaluation])
def test_report_cli_options_fail_before_data_or_training(module, monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid output must fail before data loading or search")

    monkeypatch.setattr(module, "load_data", unexpected)
    with pytest.raises(SystemExit):
        module.main(["--run-id", "orphan"])
    with pytest.raises(SystemExit):
        module.main(["--report-dir", str(tmp_path), "--output", str(tmp_path / "legacy.json")])
    reporting.run_path(tmp_path, "existing").mkdir()
    with pytest.raises(FileExistsError):
        module.main(["--report-dir", str(tmp_path), "--run-id", "existing"])


def test_cli_preserves_failure_report_then_exits_nonzero(optuna, training_rows, monkeypatch, tmp_path, capsys):
    X, y = training_rows
    monkeypatch.setattr(automl, "load_data", lambda path: None)
    monkeypatch.setattr(automl, "training_partition", lambda data: (X, y))
    source = tmp_path / "data.csv"
    source.write_text("synthetic input", encoding="utf-8")
    real = automl.cv_scores

    def fail(pipeline, *args):
        if type(pipeline.named_steps["model"]).__name__ != "DummyClassifier":
            raise RuntimeError("synthetic fit failure")
        return real(pipeline, *args)

    monkeypatch.setattr(automl, "cv_scores", fail)
    with pytest.raises(SystemExit) as exc:
        automl.main(["--data", str(source), "--n-trials", "1", "--folds", "3",
                     "--report-dir", str(tmp_path / "reports"), "--run-id", "failed-run"])
    assert exc.value.code == 1
    report = json.loads((tmp_path / "reports/failed-run/result.json").read_text(encoding="utf-8"))
    assert report["status"] == "no_successful_trials" and report["best"] is None
    assert report["trial_records"][0]["failure"]["type"] == "RuntimeError"
    assert json.loads(capsys.readouterr().out)["trial_records"] == report["trial_records"]


@pytest.mark.parametrize("fail", [False, True])
def test_nested_report_cli_retains_histories_and_preserves_baseline(optuna, monkeypatch, tmp_path, raw_data,
                                                                   capsys, fail):
    source = tmp_path / "synthetic.csv"
    data = raw_data.head(120).copy()
    data["Attrition"] = np.where(np.arange(len(data)) % 3 == 0, "Yes", "No")
    data.to_csv(source, index=False)
    monkeypatch.setattr(automl, "FAMILIES", ("logistic_regression",))
    real_candidates = automl_evaluation.candidate_pipelines
    monkeypatch.setattr(automl_evaluation, "candidate_pipelines",
                        lambda X: {"logistic_regression": real_candidates(X)["logistic_regression"]})
    if fail:
        real_scores = automl.cv_scores

        def failure(pipeline, *args):
            if type(pipeline.named_steps["model"]).__name__ != "DummyClassifier":
                raise ValueError("synthetic nested failure")
            return real_scores(pipeline, *args)

        monkeypatch.setattr(automl, "cv_scores", failure)
    canonical = [ROOT / "README.md", *sorted((ROOT / "reports").rglob("*")),
                 *sorted((ROOT / "models").rglob("*"))]
    before = {str(path): path.read_bytes() for path in canonical if path.is_file()}
    args = ["--data", str(source), "--n-trials", "1", "--outer-folds", "2", "--timeout", "120",
            "--report-dir", str(tmp_path / "experiments"), "--run-id", "nested-run"]
    if fail:
        with pytest.raises(SystemExit) as exc:
            automl_evaluation.main(args)
        assert exc.value.code == 1
    else:
        automl_evaluation.main(args)
    path = tmp_path / "experiments/nested-run"
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert result == {**stdout, "report_schema_version": 1}
    assert result["run_id"] == "nested-run"
    assert result["holdout_used"] is False and result["served_model_changed"] is False
    assert result["status"] == ("incomplete" if fail else "complete")
    assert len(result["searches"]) == (1 if fail else 2)
    trials = list(csv.DictReader((path / "trials.csv").read_text(encoding="utf-8").splitlines()))
    assert len(trials) == (1 if fail else 2)
    assert trials[0]["state"] == ("FAIL" if fail else "COMPLETE")
    comparisons = list(csv.DictReader((path / "comparison.csv").read_text(encoding="utf-8").splitlines()))
    if fail:
        assert comparisons == [] and "procedures" not in result
        assert result["failure"]["outer_fold"] == 0
        assert "Nested evaluation incomplete" in (path / "report.md").read_text(encoding="utf-8")
    else:
        assert {row["metric"] for row in comparisons} == set(result["comparison"]["metrics"])
        for row in comparisons:
            assert float(row["baseline_mean"]) == result["procedures"]["baseline_fixed"][row["metric"]]["mean"]
            assert float(row["paired_difference"]) == result["comparison"]["metrics"][row["metric"]]["mean"]
    after = {str(path): path.read_bytes() for path in canonical if path.is_file()}
    assert before == after
    # CLI output-directory collision is caught before loading data or starting another run.
    with pytest.raises(FileExistsError):
        automl_evaluation.main(args)


def test_nonfinite_trial_score_is_recorded_as_failure(optuna, training_rows, monkeypatch):
    X, y = training_rows
    real = automl.cv_scores

    def nonfinite(pipeline, *args):
        if type(pipeline.named_steps["model"]).__name__ != "DummyClassifier":
            return [float("nan")] * 3
        return real(pipeline, *args)

    monkeypatch.setattr(automl, "cv_scores", nonfinite)
    result = automl.search(X, y, n_trials=1, timeout=120, seed=7, folds=3)
    assert result["best"] is None and result["trial_records"][0]["state"] == "FAIL"
    assert result["trial_records"][0]["failure"]["message"] == "Trial produced non-finite CV scores."
    reporting.report_files(result)  # Strict JSON remains serializable.


def test_canonical_directory_symlink_is_protected(tmp_path, monkeypatch):
    root, external = tmp_path / "checkout", tmp_path / "external-metrics"
    external.mkdir()
    (root / "reports").mkdir(parents=True)
    canonical = root / "reports/metrics"
    try:
        canonical.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable on this host: {exc}")
    monkeypatch.setattr(reporting, "ROOT", root)
    with pytest.raises(ValueError):
        reporting.check_destination(external / "new.json")
    with pytest.raises(ValueError):
        reporting.check_destination(canonical / "new.json")
