"""FLAML benchmark: same space, folds, and boundaries as AutoML; optional, isolated, and never touches the holdout."""

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from attrition.config import (
    CATEGORIES,
    CONSTANT_COLUMNS,
    CV_FOLDS,
    ID_COLUMN,
    NUMERIC_BOUNDS,
    RANDOM_STATE,
    ROOT,
    TARGET,
)
from attrition.data.load_data import split_data
from attrition.features.preprocess import build_pipeline
from attrition.models import automl, automl_evaluation, nested, train
from attrition.models import automl_reporting as reporting
from attrition.models import flaml_benchmark as benchmark
from attrition.models.nested import NESTED_METRICS

# One trial and two outer folds keep each nested run in CI to seconds; each fold still checks every boundary.
OUTER, TRIALS, SEED = 2, 1, 7
FAMILIES = ("logistic_regression", "random_forest", "hist_gradient_boosting")


@pytest.fixture
def optuna():
    return pytest.importorskip("optuna", reason=f"The FLAML benchmark also runs Optuna: {automl.INSTALL_HINT}")


@pytest.fixture
def flaml():
    try:
        return benchmark.require_flaml()
    except ModuleNotFoundError:
        pytest.skip(f"FLAML benchmark tests need FLAML: {benchmark.INSTALL_HINT}")


@pytest.fixture
def training_rows(partitions, monkeypatch):
    X = partitions[0].head(90).copy()
    y = pd.Series((np.arange(len(X)) % 3 == 0).astype(int), index=X.index)
    monkeypatch.setattr(automl, "FAMILIES", ("logistic_regression",))
    return X, y


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(29)
    count = 150
    frame = pd.DataFrame({
        name: rng.integers(lower, (upper if upper is not None else lower + 20) + 1, count)
        for name, (lower, upper) in NUMERIC_BOUNDS.items()
    })
    for name, categories in CATEGORIES.items():
        frame[name] = rng.choice(categories, count)
    for name, value in CONSTANT_COLUMNS.items():
        frame[name] = value
    frame[ID_COLUMN] = np.arange(1, count + 1)
    frame[TARGET] = np.where((frame["OverTime"] == "Yes") & (np.arange(count) % 3 != 0), "Yes", "No")
    return frame


@pytest.fixture
def logistic_only(monkeypatch):
    """Forests dominate runtime but not data flow: CI searches and compares logistic models through the same code."""
    monkeypatch.setattr(automl, "FAMILIES", ("logistic_regression",))

    def candidates(X):
        return {name: train.build_pipeline(X, LogisticRegression(max_iter=1000, random_state=42, class_weight=weight))
                for name, weight in [("logistic_regression", None), ("logistic_balanced", "balanced")]}

    monkeypatch.setattr(automl_evaluation, "candidate_pipelines", candidates)


def without_timing(result: dict) -> dict:
    """Elapsed seconds are runtime measurements; everything else must repeat."""
    result = json.loads(json.dumps(result))
    result.pop("elapsed_seconds", None)
    for search in [result, *(entry["result"] for entry in result.get("searches", []))]:
        search.pop("elapsed_seconds", None)
        for trial in search.get("trial_records", []):
            trial.pop("elapsed_seconds")
    return result


def test_importing_the_benchmark_loads_no_optional_dependency():
    code = ("import sys, attrition.models.flaml_benchmark; "
            "assert not {'flaml', 'optuna'} & set(sys.modules), sorted({'flaml', 'optuna'} & set(sys.modules))")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert completed.returncode == 0, completed.stderr


def test_missing_flaml_fails_with_the_install_command_before_loading_data(monkeypatch):
    monkeypatch.setitem(sys.modules, "flaml", None)
    with pytest.raises(ModuleNotFoundError, match=r"requirements-flaml\.txt"):
        benchmark.require_flaml()
    monkeypatch.setattr(benchmark, "load_data", lambda path: pytest.fail("data loaded without FLAML"))
    with pytest.raises(ModuleNotFoundError, match=r"requirements-flaml\.txt"):
        benchmark.main(["--n-trials", "1"])


def test_flaml_is_optional_and_absent_from_serving_and_images():
    def lines(name):
        return (ROOT / name).read_text(encoding="utf-8").lower().splitlines()

    assert any(line.startswith("flaml>=") for line in lines("requirements-flaml.txt"))
    assert "-r requirements-automl.txt" in lines("requirements-flaml.txt")
    assert "-r requirements-flaml.txt" in lines("requirements-dev.txt")
    assert "flaml==2.7.0" in lines("requirements-lock.txt")
    for name in ("requirements.txt", "requirements-serving.txt", "requirements-automl.txt", "Dockerfile"):
        assert not any("flaml" in line for line in lines(name)), name


def test_search_space_matches_the_optuna_space(optuna, flaml):
    tune, _ = flaml
    assert tuple(benchmark.SPACE) == automl.FAMILIES == FAMILIES
    space, low_cost, starts = benchmark.search_space(tune)
    assert [choice["family"] for choice in space["model"].categories] == list(FAMILIES)
    assert starts == [{"model": {"family": family, **benchmark.LOW_COST.get(family, {})}} for family in FAMILIES]
    assert len(low_cost["model"]) == len(FAMILIES)
    for family, domains in zip(FAMILIES, space["model"].categories, strict=True):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        study.enqueue_trial({"model": family})
        trial = study.ask()
        automl.suggest_estimator(trial, 0)
        assert list(trial.distributions["model"].choices) == list(FAMILIES)
        distributions = {name.split("__", 1)[1]: value for name, value in trial.distributions.items() if name != "model"}
        assert set(distributions) == set(benchmark.SPACE[family]) == set(domains) - {"family"}
        for name, distribution in distributions.items():
            kind, *spec = benchmark.SPACE[family][name]
            domain = domains[name]
            if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                assert kind == "choice" and spec == [list(distribution.choices)]
                assert [json.loads(option) for option in domain.categories] == list(distribution.choices)
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                assert kind == "int" and not distribution.log
                assert spec == [distribution.low, distribution.high, distribution.step]
                q = getattr(domain.get_sampler(), "q", None)
                # FLAML's quantized upper bound is inclusive; its plain upper bound is exclusive.
                assert (domain.lower, domain.upper, q) == ((distribution.low, distribution.high, distribution.step)
                                                           if q else (distribution.low, distribution.high + 1, None))
            else:
                assert kind == "loguniform" and distribution.log
                assert spec == [distribution.low, distribution.high] == [domain.lower, domain.upper]
                assert type(domain.get_sampler()).__name__ == "_LogUniform"


def test_decoded_configurations_are_plain_python_values():
    config = {"model": {"family": np.str_("random_forest"), "n_estimators": np.int64(150), "max_depth": 4.0,
                        "min_samples_leaf": np.int32(3), "max_features": np.str_("0.3"), "class_weight": "null"}}
    family, params = benchmark.decode(config)
    assert family == "random_forest"
    assert params == {"n_estimators": 150, "max_depth": 4, "min_samples_leaf": 3, "max_features": 0.3,
                      "class_weight": None}
    assert all(type(value) in {int, float, type(None)} for value in params.values())


def test_search_is_seeded_reconstructable_and_starts_once_per_family(optuna, flaml, partitions, monkeypatch):
    X = partitions[0].head(90).copy()
    y = pd.Series((np.arange(len(X)) % 3 == 0).astype(int), index=X.index)
    first, second = (benchmark.search(X, y, n_trials=4, timeout=300, seed=SEED, folds=3) for _ in range(2))
    assert without_timing(first) == without_timing(second)
    json.dumps(first, allow_nan=False)
    records = first["trial_records"]
    assert [record["number"] for record in records] == [0, 1, 2, 3]
    # CFO evaluates one low-cost start point per family before its local search.
    assert [record["model"] for record in records[:3]] == list(FAMILIES)
    assert records[1]["sampled_params"]["random_forest__n_estimators"] == 100
    assert records[2]["sampled_params"]["hist_gradient_boosting__max_iter"] == 50
    assert all(record["state"] == "COMPLETE" and record["failure"] is None for record in records)
    assert first["trials"] == {"completed": 4, "total": 4} and first["status"] == "complete"
    best = max(records, key=lambda record: (record["value"], -record["number"]))
    assert first["best"]["trial"] == best["number"] and first["best"]["model"] == best["model"]
    pipeline = automl.candidate_pipeline(X, first["best"]["model"], first["best"]["params"], first["sampler"]["seed"])
    assert pipeline.named_steps["model"].get_params(deep=False) == first["best"]["estimator_params"]
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    assert automl.cv_scores(pipeline, X, y, cv) == first["best"]["cv_ap"]["folds"]
    assert first["sampler"] == {"name": "CFO", "seed": SEED, "cost_attr": None,
                                "start_points": benchmark.search_space(flaml[0])[2]}
    assert first["cv"] == {"partition": "training", "folds": 3, "stratified": True, "shuffle": True,
                           "random_state": SEED}
    assert first["budget"] == {"n_trials": 4, "timeout_seconds": 300, "n_jobs": 1}
    assert first["package_versions"]["flaml"] and first["package_versions"]["optuna"]
    assert first["holdout_used"] is False and first["served_model_changed"] is False
    assert first["training_data"] == reporting.training_metadata(X, y)
    other = benchmark.search(X, y, n_trials=4, timeout=300, seed=SEED + 1, folds=3)
    assert [record["sampled_params"] for record in other["trial_records"]] != [
        record["sampled_params"] for record in records]


@pytest.mark.parametrize("all_fail", [False, True])
def test_trial_failure_is_recorded_and_never_selected(optuna, flaml, training_rows, monkeypatch, all_fail):
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
    result = benchmark.search(X, y, n_trials=2, timeout=120, seed=SEED, folds=3)
    assert result["trials"] == {"total": 2, "completed": 0 if all_fail else 1}
    failed = result["trial_records"][0]
    assert failed["state"] == "FAIL" and failed["value"] is None and failed["cv_ap"] is None
    assert failed["failure"] == {"type": "ValueError", "message": "synthetic estimator failure"}
    assert failed["elapsed_seconds"] >= 0 and failed["sampled_params"]["model"] == "logistic_regression"
    assert result["status"] == ("no_successful_trials" if all_fail else "complete")
    assert (result["best"] is None) is all_fail
    if not all_fail:
        assert result["best"]["trial"] == 1


def test_nonfinite_scores_fail_the_trial_and_unexpected_errors_abort(optuna, flaml, training_rows, monkeypatch):
    X, y = training_rows
    real = automl.cv_scores

    def nonfinite(pipeline, *args):
        if type(pipeline.named_steps["model"]).__name__ == "DummyClassifier":
            return real(pipeline, *args)
        return [float("nan")] * 3

    monkeypatch.setattr(automl, "cv_scores", nonfinite)
    result = benchmark.search(X, y, n_trials=1, timeout=120, seed=SEED, folds=3)
    assert result["trial_records"][0]["failure"]["message"] == "Trial produced non-finite CV scores."
    json.dumps(result, allow_nan=False)

    def broken(*args):
        raise TypeError("programming error")

    monkeypatch.setattr(automl, "cv_scores", broken)
    with pytest.raises(TypeError, match="programming error"):
        benchmark.search(X, y, n_trials=1, timeout=120, seed=SEED, folds=3)


@pytest.mark.parametrize("kwargs", [{"n_trials": 0}, {"folds": 1}, {"timeout": 0}])
def test_search_rejects_invalid_budgets(flaml, training_rows, kwargs):
    X, y = training_rows
    with pytest.raises(ValueError):
        benchmark.search(X, y, **{"n_trials": 1, "timeout": 10, "folds": 3, **kwargs})


class RecordFit(TransformerMixin, BaseEstimator):
    """First pipeline step; logs every fit's rows, before any learned preprocessing, and checks its labels."""

    events: ClassVar[list] = []
    labels: ClassVar[pd.Series] = pd.Series(dtype=int)

    def fit(self, X, y=None):
        np.testing.assert_array_equal(y, self.labels.loc[X.index].to_numpy())
        self.events.append(("fit", frozenset(X.index)))
        return self

    def transform(self, X):
        return X


def rows(frame: pd.DataFrame, positions) -> frozenset:
    return frozenset(frame.iloc[positions].index)


def test_every_procedure_selects_and_thresholds_only_on_outer_training_rows(optuna, flaml, logistic_only,
                                                                           monkeypatch, synthetic_data):
    X, X_test, y, _ = split_data(synthetic_data)
    events = []
    monkeypatch.setattr(RecordFit, "events", events)
    monkeypatch.setattr(RecordFit, "labels", y)

    def recorded_pipeline(*args, **kwargs):
        return Pipeline([("record", RecordFit()), *build_pipeline(*args, **kwargs).steps])

    # Both searches' trials and dummy references, the rebuilt candidates, and the baseline all start with RecordFit.
    for module in (automl, train, benchmark):
        monkeypatch.setattr(module, "build_pipeline", recorded_pipeline)

    def recorded_search(name, real):
        def search(features, labels, **kwargs):
            pd.testing.assert_series_equal(labels, y.loc[features.index])
            events.append((name, frozenset(features.index)))
            return real(features, labels, **kwargs)
        return search

    def recorded_threshold_table(real):
        def threshold_table(labels, probabilities):
            pd.testing.assert_series_equal(labels, y.loc[labels.index])
            assert len(probabilities) == len(labels)
            events.append(("threshold", frozenset(labels.index)))
            return real(labels, probabilities)
        return threshold_table

    real_evaluate = automl_evaluation.evaluate_probabilities

    def evaluate(labels, probabilities, threshold):
        events.append(("score", frozenset(labels.index)))
        return real_evaluate(labels, probabilities, threshold)

    monkeypatch.setattr(benchmark, "search", recorded_search("flaml_search", benchmark.search))
    monkeypatch.setattr(automl, "search", recorded_search("optuna_search", automl.search))
    for module in (automl_evaluation, nested):
        monkeypatch.setattr(module, "threshold_table", recorded_threshold_table(module.threshold_table))
    monkeypatch.setattr(automl_evaluation, "evaluate_probabilities", evaluate)

    result = benchmark.benchmark_folds(X, y, n_trials=TRIALS, timeout=120, seed=SEED, outer_folds=OUTER)

    expected = []
    for fit, validation in StratifiedKFold(OUTER, shuffle=True, random_state=SEED).split(X, y):
        X_fit, y_fit = X.iloc[fit], y.iloc[fit]
        inner = [("fit", rows(X_fit, train_rows))
                 for train_rows, _ in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(X_fit, y_fit)]
        tail = [("threshold", rows(X, fit)), ("fit", rows(X, fit)), ("score", rows(X, validation))]
        # Each search: trials and the dummy reference on inner folds; then OOF predictions of the frozen choice,
        # the threshold, the refit on outer training rows, and one score on the outer fold.
        for search in ("flaml_search", "optuna_search"):
            expected += [(search, rows(X, fit)), *inner * (TRIALS + 1), *inner, *tail]
        # Baseline: both candidates on the same inner folds, OOF predictions, threshold, refit, score.
        expected += [*inner * 2, *inner, *tail]
    assert events == expected
    assert all(rows_seen.isdisjoint(X_test.index) for _, rows_seen in events)
    assert [(row["procedure"], row["outer_fold"]) for row in result] == [
        (procedure, fold) for fold in range(OUTER) for procedure in benchmark.PROCEDURES]


def test_benchmark_is_deterministic_and_reports_paired_comparisons(optuna, flaml, logistic_only, synthetic_data):
    X, _, y, _ = split_data(synthetic_data)
    first, second = (benchmark.benchmark(X, y, n_trials=TRIALS, timeout=120, seed=SEED, outer_folds=OUTER)
                     for _ in range(2))
    assert without_timing(first) == without_timing(second)
    json.dumps(first, allow_nan=False)
    assert first["status"] == "complete"
    assert first["outer_cv"] == {"partition": "training", "folds": OUTER, "stratified": True, "shuffle": True,
                                 "random_state": SEED}
    assert first["inner_cv"]["folds"] == CV_FOLDS and first["inner_cv"]["shared_by"] == list(benchmark.PROCEDURES)
    assert first["flaml"]["budget_per_outer_fold"] == first["automl"]["budget_per_outer_fold"]
    assert first["holdout_used"] is False and first["served_model_changed"] is False
    assert set(first["procedures"]) == set(benchmark.PROCEDURES)
    assert [(entry["outer_fold"], entry["framework"]) for entry in first["searches"]] == [
        (fold, framework) for fold in range(OUTER) for framework in ("flaml", "automl")]
    by_fold = {(row["procedure"], row["outer_fold"]): row for row in first["folds"]}
    assert len(by_fold) == 3 * OUTER
    for name, (left, right) in {"flaml_minus_baseline": ("flaml", "baseline_fixed"),
                                "flaml_minus_automl": ("flaml", "automl"),
                                "automl_minus_baseline": ("automl", "baseline_fixed")}.items():
        for metric in NESTED_METRICS:
            differences = [by_fold[left, fold][metric] - by_fold[right, fold][metric] for fold in range(OUTER)]
            assert first["comparison"][name][metric]["mean"] == pytest.approx(np.mean(differences))
    for entry in first["searches"]:
        row = by_fold[entry["framework"], entry["outer_fold"]]
        assert row["selected_model"] == entry["result"]["best"]["model"]
        assert row["parameters"] == entry["result"]["best"]["params"]
        assert row["inner_cv_ap"] == entry["result"]["best"]["cv_ap"]["mean"]
        assert 0 < row["threshold"] < 1
    assert set(first["selection_optimism"]) == {"flaml", "automl"}


def test_generalized_paired_differences_keep_the_automl_names():
    folds = [{"procedure": procedure, "outer_fold": fold, "inner_cv_ap": 0.5, **dict.fromkeys(NESTED_METRICS, score)}
             for fold, scores in enumerate([(0.6, 0.5, 0.4), (0.4, 0.5, 0.5)])
             for procedure, score in zip(benchmark.PROCEDURES, scores, strict=True)]
    assert automl_evaluation.paired_differences(folds, "flaml", "baseline_fixed")["pr_auc"] == {
        "mean": pytest.approx(0.05), "std": pytest.approx(0.15), "min": pytest.approx(-0.1), "max": pytest.approx(0.2),
        "flaml_higher_folds": 1, "baseline_higher_folds": 1}
    assert automl_evaluation.paired_differences(folds, "flaml", "automl")["pr_auc"]["automl_higher_folds"] == 1
    assert set(automl_evaluation.paired_differences(folds)["pr_auc"]) >= {"automl_higher_folds",
                                                                          "baseline_higher_folds"}
    assert automl_evaluation.selection_optimism(folds, "flaml")["mean_outer_ap"] == pytest.approx(0.5)


def fake_result() -> dict:
    """A small completed result built from hand-written fold rows; no search or training."""
    folds = [{"procedure": procedure, "outer_fold": fold, "selected_model": "logistic_regression", "parameters": {},
              "threshold": 0.3, "inner_cv_ap": 0.5, "trials_completed": 1,
              **dict.fromkeys(NESTED_METRICS, score)}
             for fold, scores in enumerate([(0.6, 0.5, 0.4), (0.4, 0.5, 0.5)])
             for procedure, score in zip(benchmark.PROCEDURES, scores, strict=True)]
    trial = {"number": 0, "state": "COMPLETE", "model": "logistic_regression",
             "sampled_params": {"model": "logistic_regression", "logistic_regression__C": 1.0},
             "estimator_params": {"C": 1.0}, "cv_ap": {"mean": 0.5}, "value": 0.5, "failure": None,
             "elapsed_seconds": 0.1}
    searches = [{"outer_fold": fold, "framework": framework, "result": {"trial_records": [trial]}}
                for fold in (1, 0) for framework in ("automl", "flaml")]
    return {"interpretation": benchmark.INTERPRETATION, "folds": folds, "searches": searches,
            "procedures": nested.nested_summary(pd.DataFrame(folds)),
            "comparison": {"flaml_minus_baseline": automl_evaluation.paired_differences(folds, "flaml", "baseline_fixed"),
                           "flaml_minus_automl": automl_evaluation.paired_differences(folds, "flaml", "automl"),
                           "automl_minus_baseline": automl_evaluation.paired_differences(folds)}}


def test_reports_are_repeatable_ordered_and_written_once(tmp_path):
    result = fake_result()
    files = benchmark.report_files(result)
    assert files == benchmark.report_files(json.loads(json.dumps(result)))
    trials = list(csv.DictReader(io.StringIO(files["trials.csv"])))
    assert [(row["outer_fold"], row["framework"]) for row in trials] == [
        ("0", "flaml"), ("0", "automl"), ("1", "flaml"), ("1", "automl")]
    comparisons = list(csv.DictReader(io.StringIO(files["comparison.csv"])))
    assert [row["metric"] for row in comparisons] == NESTED_METRICS
    pr_auc = comparisons[0]
    assert float(pr_auc["flaml_mean"]) == pytest.approx(0.5)
    assert float(pr_auc["baseline_fixed_mean"]) == pytest.approx(0.45)
    assert float(pr_auc["flaml_minus_baseline"]) == pytest.approx(0.05)
    assert float(pr_auc["flaml_minus_automl"]) == pytest.approx(0.0)
    assert "| pr_auc | 0.500000 ± 0.100000 | 0.500000 ± 0.000000 | 0.450000 ± 0.050000 | +0.050000 |" in files[
        "report.md"]
    assert json.loads(files["result.json"])["report_schema_version"] == reporting.REPORT_SCHEMA_VERSION
    path = tmp_path / "runs/first"
    benchmark.write_report(result, path)
    assert {child.name: child.read_bytes().decode("utf-8") for child in path.iterdir()} == files
    with pytest.raises(FileExistsError):
        benchmark.write_report(result, path)
    with pytest.raises(ValueError):
        benchmark.write_report(result, ROOT / "reports/metrics/flaml")
    incomplete = {key: value for key, value in result.items() if key not in {"procedures", "comparison"}}
    incomplete_files = benchmark.report_files(incomplete)
    assert incomplete_files["comparison.csv"].strip() == ",".join(benchmark.COMPARISON_COLUMNS)
    assert "Benchmark incomplete" in incomplete_files["report.md"]


def test_cli_uses_training_rows_only_and_writes_only_a_new_output(flaml, monkeypatch, tmp_path, capsys,
                                                                  synthetic_data):
    source = tmp_path / "data.csv"
    synthetic_data.to_csv(source, index=False)
    X_train, _, _, y_test = split_data(synthetic_data)
    calls = []

    def run(X, y, **kwargs):
        pd.testing.assert_frame_equal(X, X_train)
        assert set(y.index).isdisjoint(y_test.index), "holdout labels reached the benchmark"
        calls.append(kwargs)
        return {"holdout_used": False}

    monkeypatch.setattr(benchmark, "benchmark", run)
    monkeypatch.setattr(automl, "require_optuna", lambda: None)
    output = tmp_path / "results/flaml.json"
    args = ["--data", str(source), "--n-trials", "3", "--timeout", "60", "--seed", "11", "--outer-folds", "4"]
    benchmark.main([*args, "--output", str(output)])
    assert calls == [{"n_trials": 3, "timeout": 60.0, "seed": 11, "outer_folds": 4}]
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == json.loads(capsys.readouterr().out)
    assert len(written["data_sha256"]) == 64 and written["elapsed_seconds"] >= 0
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()) == [
        "data.csv", "results/flaml.json"]
    with pytest.raises(FileExistsError):
        benchmark.main([*args, "--output", str(output)])
    assert len(calls) == 1


@pytest.mark.parametrize("argument", [["--n-trials", "0"], ["--timeout", "0"], ["--outer-folds", "1"],
                                      ["--outer-folds", "11"], ["--run-id", "orphan"]])
def test_cli_rejects_invalid_arguments(argument):
    with pytest.raises(SystemExit):
        benchmark.main(argument)


def test_cli_refuses_protected_or_existing_destinations_before_loading_data(flaml, monkeypatch, tmp_path):
    monkeypatch.setattr(automl, "require_optuna", lambda: None)
    monkeypatch.setattr(benchmark, "load_data", lambda path: pytest.fail("data loaded for a refused destination"))
    for destination in (ROOT / "models/flaml.json", ROOT / "reports/metrics/flaml.json", ROOT / "README.md"):
        with pytest.raises(ValueError):
            benchmark.main(["--output", str(destination)])
    with pytest.raises(ValueError):
        benchmark.main(["--report-dir", str(ROOT / "reports/figures"), "--run-id", "flaml"])
    reporting.run_path(tmp_path, "existing").mkdir()
    with pytest.raises(FileExistsError):
        benchmark.main(["--report-dir", str(tmp_path), "--run-id", "existing"])


@pytest.mark.parametrize("fail", [False, True])
def test_report_cli_retains_histories_and_preserves_baseline(optuna, flaml, monkeypatch, tmp_path, raw_data, capsys,
                                                             fail):
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
                raise ValueError("synthetic benchmark failure")
            return real_scores(pipeline, *args)

        monkeypatch.setattr(automl, "cv_scores", failure)
    canonical = [ROOT / "README.md", *sorted((ROOT / "reports").rglob("*")), *sorted((ROOT / "models").rglob("*"))]
    before = {str(path): path.read_bytes() for path in canonical if path.is_file()}
    args = ["--data", str(source), "--n-trials", "1", "--outer-folds", "2", "--timeout", "120",
            "--report-dir", str(tmp_path / "experiments"), "--run-id", "flaml-run"]
    if fail:
        with pytest.raises(SystemExit) as exc:
            benchmark.main(args)
        assert exc.value.code == 1
    else:
        benchmark.main(args)
    path = tmp_path / "experiments/flaml-run"
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    assert result == {**json.loads(capsys.readouterr().out), "report_schema_version": 1}
    assert result["run_id"] == "flaml-run" and result["holdout_used"] is False
    assert result["served_model_changed"] is False
    assert result["status"] == ("incomplete" if fail else "complete")
    trials = list(csv.DictReader((path / "trials.csv").read_text(encoding="utf-8").splitlines()))
    comparisons = list(csv.DictReader((path / "comparison.csv").read_text(encoding="utf-8").splitlines()))
    if fail:
        # The FLAML search runs first, fails in outer fold 0, and its evidence is kept.
        assert result["failure"]["outer_fold"] == 0 and result["failure"]["framework"] == "flaml"
        assert [entry["framework"] for entry in result["searches"]] == ["flaml"]
        assert [row["state"] for row in trials] == ["FAIL"] and comparisons == []
        assert "procedures" not in result and result["folds"] == []
    else:
        assert [(row["outer_fold"], row["framework"]) for row in trials] == [
            ("0", "flaml"), ("0", "automl"), ("1", "flaml"), ("1", "automl")]
        for row in comparisons:
            metric = row["metric"]
            assert float(row["flaml_mean"]) == result["procedures"]["flaml"][metric]["mean"]
            assert float(row["flaml_minus_automl"]) == result["comparison"]["flaml_minus_automl"][metric]["mean"]
    after = {str(path): path.read_bytes() for path in canonical if path.is_file()}
    assert before == after
    with pytest.raises(FileExistsError):
        benchmark.main(args)


@pytest.mark.skipif(os.environ.get("ATTRITION_SLOW_TESTS") != "1",
                    reason="opt-in real-data FLAML benchmark; set ATTRITION_SLOW_TESTS=1")
def test_real_data_baseline_arm_reproduces_the_training_nested_cv(optuna, flaml, partitions):
    X, _, y, _ = partitions
    report = benchmark.benchmark(X, y, n_trials=2, timeout=300, seed=RANDOM_STATE)
    published = pd.read_csv(Path(ROOT) / "reports/metrics/nested_cv_folds.csv")
    published = published[published["procedure"] == "fixed"].reset_index(drop=True)
    baseline = pd.DataFrame([row for row in report["folds"] if row["procedure"] == "baseline_fixed"])
    # Same outer folds and the same served procedure as the committed training report: identical rows.
    assert baseline["outer_fold"].tolist() == published["outer_fold"].tolist()
    assert baseline["selected_model"].tolist() == published["selected_model"].tolist()
    np.testing.assert_allclose(baseline[["threshold", *NESTED_METRICS]].to_numpy(dtype=float),
                               published[["threshold", *NESTED_METRICS]].to_numpy(dtype=float), rtol=0, atol=1e-12)
    assert report["status"] == "complete" and report["train_rows"] == len(X) and report["holdout_used"] is False
    assert {row["procedure"] for row in report["folds"]} == set(benchmark.PROCEDURES)
