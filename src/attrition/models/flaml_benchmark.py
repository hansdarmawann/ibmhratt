"""Optional FLAML benchmark on training CV: python -m attrition.models.flaml_benchmark --n-trials 30 --seed 42.

Install requirements-flaml.txt first. FLAML's CFO searcher explores the same
families and ranges as the Optuna AutoML search, and every trial is scored by the
same shared pipeline on the same seeded stratified training folds. Nested CV then
runs three procedures on identical outer folds of the training partition: the
FLAML search, the Optuna search, and the served baseline. Each one selects, picks
an out-of-fold F2 threshold, and refits inside the outer training rows, then is
scored once on the outer validation rows. Results are training-only development
estimates: the holdout is never read, and no model, bundle, canonical baseline
report, or models/current.json is written. Nothing runs at import time.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from attrition.config import CV_FOLDS, DATA_ENV, DATA_PATH, DATA_RELATIVE, RANDOM_STATE, normalize_path
from attrition.data.load_data import load_data
from attrition.features.preprocess import build_pipeline
from attrition.models import automl, automl_evaluation
from attrition.models import automl_reporting as reporting
from attrition.models.automl_evaluation import BASELINE, THRESHOLD_RULE, Choice, NoSuccessfulSearch
from attrition.models.nested import NESTED_METRICS, OUTER_FOLDS, nested_summary

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
FLAML, AUTOML = "flaml", automl_evaluation.AUTOML
PROCEDURES = (FLAML, AUTOML, BASELINE)
SEARCHER = "CFO"
# Reported to the searcher for a failed trial: below any attainable AP, so it never looks promising.
FAILED_SCORE = -1.0
INSTALL_HINT = "python -m pip install -r requirements-flaml.txt -c requirements-lock.txt"
# The Optuna search space of attrition.models.automl.suggest_estimator, written declaratively; a test
# checks that both agree. Integer bounds are inclusive, as in Optuna. Categorical options are passed to
# FLAML as JSON labels, because its local search cannot step through mixed or None-valued choices.
SPACE: dict[str, dict[str, tuple]] = {
    "logistic_regression": {
        "C": ("loguniform", 0.01, 10.0),
        "class_weight": ("choice", [None, "balanced"]),
    },
    "random_forest": {
        "n_estimators": ("int", 100, 400, 50),
        "max_depth": ("int", 3, 10, 1),
        "min_samples_leaf": ("int", 2, 20, 1),
        "max_features": ("choice", ["sqrt", 0.3, 0.5]),
        "class_weight": ("choice", [None, "balanced_subsample"]),
    },
    "hist_gradient_boosting": {
        "learning_rate": ("loguniform", 0.01, 0.2),
        "max_iter": ("int", 50, 300, 25),
        "max_leaf_nodes": ("int", 4, 31, 1),
        "min_samples_leaf": ("int", 10, 50, 1),
        "l2_regularization": ("loguniform", 1e-3, 10.0),
        "class_weight": ("choice", [None, "balanced"]),
    },
}
# FLAML's cost-frugal start values: the smallest ensembles. CFO first evaluates one start point per
# family, in FAMILIES order, then searches locally from the best; no family is favored up front.
LOW_COST = {"random_forest": {"n_estimators": 100}, "hist_gradient_boosting": {"max_iter": 50}}
INTERPRETATION = (
    "Training-only development estimate, not a final test. All three procedures are scored on the same outer "
    "folds, so per-fold differences are paired, but the folds share most of their training rows and are not "
    "independent: no standard error, interval, or significance test is reported. The FLAML and Optuna searches "
    "get the same trial budget and search space, so the comparison is between search algorithms at that budget; "
    "another budget, seed, or space can change the ranking. On this small fictional dataset, a small mean "
    "difference, or most folds favoring one procedure, is not evidence that it generalizes better, and it is not "
    "a reason to change the served model. The published holdout was already examined and must not settle the "
    "comparison; confirming a difference needs new independent data.")


def require_flaml():
    """Import FLAML's tuner on demand so neither the baseline nor the Optuna AutoML path needs it."""
    try:
        with warnings.catch_warnings():
            # The base install omits flaml[automl], which this benchmark deliberately does not use.
            warnings.filterwarnings("ignore", message="flaml.automl is not available")
            from flaml import CFO, tune
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"The FLAML benchmark needs the optional FLAML dependency: {INSTALL_HINT}") from exc
    return tune, CFO


def label(value) -> str:
    return json.dumps(value)


def search_space(tune) -> tuple[dict, dict, list[dict]]:
    """FLAML's hierarchical space over the active AutoML families, its low-cost values, and start points."""
    families = list(automl.FAMILIES)
    choices = []
    for family in families:
        domains: dict[str, Any] = {"family": family}
        for name, (kind, *spec) in SPACE[family].items():
            if kind == "loguniform":
                domains[name] = tune.loguniform(spec[0], spec[1])
            elif kind == "choice":
                domains[name] = tune.choice([label(option) for option in spec[0]])
            elif spec[2] == 1:
                domains[name] = tune.randint(spec[0], spec[1] + 1)  # FLAML's randint excludes the upper bound.
            else:
                domains[name] = tune.qrandint(spec[0], spec[1], spec[2])  # Quantized bounds are inclusive.
        choices.append(domains)
    # One low-cost dictionary per family, in choice order. FLAML ignores a start-family index for an
    # unordered choice of dictionaries, so explicit start points decide where the search begins.
    low_cost = {"model": [dict(LOW_COST.get(family, {})) for family in families]}
    starts = [{"model": {"family": family, **LOW_COST.get(family, {})}} for family in families]
    return {"model": tune.choice(choices)}, low_cost, starts


def decode(config: dict) -> tuple[str, dict]:
    """A FLAML configuration as a family and plain Python estimator arguments."""
    sampled = dict(config["model"])
    family = str(sampled.pop("family"))
    params: dict[str, Any] = {}
    for name, (kind, *_) in SPACE[family].items():
        value = sampled[name]
        if kind == "choice":
            params[name] = json.loads(str(value))
        elif kind == "int":
            params[name] = int(value)
        else:
            params[name] = float(value)
    return family, params


def flaml_version() -> str | None:
    try:
        return importlib.metadata.version("flaml")
    except importlib.metadata.PackageNotFoundError:
        return None


def search(X: pd.DataFrame, y: pd.Series, *, n_trials: int = automl.DEFAULT_TRIALS,
           timeout: float | None = automl.DEFAULT_TIMEOUT, seed: int = RANDOM_STATE, folds: int = CV_FOLDS) -> dict:
    """Run a seeded, serial FLAML CFO search on the given training rows; same result shape as automl.search.

    Each trial is rebuilt by automl.candidate_pipeline, the constructor the Optuna
    trials use, and scored by the same stratified CV AP. Recoverable estimator
    failures (ValueError/RuntimeError) are recorded, reported to CFO as a score
    below any attainable AP, and never chosen; unexpected exceptions abort. The
    best candidate is the completed trial with the highest mean AP, the earliest
    on ties. Trial costs are not fed back to CFO, so a fixed trial count
    reproduces choices under matching code and runtime; a timeout that stops the
    search early makes the number of completed trials depend on machine speed.
    """
    tune, cfo = require_flaml()
    if n_trials < 1 or folds < 2 or (timeout is not None and not timeout > 0):
        raise ValueError("Use at least one trial, two folds, and a positive timeout.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    space, low_cost, starts = search_space(tune)
    records: list[dict] = []

    def objective(config: dict) -> dict:
        started = time.perf_counter()
        family, params = decode(config)
        record: dict[str, Any] = {
            "number": len(records), "state": "FAIL", "model": family,
            "sampled_params": {"model": family, **{f"{family}__{name}": value for name, value in params.items()}},
            "estimator_params": None, "estimator_class": None, "cv_ap": None, "value": None, "failure": None}
        records.append(record)
        try:
            pipeline = automl.candidate_pipeline(X, family, params, seed)
            estimator = pipeline.named_steps["model"]
            record["estimator_params"] = estimator.get_params(deep=False)
            record["estimator_class"] = f"{type(estimator).__module__}.{type(estimator).__name__}"
            scores = automl.cv_scores(pipeline, X, y, cv)
            if not all(math.isfinite(score) for score in scores):
                raise ValueError("Trial produced non-finite CV scores.")
            summary = automl.summarize(scores)
            record.update(state="COMPLETE", cv_ap=summary, value=summary["mean"])
            return {automl.OBJECTIVE: summary["mean"]}
        except (ValueError, RuntimeError) as exc:
            record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            return {automl.OBJECTIVE: FAILED_SCORE}
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 6)

    # cost_attr=None: CFO ignores measured trial times, which would make its proposals machine-dependent.
    # CFO consumes its start-point list, so it gets a copy and the result keeps the original.
    searcher = cfo(metric=automl.OBJECTIVE, mode="max", space=space, low_cost_partial_config=low_cost,
                   points_to_evaluate=copy.deepcopy(starts), seed=seed, cost_attr=None)
    started = time.perf_counter()
    tune.run(objective, search_alg=searcher, metric=automl.OBJECTIVE, mode="max", num_samples=n_trials,
             time_budget_s=timeout, verbose=0)
    elapsed = time.perf_counter() - started
    completed = [record for record in records if record["state"] == "COMPLETE"]
    best = max(completed, key=lambda record: (record["value"], -record["number"])) if completed else None
    reference = automl.cv_scores(build_pipeline(X, DummyClassifier(strategy="prior")), X, y, cv)
    if best is not None:
        LOGGER.info("Best FLAML trial %d: %s, training CV AP %.3f", best["number"], best["model"], best["value"])
    else:
        LOGGER.error("No successful FLAML trials; returning failure records without a candidate.")
    environment = reporting.provenance()
    environment["package_versions"]["flaml"] = flaml_version()
    return {
        "schema_version": SCHEMA_VERSION,
        "framework": "flaml",
        "objective": automl.OBJECTIVE, "direction": "maximize",
        "cv": {"partition": "training", "folds": folds, "stratified": True, "shuffle": True, "random_state": seed},
        "sampler": {"name": SEARCHER, "seed": seed, "cost_attr": None, "start_points": starts},
        "search_space": {family: {name: list(spec) for name, spec in SPACE[family].items()}
                         for family in automl.FAMILIES},
        "budget": {"n_trials": n_trials, "timeout_seconds": timeout, "n_jobs": 1},
        "trials": {"completed": len(completed), "total": len(records)},
        "status": "complete" if best is not None else "no_successful_trials",
        "trial_records": records,
        "elapsed_seconds": round(elapsed, 3),
        "best": {"trial": best["number"], "model": best["model"], "params": automl.readable(best["sampled_params"]),
                 "cv_ap": best["cv_ap"], "estimator_params": best["estimator_params"],
                 "estimator_class": best["estimator_class"]} if best is not None else None,
        "reference": {"dummy_prior": {"tuned": False, "cv_ap": automl.summarize(reference)}},
        "train_rows": len(X), "train_positives": int(y.sum()),
        "provenance": environment, "package_versions": environment["package_versions"],
        "training_data": reporting.training_metadata(X, y),
        "holdout_used": False,
        "served_model_changed": False,
        "note": "Training-CV development estimate of a FLAML candidate configuration; the chosen trial's CV AP is "
                "optimistic because the same folds selected it. No final model, threshold, or holdout score.",
    }


def flaml_procedure(X: pd.DataFrame, y: pd.Series, *, n_trials: int, timeout: float | None, seed: int) -> Choice:
    """The AutoML procedure with the FLAML search: the threshold and refit steps are the shared ones."""
    return automl_evaluation.automl_procedure(X, y, n_trials=n_trials, timeout=timeout, seed=seed, search=search)


def benchmark_folds(X: pd.DataFrame, y: pd.Series, *, n_trials: int, timeout: float | None,
                    seed: int = RANDOM_STATE, outer_folds: int = OUTER_FOLDS,
                    search_history: list[dict] | None = None) -> list[dict]:
    """One row per procedure and outer fold; every procedure sees only that fold's outer training rows."""
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    pipelines = automl_evaluation.candidate_pipelines(X)
    rows: list[dict] = []
    for fold, (fit, validation) in enumerate(outer.split(X, y)):
        LOGGER.info("Outer fold %d of %d: %d training rows", fold + 1, outer_folds, len(fit))
        X_fit, y_fit = X.iloc[fit], y.iloc[fit]
        X_validation, y_validation = X.iloc[validation], y.iloc[validation]
        fold_rows = []
        for framework, procedure in ((FLAML, flaml_procedure), (AUTOML, automl_evaluation.automl_procedure)):
            try:
                choice = procedure(X_fit, y_fit, n_trials=n_trials, timeout=timeout, seed=seed)
            except NoSuccessfulSearch as exc:
                if search_history is not None:
                    search_history.append({"outer_fold": fold, "framework": framework, "result": exc.result})
                exc.outer_fold, exc.completed_folds = fold, rows
                raise
            if search_history is not None:
                search_history.append({"outer_fold": fold, "framework": framework, "result": choice.search_result})
            fold_rows.append(automl_evaluation.fold_row(framework, fold, choice, X_validation, y_validation))
        baseline = automl_evaluation.baseline_procedure(pipelines, X_fit, y_fit, seed)
        fold_rows.append(automl_evaluation.fold_row(BASELINE, fold, baseline, X_validation, y_validation))
        rows += fold_rows
    return rows


def benchmark(X: pd.DataFrame, y: pd.Series, *, n_trials: int = automl.DEFAULT_TRIALS,
              timeout: float | None = automl.DEFAULT_TIMEOUT, seed: int = RANDOM_STATE,
              outer_folds: int = OUTER_FOLDS) -> dict:
    """Nested CV of the FLAML, Optuna, and baseline procedures on the given training rows, with paired comparisons."""
    searches: list[dict] = []
    failure = None
    try:
        rows = benchmark_folds(X, y, n_trials=n_trials, timeout=timeout, seed=seed, outer_folds=outer_folds,
                               search_history=searches)
    except NoSuccessfulSearch as exc:
        rows = exc.completed_folds
        # The failed search is always the last one recorded.
        failure = {"outer_fold": exc.outer_fold, "framework": searches[-1]["framework"], "message": str(exc)}
    environment = reporting.provenance()
    environment["package_versions"]["flaml"] = flaml_version()
    budget = {"n_trials": n_trials, "timeout_seconds": timeout, "n_jobs": 1}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if failure is None else "incomplete",
        "objective": automl.OBJECTIVE,
        "method": "Nested CV on the training partition: each outer training fold runs every complete procedure on "
                  "its own inner folds; the outer validation rows score each frozen result once.",
        "outer_cv": {"partition": "training", "folds": outer_folds, "stratified": True, "shuffle": True,
                     "random_state": seed},
        "inner_cv": {"partition": "outer training fold", "folds": CV_FOLDS, "stratified": True, "shuffle": True,
                     "random_state": seed, "shared_by": list(PROCEDURES)},
        "flaml": {"families": list(automl.FAMILIES), "sampler": {"name": SEARCHER, "seed": seed, "cost_attr": None},
                  "budget_per_outer_fold": budget},
        "automl": {"families": list(automl.FAMILIES), "sampler": {"name": "TPESampler", "seed": seed},
                   "budget_per_outer_fold": budget},
        "baseline": "The served procedure: fixed candidates, highest inner CV AP preferring logistic regression "
                    "within 0.01, then the inner OOF F2 threshold",
        "threshold_rule": THRESHOLD_RULE,
        **({"procedures": nested_summary(pd.DataFrame(rows)),
            "comparison": {
                "flaml_minus_baseline": automl_evaluation.paired_differences(rows, FLAML, BASELINE),
                "flaml_minus_automl": automl_evaluation.paired_differences(rows, FLAML, AUTOML),
                "automl_minus_baseline": automl_evaluation.paired_differences(rows, AUTOML, BASELINE)},
            "selection_optimism": {framework: automl_evaluation.selection_optimism(rows, framework)
                                   for framework in (FLAML, AUTOML)}} if failure is None else {"failure": failure}),
        "folds": rows, "searches": searches,
        "train_rows": len(X), "train_positives": int(y.sum()),
        "provenance": environment, "package_versions": environment["package_versions"],
        "training_data": reporting.training_metadata(X, y),
        "holdout_used": False,
        "served_model_changed": False,
        "interpretation": INTERPRETATION,
    }


TRIAL_COLUMNS = ("framework", *reporting.TRIAL_COLUMNS)
COMPARISON_COLUMNS = ("metric", *(f"{procedure}_{stat}" for procedure in PROCEDURES for stat in ("mean", "std")),
                      "flaml_minus_baseline", "flaml_minus_automl", "automl_minus_baseline")


def trial_rows(result: dict) -> list[dict]:
    order = {framework: index for index, framework in enumerate(PROCEDURES)}
    return [{"framework": entry["framework"], "outer_fold": entry["outer_fold"], **trial}
            for entry in sorted(result["searches"], key=lambda item: (item["outer_fold"], order[item["framework"]]))
            for trial in sorted(entry["result"]["trial_records"], key=lambda item: item["number"])]


def comparison_rows(result: dict) -> list[dict]:
    """Aligned nested development estimates only; never baseline holdout artifacts."""
    if "procedures" not in result:
        return []
    procedures, comparison = result["procedures"], result["comparison"]
    return [{"metric": metric,
             **{f"{procedure}_{stat}": procedures[procedure][metric][stat]
                for procedure in PROCEDURES for stat in ("mean", "std")},
             **{name: comparison[name][metric]["mean"] for name in comparison}}
            for metric in NESTED_METRICS]


def report_files(result: dict) -> dict[str, str]:
    """Pure rendering: the same result produces byte-identical UTF-8 JSON/CSV/Markdown."""
    comparisons = comparison_rows(result)
    lines = ["# FLAML benchmark development report", "", result["interpretation"], ""]
    if comparisons:
        lines += ["AP is average precision (`pr_auc`); SD is population fold SD, not a confidence interval.", "",
                  "| Metric | FLAML mean ± SD | Optuna AutoML mean ± SD | Baseline mean ± SD | FLAML − baseline "
                  "| FLAML − Optuna |", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {row['metric']} | {row['flaml_mean']:.6f} ± {row['flaml_std']:.6f} | "
                  f"{row['automl_mean']:.6f} ± {row['automl_std']:.6f} | "
                  f"{row['baseline_fixed_mean']:.6f} ± {row['baseline_fixed_std']:.6f} | "
                  f"{row['flaml_minus_baseline']:+.6f} | {row['flaml_minus_automl']:+.6f} |" for row in comparisons]
    else:
        lines += ["Benchmark incomplete: no procedure comparison estimates are reported. "
                  "Inspect the failed outer fold and its trial records in result.json and trials.csv."]
    lines += ["", "No holdout scores, model bundles, or serving changes are produced.", ""]
    return {"result.json": reporting.json_text({**result, "report_schema_version": reporting.REPORT_SCHEMA_VERSION}),
            "trials.csv": reporting.csv_text(trial_rows(result), TRIAL_COLUMNS),
            "comparison.csv": reporting.csv_text(comparisons, COMPARISON_COLUMNS),
            "report.md": "\n".join(lines)}


def write_report(result: dict, path: Path) -> None:
    """Create one new run directory, as the AutoML reports do; never reuse it."""
    files = report_files(result)  # Validate serialization before creating anything.
    path = reporting.check_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    for name, contents in files.items():
        with (path / name).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=automl.positive_int, default=automl.DEFAULT_TRIALS,
                        help=f"Trial budget of each search, per outer fold (default: {automl.DEFAULT_TRIALS})")
    parser.add_argument("--timeout", type=automl.positive_float, default=automl.DEFAULT_TIMEOUT,
                        help="Seconds after which each search starts no new trial "
                             f"(default: {automl.DEFAULT_TIMEOUT:g})")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help=f"Seed for outer and inner folds, both searchers, and estimators (default: {RANDOM_STATE})")
    parser.add_argument("--outer-folds", type=int, default=OUTER_FOLDS, choices=range(2, 11), metavar="{2..10}",
                        help=f"Stratified outer folds of the training partition (default: {OUTER_FOLDS})")
    parser.add_argument("--data", type=normalize_path, default=DATA_PATH,
                        help=f"Raw CSV (default: ${DATA_ENV} or <root>/{DATA_RELATIVE.as_posix()})")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--report-dir", type=normalize_path,
                         help="Write trial histories and paired development comparisons into a new run directory")
    parser.add_argument("--run-id", help="New report run ID (default: UUID); requires --report-dir")
    outputs.add_argument("--output", type=normalize_path, default=None,
                         help="Also write the JSON result to this new file (never inside the models directory)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.run_id is not None and args.report_dir is None:
        parser.error("--run-id requires --report-dir")
    # Fail before loading data or spending a search budget when an optional dependency is missing.
    require_flaml()
    automl.require_optuna()
    report_path = reporting.run_path(args.report_dir, args.run_id) if args.report_dir else None
    output = automl.check_output(args.output) if args.output else None
    data = load_data(args.data)
    X_train, y_train = automl.training_partition(data)
    started = time.perf_counter()
    # Single-threaded math, as in training, keeps folds and trials reproducible.
    with threadpool_limits(limits=1):
        result = benchmark(X_train, y_train, n_trials=args.n_trials, timeout=args.timeout, seed=args.seed,
                           outer_folds=args.outer_folds)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if report_path:
        result["run_id"] = report_path.name
    text = reporting.json_text(result)
    if report_path:
        write_report(result, report_path)
        LOGGER.info("Wrote report run %s", report_path)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text)
        LOGGER.info("Wrote %s", output)
    print(text, end="")
    if result.get("status") == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
