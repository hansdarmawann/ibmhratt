"""Operator-only preparation, manual approval, activation and rollback of AutoML bundles.

No operation runs on import. Preparation never publishes or evaluates the holdout.
Approval files are human attestations, not signatures; use trusted local storage.
"""

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from threadpoolctl import threadpool_limits

from attrition.config import (
    CHECKOUT_ROOT,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    ROOT,
    TARGET_MAPPING,
    normalize_path,
)
from attrition.data.load_data import load_data
from attrition.models import automl, automl_reporting
from attrition.models.artifacts import Bundle, load_bundle, publish_bundle, seal_bundle, sha256, write_json
from attrition.models.threshold import select_threshold, threshold_table
from attrition.monitoring.drift import PROFILE_FILE, reference_profile

CHECKS = ("provenance", "data_separation", "development_evaluation", "artifact_schema",
          "api_smoke", "dashboard_smoke", "rollback", "responsible_use")
EVIDENCE = ("source_review", "api_smoke", "dashboard_smoke")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def document(path: Path) -> tuple[dict, str]:
    """Read one JSON snapshot, fingerprinting the exact bytes that were parsed."""
    raw = path.read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict), "Expected a JSON object.")
    # Strict finite serialization also checks nested input before it becomes evidence.
    automl_reporting.json_text(value)
    return value, hashlib.sha256(raw).hexdigest()


def run_id(value: str) -> str:
    try:
        require(str(UUID(value)) == value, "Run ID must be a canonical UUID.")
    except (TypeError, AttributeError) as exc:
        raise ValueError("Run ID must be a canonical UUID.") from exc
    return value


def models_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    require(not any((parent / "manifest.json").exists() for parent in (path, *path.parents)),
            "Models directory must not be inside an existing bundle.")
    for root in {ROOT, CHECKOUT_ROOT}:
        if root is None:
            continue
        for name in ("data", "reports", "examples"):
            require(not path.is_relative_to((root / name).resolve()),
                    "Models directory must be outside canonical data and reports.")
    return path


def local_path(models: Path, folder: str, name: str) -> Path:
    path = models / folder / name
    require(path.resolve() == path, "Symlinked bundle or promotion paths are not allowed.")
    return path


def validate_sources(search: dict, evaluation: dict, X: pd.DataFrame, y: pd.Series,
                     data_digest: str, trial_number: int) -> tuple[dict, int, int]:
    """Reject incompatible evidence before fitting; selectors only receive training rows."""
    training = automl_reporting.training_metadata(X, y)
    versions = automl_reporting.provenance()["package_versions"]
    for result in (search, evaluation):
        require(result.get("schema_version") == 2 and result.get("status") == "complete"
                and result.get("objective") == automl.OBJECTIVE, "A complete schema-2 AutoML report is required.")
        require(result.get("holdout_used") is False and result.get("served_model_changed") is False,
                "Evidence must declare training-only selection with unchanged serving state.")
        require(result.get("data_sha256") == data_digest and result.get("training_data") == training
                and result.get("train_rows") == len(X) and result.get("train_positives") == int(y.sum()),
                "Evidence does not match the original training partition or data checksum.")
        provenance = result.get("provenance", {})
        require(isinstance(provenance, dict) and bool(provenance.get("git_revision")),
                "Evidence needs source revision provenance; preserve dirty source diffs for review.")
        require(result.get("package_versions") == versions and provenance.get("package_versions") == versions,
                "Evidence dependency versions must match this preparation environment.")
    cv = search.get("cv", {})
    seed, folds = cv.get("random_state"), cv.get("folds")
    require(type(seed) is int and type(folds) is int and 2 <= folds <= 10,
            "Invalid search seed or fold configuration.")
    require(cv == {"partition": "training", "folds": folds, "stratified": True,
                   "shuffle": True, "random_state": seed}
            and search.get("sampler") == {"name": "TPESampler", "seed": seed}
            and search.get("budget", {}).get("n_jobs") == 1, "Unsupported search protocol.")
    outer, inner = evaluation.get("outer_cv", {}), evaluation.get("inner_cv", {})
    require(outer.get("partition") == "training" and outer.get("stratified") is True
            and outer.get("shuffle") is True and outer.get("random_state") == seed
            and type(outer.get("folds")) is int and 2 <= outer["folds"] <= 10
            and inner.get("partition") == "outer training fold" and inner.get("folds") == folds
            and inner.get("random_state") == seed and inner.get("stratified") is True
            and inner.get("shuffle") is True, "Nested evaluation must use the aligned training-only protocol.")
    require(set(evaluation.get("procedures", {})) == {"automl", "baseline_fixed"}
            and bool(evaluation.get("comparison", {}).get("metrics"))
            and len(evaluation.get("folds", [])) == 2 * outer["folds"],
            "Complete nested baseline comparison evidence is required.")
    for procedure in evaluation["procedures"].values():
        metrics = procedure.get("pr_auc", {})
        require(all(type(metrics.get(key)) in (int, float) and math.isfinite(metrics[key])
                    and 0 <= metrics[key] <= 1 for key in ("mean", "std")),
                "Nested evaluation needs finite AP summaries for both procedures.")
    protocol = evaluation.get("automl", {})
    require(protocol.get("sampler") == search.get("sampler")
            and protocol.get("budget_per_outer_fold") == search.get("budget"),
            "Nested evaluation must assess the same sampler and search budget.")
    trials = [trial for trial in search.get("trial_records", []) if trial.get("number") == trial_number]
    require(len(trials) == 1 and trials[0].get("state") == "COMPLETE", "Choose an explicit completed trial.")
    trial = trials[0]
    require(trial.get("model") in automl.FAMILIES and isinstance(trial.get("sampled_params"), dict)
            and trial["sampled_params"].get("model") == trial["model"]
            and trial["model"] in protocol.get("families", []), "Invalid candidate family or parameters.")
    scores = trial.get("cv_ap", {}).get("folds", [])
    require(len(scores) == folds and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1
                                        for v in scores), "Invalid candidate CV scores.")
    require(trial["cv_ap"] == automl.summarize(scores) and trial.get("value") == trial["cv_ap"]["mean"],
            "Inconsistent candidate CV summary.")
    return trial, seed, folds


def prepare(search_path: Path, evaluation_path: Path, *, trial_number: int,
            data_path: Path = DATA_PATH, models_dir: Path = MODELS_DIR) -> Bundle:
    """Fit one explicitly selected configuration on the original training partition; never activate it."""
    models = models_root(models_dir)
    search, search_digest = document(search_path)
    evaluation, evaluation_digest = document(evaluation_path)
    data_digest = sha256(data_path)
    X, y = automl.training_partition(load_data(data_path))
    require(sha256(data_path) == data_digest, "Data changed while preparing the candidate.")
    trial, seed, folds = validate_sources(search, evaluation, X, y, data_digest, trial_number)
    params = automl.readable(trial["sampled_params"])
    pipeline = automl.candidate_pipeline(X, trial["model"], params, seed)
    estimator = pipeline.named_steps["model"]
    require(estimator.get_params(deep=False) == trial.get("estimator_params")
            and f"{type(estimator).__module__}.{type(estimator).__name__}" == trial.get("estimator_class"),
            "Reconstructed estimator does not match recorded parameters/class.")
    with threadpool_limits(limits=1):
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        thresholds = threshold_table(y, oof)
        threshold = select_threshold(thresholds)
        pipeline.fit(X, y)
    path = local_path(models, "runs", str(uuid4()))
    metadata = {
        "schema_version": 2, "run_id": path.name, "model_origin": "automl",
        "selected_model": trial["model"], "decision_threshold": threshold,
        "feature_columns": FEATURE_COLUMNS, "target_mapping": TARGET_MAPPING,
        "random_state": seed, "data_sha256": data_digest, "train_rows": len(X),
        "trained_on": "original training partition only; no holdout evaluation or refitting",
        "package_versions": {name: value for name, value in search["package_versions"].items() if value is not None},
        "promotion": {"selected_trial": trial_number, "params": params,
                      "search_sha256": search_digest, "evaluation_sha256": evaluation_digest,
                      "training_data": search["training_data"], "split_seed": RANDOM_STATE,
                      "threshold_cv": search["cv"], "holdout_used": False,
                      "preparation_provenance": automl_reporting.provenance()},
    }
    pipeline.attrition_metadata_ = metadata
    report = {**metadata, "selected_threshold": threshold, "selected_model_metrics": None,
              "selection_reason": "Candidate trial chosen explicitly by the operator; activation requires review.",
              "threshold_rule": "Maximize training OOF F2; ties: precision, then higher threshold. "
                                "These reused-fold diagnostics are not unbiased performance estimates.",
              "development_evaluation": {"procedures": evaluation["procedures"],
                                         "comparison": evaluation["comparison"],
                                         "interpretation": evaluation.get("interpretation", "Training-only development estimates.")},
              "candidate_cv_ap": trial["cv_ap"], "shap": {"status": "not_requested"}}
    path.mkdir(parents=True, exist_ok=False)
    joblib.dump(pipeline, path / "pipeline.joblib")
    write_json(metadata, path / "metadata.json")
    write_json(report, path / "metrics/model_metrics.json")
    thresholds.to_csv(path / "metrics/threshold_analysis.csv", index=False)
    write_json(search, path / "evidence/search.json")
    write_json(evaluation, path / "evidence/evaluation.json")
    write_json(X[FEATURE_COLUMNS].iloc[0].to_dict(), path / "employee.json")
    background = X[FEATURE_COLUMNS].sample(min(100, len(X)), random_state=seed)
    write_json(json.loads(background.to_json(orient="records")), path / "background.json")
    write_json(reference_profile(X[FEATURE_COLUMNS]), path / PROFILE_FILE)
    return seal_bundle(path)


def candidate(models: Path, candidate_id: str) -> Bundle:
    bundle = load_bundle(local_path(models, "runs", run_id(candidate_id)))
    require(bundle.report.get("model_origin") == "automl"
            and bundle.report.get("promotion", {}).get("holdout_used") is False
            and bundle.report.get("selected_model_metrics") is None,
            "Only a prepared AutoML candidate may enter the approval gate.")
    return bundle


def pointer_document(bundle: Bundle) -> dict:
    return {"schema_version": 2, "run_id": bundle.run_id,
            "manifest_sha256": sha256(bundle.path / "manifest.json")}


def current(models: Path) -> dict:
    pointer = models / "current.json"
    value, _ = document(pointer)
    require(value == pointer_document(load_bundle(pointer)), "Invalid current pointer snapshot.")
    return value


def review(candidate_id: str, models_dir: Path = MODELS_DIR) -> dict:
    """An unapproved draft bound to both candidate and rollback bundle identities."""
    models = models_root(models_dir)
    bundle = candidate(models, candidate_id)
    return {"decision": "pending", **pointer_document(bundle),
            "approval_schema_version": 1, "expected_current": current(models),
            "reviewer": "", "reason": "", "reviewed_at": "",
            "checks": dict.fromkeys(CHECKS, False), "evidence": dict.fromkeys(EVIDENCE, "")}


@contextmanager
def promotion_lock(models: Path):
    """Serialize these operator commands. Stop other writers (training/pruning) during maintenance."""
    lock = local_path(models, "", ".promotion.lock")
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Promotion in progress; inspect pointer and receipts before removing a stale lock.\n")
    try:
        yield
    finally:
        lock.unlink()


def switch(models: Path, target: Bundle, expected: dict, *, action: str, authorization: dict) -> Path:
    """Persist a recovery receipt before atomic publication; keep it even if publication fails."""
    require(current(models) == expected, "Current pointer changed; repeat review against the current run.")
    after = pointer_document(target)
    require(after != expected, "Target is already current.")
    receipt = local_path(models, "promotions", f"{uuid4()}.json")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "action": action, "created_at": datetime.now(UTC).isoformat(),
             "before": expected, "after": after, "authorization": authorization}
    with receipt.open("x", encoding="utf-8") as handle:
        handle.write(automl_reporting.json_text(value))
    require(current(models) == expected, "Current pointer changed before publication; no switch performed.")
    publish_bundle(target.path, models / "current.json")
    return receipt


def activate(approval_path: Path, models_dir: Path = MODELS_DIR) -> Path:
    models = models_root(models_dir)
    approval, digest = document(approval_path)
    require(approval.get("approval_schema_version") == 1 and approval.get("decision") == "approve",
            "Explicit manual approval is required.")
    for name in ("reviewer", "reason", "reviewed_at"):
        require(isinstance(approval.get(name), str) and bool(approval[name].strip()), f"Missing review {name}.")
    reviewed_at = datetime.fromisoformat(approval["reviewed_at"])
    require(reviewed_at.tzinfo is not None, "Review timestamp needs a timezone.")
    require(all(approval.get("checks", {}).get(name) is True for name in CHECKS),
            "Every manual review check must be explicitly confirmed.")
    require(all(isinstance(approval.get("evidence", {}).get(name), str)
                and bool(approval["evidence"][name].strip()) for name in EVIDENCE),
            "Source review and API/dashboard smoke evidence are required.")
    require(isinstance(approval.get("run_id"), str) and isinstance(approval.get("expected_current"), dict),
            "Approval must identify both the candidate and the expected current pointer.")
    with promotion_lock(models):
        target = candidate(models, approval["run_id"])
        identity = pointer_document(target)
        require(all(approval.get(key) == value for key, value in identity.items()),
                "Approval does not match the candidate manifest.")
        return switch(models, target, approval["expected_current"], action="activate",
                      authorization={"approval_sha256": digest, "review": approval})


def rollback(promotion_id: str, *, expected_current: str, reviewer: str, reason: str,
             models_dir: Path = MODELS_DIR) -> Path:
    """Restore the exact previous bundle from an activation receipt; never overwrite a newer run."""
    models = models_root(models_dir)
    require(bool(reviewer.strip()) and bool(reason.strip()), "Rollback reviewer and reason are required.")
    receipt, digest = document(local_path(models, "promotions", f"{run_id(promotion_id)}.json"))
    require(receipt.get("schema_version") == 1 and receipt.get("action") == "activate",
            "Rollback requires an activation receipt.")
    with promotion_lock(models):
        before, after = receipt["before"], receipt["after"]
        require(after["run_id"] == run_id(expected_current), "Rollback expected-current does not match receipt.")
        target = load_bundle(local_path(models, "runs", run_id(before["run_id"])))
        require(pointer_document(target) == before, "Rollback bundle changed since approval.")
        return switch(models, target, after, action="rollback",
                      authorization={"promotion_id": promotion_id, "receipt_sha256": digest,
                                     "reviewer": reviewer, "reason": reason})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=normalize_path, default=MODELS_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Fit an explicit candidate without changing current.json")
    prepare_parser.add_argument("--search", type=normalize_path, required=True)
    prepare_parser.add_argument("--evaluation", type=normalize_path, required=True)
    prepare_parser.add_argument("--trial", type=int, required=True)
    prepare_parser.add_argument("--data", type=normalize_path, default=DATA_PATH)
    commands.add_parser("review", help="Print an unapproved review draft").add_argument("--run-id", required=True)
    commands.add_parser("activate", help="Activate a manually approved candidate").add_argument(
        "--approval", type=normalize_path, required=True)
    rollback_parser = commands.add_parser("rollback", help="Restore a previous bundle from an activation receipt")
    for option in ("promotion-id", "expected-current", "reviewer", "reason"):
        rollback_parser.add_argument(f"--{option}", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        bundle = prepare(args.search, args.evaluation, trial_number=args.trial, data_path=args.data,
                         models_dir=args.models_dir)
        result = {"run_id": bundle.run_id, "path": str(bundle.path), "activated": False}
    elif args.command == "review":
        result = review(args.run_id, args.models_dir)
    elif args.command == "activate":
        result = {"receipt": str(activate(args.approval, args.models_dir))}
    else:
        result = {"receipt": str(rollback(args.promotion_id, expected_current=args.expected_current,
                                         reviewer=args.reviewer, reason=args.reason, models_dir=args.models_dir))}
    print(automl_reporting.json_text(result), end="")


if __name__ == "__main__":
    main()
