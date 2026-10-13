"""Incompatible, mixed, missing, or corrupt artifacts are refused before anything is served."""

import copy
import shutil
import warnings
from uuid import uuid4

import joblib
import numpy as np
import pytest
from sklearn.exceptions import InconsistentVersionWarning

from attrition.config import FEATURE_COLUMNS
from attrition.models.artifacts import (
    active_run_path,
    load_bundle,
    load_legacy_pipeline,
    load_pickle,
    publish_bundle,
    read_json,
    seal_bundle,
    validate_pipeline,
    write_json,
)
from attrition.models.prune import prune_runs


def with_metadata(pipeline, **changes):
    clone = copy.deepcopy(pipeline)
    clone.attrition_metadata_ = {**clone.attrition_metadata_, **changes}
    return clone


@pytest.mark.parametrize("threshold", [0, 1, 0.0, 0.5, 1.0])
def test_threshold_endpoints_are_valid(fitted_pipeline, threshold):
    assert validate_pipeline(with_metadata(fitted_pipeline, decision_threshold=threshold))["decision_threshold"] == threshold


@pytest.mark.parametrize("threshold", [-0.01, 1.01, 2, float("nan"), float("inf"), -float("inf"),
                                       True, False, "0.5", None])
def test_invalid_thresholds_are_rejected(fitted_pipeline, threshold):
    with pytest.raises(ValueError, match="threshold"):
        validate_pipeline(with_metadata(fitted_pipeline, decision_threshold=threshold))


@pytest.mark.parametrize("version", [0, 3, "2", None, 1.5])
def test_unknown_schema_versions_are_rejected(fitted_pipeline, version):
    with pytest.raises(ValueError, match="schema"):
        validate_pipeline(with_metadata(fitted_pipeline, schema_version=version))


@pytest.mark.parametrize("columns", [FEATURE_COLUMNS[::-1], FEATURE_COLUMNS[:-1], [*FEATURE_COLUMNS, "Extra"], None])
def test_feature_columns_must_match_exactly_and_in_order(fitted_pipeline, columns):
    with pytest.raises(ValueError, match="schema"):
        validate_pipeline(with_metadata(fitted_pipeline, feature_columns=columns))


def test_class_order_must_be_negative_then_positive(fitted_pipeline):
    flipped = copy.deepcopy(fitted_pipeline)
    flipped.steps[-1][1].classes_ = np.array([1, 0])
    with pytest.raises(ValueError, match="classes"):
        validate_pipeline(flipped)


def test_objects_without_metadata_are_not_pipelines(fitted_pipeline):
    bare = copy.deepcopy(fitted_pipeline)
    del bare.attrition_metadata_
    for candidate in (bare, {"pipeline": 1}, None, fitted_pipeline.steps[-1][1]):
        with pytest.raises(ValueError, match="Invalid model artifact"):
            validate_pipeline(candidate)
    bare.attrition_metadata_ = "not a dict"
    with pytest.raises(ValueError, match="Invalid model artifact"):
        validate_pipeline(bare)


def test_artifacts_from_another_scikit_learn_version_are_refused(tmp_path, monkeypatch):
    def stale_load(path):
        warnings.warn(InconsistentVersionWarning(estimator_name="Pipeline", current_sklearn_version="9.9",
                                                 original_sklearn_version="1.0"), stacklevel=1)
        return "unreachable"

    monkeypatch.setattr("attrition.models.artifacts.joblib.load", stale_load)
    with pytest.raises(InconsistentVersionWarning):
        load_pickle(tmp_path / "any.joblib")


def test_legacy_loader_returns_only_validated_schema_one_pipelines(fitted_pipeline, tmp_path):
    good = tmp_path / "good.joblib"
    joblib.dump(fitted_pipeline, good)
    assert load_legacy_pipeline(good).attrition_metadata_["schema_version"] == 1
    for name, artifact in {"dict": {"a": 1}, "wrong_schema": with_metadata(fitted_pipeline, schema_version=2),
                           "bad_columns": with_metadata(fitted_pipeline, feature_columns=["Age"])}.items():
        path = tmp_path / f"{name}.joblib"
        joblib.dump(artifact, path)
        with pytest.raises(ValueError):
            load_legacy_pipeline(path)


@pytest.mark.parametrize("content", [b"", b"not a pickle", b"\x80\x04\x95"])
def test_corrupt_legacy_files_never_load(tmp_path, content):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(content)
    with pytest.raises(Exception):  # noqa: B017 - the exception type is not yet a stable contract
        load_legacy_pipeline(path)


@pytest.mark.xfail(strict=True, reason="Known gap: corrupt pickles surface raw struct/Key/EOF errors, not ValueError")
@pytest.mark.parametrize("content", [b"", b"not a pickle", b"\x80\x04\x95"])
def test_corrupt_legacy_files_should_raise_a_clear_value_error(tmp_path, content):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        load_legacy_pipeline(path)


@pytest.fixture
def pointer(tmp_path):
    return tmp_path / "models/current.json"


def edit_manifest(bundle, mutate):
    manifest = read_json(bundle.path / "manifest.json")
    mutate(manifest)
    write_json(manifest, bundle.path / "manifest.json")


def test_pointer_requires_schema_two_canonical_run_id_and_matching_manifest(bundle_factory, pointer):
    bundle = bundle_factory()
    good = read_json(pointer)
    assert active_run_path(pointer) == bundle.path
    for change, message in [({"schema_version": 1}, "Invalid active-run pointer"),
                            ({"schema_version": None}, "Invalid active-run pointer"),
                            ({"run_id": str(uuid4())}, "manifest.json"),
                            ({"run_id": bundle.run_id.upper()}, "Invalid active-run pointer"),
                            ({"run_id": None}, "Invalid active-run pointer"),
                            ({"run_id": 7}, "Invalid active-run pointer"),
                            ({"run_id": "../" + bundle.run_id}, "Invalid active-run pointer"),
                            ({"manifest_sha256": "0" * 64}, "manifest checksum"),
                            ({"manifest_sha256": None}, "manifest checksum")]:
        write_json({**good, **change}, pointer)
        with pytest.raises((ValueError, FileNotFoundError), match=message):
            active_run_path(pointer)
    write_json({key: value for key, value in good.items() if key != "manifest_sha256"}, pointer)
    with pytest.raises(ValueError, match="manifest checksum"):
        active_run_path(pointer)


@pytest.mark.parametrize("text", ["[]", "null", '"text"', "5"])
def test_pointer_that_is_not_an_object_is_never_followed(bundle_factory, pointer, text):
    bundle_factory()
    pointer.write_text(text, encoding="utf-8")
    with pytest.raises(Exception):  # noqa: B017 - see the strict xfail below for the intended contract
        active_run_path(pointer)
    with pytest.raises(Exception):  # noqa: B017
        prune_runs(1, pointer=pointer)
    assert len(list(pointer.parent.joinpath("runs").iterdir())) == 1


@pytest.mark.xfail(strict=True, reason="Known gap: non-object pointer JSON raises AttributeError, not ValueError")
@pytest.mark.parametrize("text", ["[]", "null", '"text"', "5"])
def test_pointer_that_is_not_an_object_should_raise_the_documented_error(bundle_factory, pointer, text):
    bundle_factory()
    pointer.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid active-run pointer"):
        active_run_path(pointer)


def test_malformed_pointer_json_removes_nothing(bundle_factory, pointer):
    bundle_factory()
    pointer.write_text("{truncated", encoding="utf-8")
    with pytest.raises(ValueError):
        active_run_path(pointer)
    with pytest.raises(ValueError):
        prune_runs(1, pointer=pointer)
    assert len(list(pointer.parent.joinpath("runs").iterdir())) == 1


def test_pointer_to_a_removed_run_is_an_error_not_a_fallback(bundle_factory, pointer):
    older = bundle_factory()
    newer = bundle_factory(activate=False)
    publish_bundle(newer.path, pointer)
    shutil.rmtree(newer.path)
    with pytest.raises(FileNotFoundError):
        load_bundle(pointer)
    assert older.path.is_dir()


def test_manifest_version_and_run_id_must_match_the_directory(bundle_factory):
    bundle = bundle_factory()
    original = read_json(bundle.path / "manifest.json")
    for change in ({"schema_version": 1}, {"schema_version": "2"}, {"run_id": str(uuid4())}, {"run_id": None}):
        write_json({**original, **change}, bundle.path / "manifest.json")
        with pytest.raises(ValueError, match="Invalid bundle"):
            load_bundle(bundle.path)
    write_json(original, bundle.path / "manifest.json")
    assert load_bundle(bundle.path).run_id == bundle.run_id


def test_renamed_bundle_directory_is_refused(bundle_factory):
    bundle = bundle_factory()
    moved = bundle.path.with_name(str(uuid4()))
    bundle.path.rename(moved)
    with pytest.raises(ValueError, match="run ID"):
        load_bundle(moved)


def test_manifest_must_list_every_required_file(bundle_factory):
    bundle = bundle_factory()
    for required in ("pipeline.joblib", "metadata.json", "metrics/model_metrics.json", "employee.json",
                     "background.json"):
        edit_manifest(bundle, lambda manifest, name=required: manifest["files"].pop(name))
        with pytest.raises(ValueError, match="Incomplete"):
            load_bundle(bundle.path)
        seal_bundle(bundle.path)
    edit_manifest(bundle, lambda manifest: manifest.update(files=["pipeline.joblib"]))
    with pytest.raises(ValueError, match="Incomplete"):
        load_bundle(bundle.path)


def test_unlisted_and_escaping_files_are_refused(bundle_factory, tmp_path):
    bundle = bundle_factory()
    (bundle.path / "extra.txt").write_text("not in the manifest", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        load_bundle(bundle.path)
    (bundle.path / "extra.txt").unlink()
    (tmp_path / "outside.json").write_text("{}", encoding="utf-8")
    edit_manifest(bundle, lambda manifest: manifest["files"].update({"../../outside.json": "0" * 64}))
    with pytest.raises(ValueError, match="do not match"):
        load_bundle(bundle.path)


def test_a_pipeline_from_another_run_cannot_be_resealed_into_a_bundle(bundle_factory):
    first = bundle_factory()
    second = bundle_factory(activate=False)
    shutil.copyfile(second.path / "pipeline.joblib", first.path / "pipeline.joblib")
    with pytest.raises(ValueError, match="do not match"):
        seal_bundle(first.path)


def test_report_threshold_must_match_the_model_even_with_fresh_checksums(bundle_factory):
    bundle = bundle_factory()
    report = read_json(bundle.path / "metrics/model_metrics.json")
    report["selected_threshold"] = 0.99
    write_json(report, bundle.path / "metrics/model_metrics.json")
    with pytest.raises(ValueError, match="do not match"):
        seal_bundle(bundle.path)


def test_report_values_that_disagree_with_metadata_are_refused(bundle_factory):
    bundle = bundle_factory()
    report = read_json(bundle.path / "metrics/model_metrics.json")
    report["selected_model"] = "random_forest"
    write_json(report, bundle.path / "metrics/model_metrics.json")
    with pytest.raises(ValueError, match="do not match"):
        seal_bundle(bundle.path)


@pytest.mark.parametrize("name", ["employee.json", "background.json"])
def test_embedded_examples_must_satisfy_the_input_schema(bundle_factory, name):
    bundle = bundle_factory()
    value = read_json(bundle.path / name)
    record = value if isinstance(value, dict) else value[0]
    record["Age"] = 5
    write_json(value, bundle.path / name)
    with pytest.raises(ValueError, match="allowed range"):
        seal_bundle(bundle.path)


def test_a_bundle_cannot_be_published_from_outside_the_pointers_runs_directory(bundle_factory, tmp_path):
    bundle = bundle_factory(activate=False)
    elsewhere = tmp_path / "other/current.json"
    with pytest.raises(ValueError, match="inside"):
        publish_bundle(bundle.path, elsewhere)
    assert not elsewhere.parent.exists()


def test_write_json_refuses_non_finite_numbers_and_leaves_no_partial_file(tmp_path):
    target = tmp_path / "nested/report.json"
    with pytest.raises(ValueError):
        write_json({"metric": float("nan")}, target)
    assert not target.exists()


def test_json_roundtrip_is_utf8_with_trailing_newline(tmp_path):
    target = tmp_path / "nested/report.json"
    value = {"label": "Café – naïve", "n": 3}
    write_json(value, target)
    assert read_json(target) == value
    assert target.read_bytes().endswith(b"\n")
    target.read_bytes().decode("utf-8")
