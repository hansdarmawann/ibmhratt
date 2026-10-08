"""A failed or mixed experiment must never replace a healthy active run."""

import copy

import joblib
import pytest

from attrition.models.artifacts import load_bundle, publish_bundle, read_json, seal_bundle, write_json
from attrition.models.predict import load_pipeline, predict


def test_bundle_roundtrip_and_atomic_switch(bundle_factory, tmp_path, employee):
    first = bundle_factory()
    pointer = tmp_path / "models/current.json"
    loaded = load_bundle(pointer)
    assert loaded.run_id == first.run_id
    assert predict(employee, pipeline=first.pipeline) == predict(employee, pipeline=load_pipeline(pointer))
    second = bundle_factory(threshold=0.8, activate=False)
    assert load_bundle(pointer).run_id == first.run_id
    publish_bundle(second.path, pointer)
    assert load_bundle(pointer).run_id == second.run_id
    assert load_bundle(first.path).pipeline.attrition_metadata_["decision_threshold"] == 0.35


@pytest.mark.parametrize("name", ["pipeline.joblib", "metadata.json", "metrics/model_metrics.json", "employee.json"])
def test_corruption_prevents_publication(bundle_factory, tmp_path, name):
    first = bundle_factory()
    second = bundle_factory(activate=False)
    target = second.path / name
    target.write_bytes(target.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        publish_bundle(second.path, tmp_path / "models/current.json")
    assert load_bundle(tmp_path / "models/current.json").run_id == first.run_id


def test_metadata_mismatch_even_with_updated_checksums(bundle_factory):
    bundle = bundle_factory()
    metadata = read_json(bundle.path / "metadata.json")
    metadata["decision_threshold"] = 0.7
    write_json(metadata, bundle.path / "metadata.json")
    with pytest.raises(ValueError, match="do not match"):
        seal_bundle(bundle.path)


def test_missing_file_and_changed_manifest(bundle_factory, tmp_path):
    first = bundle_factory()
    (first.path / "background.json").unlink()
    with pytest.raises(ValueError, match="files"):
        load_bundle(first.path)
    second = bundle_factory()
    manifest = second.path / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest checksum"):
        load_bundle(tmp_path / "models/current.json")


def test_failed_pointer_write_keeps_old_run(bundle_factory, tmp_path, monkeypatch):
    first = bundle_factory()
    second = bundle_factory(activate=False)

    def fail_replace(*args):
        raise OSError("Simulated interrupted publication")

    monkeypatch.setattr("pathlib.Path.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        publish_bundle(second.path, tmp_path / "models/current.json")
    assert load_bundle(tmp_path / "models/current.json").run_id == first.run_id
    assert not list((tmp_path / "models").glob(".current-*.tmp"))


def test_failed_training_keeps_old_run(bundle_factory, tmp_path, monkeypatch):
    from attrition.models.train import train

    first = bundle_factory()

    def fail_loading(*args):
        raise ValueError("Invalid training data")

    monkeypatch.setattr("attrition.models.train.load_data", fail_loading)
    with pytest.raises(ValueError, match="training data"):
        train(output_root=tmp_path)
    assert load_bundle(tmp_path / "models/current.json").run_id == first.run_id


def test_v2_requires_bundle_and_legacy_schema_is_validated(bundle_factory, fitted_pipeline, tmp_path):
    bundle = bundle_factory()
    with pytest.raises(ValueError, match="verified bundle"):
        load_pipeline(bundle.path / "pipeline.joblib")
    invalid = copy.deepcopy(fitted_pipeline)
    invalid.attrition_metadata_["decision_threshold"] = float("nan")
    path = tmp_path / "invalid.joblib"
    joblib.dump(invalid, path)
    with pytest.raises(ValueError, match="threshold"):
        load_pipeline(path)


def test_artifact_from_renamed_package_asks_for_retraining(tmp_path):
    # Protocol-0 pickle of a function at the pre-rename module path.
    path = tmp_path / "old.joblib"
    path.write_bytes(b"csrc.features.preprocess\nnormalize_missing\np0\n.")
    with pytest.raises(ValueError, match="retrain"):
        load_pipeline(path)
