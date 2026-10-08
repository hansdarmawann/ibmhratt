"""Run retention never touches the active run or unrelated folders."""

import os

import pytest

from attrition.models.artifacts import write_json
from attrition.models.prune import prune_runs


@pytest.fixture
def four_runs(bundle_factory, tmp_path):
    """Oldest run is active; modification times increase from first to last."""
    runs = [bundle_factory(activate=index == 0) for index in range(4)]
    for age, bundle in enumerate(runs):
        os.utime(bundle.path, (1_000_000 + age, 1_000_000 + age))
    (tmp_path / "models/runs/notes").mkdir()
    return runs


def remaining(tmp_path) -> set[str]:
    return {path.name for path in (tmp_path / "models/runs").iterdir()}


def test_keeps_newest_and_active_runs(four_runs, tmp_path):
    pointer = tmp_path / "models/current.json"
    preview = prune_runs(1, pointer=pointer, dry_run=True)
    assert [run.name for run in preview] == [four_runs[2].run_id, four_runs[1].run_id]
    assert len(remaining(tmp_path)) == 5
    removed = prune_runs(1, pointer=pointer)
    assert removed == preview
    assert remaining(tmp_path) == {four_runs[0].run_id, four_runs[3].run_id, "notes"}
    assert prune_runs(1, pointer=pointer) == []


def test_without_pointer_keeps_only_newest(four_runs, tmp_path):
    (tmp_path / "models/current.json").unlink()
    prune_runs(2, pointer=tmp_path / "models/current.json")
    assert remaining(tmp_path) == {four_runs[2].run_id, four_runs[3].run_id, "notes"}


def test_invalid_pointer_removes_nothing(four_runs, tmp_path):
    pointer = tmp_path / "models/current.json"
    write_json({"schema_version": 2, "run_id": "not-a-uuid"}, pointer)
    with pytest.raises(ValueError, match="pointer"):
        prune_runs(1, pointer=pointer)
    assert len(remaining(tmp_path)) == 5


def test_keep_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        prune_runs(0, pointer=tmp_path / "models/current.json")
