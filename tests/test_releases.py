"""Release gates reject invalid refs and unreviewed commits without Git writes."""

import subprocess
from pathlib import Path

import pytest

from scripts import check_release


@pytest.fixture
def release_tree(tmp_path):
    source = tmp_path / "src/attrition"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text('__version__ = "1.1.0"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("## [1.1.0] - 2026-10-13\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def reviewed_git(monkeypatch):
    """Instrument read-only Git calls; no test creates commits, tags, or branches."""
    commands = []

    def run(command, **kwargs):
        commands.append(command[1:])
        if command[1] == "rev-parse":
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
        assert command[1:] == ["merge-base", "--is-ancestor", "a" * 40, "origin/main"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(check_release.subprocess, "run", run)
    return commands


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/tags/v1.1.0"])
def test_reviewed_publication_is_eligible(release_tree, reviewed_git, ref):
    assert check_release.check_release(release_tree, ref) == "1.1.0"
    assert reviewed_git[-1][0] == "merge-base"
    assert all(command[0] in {"rev-parse", "merge-base"} for command in reviewed_git)


@pytest.mark.parametrize("ref", [
    "refs/heads/feature", "refs/pull/10/merge", "v1.1.0", "refs/tags/v9.0.0",
    "refs/tags/v1.1.0-rc.1", "refs/tags/v1.1.0+build", "refs/tags/v01.1.0",
    "refs/tags/vanything", "refs/tags/v1.1", "refs/tags/v1.1.0/extra",
])
def test_bad_ref_rejected_before_git(release_tree, reviewed_git, ref):
    with pytest.raises(ValueError, match="Publication requires"):
        check_release.check_release(release_tree, ref)
    assert reviewed_git == []


@pytest.mark.parametrize("version", ["01.1.0", "1.1", "1.1.0rc1", "1.1.0+build", ""])
def test_invalid_source_version_fails_closed(release_tree, reviewed_git, version):
    (release_tree / "src/attrition/__init__.py").write_text(f"__version__ = {version!r}", encoding="utf-8")
    with pytest.raises(ValueError, match="Package version"):
        check_release.check_release(release_tree, "refs/heads/main")
    assert reviewed_git == []


def test_version_reader_does_not_execute_package(release_tree):
    (release_tree / "src/attrition/__init__.py").write_text(
        'raise RuntimeError("must not import")\n__version__ = "1.1.0"\n', encoding="utf-8"
    )
    assert check_release.package_version(release_tree) == "1.1.0"


def test_unmerged_tag_or_missing_main_fails_closed(release_tree, monkeypatch):
    def run(command, **kwargs):
        if command[1] == "rev-parse":
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(check_release.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        check_release.check_release(release_tree, "refs/tags/v1.1.0")


def test_wrong_checkout_fails_before_ancestry(release_tree, monkeypatch):
    def run(command, **kwargs):
        assert command[1] == "rev-parse"
        sha = "a" if command[-1] == "HEAD^{commit}" else "b"
        return subprocess.CompletedProcess(command, 0, sha * 40 + "\n", "")

    monkeypatch.setattr(check_release.subprocess, "run", run)
    with pytest.raises(ValueError, match="HEAD does not match"):
        check_release.check_release(release_tree, "refs/tags/v1.1.0")


@pytest.mark.parametrize("notes", ["## [Unreleased]\n", "## [1.0.0] - 2026-10-13\n", "## [1.1.0]\n"])
def test_tag_requires_dated_release_notes(release_tree, reviewed_git, notes):
    (release_tree / "CHANGELOG.md").write_text(notes, encoding="utf-8")
    with pytest.raises(ValueError, match="dated changelog"):
        check_release.check_release(release_tree, "refs/tags/v1.1.0")


def test_main_does_not_require_a_release_entry(release_tree, reviewed_git):
    (release_tree / "CHANGELOG.md").write_text("## [Unreleased]\n", encoding="utf-8")
    assert check_release.check_release(release_tree, "refs/heads/main") == "1.1.0"


def test_real_checkout_version_agrees_with_package():
    from attrition import __version__

    assert check_release.package_version(Path(__file__).resolve().parents[1]) == __version__
