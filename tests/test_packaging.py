"""Build the wheel from a minimal source copy, install it elsewhere, and run it outside the checkout."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from attrition import __version__, config

CHECKOUT = Path(__file__).resolve().parents[1]
# Everything the build backend reads; building from this copy proves nothing else is needed.
BUILD_INPUTS = ["pyproject.toml", "requirements-serving.txt", "README.md"]


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, **kwargs)
    assert result.returncode == 0, f"{command} failed:\n{result.stdout}\n{result.stderr}"
    return result


def local_backend_available() -> bool:
    """Use the installed setuptools when it satisfies pyproject; otherwise pip fetches it in isolation."""
    try:
        return int(importlib.metadata.version("setuptools").split(".")[0]) >= 75
    except importlib.metadata.PackageNotFoundError:
        return False


def requirements(lines) -> set[tuple[str, str]]:
    parsed = [Requirement(line) for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return {(requirement.name, str(requirement.specifier)) for requirement in parsed}


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    tree = tmp_path_factory.mktemp("source")
    for name in BUILD_INPUTS:
        shutil.copy2(CHECKOUT / name, tree / name)
    shutil.copytree(CHECKOUT / "src", tree / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    output = tmp_path_factory.mktemp("dist")
    command = [sys.executable, "-m", "pip", "wheel", str(tree), "--no-deps", "--wheel-dir", str(output), "--quiet"]
    if local_backend_available():
        command.append("--no-build-isolation")
    run(command)
    (path,) = output.glob("ibmhratt-*.whl")
    return path


def test_wheel_contains_package_and_serving_dependencies(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        info = f"ibmhratt-{__version__}.dist-info/"
        metadata = Parser().parsestr(archive.read(info + "METADATA").decode("utf-8"))
    assert {"attrition/config.py", "attrition/models/predict.py", "attrition/models/train.py"} <= set(names)
    assert all(name.startswith(("attrition/", info)) for name in names), "Only the package belongs in the wheel"
    assert metadata["Version"] == __version__
    serving = (CHECKOUT / "requirements-serving.txt").read_text(encoding="utf-8").splitlines()
    assert requirements(metadata.get_all("Requires-Dist")) == requirements(serving)


def test_installed_wheel_runs_outside_checkout(wheel, bundle_factory, employee, tmp_path):
    site, work = tmp_path / "site", tmp_path / "work"
    work.mkdir()
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--quiet",
         "--target", str(site), str(wheel)])
    # PYTHONPATH precedes site-packages, so the wheel's copy shadows any editable checkout install.
    env = {key: value for key, value in os.environ.items()
           if key not in (config.ROOT_ENV, config.DATA_ENV, config.MODELS_ENV)}
    env["PYTHONPATH"] = str(site)
    probe = run([sys.executable, "-c", "import json; from attrition import config; print(json.dumps("
                 "[config.__file__, config.CHECKOUT_ROOT and str(config.CHECKOUT_ROOT), str(config.ROOT)]))"],
                env=env, cwd=work)
    module, checkout_root, root = json.loads(probe.stdout)
    assert Path(module).resolve().is_relative_to(site.resolve())
    assert checkout_root is None
    assert Path(root) == work.resolve()

    bundle = bundle_factory()
    request = work / "employee.json"
    request.write_text(json.dumps(employee), encoding="utf-8")
    result = run([sys.executable, "-m", "attrition.models.predict", str(request), "--model", str(bundle.path)],
                 env=env, cwd=work)
    prediction = json.loads(result.stdout)
    assert prediction["run_id"] == bundle.run_id
    assert prediction["predicted_class"] in {"Yes", "No"}
