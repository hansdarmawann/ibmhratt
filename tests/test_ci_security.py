"""Static guards for supply-chain and container protections that CI and the image rely on."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github/workflows").glob("*.y*ml"))
PINNED_ACTION = re.compile(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}|\./.+|docker://[^@\s]+@sha256:[0-9a-f]{64}")


def indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def step_containing(lines: list[str], index: int) -> list[str]:
    """The workflow step (list item) that contains lines[index]."""
    start = next(i for i in range(index, -1, -1)
                 if lines[i].lstrip().startswith("- ") and indent(lines[i]) <= indent(lines[index]))
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() and indent(lines[i]) <= indent(lines[start])), len(lines))
    return lines[start:end]


def test_workflows_exist():
    assert WORKFLOWS


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_actions_are_pinned_to_commit_shas(workflow):
    references = re.findall(r"^\s*(?:-\s+)?uses:\s*['\"]?([^\s'\"#]+)", workflow.read_text(encoding="utf-8"),
                            flags=re.MULTILINE)
    unpinned = [reference for reference in references if not PINNED_ACTION.fullmatch(reference)]
    assert not unpinned


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_default_token_permissions_are_read_only(workflow):
    """Write scopes may only be granted per job, never as the workflow default."""
    lines = workflow.read_text(encoding="utf-8").splitlines()
    assert "write-all" not in "\n".join(lines)
    [index] = [i for i, line in enumerate(lines) if re.match(r"permissions:", line)]
    inline = lines[index].split(":", 1)[1].strip()
    if inline:
        assert inline in {"read-all", "{}"}
        return
    scopes = []
    for line in lines[index + 1:]:
        if line.strip() and indent(line) == 0:
            break
        if line.strip() and not line.lstrip().startswith("#"):
            scopes.append(line.split(":", 1)[1].split("#")[0].strip())
    assert scopes and set(scopes) <= {"read", "none"}


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_checkouts_do_not_persist_credentials(workflow):
    lines = workflow.read_text(encoding="utf-8").splitlines()
    checkouts = [i for i, line in enumerate(lines) if re.search(r"uses:\s*actions/checkout@", line)]
    for index in checkouts:
        assert any(re.search(r"persist-credentials:\s*false", line) for line in step_containing(lines, index))


def test_ci_keeps_dependency_audit_image_scan_and_container_smoke_test():
    text = "\n".join(path.read_text(encoding="utf-8") for path in WORKFLOWS)
    assert "pip_audit -r requirements-lock.txt" in text
    assert re.search(r"aquasec/trivy:[\w.]+@sha256:[0-9a-f]{64}", text)
    assert "--severity HIGH,CRITICAL" in text and "--exit-code 1" in text
    assert "python -m scripts.verify_container" in text


def test_image_is_pinned_runs_unprivileged_and_contains_no_api_key():
    lines = [line.strip() for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()]
    instructions = [line for line in lines if line and not line.startswith("#")]
    bases = [line for line in instructions if line.upper().startswith("FROM ")]
    assert bases and all(re.search(r"@sha256:[0-9a-f]{64}\b", line) for line in bases)
    runtime = instructions[instructions.index(bases[-1]):]
    users = [line.split(maxsplit=1)[1] for line in runtime if line.upper().startswith("USER ")]
    assert users and users[-1].split(":")[0] not in {"root", "0"}
    assert any(line.upper().startswith("HEALTHCHECK ") for line in runtime)
    # Keys are runtime secrets: never build arguments or image environment.
    assert "ATTRITION_API_KEY" not in "\n".join(instructions)
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert ".env" in ignored
