"""Read-only release eligibility checks; never tags, trains, or publishes anything."""

import argparse
import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def package_version(root: Path) -> str:
    """Read the authoritative version without importing the training/serving stack."""
    module = ast.parse((root / "src/attrition/__init__.py").read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            version = ast.literal_eval(node.value)
            if isinstance(version, str) and VERSION.fullmatch(version):
                return version
            break
    raise ValueError("Package version must be a stable MAJOR.MINOR.PATCH string.")


def validate_ref(ref: str, version: str) -> None:
    if ref == "refs/heads/main":
        return
    if ref != f"refs/tags/v{version}" or not VERSION.fullmatch(version):
        raise ValueError("Publication requires main or a stable vMAJOR.MINOR.PATCH tag matching __version__.")


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def check_release(root: Path, ref: str, main_ref: str = "origin/main") -> str:
    version = package_version(root)
    validate_ref(ref, version)
    head = git(root, "rev-parse", "--verify", "HEAD^{commit}")
    target = git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if head != target:
        raise ValueError("Checked-out HEAD does not match the publication ref.")
    # Fails closed for a missing main ref or a tag on an unmerged feature branch.
    git(root, "merge-base", "--is-ancestor", head, main_ref)
    if ref.startswith("refs/tags/"):
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        if not re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.MULTILINE):
            raise ValueError("A dated changelog section for this release is required before tagging.")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Full publication ref: refs/heads/main or refs/tags/vX.Y.Z")
    parser.add_argument("--main-ref", default="origin/main", help="Locally available reviewed-main ref")
    args = parser.parse_args()
    try:
        version = check_release(ROOT, args.ref, args.main_ref)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Publication blocked: {exc}\n")
    print(f"Publication eligible: {args.ref}, package version {version}; HEAD belongs to {args.main_ref}.")


if __name__ == "__main__":
    main()
