"""Remove old experiment bundles: python -m attrition.models.prune --keep 5 [--dry-run].

The active run is never removed. Only directories named by a canonical UUID are
considered, so unrelated folders under models/runs/ are left alone. A running API
keeps serving the bundle it loaded at startup even if that run is pruned later.
"""

import argparse
import logging
import shutil
from pathlib import Path
from uuid import UUID

from attrition.config import CURRENT_RUN
from attrition.models.artifacts import active_run_path

LOGGER = logging.getLogger(__name__)


def is_run_directory(path: Path) -> bool:
    try:
        return path.is_dir() and not path.is_symlink() and str(UUID(path.name)) == path.name
    except ValueError:
        return False


def prune_runs(keep: int, *, pointer: Path = CURRENT_RUN, dry_run: bool = False) -> list[Path]:
    """Keep the `keep` most recently written runs plus the active run; return the runs removed.

    Without a pointer file no run is active. A pointer that exists but cannot be
    verified stops pruning, because the active run cannot be identified safely.
    """
    if keep < 1:
        raise ValueError("Keep at least one run.")
    active = active_run_path(pointer).resolve() if pointer.is_file() else None
    runs_dir = pointer.parent / "runs"
    runs = [path for path in runs_dir.iterdir() if is_run_directory(path)] if runs_dir.is_dir() else []
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    removable = [run for run in runs[keep:] if run.resolve() != active]
    for run in removable:
        LOGGER.info("%s run %s", "Would remove" if dry_run else "Removing", run.name)
        if not dry_run:
            shutil.rmtree(run)
    return removable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", type=int, default=5, help="Number of most recent runs to keep (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="List runs that would be removed without deleting")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    removed = prune_runs(args.keep, dry_run=args.dry_run)
    LOGGER.info("%d run(s) %s", len(removed), "would be removed" if args.dry_run else "removed")


if __name__ == "__main__":
    main()
