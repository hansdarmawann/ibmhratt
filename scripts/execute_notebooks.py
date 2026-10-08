"""Execute notebooks in the current (ibmhratt) Python environment."""

import logging
import os

import nbformat
from nbclient import NotebookClient

from src.config import ROOT

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run every cell with the current environment's default Python kernel."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        LOGGER.info("Executing %s", path.name)
        notebook = nbformat.read(path, as_version=4)
        # Execution timestamps change on every run; omitting them keeps executed notebooks diff-free.
        for cell in notebook.cells:
            cell.metadata.pop("execution", None)
        client = NotebookClient(notebook, timeout=180, kernel_name="python3", record_timing=False,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute(env={**os.environ, "MPLBACKEND": "Agg"})
        nbformat.write(notebook, path)
        LOGGER.info("Saved executed notebook %s", path.name)


if __name__ == "__main__":
    main()
