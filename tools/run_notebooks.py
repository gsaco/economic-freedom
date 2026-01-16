from __future__ import annotations

import subprocess
from pathlib import Path

import papermill as pm


def sync_notebooks(notebooks_dir: Path) -> None:
    for py_path in sorted(notebooks_dir.glob("*.py")):
        subprocess.run(["jupytext", "--sync", str(py_path)], check=True)


def execute_notebooks(notebooks_dir: Path) -> None:
    for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
        pm.execute_notebook(str(nb_path), str(nb_path), log_output=True, kernel_name="python3")


def main() -> None:
    notebooks_dir = Path("notebooks")
    sync_notebooks(notebooks_dir)
    execute_notebooks(notebooks_dir)


if __name__ == "__main__":
    main()
