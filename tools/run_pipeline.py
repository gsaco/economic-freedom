#!/usr/bin/env python
"""Run the full EFW quinquennial pipeline (data + notebooks)."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


NOTEBOOKS = [
    "01_efw_macro_quinquennial_analysis",
    "02_components_bundles_crises",
    "03_q1_candidate_evidence",
    "04_construct_shocks",
    "05_build_stacked_event_data",
    "06_estimate_lp_stacked",
    "07_inference_bands_placebos",
]


def run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full notebook pipeline.")
    parser.add_argument("--skip-jupytext", action="store_true", help="Skip syncing .py to .ipynb")
    parser.add_argument("--skip-exec", action="store_true", help="Skip executing notebooks")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    if not args.skip_jupytext:
        for nb in NOTEBOOKS:
            py_path = root / "notebooks" / f"{nb}.py"
            run([sys.executable, "-m", "jupytext", "--to", "notebook", str(py_path)], cwd=root)

    if not args.skip_exec:
        for nb in NOTEBOOKS:
            ipynb_path = root / "notebooks" / f"{nb}.ipynb"
            run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--execute",
                    "--to",
                    "notebook",
                    "--inplace",
                    str(ipynb_path),
                ],
                cwd=root,
            )


if __name__ == "__main__":
    main()
