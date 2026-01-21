from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import ensure_directories

NOTEBOOK_ORDER = [
    "00_env_setup",
    "01_ingest_fraser",
    "02_pull_worldbank",
    "03_build_quinquennial_panel",
    "04_descriptive_coverage_and_missingness",
    "05_global_trends_and_distribution",
    "06_maps_levels_and_changes",
    "07_components_and_mobility",
    "08_macro_co_movement",
    "09_shock_episodes",
]

STAGE_NOTEBOOKS = {
    "data": NOTEBOOK_ORDER[:3],
    "build": [NOTEBOOK_ORDER[3]],
    "notebooks": NOTEBOOK_ORDER,
    "all": NOTEBOOK_ORDER,
    "ci": NOTEBOOK_ORDER,
}


def run_notebooks(notebook_stems: list[str]) -> None:
    if not notebook_stems:
        return
    cmd = [
        sys.executable,
        "tools/run_notebooks.py",
        "--notebooks",
        ",".join(notebook_stems),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Notebook execution failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def run_docs() -> None:
    cmd = [sys.executable, "tools/build_report.py"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Report build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["data", "build", "estimate", "docs", "notebooks", "all", "ci"],
        default="all",
    )
    args = parser.parse_args()

    ensure_directories()

    if args.stage == "estimate":
        raise SystemExit(
            "Inference/estimation is disabled for the descriptive atlas pipeline."
        )

    if args.stage in STAGE_NOTEBOOKS:
        run_notebooks(STAGE_NOTEBOOKS[args.stage])

    if args.stage in {"docs", "all"}:
        run_docs()

    if args.stage == "ci":
        print("CI run complete")


if __name__ == "__main__":
    main()
