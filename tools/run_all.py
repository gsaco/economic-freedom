from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ensure_directories
from src.ingest.efw import ingest_efw
from src.ingest.wdi import ingest_wdi
from src.ingest.pwt import ingest_pwt
from src.ingest.eu_events import ingest_eu_events
from src.ingest.wto_events import ingest_wto_events
from src.ingest.wto_acdb import ingest_wto_acdb
from src.build.outcomes import build_macro_outcomes
from src.build.exposure import build_exposures
from src.build.reforms import build_reforms
from src.build.panel import build_panel
from src.build.qa import (
    check_baseline_no_leakage,
    check_delta5,
    check_exposure_bounds,
    check_quinquennial_years,
    check_unique_key,
)
from src.analysis.main_results import run_main
from src.analysis.robustness import run_robustness
from src.analysis.episodes import run_episode_robustness
from src.analysis.heterogeneity import run_heterogeneity
from src.viz.figures import generate_all_figures
from src.viz.tables import table_main_effects
from src.analysis.reporting import write_reports


def run_data() -> None:
    ingest_efw()
    ingest_wdi()
    ingest_pwt()
    ingest_eu_events()
    ingest_wto_events()
    ingest_wto_acdb()


def run_build() -> None:
    build_macro_outcomes()
    build_exposures()
    build_reforms()
    panel_path = build_panel()

    panel = Path(panel_path)
    if panel.exists():
        df = __import__("pandas").read_parquet(panel)
        check_unique_key(df, ["iso3", "year"])
        check_quinquennial_years(df)
        check_exposure_bounds(df, ["EU_neg_share", "EU_post_share", "WTO_neg_share", "WTO_post_share"])
        if "efw_delta5" in df.columns:
            check_delta5(df, "efw_overall", "efw_delta5")
        check_baseline_no_leakage(df, "baseline_efw_eu", "EU_neg_share")


def run_estimate() -> None:
    diagnostics = run_main()
    run_robustness()
    run_episode_robustness()
    run_heterogeneity()
    generate_all_figures()
    table_main_effects()
    print(f"Plan used: {diagnostics.get('plan_used')}")


def run_docs() -> None:
    write_reports()


def run_notebooks() -> None:
    import subprocess

    cmd = [sys.executable, "tools/run_notebooks.py"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Notebook execution failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
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

    if args.stage in {"data", "all", "ci"}:
        run_data()
    if args.stage in {"build", "all", "ci"}:
        run_build()
    if args.stage in {"estimate", "all", "ci"}:
        run_estimate()
    if args.stage in {"docs", "all"}:
        run_docs()
    if args.stage in {"notebooks", "all"}:
        run_notebooks()

    if args.stage == "ci":
        print("CI run complete")


if __name__ == "__main__":
    main()
