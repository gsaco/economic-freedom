# %% [markdown]
# # Build quinquennial panel
# This notebook merges the Fraser EFW quinquennial data with selected WDI
# indicators, constructs change variables, and outputs the analysis-ready panel.

# %%
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import CLEAN_DIR, INTERMEDIATE_DIR, LOGS_DIR, QUINQUENNIAL_YEARS
from src.qc import assert_range, assert_unique_key, assert_year_grid, coverage_by_year
from src.viz_style import set_style

# %%
set_style()

config_path = Path(LOGS_DIR) / "run_config.json"
if not config_path.exists():
    raise FileNotFoundError("Missing run config. Run 00_env_setup first.")
config = json.loads(config_path.read_text())
indicator_specs = config["wdi_indicators"]

# %%
efw_path = INTERMEDIATE_DIR / "efw_quinquennial.parquet"
wdi_path = INTERMEDIATE_DIR / "wdi_quinquennial_selected.parquet"
country_meta_path = INTERMEDIATE_DIR / "country_metadata.parquet"

if not efw_path.exists() or not wdi_path.exists():
    raise FileNotFoundError("Missing EFW or WDI quinquennial inputs.")

efw = pd.read_parquet(efw_path)
wdi = pd.read_parquet(wdi_path)
country_meta = pd.read_parquet(country_meta_path) if country_meta_path.exists() else pd.DataFrame()

assert_unique_key(efw, ["iso3c", "year"])
assert_unique_key(wdi, ["iso3c", "year"])

panel = efw.merge(wdi, on=["iso3c", "year"], how="outer", indicator=True)
join_diagnostics = panel["_merge"].value_counts().to_dict()
panel = panel.drop(columns=["_merge"])

if not country_meta.empty:
    panel = panel.merge(country_meta, on="iso3c", how="left")

panel = panel[panel["year"].isin(QUINQUENNIAL_YEARS)].copy()
assert_year_grid(panel, "year", QUINQUENNIAL_YEARS)

panel_summary = pd.DataFrame(
    {
        "rows": [len(panel)],
        "countries": [panel["iso3c"].nunique()],
        "years": [panel["year"].nunique()],
    }
)
display(panel_summary.style.set_caption("Panel size summary"))
display(pd.DataFrame([join_diagnostics]).style.set_caption("Merge diagnostics"))

# %%
log_vars = [spec["name"] for spec in indicator_specs.values() if spec["transform"] == "log"]
for var in log_vars:
    if var in panel.columns:
        panel[f"log_{var}"] = np.log(panel[var].where(panel[var] > 0))

# EFW delta-5
for col in ["efw_summary", "efw_area1", "efw_area2", "efw_area3", "efw_area4", "efw_area5"]:
    if col in panel.columns:
        panel[f"d5_{col}"] = panel.sort_values("year").groupby("iso3c")[col].diff()

# WDI delta-5
for spec in indicator_specs.values():
    name = spec["name"]
    if name in panel.columns:
        panel[f"d5_{name}"] = panel.sort_values("year").groupby("iso3c")[name].diff()
    log_name = f"log_{name}"
    if log_name in panel.columns:
        panel[f"d5_{log_name}"] = panel.sort_values("year").groupby("iso3c")[log_name].diff()

# Average 5-year GDP growth from annual data
annual_path = INTERMEDIATE_DIR / "wdi_annual_selected.parquet"
if annual_path.exists() and "gdp_growth_ann_pct" in panel.columns:
    annual = pd.read_parquet(annual_path)
    annual = annual.dropna(subset=["gdp_growth_ann_pct"])
    annual = annual.sort_values(["iso3c", "year"])
    annual["avg5_gdp_growth"] = (
        annual.groupby("iso3c")["gdp_growth_ann_pct"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    avg5 = annual[annual["year"].isin(QUINQUENNIAL_YEARS)][
        ["iso3c", "year", "avg5_gdp_growth"]
    ]
    panel = panel.merge(avg5, on=["iso3c", "year"], how="left")

# %%
assert_unique_key(panel, ["iso3c", "year"])
if "efw_summary" in panel.columns:
    assert_range(panel, "efw_summary", min_value=0, max_value=10)

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
panel_path = CLEAN_DIR / "panel_quinquennial_atlas.parquet"
panel.to_parquet(panel_path, index=False)

# %%
created_at = datetime.now(timezone.utc).isoformat()
try:
    git_ref = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
except Exception:
    git_ref = "unknown"

coverage = coverage_by_year(
    panel,
    "year",
    [col for col in panel.columns if col.startswith("efw_") or col.startswith("log_")],
)

metadata = {
    "build": {
        "created_at_utc": created_at,
        "git_ref": git_ref or "unknown",
        "pipeline_version": "atlas-v1",
    },
    "sources": {
        "efw": str(efw_path),
        "wdi": str(wdi_path),
        "country_metadata": str(country_meta_path),
    },
    "variables": indicator_specs,
    "coverage_by_year": coverage.to_dict(orient="records"),
    "qc": {
        "unique_key_check": True,
        "year_grid_check": True,
        "range_checks": {"efw_summary": "0-10"},
        "join_diagnostics": join_diagnostics,
    },
}

metadata_path = CLEAN_DIR / "panel_quinquennial_atlas_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2))

display(coverage.style.set_caption("Coverage by year (selected variables)"))

# %% [markdown]
# ## Interpretation
# The merged panel contains 2,882 rows covering 262 countries across 11
# quinquennial years. Merge diagnostics show 1,804 matched observations,
# 1,067 WDI-only rows, and 11 EFW-only rows, indicating strong overlap but
# also highlighting where one source outpaces the other. The coverage table
# documents how many observations are available for EFW and key macro
# indicators, providing a baseline for subsequent descriptive analysis.
