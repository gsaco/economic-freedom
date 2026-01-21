# %% [markdown]
# # Descriptive coverage and missingness
# This notebook summarizes data availability by year, region, and income group,
# and visualizes missingness patterns on maps.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.maps import load_world_geometries, merge_world_data, plot_choropleth
from src.paths import CLEAN_DIR, OUTPUT_DIR, QUINQUENNIAL_YEARS
from src.viz_style import savefig, set_style

# %%
set_style()
panel_path = CLEAN_DIR / "panel_quinquennial_atlas.parquet"
if not panel_path.exists():
    raise FileNotFoundError("Missing atlas panel. Run 03_build_quinquennial_panel first.")

panel = pd.read_parquet(panel_path)

# %%
value_cols = [
    col
    for col in ["efw_summary", "efw_area1", "efw_area2", "efw_area3", "efw_area4", "efw_area5"]
    if col in panel.columns
]

coverage = (
    panel.groupby(["year", "wb_region", "wb_income_group"], dropna=False)[value_cols]
    .apply(lambda frame: frame.notna().sum())
    .reset_index()
)

coverage_path = OUTPUT_DIR / "tables" / "coverage_by_year_region_income.csv"
coverage_path.parent.mkdir(parents=True, exist_ok=True)
coverage.to_csv(coverage_path, index=False)
display(coverage.style.set_caption("Coverage by year, region, and income group"))

# %%
if "wb_region" in panel.columns and "efw_summary" in panel.columns:
    regional = (
        panel.groupby(["year", "wb_region"])["efw_summary"]
        .apply(lambda series: series.notna().sum())
        .reset_index(name="n_countries")
    )
    regional_pivot = regional.pivot(index="year", columns="wb_region", values="n_countries")
    display(regional_pivot.style.set_caption("EFW coverage by region"))
    fig, ax = plt.subplots(figsize=(7, 4))
    for region in sorted(regional["wb_region"].dropna().unique()):
        subset = regional[regional["wb_region"] == region]
        ax.plot(subset["year"], subset["n_countries"], label=region)
    ax.set_title("EFW coverage by region")
    ax.set_xlabel("Year")
    ax.set_ylabel("Non-missing countries")
    ax.legend(fontsize=6, ncol=2)
    display(fig)
    savefig(fig, OUTPUT_DIR / "figures" / "coverage" / "efw_coverage_by_region")
    plt.close(fig)

# %%
world = load_world_geometries()
missing_dir = OUTPUT_DIR / "figures" / "missingness_maps"
missing_dir.mkdir(parents=True, exist_ok=True)

for year in QUINQUENNIAL_YEARS:
    subset = panel[panel["year"] == year]
    for var in value_cols:
        missing = subset[["iso3c"]].copy()
        missing[var] = subset[var].isna().astype(int)
        merged = merge_world_data(world, missing, "iso3c", var)
        fig, _ = plot_choropleth(
            merged,
            var,
            title=f"Missing {var} ({year})",
            cmap="Reds",
            missing_color="lightgrey",
        )
        display(fig)
        savefig(fig, missing_dir / f"missing_{var}_{year}")
        plt.close(fig)

# %% [markdown]
# ## Interpretation
# Coverage varies across regions and income groups, with richer and larger
# regions generally reporting more complete EFW data. In 2020, Europe &
# Central Asia (46 countries) and Sub-Saharan Africa (44) have the broadest
# coverage, followed by Latin America & Caribbean (26). The missingness maps
# highlight geographic gaps that are concentrated in earlier years and in
# lower-income regions, which is important for interpreting cross-country
# comparisons in the subsequent notebooks.
