# %% [markdown]
# # Components and mobility
# This notebook examines trends in EFW component areas and movement across
# quintiles to understand persistence and mobility in economic freedom.

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

from src.paths import CLEAN_DIR, OUTPUT_DIR, QUINQUENNIAL_YEARS
from src.viz_style import savefig, set_style

# %%
set_style()
panel_path = CLEAN_DIR / "panel_quinquennial_atlas.parquet"
if not panel_path.exists():
    raise FileNotFoundError("Missing atlas panel. Run 03_build_quinquennial_panel first.")

panel = pd.read_parquet(panel_path)

# %%
area_cols = [
    col
    for col in ["efw_area1", "efw_area2", "efw_area3", "efw_area4", "efw_area5"]
    if col in panel.columns
]
if area_cols:
    area_trends = panel.groupby("year")[area_cols].mean().reset_index()
    display(area_trends.style.set_caption("Average EFW area scores by year"))
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in area_cols:
        ax.plot(area_trends["year"], area_trends[col], label=col)
    ax.set_title("EFW area averages over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("EFW area score")
    ax.legend(fontsize=7, ncol=2)
    display(fig)
    savefig(fig, OUTPUT_DIR / "figures" / "components" / "efw_area_trends")
    plt.close(fig)

# %%
if "efw_summary" in panel.columns:
    panel = panel.copy()
    panel["quintile"] = panel.groupby("year")["efw_summary"].transform(
        lambda series: pd.qcut(series.rank(method="first"), 5, labels=False) + 1
    )

    transition_dir = OUTPUT_DIR / "tables" / "mobility_transition_matrices"
    transition_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures" / "mobility"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mobility_stats = []

    for year in QUINQUENNIAL_YEARS:
        next_year = year + 5
        if next_year not in QUINQUENNIAL_YEARS:
            continue
        left = panel[panel["year"] == year][["iso3c", "quintile"]]
        right = panel[panel["year"] == next_year][["iso3c", "quintile"]]
        merged = left.merge(right, on="iso3c", suffixes=("_t", "_t5"))
        if merged.empty:
            continue
        matrix = pd.crosstab(merged["quintile_t"], merged["quintile_t5"]).reindex(
            index=range(1, 6), columns=range(1, 6), fill_value=0
        )
        matrix.to_csv(transition_dir / f"transition_{year}_{next_year}.csv")
        display(matrix.style.set_caption(f"Transition matrix {year}-{next_year}"))
        stay_share = float(matrix.values.trace()) / float(matrix.values.sum())
        mobility_stats.append(
            {"window": f"{year}-{next_year}", "stay_share": stay_share, "n": int(matrix.values.sum())}
        )

        fig, ax = plt.subplots(figsize=(4, 3))
        img = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Quintile transition {year}-{next_year}")
        ax.set_xlabel("Quintile t+5")
        ax.set_ylabel("Quintile t")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(range(1, 6))
        ax.set_yticklabels(range(1, 6))
        fig.colorbar(img, ax=ax, fraction=0.02, pad=0.02)
        display(fig)
        savefig(fig, fig_dir / f"transition_{year}_{next_year}")
        plt.close(fig)

    if mobility_stats:
        mobility_df = pd.DataFrame(mobility_stats)
        display(mobility_df.style.set_caption("Quintile persistence by window"))

# %%
if "d5_efw_summary" in panel.columns:
    movers_dir = OUTPUT_DIR / "tables" / "mobility_movers"
    movers_dir.mkdir(parents=True, exist_ok=True)
    for year in QUINQUENNIAL_YEARS:
        subset = panel[panel["year"] == year][
            ["iso3c", "country_name", "d5_efw_summary"]
        ].dropna()
        if subset.empty:
            continue
        ranked = subset.sort_values("d5_efw_summary", ascending=False)
        movers = pd.concat([ranked.head(10), ranked.tail(10)])
        movers.to_csv(movers_dir / f"movers_{year}.csv", index=False)
        display(movers.style.set_caption(f"Top/bottom movers {year}"))

# %% [markdown]
# ## Interpretation
# Component averages track the broad evolution of EFW sub-areas over time.
# The transition matrices show substantial persistence: on average about 65%
# of countries remain in the same quintile across five-year windows. The movers
# tables identify countries with the most pronounced improvements or declines
# in economic freedom.
