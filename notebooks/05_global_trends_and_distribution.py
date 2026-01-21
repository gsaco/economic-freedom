# %% [markdown]
# # Global trends and distribution
# This notebook tracks global EFW summary trends, distributions, and
# year-by-year rankings to highlight broad cross-country patterns.

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
trend = (
    panel.groupby("year")["efw_summary"]
    .agg(mean="mean", median="median", n="count")
    .reset_index()
)
display(trend.style.set_caption("Global EFW trend summary"))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trend["year"], trend["mean"], marker="o", label="Mean")
ax.plot(trend["year"], trend["median"], marker="s", label="Median")
for _, row in trend.iterrows():
    ax.annotate(int(row["n"]), (row["year"], row["mean"]), textcoords="offset points", xytext=(0, 6), fontsize=7)
ax.set_title("Global EFW summary trend")
ax.set_xlabel("Year")
ax.set_ylabel("EFW summary")
ax.legend()

display(fig)
savefig(fig, OUTPUT_DIR / "figures" / "trends" / "efw_summary_trend")
plt.close(fig)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
panel.boxplot(column="efw_summary", by="year", ax=ax)
ax.set_title("EFW summary distribution by year")
ax.set_xlabel("Year")
ax.set_ylabel("EFW summary")
fig.suptitle("")

display(fig)
savefig(fig, OUTPUT_DIR / "figures" / "trends" / "efw_summary_boxplot")
plt.close(fig)

# %%
rank_dir = OUTPUT_DIR / "tables"
rank_dir.mkdir(parents=True, exist_ok=True)
for year in QUINQUENNIAL_YEARS:
    subset = panel[panel["year"] == year][["iso3c", "country_name", "efw_summary"]].dropna()
    if subset.empty:
        continue
    ranked = subset.sort_values("efw_summary", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    top_bottom = pd.concat([ranked.head(10), ranked.tail(10)])
    top_bottom.to_csv(rank_dir / f"global_rankings_{year}.csv", index=False)
    display(top_bottom.style.set_caption(f"Top/bottom EFW rankings, {year}"))

# %% [markdown]
# ## Interpretation
# The global trend plot shows mean EFW rising from 6.01 in 1970 to 6.58 in 2020,
# while the median increases from 5.84 to 6.62 as coverage expands from 98 to
# 165 countries. Boxplots reveal persistent dispersion around these averages,
# and the rankings highlight which countries consistently lead or lag,
# providing context for interpreting subsequent regional and component analyses.
