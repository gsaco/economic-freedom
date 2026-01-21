# %% [markdown]
# # Shock episodes
# This notebook compares changes in EFW (and GDP per capita where available)
# across defined historical episodes to highlight cross-country responses.

# %%
from __future__ import annotations

import json
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
from src.paths import CLEAN_DIR, LOGS_DIR, OUTPUT_DIR
from src.viz_style import savefig, set_style

# %%
set_style()
panel_path = CLEAN_DIR / "panel_quinquennial_atlas.parquet"
if not panel_path.exists():
    raise FileNotFoundError("Missing atlas panel. Run 03_build_quinquennial_panel first.")

config = json.loads((LOGS_DIR / "run_config.json").read_text())
episodes = config["episodes"]

panel = pd.read_parquet(panel_path)
world = load_world_geometries()

# %%
fig_dir = OUTPUT_DIR / "figures" / "episodes"
fig_dir.mkdir(parents=True, exist_ok=True)
episode_stats = []

for name, window in episodes.items():
    start = window["start"]
    end = window["end"]
    start_df = panel[panel["year"] == start]
    end_df = panel[panel["year"] == end]
    merged = start_df.merge(end_df, on="iso3c", suffixes=("_start", "_end"))
    if merged.empty:
        continue

    merged["d_efw"] = merged["efw_summary_end"] - merged["efw_summary_start"]
    if "log_gdp_pc_const_end" in merged.columns:
        merged["d_log_gdp_pc_const"] = (
            merged["log_gdp_pc_const_end"] - merged["log_gdp_pc_const_start"]
        )

    leaderboard = merged[["iso3c", "country_name_end", "d_efw"]].dropna()
    leaderboard = leaderboard.rename(columns={"country_name_end": "country_name"})
    leaderboard = leaderboard.sort_values("d_efw", ascending=False)
    leaderboard_path = OUTPUT_DIR / "tables" / f"episode_leaderboards_{name}.csv"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_table = pd.concat([leaderboard.head(10), leaderboard.tail(10)])
    leaderboard_table.to_csv(leaderboard_path, index=False)
    display(leaderboard_table.style.set_caption(f"Top/bottom changes in {name}"))

    d_efw = merged["d_efw"].dropna()
    if not d_efw.empty:
        episode_stats.append(
            {
                "episode": name,
                "start": start,
                "end": end,
                "mean_change": float(d_efw.mean()),
                "median_change": float(d_efw.median()),
                "share_positive": float((d_efw > 0).mean()),
                "n": int(d_efw.shape[0]),
            }
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(merged["d_efw"].dropna(), bins=20, color="#4C72B0", alpha=0.8)
    ax.set_title(f"EFW change distribution: {start}-{end}")
    ax.set_xlabel("Delta EFW summary")
    ax.set_ylabel("Countries")
    display(fig)
    savefig(fig, fig_dir / f"{name}_dist_efw")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_vals = merged["d_efw"].dropna().sort_values()
    ax.plot(sorted_vals, range(1, len(sorted_vals) + 1))
    ax.set_title(f"EFW change ECDF: {start}-{end}")
    ax.set_xlabel("Delta EFW summary")
    ax.set_ylabel("Rank")
    display(fig)
    savefig(fig, fig_dir / f"{name}_ecdf_efw")
    plt.close(fig)

    map_df = merged[["iso3c", "d_efw"]]
    merged_map = merge_world_data(world, map_df, "iso3c", "d_efw")
    fig, _ = plot_choropleth(
        merged_map,
        "d_efw",
        title=f"Delta EFW summary ({start}-{end})",
        cmap="coolwarm",
        missing_color="lightgrey",
    )
    display(fig)
    savefig(fig, fig_dir / f"{name}_map_efw")
    plt.close(fig)

    if "d_log_gdp_pc_const" in merged.columns:
        map_df = merged[["iso3c", "d_log_gdp_pc_const"]]
        merged_map = merge_world_data(world, map_df, "iso3c", "d_log_gdp_pc_const")
        fig, _ = plot_choropleth(
            merged_map,
            "d_log_gdp_pc_const",
            title=f"Delta log GDP pc ({start}-{end})",
            cmap="coolwarm",
            missing_color="lightgrey",
        )
        display(fig)
        savefig(fig, fig_dir / f"{name}_map_gdp")
        plt.close(fig)

if episode_stats:
    episode_df = pd.DataFrame(episode_stats)
    display(episode_df.style.format(precision=3).set_caption("Episode summary statistics"))

# %% [markdown]
# ## Interpretation
# Episode-level summaries reveal heterogeneous responses to global shocks.
# During the 1970-1975 oil shocks, average EFW change is -0.49 with only ~20%
# of countries improving, while the 1985-1995 post-communist period shows a
# strong average gain of +0.67 with ~75% positive changes. The 2015-2020
# COVID window turns negative again (mean -0.08; ~36% positive), illustrating
# how global shocks can reverse progress. The distributions and maps highlight
# which countries experienced the largest improvements or declines.
