# %% [markdown]
# # RD first stage and reduced form (CCT)
# Estimate EFW first stage and reduced-form impacts at the close-election cutoff
# using robust bias-corrected local polynomial RD.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import ANALYSIS_DIR, PAPER_FIGURES_DIR, PAPER_TABLES_DIR
from src.rd import rd_binned_means, rd_estimate, select_bandwidth
from src.viz_style import savefig, set_style

# %%
set_style()

sample_path = ANALYSIS_DIR / "rd_event_panel.parquet"
if not sample_path.exists():
    raise FileNotFoundError("Missing event panel. Run 13_construct_efw_shocks_and_outcomes first.")

sample = pd.read_parquet(sample_path)

RUNNING = "running_var_vote"
HORIZONS = [0, 1, 2, 3, 4, 5]
PRIMARY_HORIZONS = [1, 2, 4]

bandwidth = select_bandwidth(sample[RUNNING], quantile=0.3, max_bw=0.2)
if pd.isna(bandwidth):
    raise ValueError("Unable to select RD bandwidth for plotting.")

# %%
# Construct EFW excluding Sound Money (Area 3)
area_cols = sorted([col for col in sample.columns if col.startswith("efw_area")])
area_ex_sound = [col for col in area_cols if col != "efw_area3"]
for h in HORIZONS:
    needed = [f"{col}_path_h{h}" for col in area_ex_sound]
    if all(col in sample.columns for col in needed):
        sample[f"efw_ex_sound_path_h{h}"] = sample[needed].mean(axis=1)

# %%
# First stage: EFW (overall + areas)
first_stage_rows = []

for h in HORIZONS:
    # overall EFW
    col = f"efw_path_h{h}"
    if col in sample.columns:
        subset = sample.dropna(subset=[col, RUNNING])
        if subset.empty:
            continue
        try:
            est = rd_estimate(subset, col, RUNNING, cluster="iso3c")
        except Exception:
            continue
        first_stage_rows.append(
            {
                "outcome": "efw_summary",
                "horizon": h,
                "coef": est.coef,
                "se": est.se,
                "pvalue": est.pvalue,
                "n_obs": est.n_obs,
                "n_left": est.n_left,
                "n_right": est.n_right,
                "bandwidth": est.bandwidth,
                "method": est.method,
            }
        )

    # EFW areas
    for area in area_cols:
        area_col = f"{area}_path_h{h}"
        if area_col not in sample.columns:
            continue
        subset = sample.dropna(subset=[area_col, RUNNING])
        if subset.empty:
            continue
        try:
            est = rd_estimate(subset, area_col, RUNNING, cluster="iso3c")
        except Exception:
            continue
        first_stage_rows.append(
            {
                "outcome": area,
                "horizon": h,
                "coef": est.coef,
                "se": est.se,
                "pvalue": est.pvalue,
                "n_obs": est.n_obs,
                "n_left": est.n_left,
                "n_right": est.n_right,
                "bandwidth": est.bandwidth,
                "method": est.method,
            }
        )

    # EFW excluding Sound Money
    ex_col = f"efw_ex_sound_path_h{h}"
    if ex_col in sample.columns:
        subset = sample.dropna(subset=[ex_col, RUNNING])
        if subset.empty:
            continue
        try:
            est = rd_estimate(subset, ex_col, RUNNING, cluster="iso3c")
        except Exception:
            continue
        first_stage_rows.append(
            {
                "outcome": "efw_ex_sound",
                "horizon": h,
                "coef": est.coef,
                "se": est.se,
                "pvalue": est.pvalue,
                "n_obs": est.n_obs,
                "n_left": est.n_left,
                "n_right": est.n_right,
                "bandwidth": est.bandwidth,
                "method": est.method,
            }
        )

first_stage_df = pd.DataFrame(first_stage_rows)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
first_stage_path = PAPER_TABLES_DIR / "rd_first_stage.csv"
first_stage_df.to_csv(first_stage_path, index=False)

display(first_stage_df.style.set_caption("RD first stage: Z -> EFW"))

# %%
# Reduced-form macro outcomes
macro_rows = []

for h in HORIZONS:
    for outcome, label in [
        (f"log_gdp_cum_h{h}", "log_gdp_cum"),
        (f"inv_share_avg_h{h}", "inv_share_avg"),
        (f"inflation_path_h{h}", "inflation_path"),
    ]:
        if outcome not in sample.columns:
            continue
        subset = sample.dropna(subset=[outcome, RUNNING])
        if subset.empty:
            continue
        try:
            est = rd_estimate(subset, outcome, RUNNING, cluster="iso3c")
        except Exception:
            continue
        macro_rows.append(
            {
                "outcome": label,
                "horizon": h,
                "coef": est.coef,
                "se": est.se,
                "pvalue": est.pvalue,
                "n_obs": est.n_obs,
                "n_left": est.n_left,
                "n_right": est.n_right,
                "bandwidth": est.bandwidth,
                "method": est.method,
            }
        )

rf_df = pd.DataFrame(macro_rows)
rf_path = PAPER_TABLES_DIR / "rd_reduced_form.csv"
rf_df.to_csv(rf_path, index=False)

display(rf_df.style.set_caption("RD reduced form: Z -> macro outcomes"))

# %%
# Tail outcomes (primary horizons)

TAIL_OUTCOMES = [
    ("worst_growth", "worst_growth"),
    ("infl_spike_20", "infl_spike_20"),
    ("infl_spike_40", "infl_spike_40"),
    ("max_drawdown", "max_drawdown"),
    ("crisis_start", "crisis_start"),
]

tail_rows = []
for h in PRIMARY_HORIZONS:
    for base, label in TAIL_OUTCOMES:
        col = f"{base}_h{h}"
        if col not in sample.columns:
            continue
        subset = sample.dropna(subset=[col, RUNNING])
        if subset.empty:
            continue
        try:
            est = rd_estimate(subset, col, RUNNING, cluster="iso3c")
        except Exception:
            continue
        tail_rows.append(
            {
                "outcome": label,
                "horizon": h,
                "coef": est.coef,
                "se": est.se,
                "pvalue": est.pvalue,
                "n_obs": est.n_obs,
                "n_left": est.n_left,
                "n_right": est.n_right,
                "bandwidth": est.bandwidth,
                "method": est.method,
            }
        )

tail_df = pd.DataFrame(tail_rows)
if not tail_df.empty:
    tail_path = PAPER_TABLES_DIR / "rd_tail_outcomes.csv"
    tail_df.to_csv(tail_path, index=False)
    display(tail_df.style.set_caption("RD tail outcomes"))

# %%
# RD plots (selected)

def plot_rd(outcome: str, label: str) -> None:
    subset = sample.dropna(subset=[outcome])
    bins = rd_binned_means(subset, outcome, RUNNING, bandwidth=bandwidth)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(bins["bin_center"], bins["mean_outcome"], s=18, color="#2A9D8F")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(label)
    ax.set_xlabel("Vote margin (market bloc)")
    ax.set_ylabel(label)
    savefig(fig, PAPER_FIGURES_DIR / "first_stage" / f"rd_{outcome}")
    plt.close(fig)

# Figure 1: EFW h=1
if "efw_path_h1" in sample.columns:
    plot_rd("efw_path_h1", "EFW change (t+1 - t-1)")

# Figure 2: EFW areas h=1
for area in area_cols:
    col = f"{area}_path_h1"
    if col in sample.columns:
        plot_rd(col, f"{area} change (t+1 - t-1)")

# Reduced-form example plots
if "log_gdp_cum_h4" in sample.columns:
    plot_rd("log_gdp_cum_h4", "Log GDP per capita (h=4)")
if "inv_share_avg_h4" in sample.columns:
    plot_rd("inv_share_avg_h4", "Investment share (avg h=4)")

# %% [markdown]
# ## Interpretation
# These estimates provide the Part‑1 RD first stage and reduced‑form effects
# using robust bias‑corrected local polynomial inference.
