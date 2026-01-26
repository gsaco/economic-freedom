# %% [markdown]
# # Exclusion sensitivity bounds
# Compute bounds for RD-IV estimates under direct winner effects (delta).

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
from src.rd import rd_estimate
from src.viz_style import savefig, set_style

# %%
set_style()

panel_path = ANALYSIS_DIR / "rd_event_panel.parquet"
first_stage_path = PAPER_TABLES_DIR / "rd_first_stage.csv"
reduced_form_path = PAPER_TABLES_DIR / "rd_reduced_form.csv"

for path in [panel_path, first_stage_path, reduced_form_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

panel = pd.read_parquet(panel_path)
first_stage = pd.read_csv(first_stage_path)
reduced_form = pd.read_csv(reduced_form_path)

RUNNING = "running_var_vote"
PRIMARY_HORIZONS = [1, 2, 4]

# %%
# Calibrate delta bounds using pre-trend RD estimates
pretrend_cols = ["pretrend_h1", "pretrend_h3"]
pretrend_rows = []
for col in pretrend_cols:
    if col not in panel.columns:
        continue
    est = rd_estimate(panel.dropna(subset=[col]), col, RUNNING, cluster="iso3c")
    pretrend_rows.append({"outcome": col, "coef": est.coef})

pretrend_df = pd.DataFrame(pretrend_rows)
if pretrend_df.empty:
    delta_max = 0.0
else:
    delta_max = float(pretrend_df["coef"].abs().max())

# %%
# Bounds grid
if delta_max == 0.0:
    deltas = np.array([0.0])
else:
    deltas = np.linspace(-delta_max, delta_max, 41)

bounds_rows = []
summary_rows = []

for outcome in ["log_gdp_cum", "inv_share_avg"]:
    for h in PRIMARY_HORIZONS:
        tau_e = first_stage.loc[
            (first_stage["outcome"] == "efw_summary") & (first_stage["horizon"] == h), "coef"
        ]
        tau_y = reduced_form.loc[
            (reduced_form["outcome"] == outcome) & (reduced_form["horizon"] == h), "coef"
        ]
        if tau_e.empty or tau_y.empty:
            continue
        tau_e_val = float(tau_e.iloc[0])
        tau_y_val = float(tau_y.iloc[0])
        if not np.isfinite(tau_e_val) or tau_e_val == 0:
            continue

        betas = []
        for delta in deltas:
            beta = (tau_y_val - delta) / tau_e_val
            bounds_rows.append(
                {
                    "outcome": outcome,
                    "horizon": h,
                    "delta": float(delta),
                    "beta": float(beta),
                }
            )
            betas.append(beta)

        summary_rows.append(
            {
                "outcome": outcome,
                "horizon": h,
                "delta_max": float(delta_max),
                "beta_min": float(np.min(betas)) if betas else np.nan,
                "beta_max": float(np.max(betas)) if betas else np.nan,
                "beta_width": float(np.max(betas) - np.min(betas)) if betas else np.nan,
            }
        )

bounds_df = pd.DataFrame(bounds_rows)
summary_df = pd.DataFrame(summary_rows)

PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
bounds_path = PAPER_TABLES_DIR / "rd_exclusion_bounds.csv"
summary_path = PAPER_TABLES_DIR / "rd_exclusion_bounds_summary.csv"

bounds_df.to_csv(bounds_path, index=False)
summary_df.to_csv(summary_path, index=False)

display(summary_df.style.set_caption("Exclusion sensitivity bounds summary"))

# %%
# Plot: beta vs delta for growth (log_gdp_cum) at h=4 if available
plot_df = bounds_df[(bounds_df["outcome"] == "log_gdp_cum") & (bounds_df["horizon"] == 4)]
if not plot_df.empty:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(plot_df["delta"], plot_df["beta"], marker="o", linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Direct effect (delta)")
    ax.set_ylabel("IV estimate (beta)")
    ax.set_title("Exclusion sensitivity: log GDP (h=4)")
    savefig(fig, PAPER_FIGURES_DIR / "robustness" / "exclusion_bounds_h4")
    plt.close(fig)

# %% [markdown]
# ## Interpretation
# Identified sets widen as allowed direct effects grow. The delta bounds are
# calibrated from pre-trend RD estimates as a conservative placebo benchmark.
