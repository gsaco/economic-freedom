# %% [markdown]
# # Nonlinearity by magnitude and state
# Estimate IRFs by ideology distance (LargeShift) and initial state (EFW, income).

# %%
from __future__ import annotations

import json
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

from src.lpiv import iv_estimate
from src.paths import ANALYSIS_DIR, PAPER_FIGURES_DIR, PAPER_LOGS_DIR, PAPER_TABLES_DIR
from src.viz_style import savefig, set_style

# %%
set_style()

pos_path = ANALYSIS_DIR / "rd_event_panel_pos.parquet"
neg_path = ANALYSIS_DIR / "rd_event_panel_neg.parquet"

for path in [pos_path, neg_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing event panel: {path}")

panel_pos = pd.read_parquet(pos_path)
panel_neg = pd.read_parquet(neg_path)

window_choice_path = PAPER_LOGS_DIR / "rd_window_choice.json"
window_choice = json.loads(window_choice_path.read_text()) if window_choice_path.exists() else {}

window_pos = window_choice.get("positive", {}).get("window") or 0.03
window_neg = window_choice.get("negative", {}).get("window") or 0.03

RUNNING = "running_var_vote"
HORIZONS = [0, 1, 2, 3, 4, 5]
controls = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
]

# %%

def irf_by_group(
    frame: pd.DataFrame,
    *,
    instrument_col: str,
    window: float,
    group_col: str,
    group_value: int,
    label: str,
    sample_name: str,
) -> pd.DataFrame:
    subset = frame[frame[group_col] == group_value].copy()
    rows = []
    for h in HORIZONS:
        try:
            est = iv_estimate(
                subset,
                outcome_col=f"log_gdp_cum_h{h}",
                endog_col="shock_efw",
                running_col=RUNNING,
                instrument_col=instrument_col,
                controls=controls,
                window=window,
                cluster="iso3c",
            )
        except ValueError:
            est = iv_estimate(
                subset.iloc[0:0],
                outcome_col=f"log_gdp_cum_h{h}",
                endog_col="shock_efw",
                running_col=RUNNING,
                instrument_col=instrument_col,
                controls=controls,
                window=window,
                cluster="iso3c",
            )
        rows.append(
            {
                "sample": sample_name,
                "group": label,
                "group_col": group_col,
                "group_value": group_value,
                "horizon": h,
                "coef": est.coef,
                "se": est.se,
                "n_obs": est.n_obs,
                "window": est.window,
            }
        )
    return pd.DataFrame(rows)


results = []
# Magnitude splits
results.append(
    irf_by_group(
        panel_pos,
        instrument_col="z_pos",
        window=window_pos,
        group_col="large_shift",
        group_value=1,
        label="LargeShift",
        sample_name="positive",
    )
)
results.append(
    irf_by_group(
        panel_pos,
        instrument_col="z_pos",
        window=window_pos,
        group_col="large_shift",
        group_value=0,
        label="SmallShift",
        sample_name="positive",
    )
)
results.append(
    irf_by_group(
        panel_neg,
        instrument_col="z_neg",
        window=window_neg,
        group_col="large_shift",
        group_value=1,
        label="LargeShift",
        sample_name="negative",
    )
)
results.append(
    irf_by_group(
        panel_neg,
        instrument_col="z_neg",
        window=window_neg,
        group_col="large_shift",
        group_value=0,
        label="SmallShift",
        sample_name="negative",
    )
)

# State dependence splits
results.append(
    irf_by_group(
        panel_pos,
        instrument_col="z_pos",
        window=window_pos,
        group_col="low_efw",
        group_value=1,
        label="LowEFW",
        sample_name="positive",
    )
)
results.append(
    irf_by_group(
        panel_pos,
        instrument_col="z_pos",
        window=window_pos,
        group_col="low_efw",
        group_value=0,
        label="HighEFW",
        sample_name="positive",
    )
)
results.append(
    irf_by_group(
        panel_neg,
        instrument_col="z_neg",
        window=window_neg,
        group_col="low_efw",
        group_value=1,
        label="LowEFW",
        sample_name="negative",
    )
)
results.append(
    irf_by_group(
        panel_neg,
        instrument_col="z_neg",
        window=window_neg,
        group_col="low_efw",
        group_value=0,
        label="HighEFW",
        sample_name="negative",
    )
)

results.append(
    irf_by_group(
        panel_pos,
        instrument_col="z_pos",
        window=window_pos,
        group_col="low_income",
        group_value=1,
        label="LowIncome",
        sample_name="positive",
    )
)
results.append(
    irf_by_group(
        panel_pos,
        instrument_col="z_pos",
        window=window_pos,
        group_col="low_income",
        group_value=0,
        label="HighIncome",
        sample_name="positive",
    )
)
results.append(
    irf_by_group(
        panel_neg,
        instrument_col="z_neg",
        window=window_neg,
        group_col="low_income",
        group_value=1,
        label="LowIncome",
        sample_name="negative",
    )
)
results.append(
    irf_by_group(
        panel_neg,
        instrument_col="z_neg",
        window=window_neg,
        group_col="low_income",
        group_value=0,
        label="HighIncome",
        sample_name="negative",
    )
)

results_df = pd.concat(results, ignore_index=True)

# %%
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
output_path = PAPER_TABLES_DIR / "nonlinear_effects.csv"
results_df.to_csv(output_path, index=False)

# %%
# Plot magnitude IRFs
mag_df = results_df[results_df["group_col"] == "large_shift"]

fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for ax, sample_name, color in zip(axes, ["positive", "negative"], ["#2A9D8F", "#E76F51"]):
    for label, linestyle in [("LargeShift", "-"), ("SmallShift", "--")]:
        subset = mag_df[(mag_df["sample"] == sample_name) & (mag_df["group"] == label)].sort_values("horizon")
        ax.plot(subset["horizon"], subset["coef"], linestyle=linestyle, marker="o", label=label)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"{sample_name.capitalize()} shocks")
    ax.set_xlabel("Horizon")
axes[0].set_ylabel("Log GDP per capita (cum)")
axes[0].legend(frameon=False)

savefig(fig, PAPER_FIGURES_DIR / "nonlinear" / "irf_by_magnitude")
plt.close(fig)

# %%
# Plot state dependence
state_df = results_df[results_df["group_col"].isin(["low_efw", "low_income"])]

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
state_specs = [("low_efw", "LowEFW", "HighEFW"), ("low_income", "LowIncome", "HighIncome")]
for row, (group_col, low_label, high_label) in enumerate(state_specs):
    for col, sample_name in enumerate(["positive", "negative"]):
        ax = axes[row, col]
        for label, linestyle in [(low_label, "-"), (high_label, "--")]:
            subset = state_df[
                (state_df["group_col"] == group_col)
                & (state_df["sample"] == sample_name)
                & (state_df["group"] == label)
            ].sort_values("horizon")
            ax.plot(subset["horizon"], subset["coef"], linestyle=linestyle, marker="o", label=label)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{sample_name.capitalize()} - {group_col}")
        ax.set_xlabel("Horizon")
        if row == 0 and col == 0:
            ax.legend(frameon=False)

axes[0, 0].set_ylabel("Log GDP per capita (cum)")
axes[1, 0].set_ylabel("Log GDP per capita (cum)")

savefig(fig, PAPER_FIGURES_DIR / "nonlinear" / "irf_by_state")
plt.close(fig)

# %%
display(results_df.head(10).style.set_caption("Nonlinear IRF sample"))

# %% [markdown]
# ## Interpretation
# Heterogeneity splits test whether larger ideological swings or weaker initial
# institutions/income amplify the growth response to EFW shocks.
