# %% [markdown]
# # LP-IV impulse responses (positive vs negative shocks)
# Estimate local-projection IV IRFs for GDP per capita after EFW shocks.

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

from src.lpiv import first_stage, iv_estimate, reduced_form
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
window_choice = {}
if window_choice_path.exists():
    window_choice = json.loads(window_choice_path.read_text())

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

def run_irf(frame: pd.DataFrame, *, instrument_col: str, window: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fs_rows = []
    rf_rows = []
    iv_rows = []

    fs_shock = first_stage(
        frame,
        outcome_col="shock_efw",
        running_col=RUNNING,
        instrument_col=instrument_col,
        controls=controls,
        window=window,
        cluster="iso3c",
    )
    fs_rows.append(
        {
            "outcome": "shock_efw",
            "horizon": np.nan,
            "coef": fs_shock.coef,
            "se": fs_shock.se,
            "pvalue": fs_shock.pvalue,
            "n_obs": fs_shock.n_obs,
            "n_left": fs_shock.n_left,
            "n_right": fs_shock.n_right,
            "window": fs_shock.window,
        }
    )

    for h in HORIZONS:
        fs_path = first_stage(
            frame,
            outcome_col=f"efw_path_h{h}",
            running_col=RUNNING,
            instrument_col=instrument_col,
            controls=controls,
            window=window,
            cluster="iso3c",
        )
        fs_rows.append(
            {
                "outcome": f"efw_path_h{h}",
                "horizon": h,
                "coef": fs_path.coef,
                "se": fs_path.se,
                "pvalue": fs_path.pvalue,
                "n_obs": fs_path.n_obs,
                "n_left": fs_path.n_left,
                "n_right": fs_path.n_right,
                "window": fs_path.window,
            }
        )

        rf = reduced_form(
            frame,
            outcome_col=f"log_gdp_cum_h{h}",
            running_col=RUNNING,
            instrument_col=instrument_col,
            controls=controls,
            window=window,
            cluster="iso3c",
        )
        rf_rows.append(
            {
                "outcome": f"log_gdp_cum_h{h}",
                "horizon": h,
                "coef": rf.coef,
                "se": rf.se,
                "pvalue": rf.pvalue,
                "n_obs": rf.n_obs,
                "n_left": rf.n_left,
                "n_right": rf.n_right,
                "window": rf.window,
            }
        )

        iv = iv_estimate(
            frame,
            outcome_col=f"log_gdp_cum_h{h}",
            endog_col="shock_efw",
            running_col=RUNNING,
            instrument_col=instrument_col,
            controls=controls,
            window=window,
            cluster="iso3c",
        )
        iv_rows.append(
            {
                "outcome": f"log_gdp_cum_h{h}",
                "horizon": h,
                "coef": iv.coef,
                "se": iv.se,
                "pvalue": iv.pvalue,
                "n_obs": iv.n_obs,
                "n_left": iv.n_left,
                "n_right": iv.n_right,
                "window": iv.window,
            }
        )

    return pd.DataFrame(fs_rows), pd.DataFrame(rf_rows), pd.DataFrame(iv_rows)


fs_pos, rf_pos, iv_pos = run_irf(panel_pos, instrument_col="z_pos", window=window_pos)
fs_neg, rf_neg, iv_neg = run_irf(panel_neg, instrument_col="z_neg", window=window_neg)

# %%
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

fs_pos.to_csv(PAPER_TABLES_DIR / "irf_first_stage_pos.csv", index=False)
fs_neg.to_csv(PAPER_TABLES_DIR / "irf_first_stage_neg.csv", index=False)
rf_pos.to_csv(PAPER_TABLES_DIR / "irf_reduced_form_pos.csv", index=False)
rf_neg.to_csv(PAPER_TABLES_DIR / "irf_reduced_form_neg.csv", index=False)
iv_pos.to_csv(PAPER_TABLES_DIR / "irf_iv_pos.csv", index=False)
iv_neg.to_csv(PAPER_TABLES_DIR / "irf_iv_neg.csv", index=False)

# %%
# IRF plots

def plot_irf(iv_df: pd.DataFrame, label: str, color: str, path_suffix: str) -> None:
    plot_df = iv_df.dropna(subset=["horizon", "coef"]).sort_values("horizon")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(plot_df["horizon"], plot_df["coef"], marker="o", color=color, label=label)
    ax.fill_between(
        plot_df["horizon"],
        plot_df["coef"] - 1.96 * plot_df["se"],
        plot_df["coef"] + 1.96 * plot_df["se"],
        color=color,
        alpha=0.2,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Log GDP per capita (cum)")
    ax.set_title(f"IV IRF: {label}")
    ax.legend(frameon=False)
    savefig(fig, PAPER_FIGURES_DIR / "irfs" / path_suffix)
    plt.close(fig)


plot_irf(iv_pos, "Positive shocks", "#2A9D8F", "irf_iv_pos")
plot_irf(iv_neg, "Negative shocks", "#E76F51", "irf_iv_neg")

# %%
# Comparison plot
pos_plot = iv_pos.dropna(subset=["horizon"]).sort_values("horizon")
neg_plot = iv_neg.dropna(subset=["horizon"]).sort_values("horizon")

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(pos_plot["horizon"], pos_plot["coef"], marker="o", color="#2A9D8F", label="Positive")
ax.plot(neg_plot["horizon"], neg_plot["coef"], marker="o", color="#E76F51", label="Negative")
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Horizon (years)")
ax.set_ylabel("Log GDP per capita (cum)")
ax.set_title("IV IRF comparison")
ax.legend(frameon=False)

savefig(fig, PAPER_FIGURES_DIR / "irfs" / "irf_compare_pos_vs_neg")
plt.close(fig)

# %%
display(iv_pos.style.set_caption("IV IRF (positive shocks)"))
display(iv_neg.style.set_caption("IV IRF (negative shocks)"))

# %% [markdown]
# ## Interpretation
# These IRFs trace the causal effect of EFW shocks on log GDP per capita for
# close-election compliers, separately for positive and negative shocks.
