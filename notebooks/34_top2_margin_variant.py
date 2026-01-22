# %% [markdown]
# # Top-2 margin variant (bloc winner vs runner-up)
# Evaluate RD validity and LP-IV IRFs using top-2 vote-share margin between blocs.

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
from src.rd import density_discontinuity
from src.rd_localrand import select_window_by_balance
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

RUNNING = "running_var_top2"
controls = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
]
HORIZONS = [0, 1, 2, 3, 4, 5]

# %%
# Define instruments based on top-2 margin
panel_pos = panel_pos.copy()
panel_neg = panel_neg.copy()

panel_pos["z_pos_top2"] = (panel_pos[RUNNING] > 0).astype(int)
panel_neg["z_neg_top2"] = (panel_neg[RUNNING] < 0).astype(int)

# %%
# Density diagnostics

def density_block(frame: pd.DataFrame, label: str) -> dict:
    series = frame[RUNNING].dropna()
    bandwidth = float(series.abs().quantile(0.3)) if not series.empty else 0.05
    bandwidth = min(bandwidth, 0.1)
    stats = density_discontinuity(series, bandwidth=bandwidth)
    stats["sample"] = label
    stats["bandwidth"] = bandwidth
    return stats


density_pos = density_block(panel_pos, "pos_top2")
density_neg = density_block(panel_neg, "neg_top2")

density_df = pd.DataFrame([density_pos, density_neg])
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
density_path = PAPER_TABLES_DIR / "rd_density_top2.csv"
density_df.to_csv(density_path, index=False)

# %%
# Balance windows
balance_vars = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
    "lag1_inv_share_gdp",
]
windows = [0.01, 0.02, 0.03, 0.04, 0.05]

choice_pos, table_pos = select_window_by_balance(
    panel_pos,
    RUNNING,
    balance_vars,
    windows=windows,
    p_threshold=0.15,
    cluster="iso3c",
)

choice_neg, table_neg = select_window_by_balance(
    panel_neg,
    RUNNING,
    balance_vars,
    windows=windows,
    p_threshold=0.15,
    cluster="iso3c",
)

valid_pos_path = PAPER_TABLES_DIR / "rd_validity_top2_pos.csv"
valid_neg_path = PAPER_TABLES_DIR / "rd_validity_top2_neg.csv"

table_pos.to_csv(valid_pos_path, index=False)
table_neg.to_csv(valid_neg_path, index=False)

window_choice = {
    "positive": {
        "window": choice_pos.window,
        "p_threshold": choice_pos.p_threshold,
        "windows_tested": choice_pos.windows_tested,
    },
    "negative": {
        "window": choice_neg.window,
        "p_threshold": choice_neg.p_threshold,
        "windows_tested": choice_neg.windows_tested,
    },
}

PAPER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
choice_path = PAPER_LOGS_DIR / "rd_window_choice_top2.json"
choice_path.write_text(json.dumps(window_choice, indent=2))

window_pos = choice_pos.window or 0.03
window_neg = choice_neg.window or 0.03

# %%
# Helper to avoid rank errors

def safe_iv(frame: pd.DataFrame, **kwargs):
    try:
        return iv_estimate(frame, **kwargs)
    except ValueError:
        return iv_estimate(frame.iloc[0:0], **kwargs)


# First stage, reduced form, IV
fs_rows = []
rf_rows = []
iv_rows = []

for sample_name, frame, instrument, window in [
    ("pos", panel_pos, "z_pos_top2", window_pos),
    ("neg", panel_neg, "z_neg_top2", window_neg),
]:
    fs = first_stage(
        frame,
        outcome_col="shock_efw",
        running_col=RUNNING,
        instrument_col=instrument,
        controls=controls,
        window=window,
        cluster="iso3c",
    )
    fs_rows.append(
        {
            "sample": sample_name,
            "outcome": "shock_efw",
            "coef": fs.coef,
            "se": fs.se,
            "pvalue": fs.pvalue,
            "n_obs": fs.n_obs,
            "window": fs.window,
        }
    )

    for h in HORIZONS:
        rf = reduced_form(
            frame,
            outcome_col=f"log_gdp_cum_h{h}",
            running_col=RUNNING,
            instrument_col=instrument,
            controls=controls,
            window=window,
            cluster="iso3c",
        )
        rf_rows.append(
            {
                "sample": sample_name,
                "horizon": h,
                "coef": rf.coef,
                "se": rf.se,
                "pvalue": rf.pvalue,
                "n_obs": rf.n_obs,
                "window": rf.window,
            }
        )

        iv = safe_iv(
            frame,
            outcome_col=f"log_gdp_cum_h{h}",
            endog_col="shock_efw",
            running_col=RUNNING,
            instrument_col=instrument,
            controls=controls,
            window=window,
            cluster="iso3c",
        )
        iv_rows.append(
            {
                "sample": sample_name,
                "horizon": h,
                "coef": iv.coef,
                "se": iv.se,
                "pvalue": iv.pvalue,
                "n_obs": iv.n_obs,
                "window": iv.window,
            }
        )

fs_df = pd.DataFrame(fs_rows)
rf_df = pd.DataFrame(rf_rows)
iv_df = pd.DataFrame(iv_rows)

PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
fs_df.to_csv(PAPER_TABLES_DIR / "irf_first_stage_top2.csv", index=False)
rf_df.to_csv(PAPER_TABLES_DIR / "irf_reduced_form_top2.csv", index=False)
iv_df.to_csv(PAPER_TABLES_DIR / "irf_iv_top2.csv", index=False)

# %%
# IRF plots

def plot_irf(sample: str, color: str, path_suffix: str) -> None:
    subset = iv_df[iv_df["sample"] == sample].dropna(subset=["horizon", "coef"]).sort_values("horizon")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(subset["horizon"], subset["coef"], marker="o", color=color, label=sample)
    ax.fill_between(
        subset["horizon"],
        subset["coef"] - 1.96 * subset["se"],
        subset["coef"] + 1.96 * subset["se"],
        color=color,
        alpha=0.2,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Log GDP per capita (cum)")
    ax.set_title(f"Top-2 margin IV IRF ({sample})")
    ax.legend(frameon=False)
    savefig(fig, PAPER_FIGURES_DIR / "irfs_top2" / path_suffix)
    plt.close(fig)


plot_irf("pos", "#2A9D8F", "irf_iv_pos_top2")
plot_irf("neg", "#E76F51", "irf_iv_neg_top2")

# Comparison plot
pos_plot = iv_df[iv_df["sample"] == "pos"].sort_values("horizon")
neg_plot = iv_df[iv_df["sample"] == "neg"].sort_values("horizon")
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(pos_plot["horizon"], pos_plot["coef"], marker="o", color="#2A9D8F", label="Positive")
ax.plot(neg_plot["horizon"], neg_plot["coef"], marker="o", color="#E76F51", label="Negative")
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Horizon (years)")
ax.set_ylabel("Log GDP per capita (cum)")
ax.set_title("Top-2 margin IV IRF comparison")
ax.legend(frameon=False)

savefig(fig, PAPER_FIGURES_DIR / "irfs_top2" / "irf_compare_top2")
plt.close(fig)

# %%
# Summary comparison vs baseline
summary_rows = []

baseline_pos = pd.read_csv(PAPER_TABLES_DIR / "irf_first_stage_pos.csv")
baseline_neg = pd.read_csv(PAPER_TABLES_DIR / "irf_first_stage_neg.csv")

for sample_name, df in [("pos", baseline_pos), ("neg", baseline_neg)]:
    row = df.loc[df["outcome"] == "shock_efw"].iloc[0]
    summary_rows.append(
        {
            "variant": "vote_margin",
            "sample": sample_name,
            "coef": row["coef"],
            "se": row["se"],
            "pvalue": row["pvalue"],
            "n_obs": row["n_obs"],
        }
    )

for sample_name in ["pos", "neg"]:
    row = fs_df.loc[fs_df["sample"] == sample_name].iloc[0]
    summary_rows.append(
        {
            "variant": "top2_margin",
            "sample": sample_name,
            "coef": row["coef"],
            "se": row["se"],
            "pvalue": row["pvalue"],
            "n_obs": row["n_obs"],
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_path = PAPER_TABLES_DIR / "top2_vs_vote_margin_summary.csv"
summary_df.to_csv(summary_path, index=False)

# %%
display(summary_df.style.set_caption("First-stage comparison: vote margin vs top-2 margin"))
