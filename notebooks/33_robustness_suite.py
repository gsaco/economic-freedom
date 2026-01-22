# %% [markdown]
# # Robustness suite
# Bandwidth sensitivity, donut RD, placebo cutoffs, pre-trend tests, alternative
# thresholds, outcome variants, and crisis exclusions.

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

from src.lpiv import iv_estimate, reduced_form
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
controls = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
]

# %%
rows = []


def safe_iv_estimate(frame: pd.DataFrame, **kwargs):
    try:
        return iv_estimate(frame, **kwargs)
    except ValueError:
        return iv_estimate(frame.iloc[0:0], **kwargs)

# Bandwidth sensitivity (h=3)
windows = [0.01, 0.02, 0.03, 0.04, 0.05]
for window in windows:
    for sample_name, frame, instrument in [
        ("positive", panel_pos, "z_pos"),
        ("negative", panel_neg, "z_neg"),
    ]:
        est = safe_iv_estimate(
            frame,
            outcome_col="log_gdp_cum_h3",
            endog_col="shock_efw",
            running_col=RUNNING,
            instrument_col=instrument,
            controls=controls,
            window=window,
            cluster="iso3c",
        )
        rows.append(
            {
                "check": "bandwidth",
                "sample": sample_name,
                "horizon": 3,
                "window": window,
                "coef": est.coef,
                "se": est.se,
                "n_obs": est.n_obs,
            }
        )

# Donut RD (exclude |m| < 0.005)
for sample_name, frame, instrument, window in [
    ("positive", panel_pos, "z_pos", window_pos),
    ("negative", panel_neg, "z_neg", window_neg),
]:
    donut = frame.loc[frame[RUNNING].abs() >= 0.005].copy()
    est = safe_iv_estimate(
        donut,
        outcome_col="log_gdp_cum_h3",
        endog_col="shock_efw",
        running_col=RUNNING,
        instrument_col=instrument,
        controls=controls,
        window=window,
        cluster="iso3c",
    )
    rows.append(
        {
            "check": "donut",
            "sample": sample_name,
            "horizon": 3,
            "window": window,
            "coef": est.coef,
            "se": est.se,
            "n_obs": est.n_obs,
        }
    )

# Placebo cutoffs (shift cutoff)
for cutoff in [-0.05, 0.05]:
    for sample_name, frame, instrument, window in [
        ("positive", panel_pos, "z_pos", window_pos),
        ("negative", panel_neg, "z_neg", window_neg),
    ]:
        est = safe_iv_estimate(
            frame,
            outcome_col="log_gdp_cum_h3",
            endog_col="shock_efw",
            running_col=RUNNING,
            instrument_col=instrument,
            controls=controls,
            window=window,
            cutoff=cutoff,
            cluster="iso3c",
        )
        rows.append(
            {
                "check": f"placebo_cutoff_{cutoff:+.2f}",
                "sample": sample_name,
                "horizon": 3,
                "window": window,
                "coef": est.coef,
                "se": est.se,
                "n_obs": est.n_obs,
            }
        )

# Pre-trend placebo (reduced form)
for sample_name, frame, instrument, window in [
    ("positive", panel_pos, "z_pos", window_pos),
    ("negative", panel_neg, "z_neg", window_neg),
]:
    for h in [1, 3]:
        rf = reduced_form(
            frame,
            outcome_col=f"pretrend_h{h}",
            running_col=RUNNING,
            instrument_col=instrument,
            controls=controls,
            window=window,
            cluster="iso3c",
        )
        rows.append(
            {
                "check": f"pretrend_h{h}",
                "sample": sample_name,
                "horizon": -h,
                "window": window,
                "coef": rf.coef,
                "se": rf.se,
                "n_obs": rf.n_obs,
            }
        )

# Alternative threshold (vote share alt)
for sample_name, frame, instrument_base, window in [
    ("positive", panel_pos, "z_pos", window_pos),
    ("negative", panel_neg, "z_neg", window_neg),
]:
    alt = frame.copy()
    alt["z_alt"] = (alt["running_var_vote_alt"] > 0).astype(int)
    if sample_name == "negative":
        alt["z_alt"] = (alt["running_var_vote_alt"] < 0).astype(int)
    est = safe_iv_estimate(
        alt,
        outcome_col="log_gdp_cum_h3",
        endog_col="shock_efw",
        running_col="running_var_vote_alt",
        instrument_col="z_alt",
        controls=controls,
        window=window,
        cluster="iso3c",
    )
    rows.append(
        {
            "check": "alt_threshold",
            "sample": sample_name,
            "horizon": 3,
            "window": window,
            "coef": est.coef,
            "se": est.se,
            "n_obs": est.n_obs,
        }
    )

# Outcome variant: GDP growth average
for sample_name, frame, instrument, window in [
    ("positive", panel_pos, "z_pos", window_pos),
    ("negative", panel_neg, "z_neg", window_neg),
]:
    for h in [3, 5]:
        est = safe_iv_estimate(
            frame,
            outcome_col=f"gdp_growth_avg_h{h}",
            endog_col="shock_efw",
            running_col=RUNNING,
            instrument_col=instrument,
            controls=controls,
            window=window,
            cluster="iso3c",
        )
        rows.append(
            {
                "check": f"growth_avg_h{h}",
                "sample": sample_name,
                "horizon": h,
                "window": window,
                "coef": est.coef,
                "se": est.se,
                "n_obs": est.n_obs,
            }
        )

# Exclude crisis years (2008-2010, 2019-2020)
crisis_years = set(range(2008, 2011)) | {2019, 2020}
for sample_name, frame, instrument, window in [
    ("positive", panel_pos, "z_pos", window_pos),
    ("negative", panel_neg, "z_neg", window_neg),
]:
    subset = frame.loc[~frame["election_year"].isin(crisis_years)].copy()
    est = safe_iv_estimate(
        subset,
        outcome_col="log_gdp_cum_h3",
        endog_col="shock_efw",
        running_col=RUNNING,
        instrument_col=instrument,
        controls=controls,
        window=window,
        cluster="iso3c",
    )
    rows.append(
        {
            "check": "exclude_crisis",
            "sample": sample_name,
            "horizon": 3,
            "window": window,
            "coef": est.coef,
            "se": est.se,
            "n_obs": est.n_obs,
        }
    )

# Year fixed effects
for sample_name, frame, instrument, window in [
    ("positive", panel_pos, "z_pos", window_pos),
    ("negative", panel_neg, "z_neg", window_neg),
]:
    est = safe_iv_estimate(
        frame,
        outcome_col="log_gdp_cum_h3",
        endog_col="shock_efw",
        running_col=RUNNING,
        instrument_col=instrument,
        controls=controls,
        window=window,
        cluster="iso3c",
        year_fe=True,
        year_col="election_year",
    )
    rows.append(
        {
            "check": "year_fe",
            "sample": sample_name,
            "horizon": 3,
            "window": window,
            "coef": est.coef,
            "se": est.se,
            "n_obs": est.n_obs,
        }
    )

robust_df = pd.DataFrame(rows)

# %%
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
robust_path = PAPER_TABLES_DIR / "robustness_suite.csv"
robust_df.to_csv(robust_path, index=False)

# %%
# Plot bandwidth sensitivity
fig, ax = plt.subplots(figsize=(6, 3))
for sample_name, color in [("positive", "#2A9D8F"), ("negative", "#E76F51")]:
    subset = robust_df[(robust_df["check"] == "bandwidth") & (robust_df["sample"] == sample_name)]
    ax.plot(subset["window"], subset["coef"], marker="o", label=sample_name.capitalize(), color=color)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Window")
ax.set_ylabel("IV coef (h=3)")
ax.set_title("Bandwidth sensitivity")
ax.legend(frameon=False)

savefig(fig, PAPER_FIGURES_DIR / "robustness" / "bandwidth_sensitivity")
plt.close(fig)

# %%
display(robust_df.head(10).style.set_caption("Robustness suite (sample)"))

# %% [markdown]
# ## Interpretation
# Robustness checks assess sensitivity to window choice, donut exclusions,
# placebo cutoffs, pre-trends, alternative thresholds, outcome measures, crisis
# periods, and year fixed effects.
