# %% [markdown]
# # Longer post windows for shocks (t+1..t+5, t+1..t+7)
# Rebuild shocks with longer post windows and compare first stage / IRFs.

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

from src.lpiv import first_stage, iv_estimate
from src.paths import ANALYSIS_DIR, CLEAN_DIR, PAPER_FIGURES_DIR, PAPER_LOGS_DIR, PAPER_TABLES_DIR
from src.shocks import ShockSpec, build_event_panel
from src.viz_style import savefig, set_style

# %%
set_style()

panel_path = CLEAN_DIR / "panel_annual_atlas.parquet"
pos_path = ANALYSIS_DIR / "rd_sample_pos.parquet"
neg_path = ANALYSIS_DIR / "rd_sample_neg.parquet"

for path in [panel_path, pos_path, neg_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

panel = pd.read_parquet(panel_path)
sample_pos = pd.read_parquet(pos_path)
sample_neg = pd.read_parquet(neg_path)

# %%
# Build event panels with longer post windows
spec_post5 = ShockSpec(pre_window=(-3, -1), post_window=(1, 5), min_pre_obs=2, min_post_obs=3)
spec_post7 = ShockSpec(pre_window=(-3, -1), post_window=(1, 7), min_pre_obs=2, min_post_obs=4)

horizons = (0, 1, 2, 3, 4, 5)

panel_pos_15 = build_event_panel(
    panel,
    sample_pos,
    event_year_col="election_year",
    horizons=horizons,
    shock_spec=spec_post5,
)
panel_neg_15 = build_event_panel(
    panel,
    sample_neg,
    event_year_col="election_year",
    horizons=horizons,
    shock_spec=spec_post5,
)

panel_pos_17 = build_event_panel(
    panel,
    sample_pos,
    event_year_col="election_year",
    horizons=horizons,
    shock_spec=spec_post7,
)
panel_neg_17 = build_event_panel(
    panel,
    sample_neg,
    event_year_col="election_year",
    horizons=horizons,
    shock_spec=spec_post7,
)

# Instruments
for frame in [panel_pos_15, panel_pos_17]:
    frame["z_pos"] = (frame["running_var_vote"] > 0).astype(int)
for frame in [panel_neg_15, panel_neg_17]:
    frame["z_neg"] = (frame["running_var_vote"] < 0).astype(int)

RUNNING = "running_var_vote"
controls = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
]

# Windows from baseline
window_choice_path = PAPER_LOGS_DIR / "rd_window_choice.json"
if window_choice_path.exists():
    window_choice = json.loads(window_choice_path.read_text())
    window_pos = window_choice.get("positive", {}).get("window") or 0.05
    window_neg = window_choice.get("negative", {}).get("window") or 0.02
else:
    window_pos, window_neg = 0.05, 0.02

# %%
# Helper for rank failures

def safe_iv(frame: pd.DataFrame, **kwargs):
    try:
        return iv_estimate(frame, **kwargs)
    except ValueError:
        return iv_estimate(frame.iloc[0:0], **kwargs)


# First stage comparison
rows = []
for label, frame, instrument, window in [
    ("pos_post5", panel_pos_15, "z_pos", window_pos),
    ("neg_post5", panel_neg_15, "z_neg", window_neg),
    ("pos_post7", panel_pos_17, "z_pos", window_pos),
    ("neg_post7", panel_neg_17, "z_neg", window_neg),
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
    rows.append(
        {
            "variant": label,
            "coef": fs.coef,
            "se": fs.se,
            "pvalue": fs.pvalue,
            "n_obs": fs.n_obs,
            "window": fs.window,
        }
    )

fs_df = pd.DataFrame(rows)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
fs_path = PAPER_TABLES_DIR / "irf_first_stage_longer_post.csv"
fs_df.to_csv(fs_path, index=False)

# %%
# IRFs (h=0..5) for post5 and post7
iv_rows = []
for h in horizons:
    for label, frame, instrument, window in [
        ("pos_post5", panel_pos_15, "z_pos", window_pos),
        ("neg_post5", panel_neg_15, "z_neg", window_neg),
        ("pos_post7", panel_pos_17, "z_pos", window_pos),
        ("neg_post7", panel_neg_17, "z_neg", window_neg),
    ]:
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
                "variant": label,
                "horizon": h,
                "coef": iv.coef,
                "se": iv.se,
                "pvalue": iv.pvalue,
                "n_obs": iv.n_obs,
                "window": iv.window,
            }
        )

iv_df = pd.DataFrame(iv_rows)
iv_path = PAPER_TABLES_DIR / "irf_iv_longer_post.csv"
iv_df.to_csv(iv_path, index=False)

# %%
# Plot comparison for post5 vs post7 (positives)
fig, ax = plt.subplots(figsize=(6, 3))
for label, color in [("pos_post5", "#2A9D8F"), ("pos_post7", "#1F4E79")]:
    subset = iv_df[iv_df["variant"] == label].sort_values("horizon")
    ax.plot(subset["horizon"], subset["coef"], marker="o", label=label, color=color)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Horizon")
ax.set_ylabel("Log GDP per capita (cum)")
ax.set_title("Longer post windows (positive shocks)")
ax.legend(frameon=False)

savefig(fig, PAPER_FIGURES_DIR / "irfs_longer_post" / "irf_pos_post5_post7")
plt.close(fig)

# %%
# Plot comparison for post5 vs post7 (negatives)
fig, ax = plt.subplots(figsize=(6, 3))
for label, color in [("neg_post5", "#E76F51"), ("neg_post7", "#A23B2A")]:
    subset = iv_df[iv_df["variant"] == label].sort_values("horizon")
    ax.plot(subset["horizon"], subset["coef"], marker="o", label=label, color=color)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Horizon")
ax.set_ylabel("Log GDP per capita (cum)")
ax.set_title("Longer post windows (negative shocks)")
ax.legend(frameon=False)

savefig(fig, PAPER_FIGURES_DIR / "irfs_longer_post" / "irf_neg_post5_post7")
plt.close(fig)

# %%
# Compare first stage vs baseline
baseline_pos = pd.read_csv(PAPER_TABLES_DIR / "irf_first_stage_pos.csv")
baseline_neg = pd.read_csv(PAPER_TABLES_DIR / "irf_first_stage_neg.csv")

compare_rows = []
for sample_name, df in [("pos_baseline", baseline_pos), ("neg_baseline", baseline_neg)]:
    row = df.loc[df["outcome"] == "shock_efw"].iloc[0]
    compare_rows.append(
        {
            "variant": sample_name,
            "coef": row["coef"],
            "se": row["se"],
            "pvalue": row["pvalue"],
            "n_obs": row["n_obs"],
        }
    )

compare_rows += fs_df.to_dict(orient="records")
compare_df = pd.DataFrame(compare_rows)
compare_path = PAPER_TABLES_DIR / "longer_post_vs_baseline_summary.csv"
compare_df.to_csv(compare_path, index=False)

# %%
display(compare_df.style.set_caption("First-stage comparison: baseline vs longer post"))
