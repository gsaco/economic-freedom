# %% [markdown]
# # Postmortem: seat-share RD failure
# Diagnose manipulation and balance issues when using seat shares as the running variable.

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

from src.paths import ANALYSIS_DIR, CLEAN_DIR, PAPER_FIGURES_DIR, PAPER_TABLES_DIR
from src.rd import density_discontinuity, rd_estimate, select_bandwidth
from src.shocks import build_event_panel
from src.viz_style import savefig, set_style

# %%
set_style()

panel_path = CLEAN_DIR / "panel_annual_atlas.parquet"
if not panel_path.exists():
    raise FileNotFoundError("Missing annual panel. Run 03b_build_annual_panel first.")

sample_path = ANALYSIS_DIR / "close_elections_vote_margin.parquet"
if not sample_path.exists():
    raise FileNotFoundError("Missing close-election sample. Run 11_construct_close_elections_rd_sample first.")

panel = pd.read_parquet(panel_path)
events = pd.read_parquet(sample_path)

# %%
# Attach lagged covariates for balance checks
panel_with_cov = build_event_panel(panel, events, event_year_col="election_year", horizons=(0,))

RUNNING = "running_var_seat"
bandwidth = select_bandwidth(panel_with_cov[RUNNING], quantile=0.3, max_bw=0.2)
if pd.isna(bandwidth):
    bandwidth = 0.1

# %%
subset = panel_with_cov[panel_with_cov[RUNNING].abs() <= bandwidth]
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(subset[RUNNING], bins=40, color="#E76F51", edgecolor="white")
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Seat-share running variable density")
ax.set_xlabel("Seat share margin (market bloc)")
ax.set_ylabel("Count")

savefig(fig, PAPER_FIGURES_DIR / "postmortem" / "seatshare_density")
plt.close(fig)

stats = density_discontinuity(panel_with_cov[RUNNING], bandwidth=bandwidth)
stats_df = pd.DataFrame([stats])

# %%
# Balance checks
balance_vars = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
]

balance_rows = []
for var in balance_vars:
    if var not in panel_with_cov.columns:
        continue
    estimate = rd_estimate(panel_with_cov.dropna(subset=[var]), var, RUNNING, bandwidth=bandwidth, cluster="iso3c")
    balance_rows.append(
        {
            "variable": var,
            "coef": estimate.coef,
            "se": estimate.se,
            "pvalue": estimate.pvalue,
            "n_obs": estimate.n_obs,
            "bandwidth": estimate.bandwidth,
        }
    )

balance_df = pd.DataFrame(balance_rows)

# %%
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
summary_path = PAPER_TABLES_DIR / "postmortem_summary.csv"
summary_df = pd.concat(
    [
        stats_df.assign(metric="density"),
        balance_df.assign(metric="balance"),
    ],
    ignore_index=True,
)
summary_df.to_csv(summary_path, index=False)

display(stats_df.style.set_caption("Seat-share density discontinuity"))
display(balance_df.style.set_caption("Seat-share balance checks"))

# %% [markdown]
# ## Interpretation
# Seat shares exhibit mechanical heaping around the cutoff and generate
# covariate imbalances, undermining the continuity assumptions required for
# causal RD interpretation.
