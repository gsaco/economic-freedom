# %% [markdown]
# # RD-IV main results
# This notebook estimates the local IV (fuzzy RD) effect of EFW on medium-run
# outcomes using close elections as the instrument.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import ANALYSIS_DIR, PAPER_TABLES_DIR
from src.rd import rd_iv, select_bandwidth
from src.viz_style import set_style

# %%
set_style()

sample_path = ANALYSIS_DIR / "close_elections_sample.parquet"
if not sample_path.exists():
    raise FileNotFoundError("Missing close-election sample. Run 11_construct_close_elections_rd_sample first.")

sample = pd.read_parquet(sample_path)

RUNNING = "running_var"
bandwidth = select_bandwidth(sample[RUNNING], quantile=0.3, max_bw=0.2)
if pd.isna(bandwidth):
    raise ValueError("Unable to select RD bandwidth.")

# %%
outcomes = [
    "gdp_growth_h3",
    "gdp_growth_h5",
    "inv_share_avg_h3",
    "inv_share_avg_h5",
]

rows = []
for outcome in outcomes:
    if outcome not in sample.columns:
        continue
    estimate = rd_iv(
        sample.dropna(subset=[outcome, "efw_post_1_3"]),
        outcome,
        RUNNING,
        endogenous="efw_post_1_3",
        bandwidth=bandwidth,
        cluster="iso3c",
    )
    rows.append(
        {
            "outcome": outcome,
            "coef": estimate.coef,
            "se": estimate.se,
            "pvalue": estimate.pvalue,
            "n_obs": estimate.n_obs,
            "bandwidth": estimate.bandwidth,
        }
    )

iv_df = pd.DataFrame(rows)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
iv_path = PAPER_TABLES_DIR / "rd_iv_main.csv"
iv_df.to_csv(iv_path, index=False)
display(iv_df.style.set_caption("RD-IV estimates (EFW -> outcomes)"))

# %% [markdown]
# ## Interpretation
# The IV estimates translate the discontinuity in EFW induced by close
# elections into local effects on growth and investment, interpreted as a
# LATE for compliers near the cutoff.
