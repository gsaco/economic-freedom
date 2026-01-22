# %% [markdown]
# # Robustness and placebo checks
# This notebook explores bandwidth sensitivity, polynomial order, donut RD,
# placebo cutoffs, and alternative market-orientation thresholds.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import ANALYSIS_DIR, CLEAN_DIR, PAPER_TABLES_DIR
from src.rd import rd_estimate, rd_iv, select_bandwidth
from src.viz_style import set_style

# %%
set_style()

sample_path = ANALYSIS_DIR / "close_elections_sample.parquet"
panel_path = CLEAN_DIR / "panel_annual_atlas.parquet"

if not sample_path.exists():
    raise FileNotFoundError("Missing close-election sample. Run 11_construct_close_elections_rd_sample first.")
if not panel_path.exists():
    raise FileNotFoundError("Missing annual panel. Run 03b_build_annual_panel first.")

sample = pd.read_parquet(sample_path)
panel = pd.read_parquet(panel_path)

RUNNING = "running_var"
bandwidth_main = select_bandwidth(sample[RUNNING], quantile=0.3, max_bw=0.2)
if pd.isna(bandwidth_main):
    raise ValueError("Unable to select RD bandwidth.")

# %%
bandwidth_grid = [0.02, 0.05, 0.1, bandwidth_main]
bandwidth_grid = sorted({round(float(bw), 3) for bw in bandwidth_grid if bw and not np.isnan(bw)})

rows = []
for bw in bandwidth_grid:
    estimate = rd_estimate(
        sample.dropna(subset=["gdp_growth_h3"]),
        "gdp_growth_h3",
        RUNNING,
        bandwidth=bw,
        cluster="iso3c",
    )
    rows.append(
        {
            "bandwidth": bw,
            "outcome": "gdp_growth_h3",
            "coef": estimate.coef,
            "se": estimate.se,
            "pvalue": estimate.pvalue,
            "n_obs": estimate.n_obs,
        }
    )

bandwidth_df = pd.DataFrame(rows)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
bandwidth_df.to_csv(PAPER_TABLES_DIR / "rd_bandwidth_sensitivity.csv", index=False)
display(bandwidth_df.style.set_caption("Bandwidth sensitivity (reduced form)"))

# %%
poly_rows = []
for order in [1, 2]:
    estimate = rd_estimate(
        sample.dropna(subset=["gdp_growth_h3"]),
        "gdp_growth_h3",
        RUNNING,
        bandwidth=bandwidth_main,
        order=order,
        cluster="iso3c",
    )
    poly_rows.append(
        {
            "order": order,
            "outcome": "gdp_growth_h3",
            "coef": estimate.coef,
            "se": estimate.se,
            "pvalue": estimate.pvalue,
            "n_obs": estimate.n_obs,
        }
    )

poly_df = pd.DataFrame(poly_rows)
poly_df.to_csv(PAPER_TABLES_DIR / "rd_polynomial_order.csv", index=False)
display(poly_df.style.set_caption("Polynomial order sensitivity"))

# %%
donut = sample[sample[RUNNING].abs() >= 0.01].copy()
donut_est = rd_estimate(
    donut.dropna(subset=["gdp_growth_h3"]),
    "gdp_growth_h3",
    RUNNING,
    bandwidth=bandwidth_main,
    cluster="iso3c",
)
donut_df = pd.DataFrame(
    [
        {
            "outcome": "gdp_growth_h3",
            "coef": donut_est.coef,
            "se": donut_est.se,
            "pvalue": donut_est.pvalue,
            "n_obs": donut_est.n_obs,
        }
    ]
)
donut_df.to_csv(PAPER_TABLES_DIR / "rd_donut.csv", index=False)
display(donut_df.style.set_caption("Donut RD (|m| >= 0.01)"))

# %%
placebo_cutoffs = [-0.05, -0.02, 0.02, 0.05]
placebo_rows = []
for cutoff in placebo_cutoffs:
    estimate = rd_estimate(
        sample.dropna(subset=["gdp_growth_h3"]),
        "gdp_growth_h3",
        RUNNING,
        bandwidth=bandwidth_main,
        cutoff=cutoff,
        cluster="iso3c",
    )
    placebo_rows.append(
        {
            "cutoff": cutoff,
            "outcome": "gdp_growth_h3",
            "coef": estimate.coef,
            "se": estimate.se,
            "pvalue": estimate.pvalue,
            "n_obs": estimate.n_obs,
        }
    )

placebo_df = pd.DataFrame(placebo_rows)
placebo_df.to_csv(PAPER_TABLES_DIR / "rd_placebo_cutoffs.csv", index=False)
display(placebo_df.style.set_caption("Placebo cutoffs"))

# %%
alt_rows = []
for running in ["running_var", "running_var_alt"]:
    estimate = rd_iv(
        sample.dropna(subset=["gdp_growth_h3", "efw_post_1_3"]),
        "gdp_growth_h3",
        running,
        endogenous="efw_post_1_3",
        bandwidth=bandwidth_main,
        cluster="iso3c",
    )
    alt_rows.append(
        {
            "running": running,
            "coef": estimate.coef,
            "se": estimate.se,
            "pvalue": estimate.pvalue,
            "n_obs": estimate.n_obs,
        }
    )

alt_df = pd.DataFrame(alt_rows)
alt_df.to_csv(PAPER_TABLES_DIR / "rd_alt_threshold_iv.csv", index=False)
display(alt_df.style.set_caption("Alternative market threshold (IV)"))

# %%
panel = panel.copy()
panel["log_pop_total"] = np.log(panel["pop_total"].where(panel["pop_total"] > 0))
panel_index = panel.set_index(["iso3c", "year"])

def _pop_growth(row: pd.Series, horizon: int = 3) -> float:
    iso3c = row["iso3c"]
    year = int(row["election_year"])
    try:
        pre = panel_index.at[(iso3c, year - 1), "log_pop_total"]
        post = panel_index.at[(iso3c, year + horizon), "log_pop_total"]
    except KeyError:
        return np.nan
    if pd.isna(pre) or pd.isna(post):
        return np.nan
    return float(post - pre)


sample["pop_growth_h3"] = sample.apply(_pop_growth, axis=1)
neg_ctrl_est = rd_estimate(
    sample.dropna(subset=["pop_growth_h3"]),
    "pop_growth_h3",
    RUNNING,
    bandwidth=bandwidth_main,
    cluster="iso3c",
)
neg_ctrl_df = pd.DataFrame(
    [
        {
            "outcome": "pop_growth_h3",
            "coef": neg_ctrl_est.coef,
            "se": neg_ctrl_est.se,
            "pvalue": neg_ctrl_est.pvalue,
            "n_obs": neg_ctrl_est.n_obs,
        }
    ]
)
neg_ctrl_df.to_csv(PAPER_TABLES_DIR / "rd_negative_control.csv", index=False)
display(neg_ctrl_df.style.set_caption("Negative control: population growth"))

# %% [markdown]
# ## Interpretation
# Robustness checks probe the sensitivity of the RD-IV estimates to bandwidth,
# polynomial order, donut exclusions, placebo cutoffs, and ideology thresholds.
