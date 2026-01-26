# %% [markdown]
# # Construct EFW shocks and LP outcomes
# Build event panels with pre/post EFW shocks and local-projection outcomes.

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

from src.crisis import load_systemic_crisis_years
from src.paths import ANALYSIS_DIR, CLEAN_DIR, INTERMEDIATE_DIR, PAPER_TABLES_DIR, RAW_DIR
from src.shocks import ShockSpec, build_event_panel
from src.viz_style import set_style

# %%
set_style()

panel_path = CLEAN_DIR / "panel_annual_atlas.parquet"
if not panel_path.exists():
    raise FileNotFoundError("Missing annual panel. Run 03b_build_annual_panel first.")

pos_path = ANALYSIS_DIR / "rd_sample_pos.parquet"
neg_path = ANALYSIS_DIR / "rd_sample_neg.parquet"
full_path = ANALYSIS_DIR / "close_elections_vote_margin.parquet"

for path in [pos_path, neg_path, full_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing RD sample: {path}")

panel = pd.read_parquet(panel_path)
sample_pos = pd.read_parquet(pos_path)
sample_neg = pd.read_parquet(neg_path)
sample_all = pd.read_parquet(full_path)
efw_area_cols = [col for col in panel.columns if col.startswith("efw_area")]

crisis_path = RAW_DIR / "crisis" / "laeven_valencia_2020.xlsx"
country_meta_path = INTERMEDIATE_DIR / "country_metadata.parquet"
country_meta = pd.read_parquet(country_meta_path) if country_meta_path.exists() else pd.DataFrame()
crisis_years = (
    load_systemic_crisis_years(crisis_path, country_meta=country_meta)
    if crisis_path.exists()
    else pd.DataFrame()
)

# %%
shock_spec = ShockSpec(pre_window=(-3, -1), post_window=(1, 3), min_pre_obs=2, min_post_obs=2)

horizons = (0, 1, 2, 3, 4, 5)
pretrend_horizons = (1, 3)

panel_pos = build_event_panel(
    panel,
    sample_pos,
    event_year_col="election_year",
    horizons=horizons,
    pretrend_horizons=pretrend_horizons,
    shock_spec=shock_spec,
    crisis_years=crisis_years,
    efw_extra_cols=tuple(efw_area_cols),
)

panel_neg = build_event_panel(
    panel,
    sample_neg,
    event_year_col="election_year",
    horizons=horizons,
    pretrend_horizons=pretrend_horizons,
    shock_spec=shock_spec,
    crisis_years=crisis_years,
    efw_extra_cols=tuple(efw_area_cols),
)

panel_all = build_event_panel(
    panel,
    sample_all,
    event_year_col="election_year",
    horizons=horizons,
    pretrend_horizons=pretrend_horizons,
    shock_spec=shock_spec,
    crisis_years=crisis_years,
    efw_extra_cols=tuple(efw_area_cols),
)

# %%
# Pre-registered heterogeneity splits
combined = pd.concat([panel_pos, panel_neg], ignore_index=True)
median_lr_distance = combined["lr_distance_abs"].median()
median_efw = combined["lag1_efw_summary"].median()
median_income = combined["lag1_log_gdp_pc_const"].median()

for frame in [panel_pos, panel_neg]:
    frame["large_shift"] = (frame["lr_distance_abs"] >= median_lr_distance).astype(int)
    frame["low_efw"] = (frame["lag1_efw_summary"] <= median_efw).astype(int)
    frame["low_income"] = (frame["lag1_log_gdp_pc_const"] <= median_income).astype(int)

# %%
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

pos_event_path = ANALYSIS_DIR / "rd_event_panel_pos.parquet"
neg_event_path = ANALYSIS_DIR / "rd_event_panel_neg.parquet"
all_event_path = ANALYSIS_DIR / "rd_event_panel.parquet"

panel_pos.to_parquet(pos_event_path, index=False)
panel_neg.to_parquet(neg_event_path, index=False)
panel_all.to_parquet(all_event_path, index=False)

# %%
summary_rows = []
for name, frame in [("positive", panel_pos), ("negative", panel_neg), ("full", panel_all)]:
    shock = frame["shock_efw"]
    summary_rows.append(
        {
            "sample": name,
            "rows": len(frame),
            "shock_mean": shock.mean(),
            "shock_median": shock.median(),
            "shock_std": shock.std(),
            "share_shock_pos": (shock > 0).mean(),
            "share_shock_neg": (shock < 0).mean(),
            "shock_nonmissing": shock.notna().mean(),
        }
    )

summary_df = pd.DataFrame(summary_rows)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
summary_path = PAPER_TABLES_DIR / "efw_shock_summary_pos_neg.csv"
summary_df.to_csv(summary_path, index=False)

# %%
display(summary_df.style.set_caption("EFW shock summary (pre/post averages)"))

# %% [markdown]
# ## Interpretation
# The event panels now include pre/post EFW averages, the discrete EFW shock,
# and cumulative log-GDP outcomes required for local projection IV estimates.
# Pre-registered heterogeneity splits are attached for later asymmetry tests.
