# %% [markdown]
# # Define positive and negative shock samples
# Split the close-election sample by pre-election incumbency orientation,
# creating positive (reform) and negative (reversal) samples.

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
from src.qc import assert_unique_key
from src.viz_style import set_style

# %%
set_style()

sample_path = ANALYSIS_DIR / "close_elections_vote_margin.parquet"
if not sample_path.exists():
    raise FileNotFoundError("Missing close-election sample. Run 11_construct_close_elections_rd_sample first.")

sample = pd.read_parquet(sample_path)

required_cols = ["running_var_vote", "incumbent_market", "winner_market"]
missing = [col for col in required_cols if col not in sample.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# %%
# Define instruments
sample = sample.copy()
sample["z_pos"] = (sample["running_var_vote"] > 0).astype(int)
sample["z_neg"] = (sample["running_var_vote"] < 0).astype(int)

# Define samples
sample_pos = sample[sample["incumbent_market"] == 0].copy()
sample_neg = sample[sample["incumbent_market"] == 1].copy()

assert_unique_key(sample_pos, ["iso3c", "election_year"])
assert_unique_key(sample_neg, ["iso3c", "election_year"])

# %%
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

pos_path = ANALYSIS_DIR / "rd_sample_pos.parquet"
neg_path = ANALYSIS_DIR / "rd_sample_neg.parquet"

sample_pos.to_parquet(pos_path, index=False)
sample_neg.to_parquet(neg_path, index=False)

summary = pd.DataFrame(
    [
        {
            "sample": "positive_shocks",
            "rows": len(sample_pos),
            "countries": sample_pos["iso3c"].nunique(),
            "min_year": sample_pos["election_year"].min(),
            "max_year": sample_pos["election_year"].max(),
            "share_market_win": sample_pos["z_pos"].mean(),
        },
        {
            "sample": "negative_shocks",
            "rows": len(sample_neg),
            "countries": sample_neg["iso3c"].nunique(),
            "min_year": sample_neg["election_year"].min(),
            "max_year": sample_neg["election_year"].max(),
            "share_market_loss": sample_neg["z_neg"].mean(),
        },
    ]
)

PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
summary_path = PAPER_TABLES_DIR / "sample_sizes_pos_neg.csv"
summary.to_csv(summary_path, index=False)

# %%
display(summary.style.set_caption("Positive vs negative shock samples"))

# %% [markdown]
# ## Interpretation
# The sample split isolates reform (positive) and reversal (negative) settings
# using pre-election incumbency, keeping treatment assignment tied to close
# election outcomes rather than ex-post EFW movements.
