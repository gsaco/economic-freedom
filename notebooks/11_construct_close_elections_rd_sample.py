# %% [markdown]
# # Construct close-election RD sample (vote-margin)
# This notebook defines market-oriented blocs, constructs vote-share running
# variables, and links post-election cabinets for the close-election design.

# %%
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.elections_parlgov import (
    compute_bloc_ideology_distance,
    compute_cabinet_ideology,
    compute_market_seat_shares,
    compute_market_vote_shares,
    compute_top2_margin_by_bloc,
    select_post_election_cabinets,
)
from src.paths import ANALYSIS_DIR, CLEAN_DIR, INTERMEDIATE_DIR, PAPER_LOGS_DIR
from src.qc import assert_unique_key
from src.viz_style import set_style

# %%
set_style()

panel_path = CLEAN_DIR / "panel_annual_atlas.parquet"
if not panel_path.exists():
    raise FileNotFoundError("Missing annual panel. Run 03b_build_annual_panel first.")

panel = pd.read_parquet(panel_path)


elections_path = INTERMEDIATE_DIR / "parlgov_elections.parquet"
results_path = INTERMEDIATE_DIR / "parlgov_election_results.parquet"
cabinets_path = INTERMEDIATE_DIR / "parlgov_cabinets.parquet"
cabinet_parties_path = INTERMEDIATE_DIR / "parlgov_cabinet_parties.parquet"
parties_path = INTERMEDIATE_DIR / "parlgov_parties.parquet"

for path in [
    elections_path,
    results_path,
    cabinets_path,
    cabinet_parties_path,
    parties_path,
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ParlGov output: {path}")

elections = pd.read_parquet(elections_path)
results = pd.read_parquet(results_path)
cabinets = pd.read_parquet(cabinets_path)
cabinet_parties = pd.read_parquet(cabinet_parties_path)
parties = pd.read_parquet(parties_path)

# %%
MARKET_THRESHOLD = 5.0
ALT_THRESHOLD = 6.0
START_YEAR = 2000

max_year = int(panel["year"].max()) - 3

elections = elections[elections["election_type"].str.contains("Parliament", case=False, na=False)].copy()
elections = elections[elections["year"].notna()].copy()
elections["year"] = elections["year"].astype(int)
elections = elections[(elections["year"] >= START_YEAR) & (elections["year"] <= max_year)].copy()

# %%
results = results.dropna(subset=["seats"]).copy()

seat_share_main = compute_market_seat_shares(results, parties, threshold=MARKET_THRESHOLD)
seat_share_alt = compute_market_seat_shares(results, parties, threshold=ALT_THRESHOLD).rename(
    columns={
        "seat_market": "seat_market_alt",
        "seat_share_market": "seat_share_market_alt",
    }
)

vote_share_main = compute_market_vote_shares(results, parties, threshold=MARKET_THRESHOLD)
vote_share_alt = compute_market_vote_shares(results, parties, threshold=ALT_THRESHOLD).rename(
    columns={
        "vote_share_market_raw": "vote_share_market_raw_alt",
        "vote_share_market": "vote_share_market_alt",
    }
)

ideology_distance = compute_bloc_ideology_distance(results, parties, threshold=MARKET_THRESHOLD)
top2_margin = compute_top2_margin_by_bloc(results, parties, threshold=MARKET_THRESHOLD)

# %%
elections = elections.merge(seat_share_main, on="election_id", how="left")
elections = elections.merge(
    seat_share_alt[["election_id", "seat_market_alt", "seat_share_market_alt"]],
    on="election_id",
    how="left",
)
elections = elections.merge(vote_share_main, on="election_id", how="left")
elections = elections.merge(
    vote_share_alt[["election_id", "vote_share_market_alt"]],
    on="election_id",
    how="left",
)
elections = elections.merge(
    ideology_distance[["election_id", "lr_market", "lr_nonmarket", "lr_distance", "lr_distance_abs"]],
    on="election_id",
    how="left",
)
elections = elections.merge(
    top2_margin[
        [
            "election_id",
            "top_market_share",
            "top_nonmarket_share",
            "top2_margin",
            "top2_margin_abs",
            "winner_market_top2",
        ]
    ],
    on="election_id",
    how="left",
)

# Running variables

elections["running_var_seat"] = elections["seat_share_market"] - 0.5
elections["running_var_seat_alt"] = elections["seat_share_market_alt"] - 0.5
elections["running_var_vote"] = elections["vote_share_market"] - 0.5
elections["running_var_vote_margin"] = 2 * elections["vote_share_market"] - 1
elections["running_var_vote_alt"] = elections["vote_share_market_alt"] - 0.5
elections["running_var_top2"] = elections["top2_margin"]

# Treatment indicators

elections["market_majority_seat"] = (elections["running_var_seat"] > 0).astype(int)
elections["market_majority_vote"] = (elections["running_var_vote"] > 0).astype(int)
elections["market_majority_vote_alt"] = (elections["running_var_vote_alt"] > 0).astype(int)
elections["market_majority_top2"] = (elections["running_var_top2"] > 0).astype(int)

# %%
# Cabinet ideology and incumbency
cabinet_summary = compute_cabinet_ideology(
    cabinets,
    cabinet_parties,
    parties,
    results,
    threshold=MARKET_THRESHOLD,
)

post_cabinets = select_post_election_cabinets(cabinets)
post_cabinets = post_cabinets.merge(
    cabinet_summary,
    left_on="post_cabinet_id",
    right_on="cabinet_id",
    how="left",
)
post_cabinets = post_cabinets.rename(
    columns={
        "cabinet_lr": "post_cabinet_lr",
        "cabinet_market_share": "post_cabinet_market_share",
        "cabinet_party_count": "post_cabinet_party_count",
    }
)
post_cabinets["winner_market"] = (post_cabinets["post_cabinet_lr"] >= MARKET_THRESHOLD).astype(float)

incumbent = post_cabinets.merge(
    cabinet_summary,
    left_on="previous_cabinet_id",
    right_on="cabinet_id",
    how="left",
    suffixes=("", "_incumbent"),
)
incumbent = incumbent.rename(
    columns={
        "cabinet_lr": "incumbent_cabinet_lr",
        "cabinet_market_share": "incumbent_cabinet_market_share",
        "cabinet_party_count": "incumbent_cabinet_party_count",
    }
)
incumbent["incumbent_market"] = (incumbent["incumbent_cabinet_lr"] >= MARKET_THRESHOLD).astype(float)

cabinet_cols = [
    "election_id",
    "post_cabinet_id",
    "previous_cabinet_id",
    "post_cabinet_lr",
    "post_cabinet_market_share",
    "post_cabinet_party_count",
    "winner_market",
    "incumbent_cabinet_lr",
    "incumbent_cabinet_market_share",
    "incumbent_cabinet_party_count",
    "incumbent_market",
]

elections = elections.merge(incumbent[cabinet_cols], on="election_id", how="left")
elections["market_switch"] = (elections["winner_market"] != elections["incumbent_market"]).astype(float)

# %%
# Keep one parliamentary election per country-year

elections = elections.sort_values("election_date")
elections = elections.groupby(["iso3c", "year"], as_index=False).tail(1).reset_index(drop=True)

# %%
events = elections.rename(columns={"year": "election_year"})
assert_unique_key(events, ["iso3c", "election_year"])

# %%
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
sample_path = ANALYSIS_DIR / "close_elections_vote_margin.parquet"
legacy_path = ANALYSIS_DIR / "close_elections_sample.parquet"

events.to_parquet(sample_path, index=False)
events.to_parquet(legacy_path, index=False)

coverage = pd.DataFrame(
    {
        "rows": [len(events)],
        "countries": [events["iso3c"].nunique()],
        "min_year": [events["election_year"].min()],
        "max_year": [events["election_year"].max()],
        "share_market_majority_vote": [events["market_majority_vote"].mean()],
        "share_market_majority_seat": [events["market_majority_seat"].mean()],
    }
)
display(coverage.style.set_caption("Close-election sample summary"))
display(events.head(5).style.set_caption("Sample rows"))

running = events["running_var_vote"].dropna()
heaping_share = (running.round(2) == running).mean() if not running.empty else np.nan
missing_vote_share = events["vote_share_market"].isna().mean()
missing_top2 = events["running_var_top2"].isna().mean()

meta = {
    "build": {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "rd-iv-vote-margin-v1",
    },
    "definitions": {
        "market_threshold": MARKET_THRESHOLD,
        "alt_threshold": ALT_THRESHOLD,
        "running_variable": "vote_share_market - 0.5",
    },
    "inputs": {
        "panel_annual": str(panel_path),
        "parlgov_elections": str(elections_path),
        "parlgov_results": str(results_path),
        "parlgov_cabinets": str(cabinets_path),
        "parlgov_parties": str(parties_path),
    },
    "summary": {
        **coverage.to_dict(orient="records")[0],
        "missing_vote_share": missing_vote_share,
        "missing_top2_margin": missing_top2,
    },
    "qc": {"running_var_vote_heaping_share": heaping_share},
}

meta_path = PAPER_LOGS_DIR / "close_elections_vote_margin_metadata.json"
meta_path.parent.mkdir(parents=True, exist_ok=True)
meta_path.write_text(json.dumps(meta, indent=2))

# %% [markdown]
# ## Interpretation
# The close-election sample now uses vote shares to construct a continuous
# running variable, while preserving cabinet ideology and incumbency markers
# required for positive vs negative shock splits.
