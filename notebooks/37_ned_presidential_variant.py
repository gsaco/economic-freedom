# %% [markdown]
# # NED presidential elections variant (vote shares)
# Construct a vote-share running variable from NED presidential elections by
# mapping parties to ParlGov ideology and estimating RD-IV IRFs.

# %%
from __future__ import annotations

import json
import re
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

from src.lpiv import first_stage, iv_estimate, reduced_form
from src.paths import ANALYSIS_DIR, CLEAN_DIR, PAPER_LOGS_DIR, PAPER_TABLES_DIR
from src.qc import assert_unique_key
from src.rd import density_discontinuity
from src.rd_localrand import select_window_by_balance
from src.shocks import build_event_panel
from src.viz_style import set_style

# %%
set_style()

ned_path = Path("data/01_raw/ned/presidential_elections_v2.dta")
parlgov_parties_path = Path("data/02_intermediate/parlgov_parties.parquet")
panel_path = CLEAN_DIR / "panel_annual_atlas.parquet"

for path in [ned_path, parlgov_parties_path, panel_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

ned = pd.read_stata(ned_path, convert_categoricals=False)
parties = pd.read_parquet(parlgov_parties_path)
panel = pd.read_parquet(panel_path)

# %%
MARKET_THRESHOLD = 5.0


def normalize_name(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


# Build mapping from ParlGov party names
name_cols = ["name", "name_short", "name_english", "name_ascii"]
party_names = []
for _, row in parties.iterrows():
    iso3c = row.get("iso3c")
    lr = row.get("left_right")
    for col in name_cols:
        name = row.get(col)
        norm = normalize_name(name) if isinstance(name, str) else ""
        if norm:
            party_names.append({"iso3c": iso3c, "norm": norm, "left_right": lr})

party_map = pd.DataFrame(party_names).dropna(subset=["iso3c", "left_right"])
party_map = party_map.drop_duplicates(subset=["iso3c", "norm"])

# %%
# Reshape NED parties to long format (round 1 vote shares)
party_cols = [col for col in ned.columns if col.startswith("party_")]

long_rows = []
for i in range(1, len(party_cols) + 1):
    party_col = f"party_{i}"
    vote_col = f"vote_share1_{i}"
    if party_col not in ned.columns or vote_col not in ned.columns:
        continue
    subset = ned[["country_abb", "country", "year", party_col, vote_col]].copy()
    subset = subset.rename(columns={party_col: "party", vote_col: "vote_share"})
    subset["party_rank"] = i
    long_rows.append(subset)

ned_long = pd.concat(long_rows, ignore_index=True)

# Clean
ned_long = ned_long.dropna(subset=["country_abb", "year", "party", "vote_share"])

ned_long["iso3c"] = ned_long["country_abb"].astype(str).str.upper().str.strip()
ned_long["norm_party"] = ned_long["party"].apply(normalize_name)

# Merge ideology
ned_long = ned_long.merge(
    party_map,
    left_on=["iso3c", "norm_party"],
    right_on=["iso3c", "norm"],
    how="left",
)

# %%
# Aggregate to election-level
ned_long["market_party"] = ned_long["left_right"] >= MARKET_THRESHOLD

agg = ned_long.groupby(["iso3c", "year"]).agg(
    vote_share_total=("vote_share", "sum"),
    vote_share_known=("vote_share", lambda x: x[ned_long.loc[x.index, "left_right"].notna()].sum()),
    vote_share_market=("vote_share", lambda x: x[ned_long.loc[x.index, "market_party"]].sum()),
)
agg = agg.reset_index()
agg["vote_share_nonmarket"] = agg["vote_share_known"] - agg["vote_share_market"]
agg["coverage"] = agg["vote_share_known"] / agg["vote_share_total"].replace(0, np.nan)

# Running variable and treatment
agg["running_var_ned"] = agg["vote_share_market"] - agg["vote_share_nonmarket"]
agg["winner_market"] = (agg["running_var_ned"] > 0).astype(float)

# Incumbent = previous election winner
agg = agg.sort_values(["iso3c", "year"])
agg["incumbent_market"] = agg.groupby("iso3c")["winner_market"].shift(1)

# Keep adequate coverage
agg = agg[(agg["coverage"] >= 0.8) & agg["incumbent_market"].notna()].copy()

# %%
# Build event panel with EFW shocks/outcomes
agg = agg.rename(columns={"year": "election_year"})

panel_events = build_event_panel(panel, agg, event_year_col="election_year")
assert_unique_key(panel_events, ["iso3c", "election_year"])

# Define samples
panel_events["z_pos"] = (panel_events["running_var_ned"] > 0).astype(int)
panel_events["z_neg"] = (panel_events["running_var_ned"] < 0).astype(int)

panel_pos = panel_events[panel_events["incumbent_market"] == 0].copy()
panel_neg = panel_events[panel_events["incumbent_market"] == 1].copy()

# %%
# RD validity
balance_vars = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
    "lag1_inv_share_gdp",
]
windows = [0.02, 0.03, 0.04, 0.05, 0.06]

choice_pos, table_pos = select_window_by_balance(
    panel_pos,
    "running_var_ned",
    balance_vars,
    windows=windows,
    p_threshold=0.15,
    cluster="iso3c",
)
choice_neg, table_neg = select_window_by_balance(
    panel_neg,
    "running_var_ned",
    balance_vars,
    windows=windows,
    p_threshold=0.15,
    cluster="iso3c",
)

PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
table_pos.to_csv(PAPER_TABLES_DIR / "rd_validity_ned_pres_pos.csv", index=False)
table_neg.to_csv(PAPER_TABLES_DIR / "rd_validity_ned_pres_neg.csv", index=False)

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
choice_path = PAPER_LOGS_DIR / "rd_window_choice_ned_pres.json"
choice_path.write_text(json.dumps(window_choice, indent=2))

window_pos = choice_pos.window or 0.05
window_neg = choice_neg.window or 0.05

# Density stats
stats_pos = density_discontinuity(panel_pos["running_var_ned"].dropna(), bandwidth=window_pos)
stats_pos["sample"] = "pos"
stats_neg = density_discontinuity(panel_neg["running_var_ned"].dropna(), bandwidth=window_neg)
stats_neg["sample"] = "neg"

pd.DataFrame([stats_pos, stats_neg]).to_csv(PAPER_TABLES_DIR / "rd_density_ned_pres.csv", index=False)

# %%
# Helper for IV rank issues

def safe_iv(frame: pd.DataFrame, **kwargs):
    try:
        return iv_estimate(frame, **kwargs)
    except ValueError:
        return iv_estimate(frame.iloc[0:0], **kwargs)


HORIZONS = [0, 1, 2, 3, 4, 5]
controls = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
]

fs_rows = []
rf_rows = []
iv_rows = []

for sample_name, frame, instrument, window in [
    ("pos", panel_pos, "z_pos", window_pos),
    ("neg", panel_neg, "z_neg", window_neg),
]:
    fs = first_stage(
        frame,
        outcome_col="shock_efw",
        running_col="running_var_ned",
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
            running_col="running_var_ned",
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
            running_col="running_var_ned",
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
fs_df.to_csv(PAPER_TABLES_DIR / "irf_first_stage_ned_pres.csv", index=False)
rf_df.to_csv(PAPER_TABLES_DIR / "irf_reduced_form_ned_pres.csv", index=False)
iv_df.to_csv(PAPER_TABLES_DIR / "irf_iv_ned_pres.csv", index=False)

# %%
# Summary
summary = pd.DataFrame([
    {
        "rows": len(panel_events),
        "countries": panel_events["iso3c"].nunique(),
        "coverage_mean": panel_events["coverage"].mean(),
        "pos_rows": len(panel_pos),
        "neg_rows": len(panel_neg),
    }
])
summary.to_csv(PAPER_TABLES_DIR / "ned_presidential_summary.csv", index=False)

# %%
display(summary.style.set_caption("NED presidential sample summary"))
