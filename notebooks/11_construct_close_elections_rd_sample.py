# %% [markdown]
# # Construct close-election RD sample (vote-margin)
# This notebook defines market-oriented winners, constructs signed vote-margin
# running variables, and links pre/post-election incumbency for the close-election design.

# %%
from __future__ import annotations

import json
import os
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
from src.ideology import (
    build_ideology_bundle,
    classify_market_ideology,
    classify_market_lr,
    normalize_party_name,
    prepare_vparty_positions,
)
from src.paths import ANALYSIS_DIR, CLEAN_DIR, INTERMEDIATE_DIR, PAPER_LOGS_DIR, RAW_DIR
from src.qc import assert_unique_key
from src.viz_style import set_style

# %%
set_style()

panel_path = CLEAN_DIR / "panel_annual_atlas.parquet"
if not panel_path.exists():
    raise FileNotFoundError("Missing annual panel. Run 03b_build_annual_panel first.")

panel = pd.read_parquet(panel_path)

# %%
# Source selection (primary: CLEA if present, fallback: ParlGov)
clea_elections_path = INTERMEDIATE_DIR / "clea_elections.parquet"
clea_results_path = INTERMEDIATE_DIR / "clea_results.parquet"

parlgov_elections_path = INTERMEDIATE_DIR / "parlgov_elections.parquet"
parlgov_results_path = INTERMEDIATE_DIR / "parlgov_election_results.parquet"
parlgov_cabinets_path = INTERMEDIATE_DIR / "parlgov_cabinets.parquet"
parlgov_cabinet_parties_path = INTERMEDIATE_DIR / "parlgov_cabinet_parties.parquet"
parlgov_parties_path = INTERMEDIATE_DIR / "parlgov_parties.parquet"

source_pref = os.getenv("ELECTION_SOURCE", "auto").strip().lower()
use_clea = clea_elections_path.exists() and clea_results_path.exists()
if source_pref == "clea":
    use_clea = True
if source_pref == "parlgov":
    use_clea = False
source_label = "clea" if use_clea else "parlgov"

# %%
START_YEAR = 2000
max_year = int(panel["year"].max()) - 3

# %%
# Helper: compute top-2 margin from national vote shares

def compute_top2_margin(results: pd.DataFrame, vote_share_col: str = "vote_share") -> pd.DataFrame:
    frame = results.dropna(subset=[vote_share_col]).copy()
    frame["vote_share_raw"] = frame[vote_share_col].astype(float)
    scale = 100.0 if frame["vote_share_raw"].max() > 1.5 else 1.0
    frame["vote_share_frac"] = frame["vote_share_raw"] / scale
    frame = frame.sort_values(["election_id", "vote_share_frac"], ascending=[True, False])
    top2 = frame.groupby("election_id", as_index=False).head(2).copy()
    top2["rank"] = top2.groupby("election_id").cumcount() + 1

    share = (
        top2.pivot_table(index="election_id", columns="rank", values="vote_share_frac", aggfunc="first")
        .rename(columns={1: "top1_share", 2: "top2_share"})
        .reset_index()
    )
    party = (
        top2.pivot_table(index="election_id", columns="rank", values="party_id", aggfunc="first")
        .rename(columns={1: "top1_party_id", 2: "top2_party_id"})
        .reset_index()
    )
    party_name = (
        top2.pivot_table(index="election_id", columns="rank", values="party_name", aggfunc="first")
        .rename(columns={1: "top1_party_name", 2: "top2_party_name"})
        .reset_index()
    )

    out = share.merge(party, on="election_id", how="left")
    out = out.merge(party_name, on="election_id", how="left")
    out["top2_margin"] = out["top1_share"] - out["top2_share"]
    out["top2_margin_abs"] = out["top2_margin"].abs()
    return out


# %%
if use_clea:
    elections = pd.read_parquet(clea_elections_path)
    results = pd.read_parquet(clea_results_path)

    elections = elections[elections["election_year"].notna()].copy()
    elections["election_year"] = elections["election_year"].astype(int)
    elections = elections[(elections["election_year"] >= START_YEAR) & (elections["election_year"] <= max_year)]

    top2_margin = compute_top2_margin(results)
    elections = elections.merge(top2_margin, on="election_id", how="left")

    # Ideology mapping (V-Party primary; DPI fallback for incumbency)
    dpi_path = RAW_DIR / "dpi" / "dpi2012.xls"
    ideology = build_ideology_bundle(dpi_path, RAW_DIR / "vparty")

    vparty_path = RAW_DIR / "vparty" / "CPD_V-Party_CSV_v2" / "V-Dem-CPD-Party-V2.csv"
    vparty_map = prepare_vparty_positions(vparty_path) if vparty_path.exists() else None

    results = results.copy()
    results["party_name_norm"] = results["party_name"].apply(normalize_party_name)

    if vparty_map is not None and not vparty_map.empty:
        results = results.merge(
            vparty_map,
            left_on=["iso3c", "election_year", "party_name_norm"],
            right_on=["iso3c", "year", "party_name_norm"],
            how="left",
        )
        results["market_party"] = results["ideology_lr"].apply(classify_market_lr)

        vote_totals = results.groupby("election_id")["votes"].sum().rename("vote_total")
        market_votes = (
            results.loc[results["market_party"] == 1]
            .groupby("election_id")["votes"]
            .sum()
            .rename("vote_market")
        )
        elections = elections.merge(vote_totals, on="election_id", how="left")
        elections = elections.merge(market_votes, on="election_id", how="left")
        elections["vote_share_market"] = elections["vote_market"] / elections["vote_total"].replace(0, np.nan)

        elections["winner_market"] = np.where(
            elections["vote_share_market"].notna(),
            (elections["vote_share_market"] > 0.5).astype(float),
            np.nan,
        )
        elections["running_var_vote"] = elections["vote_share_market"] - 0.5
        elections["running_var_vote_margin"] = 2 * elections["vote_share_market"] - 1
    else:
        elections["vote_share_market"] = np.nan
        elections["winner_market"] = np.nan
        elections["running_var_vote"] = np.nan
        elections["running_var_vote_margin"] = np.nan

    # DPI incumbency fallback
    if ideology.dpi_exec is not None:
        dpi = ideology.dpi_exec.copy()
        dpi["market_exec"] = dpi["exec_rlc"].apply(classify_market_ideology)
        dpi["market_gov1"] = dpi["gov1_rlc"].apply(classify_market_ideology)
        dpi["market_best"] = dpi["market_exec"].combine_first(dpi["market_gov1"])

        incumbent = dpi[["iso3c", "year", "market_best"]].copy()
        incumbent["election_year"] = incumbent["year"] + 1
        incumbent = incumbent.rename(columns={"market_best": "incumbent_market"})
        elections = elections.merge(
            incumbent[["iso3c", "election_year", "incumbent_market"]],
            on=["iso3c", "election_year"],
            how="left",
        )
    else:
        elections["incumbent_market"] = np.nan

    # Signed running variable: top-2 margin signed by winner ideology (fallback)
    elections["running_var_top2"] = elections["top2_margin"]
    elections["running_var_vote_alt"] = np.nan
    elections["running_var_seat"] = np.nan
    elections["running_var_seat_alt"] = np.nan
    elections["vote_share_market_alt"] = np.nan

    if elections["running_var_vote"].notna().sum() == 0 and elections["winner_market"].notna().sum() > 0:
        elections["running_var_vote"] = elections["top2_margin"] * (2 * elections["winner_market"] - 1)
        elections["running_var_vote_margin"] = elections["running_var_vote"]

    elections["market_majority_vote"] = np.where(
        elections["running_var_vote"].notna(),
        (elections["running_var_vote"] > 0).astype(float),
        np.nan,
    )
    elections["market_majority_vote_alt"] = np.nan
    elections["market_majority_seat"] = np.nan
    elections["market_majority_top2"] = np.where(
        elections["running_var_top2"].notna(),
        (elections["running_var_top2"] > 0).astype(float),
        np.nan,
    )

    elections["market_switch"] = np.where(
        elections["winner_market"].notna() & elections["incumbent_market"].notna(),
        (elections["winner_market"] != elections["incumbent_market"]).astype(float),
        np.nan,
    )
else:
    for path in [
        parlgov_elections_path,
        parlgov_results_path,
        parlgov_cabinets_path,
        parlgov_cabinet_parties_path,
        parlgov_parties_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing ParlGov output: {path}")

    elections = pd.read_parquet(parlgov_elections_path)
    results = pd.read_parquet(parlgov_results_path)
    cabinets = pd.read_parquet(parlgov_cabinets_path)
    cabinet_parties = pd.read_parquet(parlgov_cabinet_parties_path)
    parties = pd.read_parquet(parlgov_parties_path)

    MARKET_THRESHOLD = 5.0
    ALT_THRESHOLD = 6.0

    elections = elections[elections["election_type"].str.contains("Parliament", case=False, na=False)].copy()
    elections = elections[elections["year"].notna()].copy()
    elections["year"] = elections["year"].astype(int)
    elections = elections[(elections["year"] >= START_YEAR) & (elections["year"] <= max_year)].copy()

    results = results.dropna(subset=["seats"]).copy()

    # Enrich ParlGov party ideology with V-Party where missing
    vparty_path = RAW_DIR / "vparty" / "CPD_V-Party_CSV_v2" / "V-Dem-CPD-Party-V2.csv"
    vparty_map = prepare_vparty_positions(vparty_path) if vparty_path.exists() else None

    parties_enriched = parties.copy()
    if vparty_map is not None and not vparty_map.empty:
        vparty_party = (
            vparty_map.groupby(["iso3c", "party_name_norm"], as_index=False)["ideology_lr"].median()
        )
        vparty_party["left_right_vparty"] = vparty_party["ideology_lr"] * (10.0 / 6.0)

        name_cols = ["name_short", "name_english", "name", "name_ascii"]
        name_rows = []
        for priority, col in enumerate(name_cols):
            if col not in parties_enriched.columns:
                continue
            subset = parties_enriched[["party_id", "iso3c", col]].copy()
            subset = subset.rename(columns={col: "party_name"})
            subset["party_name_norm"] = subset["party_name"].apply(normalize_party_name)
            subset["name_priority"] = priority
            name_rows.append(subset)
        party_names = pd.concat(name_rows, ignore_index=True)
        party_names = party_names[party_names["party_name_norm"] != ""].copy()
        party_names = party_names.drop_duplicates(subset=["party_id", "party_name_norm"])

        matches = party_names.merge(vparty_party, on=["iso3c", "party_name_norm"], how="left")
        matches = matches.dropna(subset=["left_right_vparty"]).copy()
        matches = matches.sort_values(["party_id", "name_priority"])
        best_match = matches.groupby("party_id", as_index=False).first()

        parties_enriched = parties_enriched.merge(
            best_match[["party_id", "left_right_vparty"]],
            on="party_id",
            how="left",
        )
        parties_enriched["left_right_filled"] = parties_enriched["left_right"].combine_first(
            parties_enriched["left_right_vparty"]
        )

        # Optional fuzzy matching for remaining gaps
        try:
            from rapidfuzz import fuzz, process

            vparty_by_country = {}
            for iso3c, group in vparty_party.groupby("iso3c"):
                names = group["party_name_norm"].tolist()
                vparty_by_country[iso3c] = names

            vparty_lr_lookup = {
                (row.iso3c, row.party_name_norm): row.left_right_vparty
                for row in vparty_party.itertuples(index=False)
            }

            def _best_fuzzy_match(iso3c: str, candidates: list[str]) -> float | None:
                if not iso3c or iso3c not in vparty_by_country:
                    return None
                choices = vparty_by_country[iso3c]
                best_score = -1
                best_name = None
                for cand in candidates:
                    if not cand:
                        continue
                    match = process.extractOne(cand, choices, scorer=fuzz.ratio)
                    if match and match[1] > best_score:
                        best_score = match[1]
                        best_name = match[0]
                if best_score >= 90 and best_name is not None:
                    return vparty_lr_lookup.get((iso3c, best_name))
                return None

            missing_mask = parties_enriched["left_right_filled"].isna()
            if missing_mask.any():
                name_cols = ["name_short", "name_english", "name", "name_ascii"]
                for idx, row in parties_enriched.loc[missing_mask].iterrows():
                    candidates = [normalize_party_name(row.get(col)) for col in name_cols]
                    match_lr = _best_fuzzy_match(row.get("iso3c"), candidates)
                    if match_lr is not None:
                        parties_enriched.at[idx, "left_right_filled"] = match_lr
        except Exception:
            pass

        parties_enriched["left_right"] = parties_enriched["left_right_filled"]

    parties = parties_enriched

    # Build results-level ideology using election-year V-Party matches as a final fill
    results_enriched = results.merge(
        parties[["party_id", "left_right", "name_short", "name_english", "name", "name_ascii"]],
        on="party_id",
        how="left",
    )
    results_enriched["year"] = results_enriched["year"].astype(int)
    results_enriched["left_right_result"] = results_enriched["left_right"]

    if vparty_map is not None and not vparty_map.empty:
        vparty_year = (
            vparty_map.groupby(["iso3c", "year", "party_name_norm"], as_index=False)["ideology_lr"].median()
        )
        vparty_year["left_right_vparty_year"] = vparty_year["ideology_lr"] * (10.0 / 6.0)
        vparty_lookup = {
            (row.iso3c, int(row.year), row.party_name_norm): row.left_right_vparty_year
            for row in vparty_year.itertuples(index=False)
        }

        name_cols = ["name_short", "name_english", "name", "name_ascii"]

        def _exact_year_match(row: pd.Series) -> float:
            iso3c = row.get("iso3c")
            year = row.get("year")
            if not iso3c or pd.isna(year):
                return np.nan
            for col in name_cols:
                name_norm = normalize_party_name(row.get(col))
                if not name_norm:
                    continue
                value = vparty_lookup.get((iso3c, int(year), name_norm))
                if value is not None:
                    return value
            return np.nan

        missing_mask = results_enriched["left_right_result"].isna()
        if missing_mask.any():
            results_enriched.loc[missing_mask, "left_right_result"] = (
                results_enriched.loc[missing_mask].apply(_exact_year_match, axis=1)
            )

        # Optional fuzzy year matching for remaining gaps
        try:
            from rapidfuzz import fuzz, process

            vparty_by_year = {}
            for (iso3c, year), group in vparty_year.groupby(["iso3c", "year"]):
                vparty_by_year[(iso3c, int(year))] = group["party_name_norm"].tolist()

            vparty_year_lookup = {
                (row.iso3c, int(row.year), row.party_name_norm): row.left_right_vparty_year
                for row in vparty_year.itertuples(index=False)
            }

            def _best_fuzzy_year_match(row: pd.Series) -> float:
                iso3c = row.get("iso3c")
                year = row.get("year")
                if not iso3c or pd.isna(year):
                    return np.nan
                key = (iso3c, int(year))
                choices = vparty_by_year.get(key)
                if not choices:
                    return np.nan
                best_score = -1
                best_name = None
                for col in name_cols:
                    cand = normalize_party_name(row.get(col))
                    if not cand:
                        continue
                    match = process.extractOne(cand, choices, scorer=fuzz.ratio)
                    if match and match[1] > best_score:
                        best_score = match[1]
                        best_name = match[0]
                if best_score >= 90 and best_name is not None:
                    return vparty_year_lookup.get((iso3c, int(year), best_name))
                return np.nan

            missing_mask = results_enriched["left_right_result"].isna()
            if missing_mask.any():
                results_enriched.loc[missing_mask, "left_right_result"] = (
                    results_enriched.loc[missing_mask].apply(_best_fuzzy_year_match, axis=1)
                )
        except Exception:
            pass

    results = results_enriched.drop(columns=["left_right"]).rename(columns={"left_right_result": "left_right"})

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
    elections["market_majority_seat"] = np.where(
        elections["running_var_seat"].notna(),
        (elections["running_var_seat"] > 0).astype(int),
        np.nan,
    )
    elections["market_majority_vote"] = np.where(
        elections["running_var_vote"].notna(),
        (elections["running_var_vote"] > 0).astype(int),
        np.nan,
    )
    elections["market_majority_vote_alt"] = np.where(
        elections["running_var_vote_alt"].notna(),
        (elections["running_var_vote_alt"] > 0).astype(int),
        np.nan,
    )
    elections["market_majority_top2"] = np.where(
        elections["running_var_top2"].notna(),
        (elections["running_var_top2"] > 0).astype(int),
        np.nan,
    )

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
# Keep one election per country-year
if "election_date" in elections.columns:
    elections = elections.sort_values("election_date")
else:
    elections = elections.sort_values(["iso3c", "election_year", "election_id"])

if "year" in elections.columns:
    elections = elections.rename(columns={"year": "election_year"})

# Align to EFW coverage
efw_countries = panel["iso3c"].dropna().unique().tolist()

if "iso3c" not in elections.columns:
    raise ValueError("Elections data must include iso3c codes.")

elections = elections[elections["iso3c"].isin(efw_countries)].copy()

# Select last election per country-year
if "election_year" not in elections.columns:
    raise ValueError("Elections data must include election_year.")

elections = elections.groupby(["iso3c", "election_year"], as_index=False).tail(1).reset_index(drop=True)

# %%
# Finalize events
assert_unique_key(elections, ["iso3c", "election_year"])

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
sample_path = ANALYSIS_DIR / "close_elections_vote_margin.parquet"
legacy_path = ANALYSIS_DIR / "close_elections_sample.parquet"

elections.to_parquet(sample_path, index=False)
elections.to_parquet(legacy_path, index=False)

coverage = pd.DataFrame(
    {
        "rows": [len(elections)],
        "countries": [elections["iso3c"].nunique()],
        "min_year": [elections["election_year"].min()],
        "max_year": [elections["election_year"].max()],
        "share_market_majority_vote": [elections["market_majority_vote"].mean()],
        "share_market_majority_seat": [
            elections["market_majority_seat"].mean() if "market_majority_seat" in elections.columns else np.nan
        ],
    }
)

display(coverage.style.set_caption("Close-election sample summary"))
display(elections.head(5).style.set_caption("Sample rows"))

running = elections["running_var_vote"].dropna()
heaping_share = (running.round(2) == running).mean() if not running.empty else np.nan
missing_vote_share = elections["vote_share_market"].isna().mean() if "vote_share_market" in elections.columns else np.nan
missing_top2 = elections["running_var_top2"].isna().mean() if "running_var_top2" in elections.columns else np.nan

meta = {
    "build": {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "rd-iv-vote-margin-v2",
    },
    "definitions": {
        "source": source_label,
        "running_variable": (
            "vote_share_market - 0.5 (V-Party bloc)"
            if use_clea and elections["vote_share_market"].notna().any()
            else ("signed top-2 vote margin (winner ideology)" if use_clea else "vote_share_market - 0.5")
        ),
    },
    "inputs": {
        "panel_annual": str(panel_path),
        "clea_elections": str(clea_elections_path) if use_clea else None,
        "clea_results": str(clea_results_path) if use_clea else None,
        "parlgov_elections": str(parlgov_elections_path) if not use_clea else None,
        "parlgov_results": str(parlgov_results_path) if not use_clea else None,
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
# The close-election sample uses a signed vote-margin running variable that is positive
# for more-market winners and negative for less-market winners. CLEA is the primary
# source when available; ParlGov is used only as a fallback.
