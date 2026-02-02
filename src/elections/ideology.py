from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .standardize import normalize_name

try:  # optional
    from rapidfuzz import process, fuzz
except Exception:  # pragma: no cover
    process = None
    fuzz = None


def percentile_rank(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True)


def build_party_crosswalk(pf_external: pd.DataFrame, dataset_keys: list[str]) -> pd.DataFrame:
    subset = pf_external[pf_external["dataset_key"].isin(dataset_keys)].copy()
    return subset[["dataset_key", "dataset_party_id", "partyfacts_id", "country", "name", "name_english"]]


def build_ideology_obs(
    vparty: pd.DataFrame,
    ches: pd.DataFrame,
    elff: pd.DataFrame,
    parlgov_pos: pd.DataFrame,
    pf_core: pd.DataFrame,
    pf_external: pd.DataFrame,
    country_map: pd.DataFrame,
    audit_dir: Path,
    sources_enabled: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    enabled = set(sources_enabled) if sources_enabled is not None else {"vparty", "ches", "parlgov", "elff"}

    # V-Party
    if "vparty" in enabled:
        vparty_obs = vparty.copy()
        vparty_obs = vparty_obs.rename(columns={"pf_party_id": "partyfacts_id", "year": "year", "v2pariglef": "ideo_raw"})
        vparty_obs["source"] = "vparty"
        vparty_obs["ideo_scale_note"] = "vparty_v2pariglef"
        vparty_obs = vparty_obs[["partyfacts_id", "year", "ideo_raw", "source", "ideo_scale_note"]]
        vparty_obs = vparty_obs.dropna(subset=["partyfacts_id", "ideo_raw"])
        rows.append(vparty_obs)

    # CHES
    if "ches" in enabled:
        ches_map = pf_external[pf_external["dataset_key"] == "ches"].copy()
        ches_map["dataset_party_id"] = ches_map["dataset_party_id"].astype(str)
        ches_tmp = ches.copy()
        ches_tmp["party_id"] = ches_tmp["party_id"].astype(str)
        ches_tmp = ches_tmp.merge(ches_map[["dataset_party_id", "partyfacts_id"]], left_on="party_id", right_on="dataset_party_id", how="left")
        ches_obs = ches_tmp.rename(columns={"lrecon": "ideo_raw", "year": "year"})
        ches_obs["source"] = "ches"
        ches_obs["ideo_scale_note"] = "ches_lrecon"
        ches_obs = ches_obs[["partyfacts_id", "year", "ideo_raw", "source", "ideo_scale_note"]]
        ches_obs = ches_obs.dropna(subset=["partyfacts_id", "ideo_raw"])
        rows.append(ches_obs)

    # ParlGov
    if "parlgov" in enabled:
        parlgov_map = pf_external[pf_external["dataset_key"] == "parlgov"].copy()
        parlgov_map["dataset_party_id"] = parlgov_map["dataset_party_id"].astype(str)
        parlgov_tmp = parlgov_pos.copy()
        parlgov_tmp["party_id"] = parlgov_tmp["party_id"].astype(str)
        parlgov_tmp = parlgov_tmp.merge(parlgov_map[["dataset_party_id", "partyfacts_id"]], left_on="party_id", right_on="dataset_party_id", how="left")
        parlgov_obs = parlgov_tmp.rename(columns={"state_market": "ideo_raw"})
        parlgov_obs["source"] = "parlgov"
        parlgov_obs["ideo_scale_note"] = "parlgov_state_market"
        parlgov_obs["year"] = pd.NA
        parlgov_obs = parlgov_obs[["partyfacts_id", "year", "ideo_raw", "source", "ideo_scale_note"]]
        parlgov_obs = parlgov_obs.dropna(subset=["partyfacts_id", "ideo_raw"])
        rows.append(parlgov_obs)

    # ELFF (fuzzy match to PartyFacts)
    if "elff" in enabled:
        # Map ELFF countries to ISO3
        country_lookup = country_map[["iso3", "country_name_std"]].copy()
        country_lookup["country_norm"] = country_lookup["country_name_std"].map(normalize_name)
        elff = elff.copy()
        elff["country_norm"] = elff["countryname"].map(normalize_name)
        elff = elff.merge(country_lookup[["iso3", "country_norm"]], on="country_norm", how="left")

        # Build PartyFacts aliases
        pf_core = pf_core.copy()
        pf_core["iso3"] = pf_core["country"].astype(str).str.upper()
        pf_aliases = []
        for _, row in pf_core.iterrows():
            for col in ["name", "name_english", "name_short", "name_other"]:
                val = row.get(col)
                if pd.isna(val) or val is None:
                    continue
                norm = normalize_name(val)
                if not norm:
                    continue
                pf_aliases.append({
                    "partyfacts_id": row["partyfacts_id"],
                    "iso3": row["iso3"],
                    "alias": norm,
                })
        pf_aliases = pd.DataFrame(pf_aliases)

        elff_rows = []
        for _, row in elff.iterrows():
            iso3 = row.get("iso3")
            if pd.isna(iso3):
                continue
            party_norm = normalize_name(row.get("partyname"))
            if not party_norm:
                continue
            pool = pf_aliases[pf_aliases["iso3"] == iso3]
            if pool.empty:
                continue
            choices = pool["alias"].tolist()
            if process and fuzz:
                best, score, _ = process.extractOne(party_norm, choices, scorer=fuzz.token_sort_ratio)
            else:
                best = None
                score = 0
                for c in choices:
                    if c == party_norm:
                        best = c
                        score = 100
                        break
            if best is None or score < 90:
                continue
            pid = pool.loc[pool["alias"] == best, "partyfacts_id"].iloc[0]
            elff_rows.append({
                "partyfacts_id": pid,
                "year": row.get("year"),
                "ideo_raw": row.get("econlr"),
                "source": "elff",
                "ideo_scale_note": "elff_econlr",
            })
        if elff_rows:
            rows.append(pd.DataFrame(elff_rows))

    obs = pd.concat(rows, ignore_index=True)
    obs["ideo_raw"] = pd.to_numeric(obs["ideo_raw"], errors="coerce")
    obs = obs.dropna(subset=["ideo_raw", "partyfacts_id"])

    # add iso3 from partyfacts core
    obs = obs.merge(pf_core[["partyfacts_id", "country"]].rename(columns={"country": "iso3"}), on="partyfacts_id", how="left")

    # standardize within source
    obs["ideo_std"] = obs.groupby("source")["ideo_raw"].transform(percentile_rank)
    obs["ideo_std"] = obs["ideo_std"].clip(0, 1)

    return obs[["partyfacts_id", "iso3", "source", "year", "ideo_raw", "ideo_std", "ideo_scale_note"]]


def assign_ideology_to_elections(
    elections: pd.DataFrame,
    obs: pd.DataFrame,
    source_priority: list[str],
    year_windows: dict,
    stale_after_years: int | None = None,
) -> pd.DataFrame:
    df = elections.copy()
    obs = obs.copy()

    # prep lookup by source
    obs["year"] = pd.to_numeric(obs["year"], errors="coerce")

    def pick_obs(party_id: int, year: int, source: str):
        subset = obs[(obs["partyfacts_id"] == party_id) & (obs["source"] == source)]
        if subset.empty:
            return None
        window = year_windows.get(source)
        if window is not None and not pd.isna(year):
            subset = subset[(subset["year"].isna()) | (subset["year"].sub(year).abs() <= window)]
        if subset.empty:
            return None
        subset = subset.assign(year_diff=(subset["year"].sub(year)).abs())
        subset = subset.sort_values(["year_diff", "year"], ascending=[True, True], kind="mergesort")
        return subset.iloc[0]

    results = []
    for idx, row in df.iterrows():
        pid1 = row.get("party_1_id")
        pid2 = row.get("party_2_id")
        year = row.get("election_year")
        chosen_source = None
        p1_obs = None
        p2_obs = None
        if pd.notna(pid1) and pd.notna(pid2):
            for source in source_priority:
                o1 = pick_obs(int(pid1), year, source)
                o2 = pick_obs(int(pid2), year, source)
                if o1 is not None and o2 is not None:
                    chosen_source = source
                    p1_obs, p2_obs = o1, o2
                    break
        # allow mixed
        if chosen_source is None:
            if pd.notna(pid1):
                for source in source_priority:
                    o1 = pick_obs(int(pid1), year, source)
                    if o1 is not None:
                        p1_obs = o1
                        break
            if pd.notna(pid2):
                for source in source_priority:
                    o2 = pick_obs(int(pid2), year, source)
                    if o2 is not None:
                        p2_obs = o2
                        break
            if p1_obs is not None or p2_obs is not None:
                chosen_source = "mixed"
        results.append((chosen_source, p1_obs, p2_obs))

    df["ideo_source"] = [r[0] for r in results]
    df["ideo_raw_1"] = [r[1]["ideo_raw"] if r[1] is not None else pd.NA for r in results]
    df["ideo_raw_2"] = [r[2]["ideo_raw"] if r[2] is not None else pd.NA for r in results]
    df["ideo_std_1"] = [r[1]["ideo_std"] if r[1] is not None else pd.NA for r in results]
    df["ideo_std_2"] = [r[2]["ideo_std"] if r[2] is not None else pd.NA for r in results]
    df["ideo_year_1"] = [r[1]["year"] if r[1] is not None else pd.NA for r in results]
    df["ideo_year_2"] = [r[2]["year"] if r[2] is not None else pd.NA for r in results]
    election_years = df["election_year"].tolist()
    df["ideo_year_diff_1"] = [
        abs(r[1]["year"] - election_years[i]) if r[1] is not None and pd.notna(r[1]["year"]) and pd.notna(election_years[i]) else pd.NA
        for i, r in enumerate(results)
    ]
    df["ideo_year_diff_2"] = [
        abs(r[2]["year"] - election_years[i]) if r[2] is not None and pd.notna(r[2]["year"]) and pd.notna(election_years[i]) else pd.NA
        for i, r in enumerate(results)
    ]

    df["ideology_1_source_key"] = [r[1]["source"] if r[1] is not None else pd.NA for r in results]
    df["ideology_2_source_key"] = [r[2]["source"] if r[2] is not None else pd.NA for r in results]
    df["ideology_1_year"] = df["ideo_year_1"]
    df["ideology_2_year"] = df["ideo_year_2"]
    df["ideology_1_year_gap"] = df["ideo_year_diff_1"]
    df["ideology_2_year_gap"] = df["ideo_year_diff_2"]
    if stale_after_years is not None:
        df["ideology_1_stale_flag"] = df["ideology_1_year_gap"].apply(lambda x: bool(x > stale_after_years) if pd.notna(x) else pd.NA)
        df["ideology_2_stale_flag"] = df["ideology_2_year_gap"].apply(lambda x: bool(x > stale_after_years) if pd.notna(x) else pd.NA)
    else:
        df["ideology_1_stale_flag"] = pd.NA
        df["ideology_2_stale_flag"] = pd.NA

    # compute margin_market
    df["margin_market"] = pd.NA
    df["D_market_win"] = pd.NA
    df["margin_market_abs"] = pd.NA

    has_ideo = df["ideo_std_1"].notna() & df["ideo_std_2"].notna() & df["share_1"].notna() & df["share_2"].notna()
    more_market_party1 = df["ideo_std_1"].fillna(-np.inf) >= df["ideo_std_2"].fillna(-np.inf)
    df.loc[has_ideo & more_market_party1, "margin_market"] = df.loc[has_ideo & more_market_party1, "share_1"] - df.loc[has_ideo & more_market_party1, "share_2"]
    df.loc[has_ideo & (~more_market_party1), "margin_market"] = df.loc[has_ideo & (~more_market_party1), "share_2"] - df.loc[has_ideo & (~more_market_party1), "share_1"]
    df.loc[has_ideo, "D_market_win"] = (df.loc[has_ideo, "margin_market"] >= 0).astype(int)
    df.loc[has_ideo, "margin_market_abs"] = df.loc[has_ideo, "margin_market"].abs()
    df["margin_market_bin_0_5"] = df["margin_market"].apply(lambda x: round(x / 0.5) * 0.5 if pd.notna(x) else pd.NA)
    df["margin_market_bin_1"] = df["margin_market"].apply(lambda x: round(x) if pd.notna(x) else pd.NA)

    return df
