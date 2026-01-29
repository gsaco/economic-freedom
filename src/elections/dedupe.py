from __future__ import annotations

import pandas as pd
import numpy as np


def compute_merge_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    key_day = df["election_date"].dt.strftime("%Y-%m-%d")
    key_month = df["election_date"].dt.strftime("%Y-%m")
    key_year = df["election_year"].astype("Int64").astype(str)

    df["merge_strategy"] = "year"
    df.loc[df["date_precision"] == "month", "merge_strategy"] = "month"
    df.loc[df["date_precision"] == "day", "merge_strategy"] = "day"

    df["merge_key"] = (
        df["iso3"].astype(str) + "|" + df["office_type"].astype(str) + "|" +
        df.apply(lambda r: key_day[r.name] if r["merge_strategy"] == "day" else (key_month[r.name] if r["merge_strategy"] == "month" else key_year[r.name]), axis=1)
    )
    # avoid grouping when iso3 missing
    df.loc[df["iso3"].isna(), "merge_key"] = df.loc[df["iso3"].isna(), "record_id"]
    return df


def score_record(df: pd.DataFrame) -> pd.Series:
    fields = ["share_1", "share_2", "party_1_id", "party_2_id", "election_date", "nelda3", "nelda4", "nelda5"]
    score = pd.Series(0, index=df.index, dtype=int)
    for col in fields:
        if col in df.columns:
            score += df[col].notna().astype(int)
    return score


def dedupe_events(df: pd.DataFrame, source_rank: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df = compute_merge_key(df)
    df["merge_score"] = score_record(df)

    # rank sources
    rank_map = {s: i for i, s in enumerate(source_rank)}
    df["source_rank"] = df["source"].map(rank_map).fillna(len(rank_map)).astype(int)

    # assign seq within merge_key if multiple
    df = df.sort_values(["merge_key", "source_rank", "merge_score", "record_id"], ascending=[True, True, False, True], kind="mergesort")
    df["seq"] = df.groupby("merge_key").cumcount() + 1

    # election_id
    def make_election_id(row):
        if pd.isna(row.get("iso3")):
            return f"UNMAPPED_{row.get('record_id')}"
        date_key = row.get("merge_key").split("|")[-1]
        office = row.get("office_type")
        return f"{row['iso3']}_{office}_{date_key}_s{int(row['seq']):02d}"

    df["election_id"] = df.apply(make_election_id, axis=1)

    # preferred record per merge_key
    df["merge_preferred"] = False
    df["n_source_records_in_event"] = df.groupby("merge_key")["record_id"].transform("count")
    preferred_idx = df.groupby("merge_key").head(1).index
    df.loc[preferred_idx, "merge_preferred"] = True

    # canonical table
    canonical = df[df["merge_preferred"]].copy()

    return df, canonical


def attach_primary_secondary_shares(canonical: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    # For each election_id, prefer CLEA vote shares if parties align; otherwise keep canonical
    out = canonical.copy()
    out["share_metric_primary"] = out["share_metric"]
    out["share_metric_secondary"] = pd.NA
    out["share_1_secondary"] = pd.NA
    out["share_2_secondary"] = pd.NA
    out["margin_secondary"] = pd.NA

    clea = sources[sources["source"] == "clea_lc"].copy()
    if clea.empty:
        return out

    clea_best = clea.sort_values(["merge_score"], ascending=[False], kind="mergesort").groupby("election_id").head(1)
    clea_lookup = clea_best.set_index("election_id")

    for idx, row in out.iterrows():
        eid = row.get("election_id")
        if pd.isna(eid) or eid not in clea_lookup.index:
            continue
        clea_row = clea_lookup.loc[eid]

        # check party alignment (set equality for top two)
        parties_canon = {row.get("party_1_id"), row.get("party_2_id")}
        parties_clea = {clea_row.get("party_1_id"), clea_row.get("party_2_id")}
        parties_canon = {p for p in parties_canon if pd.notna(p)}
        parties_clea = {p for p in parties_clea if pd.notna(p)}

        aligned = parties_canon and parties_clea and parties_canon == parties_clea

        if aligned:
            out.at[idx, "share_metric_primary"] = clea_row.get("share_metric")
            out.at[idx, "share_metric_secondary"] = row.get("share_metric")
            out.at[idx, "share_1_secondary"] = row.get("share_1")
            out.at[idx, "share_2_secondary"] = row.get("share_2")
            out.at[idx, "margin_secondary"] = row.get("margin")

            out.at[idx, "share_1"] = clea_row.get("share_1")
            out.at[idx, "share_2"] = clea_row.get("share_2")
            out.at[idx, "margin"] = clea_row.get("margin")
        else:
            # keep canonical primary; attach CLEA as secondary if present
            out.at[idx, "share_metric_secondary"] = clea_row.get("share_metric")
            out.at[idx, "share_1_secondary"] = clea_row.get("share_1")
            out.at[idx, "share_2_secondary"] = clea_row.get("share_2")
            out.at[idx, "margin_secondary"] = clea_row.get("margin")

    return out
