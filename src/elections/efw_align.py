from __future__ import annotations

import pandas as pd
import numpy as np


def prepare_efw_panel(efw: pd.DataFrame) -> pd.DataFrame:
    df = efw.rename(columns={
        "ISO_Code": "iso3",
        "Countries": "country_name",
        "Year": "year",
        "Summary": "efw_summary",
        "Area 1": "efw_area1",
        "Area 2": "efw_area2",
        "Area 3": "efw_area3",
        "Area 4": "efw_area4",
        "Area 5": "efw_area5",
    }).copy()
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df[["iso3", "country_name", "year", "efw_summary", "efw_area1", "efw_area2", "efw_area3", "efw_area4", "efw_area5"]]


def attach_efw_baseline(elections: pd.DataFrame, efw: pd.DataFrame) -> pd.DataFrame:
    df = elections.copy()
    df["efw_baseline_year"] = df["election_year"].astype("Int64") - 1
    merged = df.merge(efw, left_on=["iso3", "efw_baseline_year"], right_on=["iso3", "year"], how="left", suffixes=("", "_efw"))
    merged = merged.rename(columns={
        "efw_summary": "efw_summary_baseline",
        "efw_area1": "efw_area1_baseline",
        "efw_area2": "efw_area2_baseline",
        "efw_area3": "efw_area3_baseline",
        "efw_area4": "efw_area4_baseline",
        "efw_area5": "efw_area5_baseline",
    })
    # match status
    merged["efw_match_status"] = "matched"
    merged.loc[merged["iso3"].isna(), "efw_match_status"] = "no_iso3"
    merged.loc[merged["efw_baseline_year"].isna(), "efw_match_status"] = "no_baseline_year"
    merged.loc[merged["efw_summary_baseline"].isna() & merged["iso3"].notna(), "efw_match_status"] = "no_efw_country"
    return merged


def build_efw_horizons(elections: pd.DataFrame, efw: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for h in horizons:
        temp = elections[[
            "election_id", "iso3", "office_type", "election_year", "D_market_win", "margin_market",
            "efw_baseline_year", "efw_summary_baseline", "efw_area1_baseline", "efw_area2_baseline", "efw_area3_baseline", "efw_area4_baseline", "efw_area5_baseline",
        ]].copy()
        temp["horizon"] = h
        temp["outcome_year"] = temp["election_year"] + h
        temp = temp.merge(efw, left_on=["iso3", "outcome_year"], right_on=["iso3", "year"], how="left")
        temp = temp.rename(columns={
            "efw_summary": "efw_outcome_summary",
            "efw_area1": "efw_outcome_area1",
            "efw_area2": "efw_outcome_area2",
            "efw_area3": "efw_outcome_area3",
            "efw_area4": "efw_outcome_area4",
            "efw_area5": "efw_outcome_area5",
        })
        # deltas
        temp["d_efw_summary"] = temp["efw_outcome_summary"] - temp["efw_summary_baseline"]
        temp["d_efw_area1"] = temp["efw_outcome_area1"] - temp["efw_area1_baseline"]
        temp["d_efw_area2"] = temp["efw_outcome_area2"] - temp["efw_area2_baseline"]
        temp["d_efw_area3"] = temp["efw_outcome_area3"] - temp["efw_area3_baseline"]
        temp["d_efw_area4"] = temp["efw_outcome_area4"] - temp["efw_area4_baseline"]
        temp["d_efw_area5"] = temp["efw_outcome_area5"] - temp["efw_area5_baseline"]

        temp["efw_horizon_match_status"] = "matched"
        temp.loc[temp["iso3"].isna(), "efw_horizon_match_status"] = "no_iso3"
        temp.loc[temp["efw_summary_baseline"].isna(), "efw_horizon_match_status"] = "missing_baseline"
        temp.loc[temp["efw_outcome_summary"].isna(), "efw_horizon_match_status"] = "missing_outcome"

        rows.append(temp)
    return pd.concat(rows, ignore_index=True)
