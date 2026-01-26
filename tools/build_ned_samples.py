from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "01_raw" / "ned"
INTERMEDIATE_DIR = ROOT / "data" / "02_intermediate"
ANALYSIS_DIR = ROOT / "data" / "04_analysis"

PRES_PATH = RAW_DIR / "presidential_elections_v2.dta"
PARL_PATH = RAW_DIR / "parliamentary_elections_v2.dta"
PARTIES_PATH = INTERMEDIATE_DIR / "parlgov_parties.parquet"

MARKET_THRESHOLD = 5.0
COVERAGE_THRESHOLDS = (0.8, 0.5)


def normalize_name(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def build_party_map(parties: pd.DataFrame) -> pd.DataFrame:
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
    return party_map


def build_ned_presidential_sample(party_map: pd.DataFrame, *, coverage_threshold: float) -> pd.DataFrame:
    ned = pd.read_stata(PRES_PATH, convert_categoricals=False)
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
    ned_long = ned_long.dropna(subset=["country_abb", "year", "party", "vote_share"])
    ned_long["iso3c"] = ned_long["country_abb"].astype(str).str.upper().str.strip()
    ned_long["norm_party"] = ned_long["party"].apply(normalize_name)

    ned_long = ned_long.merge(
        party_map,
        left_on=["iso3c", "norm_party"],
        right_on=["iso3c", "norm"],
        how="left",
    )

    ned_long["market_party"] = ned_long["left_right"] >= MARKET_THRESHOLD

    agg = ned_long.groupby(["iso3c", "year"], as_index=False).agg(
        vote_share_total=("vote_share", "sum"),
        vote_share_known=("vote_share", lambda x: x[ned_long.loc[x.index, "left_right"].notna()].sum()),
        vote_share_market=("vote_share", lambda x: x[ned_long.loc[x.index, "market_party"]].sum()),
    )
    agg["vote_share_nonmarket"] = agg["vote_share_known"] - agg["vote_share_market"]
    agg["coverage"] = agg["vote_share_known"] / agg["vote_share_total"].replace(0, np.nan)

    agg["running_var_ned"] = agg["vote_share_market"] - agg["vote_share_nonmarket"]
    agg["winner_market"] = (agg["running_var_ned"] > 0).astype(float)

    agg = agg.sort_values(["iso3c", "year"])
    agg["incumbent_market"] = agg.groupby("iso3c")["winner_market"].shift(1)
    agg["market_switch"] = (agg["winner_market"] != agg["incumbent_market"]).astype(float)

    agg = agg[(agg["coverage"] >= coverage_threshold) & agg["incumbent_market"].notna()].copy()
    agg = agg.rename(columns={"year": "election_year"})

    return agg


def _derive_seat_share(frame: pd.DataFrame, seat_col: str, total_seats: pd.Series) -> pd.Series:
    if seat_col in frame.columns:
        return frame[seat_col]
    seats_col = seat_col.replace("seat_share", "seats")
    if seats_col in frame.columns:
        return frame[seats_col] / total_seats.replace(0, np.nan) * 100
    return pd.Series([np.nan] * len(frame))


def build_ned_parliamentary_sample(party_map: pd.DataFrame, *, coverage_threshold: float) -> pd.DataFrame:
    ned = pd.read_stata(PARL_PATH, convert_categoricals=False)
    party_cols = [col for col in ned.columns if col.startswith("party_")]

    long_rows = []
    for i in range(1, len(party_cols) + 1):
        party_col = f"party_{i}"
        seat_share_col = f"seat_share_{i}"
        seats_col = f"seats_{i}"
        if party_col not in ned.columns:
            continue
        subset = ned[["country_abb", "country", "year", "total_seats", party_col]].copy()
        subset["party_rank"] = i
        if seat_share_col in ned.columns:
            subset["seat_share"] = ned[seat_share_col]
        elif seats_col in ned.columns:
            subset["seat_share"] = ned[seats_col] / ned["total_seats"].replace(0, np.nan) * 100
        else:
            subset["seat_share"] = np.nan
        subset = subset.rename(columns={party_col: "party"})
        long_rows.append(subset)

    ned_long = pd.concat(long_rows, ignore_index=True)
    ned_long = ned_long.dropna(subset=["country_abb", "year", "party", "seat_share"])
    ned_long["iso3c"] = ned_long["country_abb"].astype(str).str.upper().str.strip()
    ned_long["norm_party"] = ned_long["party"].apply(normalize_name)

    ned_long = ned_long.merge(
        party_map,
        left_on=["iso3c", "norm_party"],
        right_on=["iso3c", "norm"],
        how="left",
    )

    ned_long["market_party"] = ned_long["left_right"] >= MARKET_THRESHOLD

    agg = ned_long.groupby(["iso3c", "year"], as_index=False).agg(
        seat_share_total=("seat_share", "sum"),
        seat_share_known=("seat_share", lambda x: x[ned_long.loc[x.index, "left_right"].notna()].sum()),
        seat_share_market=("seat_share", lambda x: x[ned_long.loc[x.index, "market_party"]].sum()),
    )
    agg["seat_share_nonmarket"] = agg["seat_share_known"] - agg["seat_share_market"]
    agg["coverage"] = agg["seat_share_known"] / agg["seat_share_total"].replace(0, np.nan)

    agg["running_var_ned_seat"] = agg["seat_share_market"] - agg["seat_share_nonmarket"]
    agg["winner_market"] = (agg["running_var_ned_seat"] > 0).astype(float)

    agg = agg.sort_values(["iso3c", "year"])
    agg["incumbent_market"] = agg.groupby("iso3c")["winner_market"].shift(1)
    agg["market_switch"] = (agg["winner_market"] != agg["incumbent_market"]).astype(float)

    agg = agg[(agg["coverage"] >= coverage_threshold) & agg["incumbent_market"].notna()].copy()
    agg = agg.rename(columns={"year": "election_year"})

    return agg


def main() -> None:
    for path in [PRES_PATH, PARL_PATH, PARTIES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    parties = pd.read_parquet(PARTIES_PATH)
    party_map = build_party_map(parties)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    for threshold in COVERAGE_THRESHOLDS:
        suffix = f"cov{int(threshold * 100):02d}"
        pres_sample = build_ned_presidential_sample(party_map, coverage_threshold=threshold)
        pres_path = ANALYSIS_DIR / f"ned_presidential_sample_{suffix}.parquet"
        pres_sample.to_parquet(pres_path, index=False)

        parl_sample = build_ned_parliamentary_sample(party_map, coverage_threshold=threshold)
        parl_path = ANALYSIS_DIR / f"ned_parliamentary_sample_{suffix}.parquet"
        parl_sample.to_parquet(parl_path, index=False)

        print(f"Wrote {pres_path} ({len(pres_sample)} rows)")
        print(f"Wrote {parl_path} ({len(parl_sample)} rows)")


if __name__ == "__main__":
    main()
