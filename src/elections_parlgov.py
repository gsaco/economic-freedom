from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_seat_shares(
    results: pd.DataFrame,
    parties: pd.DataFrame,
    *,
    threshold: float,
    seat_col: str = "seats",
) -> pd.DataFrame:
    frame = results.merge(parties[["party_id", "left_right"]], on="party_id", how="left")
    frame = frame.dropna(subset=[seat_col]).copy()
    frame["market_party"] = frame["left_right"] >= threshold
    totals = frame.groupby("election_id")[seat_col].sum().rename("seat_total")
    market = frame.loc[frame["market_party"]].groupby("election_id")[seat_col].sum().rename("seat_market")
    out = pd.concat([totals, market], axis=1).reset_index()
    out["seat_share_market"] = out["seat_market"] / out["seat_total"].replace(0, np.nan)
    return out


def compute_market_vote_shares(
    results: pd.DataFrame,
    parties: pd.DataFrame,
    *,
    threshold: float,
    vote_share_col: str = "vote_share",
) -> pd.DataFrame:
    frame = results.merge(parties[["party_id", "left_right"]], on="party_id", how="left")
    frame = frame.dropna(subset=[vote_share_col]).copy()
    frame["vote_share"] = frame[vote_share_col].astype(float)
    frame["market_party"] = frame["left_right"] >= threshold
    totals = frame.groupby("election_id")["vote_share"].sum().rename("vote_share_total")
    market = (
        frame.loc[frame["market_party"]]
        .groupby("election_id")["vote_share"]
        .sum()
        .rename("vote_share_market_raw")
    )
    out = pd.concat([totals, market], axis=1).reset_index()
    out["vote_share_market"] = out["vote_share_market_raw"] / out["vote_share_total"].replace(0, np.nan)
    return out


def compute_bloc_ideology_distance(
    results: pd.DataFrame,
    parties: pd.DataFrame,
    *,
    threshold: float,
    weight_col: str = "vote_share",
) -> pd.DataFrame:
    frame = results.merge(parties[["party_id", "left_right"]], on="party_id", how="left")
    frame = frame.dropna(subset=[weight_col, "left_right"]).copy()
    frame["weight"] = frame[weight_col].astype(float)
    frame["market_party"] = frame["left_right"] >= threshold
    frame["weighted_lr"] = frame["left_right"] * frame["weight"]

    market = frame.loc[frame["market_party"]].groupby("election_id").agg(
        weight_market=("weight", "sum"),
        weighted_lr_market=("weighted_lr", "sum"),
    )
    nonmarket = frame.loc[~frame["market_party"]].groupby("election_id").agg(
        weight_nonmarket=("weight", "sum"),
        weighted_lr_nonmarket=("weighted_lr", "sum"),
    )
    out = pd.concat([market, nonmarket], axis=1).reset_index()
    out["lr_market"] = out["weighted_lr_market"] / out["weight_market"].replace(0, np.nan)
    out["lr_nonmarket"] = out["weighted_lr_nonmarket"] / out["weight_nonmarket"].replace(0, np.nan)
    out["lr_distance"] = out["lr_market"] - out["lr_nonmarket"]
    out["lr_distance_abs"] = out["lr_distance"].abs()
    return out


def compute_top2_margin_by_bloc(
    results: pd.DataFrame,
    parties: pd.DataFrame,
    *,
    threshold: float,
    vote_share_col: str = "vote_share",
) -> pd.DataFrame:
    frame = results.merge(parties[["party_id", "left_right"]], on="party_id", how="left")
    frame = frame.dropna(subset=[vote_share_col, "left_right"]).copy()
    frame["vote_share_raw"] = frame[vote_share_col].astype(float)
    scale = 100.0 if frame["vote_share_raw"].max() > 1.0 else 1.0
    frame["vote_share_frac"] = frame["vote_share_raw"] / scale
    frame["market_party"] = frame["left_right"] >= threshold

    top = (
        frame.groupby(["election_id", "market_party"])["vote_share_frac"]
        .max()
        .unstack("market_party")
        .rename(columns={False: "top_nonmarket_share", True: "top_market_share"})
        .reset_index()
    )
    top["top2_margin"] = top["top_market_share"] - top["top_nonmarket_share"]
    top["top2_margin_abs"] = top["top2_margin"].abs()
    top["winner_market_top2"] = (top["top2_margin"] > 0).astype(float)
    return top


def compute_cabinet_ideology(
    cabinets: pd.DataFrame,
    cabinet_parties: pd.DataFrame,
    parties: pd.DataFrame,
    results: pd.DataFrame,
    *,
    threshold: float,
    seat_col: str = "seats",
) -> pd.DataFrame:
    cabinets = cabinets.rename(columns={"previous_parliament_election_id": "election_id"})
    cabinet_party = cabinet_parties.merge(
        cabinets[["cabinet_id", "election_id"]],
        on="cabinet_id",
        how="left",
    )
    cabinet_party = cabinet_party.merge(parties[["party_id", "left_right"]], on="party_id", how="left")
    cabinet_party = cabinet_party.merge(
        results[["election_id", "party_id", seat_col]],
        on=["election_id", "party_id"],
        how="left",
    )

    cabinet_party["weight"] = cabinet_party[seat_col].fillna(1.0)
    cabinet_party["weight_valid"] = cabinet_party["weight"].where(cabinet_party["left_right"].notna(), 0.0)
    cabinet_party["market_party"] = cabinet_party["left_right"] >= threshold
    cabinet_party["weight_lr"] = cabinet_party["left_right"].fillna(0.0) * cabinet_party["weight_valid"]
    cabinet_party["weight_market"] = cabinet_party["market_party"].astype(float) * cabinet_party["weight_valid"]

    summary = (
        cabinet_party.groupby("cabinet_id")
        .agg(
            weight_total=("weight_valid", "sum"),
            weight_lr=("weight_lr", "sum"),
            weight_market=("weight_market", "sum"),
            cabinet_party_count=("party_id", "nunique"),
        )
        .reset_index()
    )
    summary["cabinet_lr"] = summary["weight_lr"] / summary["weight_total"].replace(0, np.nan)
    summary["cabinet_market_share"] = summary["weight_market"] / summary["weight_total"].replace(0, np.nan)
    return summary.drop(columns=["weight_total", "weight_lr", "weight_market"])


def select_post_election_cabinets(cabinets: pd.DataFrame) -> pd.DataFrame:
    frame = cabinets[cabinets["previous_parliament_election_id"].notna()].copy()
    frame = frame.rename(columns={"previous_parliament_election_id": "election_id"})
    frame["caretaker_flag"] = frame["caretaker"].fillna(0).astype(int)
    frame = frame.sort_values(["election_id", "caretaker_flag", "start_date"])
    selected = frame.groupby("election_id", as_index=False).first()
    selected = selected.rename(columns={"cabinet_id": "post_cabinet_id"})
    return selected[
        [
            "election_id",
            "post_cabinet_id",
            "previous_cabinet_id",
            "start_date",
            "caretaker_flag",
        ]
    ]
