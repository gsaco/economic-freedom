"""Stacked event-study utilities for reform shocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class EventConfig:
    pos_quantile: float
    neg_quantile: float
    sustain_epsilon: float
    sustain_horizons: List[int]
    cooldown_years: int
    window_pre: int
    window_post: int


@dataclass
class EventSet:
    events: pd.DataFrame
    config: EventConfig
    pos_threshold: float
    neg_threshold: float


def compute_thresholds(series: pd.Series, pos_q: float, neg_q: float) -> tuple[float, float]:
    pos_threshold = float(series.quantile(pos_q))
    neg_threshold = float(series.quantile(neg_q))
    return pos_threshold, neg_threshold


def _sustain_check(
    history: pd.DataFrame,
    year: int,
    level_col: str,
    event_type: str,
    epsilon: float,
    horizons: Iterable[int],
) -> bool:
    level_t = history.loc[year, level_col]
    for h in horizons:
        target_year = year + h
        if target_year not in history.index:
            return False
        level_future = history.loc[target_year, level_col]
        if pd.isna(level_future):
            return False
        if event_type == "pos" and level_future < level_t - epsilon:
            return False
        if event_type == "neg" and level_future > level_t + epsilon:
            return False
    return True


def identify_events(
    df: pd.DataFrame,
    change_col: str,
    level_col: str,
    config: EventConfig,
) -> EventSet:
    series = df[change_col].dropna()
    pos_threshold, neg_threshold = compute_thresholds(series, config.pos_quantile, config.neg_quantile)

    events = []
    for iso3, group in df.sort_values(["iso3", "year"]).groupby("iso3"):
        history = group.set_index("year")
        for _, row in group.iterrows():
            change = row.get(change_col)
            if pd.isna(change):
                continue
            event_type = None
            if change >= pos_threshold:
                event_type = "pos"
            elif change <= neg_threshold:
                event_type = "neg"
            if event_type is None:
                continue
            if not _sustain_check(
                history,
                int(row["year"]),
                level_col,
                event_type,
                config.sustain_epsilon,
                config.sustain_horizons,
            ):
                continue
            events.append(
                {
                    "iso3": iso3,
                    "year": int(row["year"]),
                    "event_type": event_type,
                    "change": float(change),
                    "level": float(row[level_col]) if pd.notna(row[level_col]) else np.nan,
                }
            )

    events_df = pd.DataFrame.from_records(events)
    if events_df.empty:
        return EventSet(events=events_df, config=config, pos_threshold=pos_threshold, neg_threshold=neg_threshold)

    # Apply cooldown per country and event type.
    filtered = []
    for (iso3, event_type), group in events_df.sort_values("year").groupby(["iso3", "event_type"]):
        last_year = None
        for _, row in group.iterrows():
            if last_year is None or (row["year"] - last_year) >= config.cooldown_years:
                filtered.append(row)
                last_year = row["year"]
    events_df = pd.DataFrame(filtered).reset_index(drop=True)
    events_df["event_id"] = np.arange(len(events_df))

    return EventSet(events=events_df, config=config, pos_threshold=pos_threshold, neg_threshold=neg_threshold)


def build_stacked_panel(
    df: pd.DataFrame,
    events: pd.DataFrame,
    window_pre: int,
    window_post: int,
    event_type: Optional[str] = None,
    exclude_any_event: bool = True,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    events = events.copy()
    if event_type is not None:
        events = events[events["event_type"] == event_type].copy()
    if events.empty:
        return pd.DataFrame()

    all_iso3 = sorted(df["iso3"].dropna().unique())
    events_by_country = events.groupby("iso3")["year"].apply(list).to_dict()
    all_events = events.groupby("iso3")["year"].apply(list).to_dict()

    stacked = []
    for _, event in events.iterrows():
        event_year = int(event["year"])
        treated_iso3 = event["iso3"]
        window_years = [event_year + 5 * k for k in range(-window_pre, window_post + 1)]

        # Clean controls: exclude countries with any event in window.
        control_iso3 = []
        for iso3 in all_iso3:
            if iso3 == treated_iso3:
                control_iso3.append(iso3)
                continue
            event_years = all_events.get(iso3, []) if exclude_any_event else events_by_country.get(iso3, [])
            if any(year in window_years for year in event_years):
                continue
            control_iso3.append(iso3)

        subset = df[df["iso3"].isin(control_iso3) & df["year"].isin(window_years)].copy()
        subset["stack_id"] = int(event["event_id"])
        subset["event_year"] = event_year
        subset["treated"] = (subset["iso3"] == treated_iso3).astype(int)
        subset["event_time"] = ((subset["year"] - event_year) / 5).round().astype(int)
        subset["event_time_years"] = subset["event_time"] * 5
        subset["event_type"] = event["event_type"]
        subset["entity_id"] = subset["stack_id"].astype(str) + ":" + subset["iso3"].astype(str)
        stacked.append(subset)

    if not stacked:
        return pd.DataFrame()
    return pd.concat(stacked, ignore_index=True)


def build_event_dummies(
    df: pd.DataFrame,
    event_time_col: str,
    treated_col: str,
    omit: int,
) -> tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    event_times = sorted(df[event_time_col].dropna().unique())
    event_times = [int(t) for t in event_times if int(t) != omit]
    columns = []
    for t in event_times:
        name = f"event_{t}"
        df[name] = ((df[event_time_col] == t) & (df[treated_col] == 1)).astype(int)
        columns.append(name)
    return df, columns
