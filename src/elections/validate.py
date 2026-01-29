from __future__ import annotations

import pandas as pd


def assert_unique(df: pd.DataFrame, cols: list[str], name: str) -> list[str]:
    errors = []
    if df.duplicated(cols).any():
        errors.append(f"{name}: duplicate keys {cols}")
    return errors


def assert_range(df: pd.DataFrame, col: str, min_val: float, max_val: float, name: str, eps: float = 1e-6) -> list[str]:
    errors = []
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if ((s < (min_val - eps)) | (s > (max_val + eps))).any():
            errors.append(f"{name}: {col} out of range [{min_val},{max_val}]")
    return errors


def validate_election_event(df: pd.DataFrame) -> list[str]:
    errors = []
    errors += assert_unique(df, ["election_id"], "election_event")
    errors += assert_range(df, "share_1", 0, 100, "election_event")
    errors += assert_range(df, "share_2", 0, 100, "election_event")
    errors += assert_range(df, "margin", 0, 100, "election_event")
    errors += assert_range(df, "margin_market", -100, 100, "election_event")
    return errors


def validate_country(df: pd.DataFrame) -> list[str]:
    errors = []
    errors += assert_unique(df, ["iso3"], "country")
    return errors


def validate_party(df: pd.DataFrame) -> list[str]:
    errors = []
    errors += assert_unique(df, ["partyfacts_id"], "party")
    return errors


def validate_horizon(df: pd.DataFrame) -> list[str]:
    errors = []
    errors += assert_unique(df, ["election_id", "horizon"], "election_event_horizon_efw")
    return errors
