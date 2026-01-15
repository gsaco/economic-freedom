"""Build merged quinquennial panel and validations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass
class MergeReport:
    rows_left: int
    rows_right: int
    rows_merged: int
    left_only: int
    right_only: int
    both: int


def assert_unique(df: pd.DataFrame, keys: List[str], label: str) -> None:
    dupes = df.duplicated(keys)
    if dupes.any():
        dup_count = int(dupes.sum())
        raise ValueError(f"Duplicate keys in {label}: {dup_count} rows")


def merge_with_report(left: pd.DataFrame, right: pd.DataFrame, keys: List[str], label: str) -> tuple[pd.DataFrame, MergeReport]:
    merged = left.merge(right, on=keys, how="left", indicator=True)
    report = MergeReport(
        rows_left=len(left),
        rows_right=len(right),
        rows_merged=len(merged),
        left_only=int((merged["_merge"] == "left_only").sum()),
        right_only=int((merged["_merge"] == "right_only").sum()),
        both=int((merged["_merge"] == "both").sum()),
    )
    merged = merged.drop(columns=["_merge"])
    return merged, report


def build_panel(
    efw_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    pwt_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[MergeReport]]:
    keys = ["iso3", "year"]
    assert_unique(efw_df, keys, "EFW")
    assert_unique(macro_df, keys, "Macro")

    merged, report1 = merge_with_report(efw_df, macro_df, keys, "EFW+Macro")

    reports = [report1]
    if pwt_df is not None:
        assert_unique(pwt_df, keys, "PWT")
        merged, report2 = merge_with_report(merged, pwt_df, keys, "EFW+Macro+PWT")
        reports.append(report2)

    return merged, reports


def validate_panel(df: pd.DataFrame, start_year: int = 1970, end_year: int = 2020) -> None:
    expected_years = set(range(start_year, end_year + 1, 5))
    years = set(df["year"].dropna().unique())
    if years - expected_years:
        raise ValueError(f"Unexpected years found: {sorted(years - expected_years)}")
    if expected_years - years:
        print(f"Warning: missing expected years: {sorted(expected_years - years)}")

    if df.duplicated(["iso3", "year"]).any():
        raise ValueError("Panel has duplicate iso3-year keys")
