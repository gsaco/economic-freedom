from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass
class QcResult:
    check: str
    passed: bool
    details: dict


def assert_unique_key(df: pd.DataFrame, keys: list[str]) -> None:
    duplicated = df.duplicated(subset=keys, keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, keys].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate keys detected for {keys}: {sample}")


def assert_year_grid(df: pd.DataFrame, year_col: str, expected_years: Iterable[int]) -> None:
    expected = set(int(year) for year in expected_years)
    observed = set(int(year) for year in df[year_col].dropna().unique())
    extra = sorted(observed - expected)
    missing = sorted(expected - observed)
    if extra:
        raise ValueError(f"Unexpected years found: {extra}")
    if missing:
        raise ValueError(f"Missing expected years: {missing}")


def assert_range(
    df: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    series = df[column].dropna()
    if min_value is not None and (series < min_value).any():
        bad = series[series < min_value].head(10).tolist()
        raise ValueError(f"{column} below {min_value}: {bad}")
    if max_value is not None and (series > max_value).any():
        bad = series[series > max_value].head(10).tolist()
        raise ValueError(f"{column} above {max_value}: {bad}")


def coverage_by_year(
    df: pd.DataFrame,
    year_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    coverage = (
        df.groupby(year_col)[value_cols]
        .apply(lambda frame: frame.notna().sum())
        .reset_index()
    )
    return coverage


def coverage_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    coverage = (
        df.groupby(group_cols)[value_cols]
        .apply(lambda frame: frame.notna().sum())
        .reset_index()
    )
    return coverage


def missingness_share(df: pd.DataFrame, value_cols: list[str]) -> pd.Series:
    return df[value_cols].isna().mean().sort_values(ascending=False)
