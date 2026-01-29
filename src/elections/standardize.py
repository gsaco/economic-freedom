from __future__ import annotations

import re
import unicodedata
from typing import Iterable
import numpy as np
import pandas as pd

_whitespace_re = re.compile(r"\s+")
_non_alnum_re = re.compile(r"[^a-z0-9]+")


def normalize_name(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", "and")
    text = text.lower()
    text = _non_alnum_re.sub(" ", text)
    text = _whitespace_re.sub(" ", text).strip()
    return text


def standardize_share(series: pd.Series, proportion_threshold: float = 1.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    maxv = s.max(skipna=True)
    if pd.isna(maxv):
        return s
    if maxv <= proportion_threshold:
        return s * 100.0
    return s


def reorder_top_two(df: pd.DataFrame, share_cols: tuple[str, str], swap_cols: Iterable[tuple[str, str]]) -> pd.DataFrame:
    df = df.copy()
    s1, s2 = share_cols
    swap_mask = df[s2] > df[s1]
    if not swap_mask.any():
        return df
    for col1, col2 in swap_cols:
        if col1 in df.columns and col2 in df.columns:
            tmp = df.loc[swap_mask, col1].copy()
            df.loc[swap_mask, col1] = df.loc[swap_mask, col2]
            df.loc[swap_mask, col2] = tmp
    return df


def parse_date_parts(date_series: pd.Series | None, year: pd.Series | None, month: pd.Series | None) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    # Returns (date, year, month, day) with best effort
    if date_series is not None:
        date = pd.to_datetime(date_series, errors="coerce")
    else:
        date = pd.Series([pd.NaT] * len(year), index=year.index)

    year_out = pd.to_numeric(year, errors="coerce").astype("Int64") if year is not None else date.dt.year.astype("Int64")
    month_out = pd.to_numeric(month, errors="coerce").astype("Int64") if month is not None else date.dt.month.astype("Int64")
    day_out = date.dt.day.astype("Int64")

    # If date is missing but year/month present, construct date with day=1
    missing_date = date.isna()
    if year_out is not None and month_out is not None:
        fill_mask = missing_date & year_out.notna() & month_out.notna()
        if fill_mask.any():
            date.loc[fill_mask] = pd.to_datetime(
                {
                    "year": year_out[fill_mask].astype(int),
                    "month": month_out[fill_mask].astype(int),
                    "day": 1,
                },
                errors="coerce",
            )
            day_out.loc[fill_mask] = 1

    return date, year_out, month_out, day_out


def date_precision_from_parts(date: pd.Series, year: pd.Series, month: pd.Series, day: pd.Series) -> pd.Series:
    prec = pd.Series(["unknown"] * len(date), index=date.index, dtype="string")
    prec.loc[year.notna() & month.notna() & day.notna()] = "day"
    prec.loc[year.notna() & month.notna() & day.isna()] = "month"
    prec.loc[year.notna() & month.isna()] = "year"
    return prec


def parse_nelda_date(year: pd.Series, mmdd: pd.Series) -> pd.Series:
    year_num = pd.to_numeric(year, errors="coerce")
    mmdd_num = pd.to_numeric(mmdd, errors="coerce")
    # mmdd is like 101 for Jan 1
    month = (mmdd_num // 100).astype("Int64")
    day = (mmdd_num % 100).astype("Int64")
    date = pd.to_datetime({
        "year": year_num,
        "month": month,
        "day": day,
    }, errors="coerce")
    return date
