from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from elections_core import normalize_name


@dataclass
class InvariantReport:
    issues: dict

    def has_failures(self) -> bool:
        return any(v for v in self.issues.values() if isinstance(v, (int, float)) and v > 0)


def _iso3_pattern_invalid(iso3: pd.Series) -> pd.Series:
    iso3 = iso3.astype("string")
    return ~iso3.str.match(r"^[A-Z]{3}$", na=True)


def _impossible_iso_mappings(df: pd.DataFrame) -> int:
    if "iso3" not in df.columns:
        return 0
    name_col = None
    for col in ["country", "country_name", "ctr_n", "ctr"]:
        if col in df.columns:
            name_col = col
            break
    if name_col is None:
        return 0
    names = df[name_col].astype("string").fillna("")
    norm = names.map(normalize_name)
    iso3 = df["iso3"].astype("string")
    # Guard against common North/South Korea swaps.
    bad_prk = norm.str.contains("korea", na=False) & norm.str.contains("dem", na=False) & (iso3 == "KOR")
    bad_kor = norm.str.contains("korea", na=False) & norm.str.contains("rep", na=False) & (iso3 == "PRK")
    bad_north = norm.str.contains("north korea", na=False) & (iso3 == "KOR")
    bad_south = norm.str.contains("south korea", na=False) & (iso3 == "PRK")
    return int((bad_prk | bad_kor | bad_north | bad_south).sum())


def invariant_checks(
    df: pd.DataFrame,
    iso3_valid: Optional[Iterable[str]] = None,
) -> InvariantReport:
    issues: dict[str, int | float | None] = {}

    # Uniqueness checks
    if "election_id" in df.columns:
        issues["dup_election_id"] = int(df["election_id"].duplicated().sum())
    key_cols = [c for c in ["iso3", "office_type", "year", "month", "seq"] if c in df.columns]
    if key_cols:
        issues["dup_rows_iso_office_year_month_seq"] = int(df.duplicated(key_cols, keep=False).sum())

    # Share bounds and ordering
    if "share_1" in df.columns and "share_2" in df.columns:
        s1 = pd.to_numeric(df["share_1"], errors="coerce")
        s2 = pd.to_numeric(df["share_2"], errors="coerce")
        issues["share_out_of_bounds_rows"] = int(((s1 < 0) | (s1 > 100) | (s2 < 0) | (s2 > 100)).sum())
        issues["winner_lt_runnerup_rows"] = int(((s1 < s2) & s1.notna() & s2.notna()).sum())

    # ISO validity
    if "iso3" in df.columns:
        invalid_pattern = _iso3_pattern_invalid(df["iso3"])
        issues["iso3_invalid_pattern_rows"] = int(invalid_pattern.sum())
        if iso3_valid is not None:
            iso_set = set(iso3_valid)
            iso_series = df["iso3"].astype("string")
            invalid = df["iso3"].notna() & ~iso_series.isin(iso_set)
            issues["iso3_invalid_membership_rows"] = int(invalid.sum())

    # NELDA tri-state checks
    if "nelda_known" in df.columns and "competitive_nelda" in df.columns:
        nelda_known = df["nelda_known"]
        competitive = df["competitive_nelda"]
        bad_unknown = (nelda_known == False) & competitive.notna()
        bad_known = (nelda_known == True) & competitive.isna()
        issues["nelda_tristate_violations"] = int((bad_unknown | bad_known).sum())

    issues["impossible_iso_mappings"] = _impossible_iso_mappings(df)

    return InvariantReport(issues=issues)
