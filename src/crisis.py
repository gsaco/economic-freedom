from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COUNTRY_FIXES = {
    "COTE D'IVOIRE": "COTE D IVOIRE",
    "CZECH REPUBLIC": "CZECHIA",
    "KOREA": "KOREA, REP.",
    "RUSSIAN FEDERATION": "RUSSIA",
    "SLOVAK REPUBLIC": "SLOVAKIA",
    "VIET NAM": "VIETNAM",
}


def _normalize_country(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().upper())


def _extract_years(value) -> list[int]:
    if pd.isna(value):
        return []
    if isinstance(value, (int, float)) and not pd.isna(value):
        year = int(value)
        return [year] if 1800 < year < 2100 else []
    text = str(value)
    years = [int(match) for match in re.findall(r"\d{4}", text)]
    return [year for year in years if 1800 < year < 2100]


def load_systemic_crisis_years(
    path: Path,
    *,
    country_meta: pd.DataFrame | None = None,
    sheet_name: str = "Crisis Years",
) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name=sheet_name)
    if "Country" not in data.columns:
        raise ValueError("Expected 'Country' column in crisis dataset.")
    col = "Systemic Banking Crisis (starting date)"
    if col not in data.columns:
        raise ValueError(f"Expected '{col}' column in crisis dataset.")

    rows = []
    for _, row in data.iterrows():
        country = row["Country"]
        if pd.isna(country):
            continue
        years = _extract_years(row[col])
        for year in years:
            rows.append({"country_name": str(country), "year": int(year)})

    crisis = pd.DataFrame(rows)
    if crisis.empty:
        return crisis

    crisis["country_norm"] = crisis["country_name"].apply(_normalize_country)
    if country_meta is not None and not country_meta.empty:
        meta = country_meta.copy()
        meta = meta.dropna(subset=["iso3c", "country_name"]).copy()
        meta["country_norm"] = meta["country_name"].apply(_normalize_country)
        meta["country_norm"] = meta["country_norm"].replace(COUNTRY_FIXES)
        crisis = crisis.merge(meta[["iso3c", "country_norm"]], on="country_norm", how="left")

    return crisis[["iso3c", "country_name", "year"]]
