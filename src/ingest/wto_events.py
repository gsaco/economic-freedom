from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import pycountry
import requests

from src.config import PROCESSED_DIR, RAW_DIR

WTO_LIST_URLS = [
    "https://en.wikipedia.org/wiki/Member_states_of_the_World_Trade_Organization",
    "https://en.wikipedia.org/wiki/World_Trade_Organization",
]

NAME_FIXES = {
    "Niger": "NER",
    "Nigeria": "NGA",
    "Congo, Democratic Republic of the": "COD",
    "Congo, Republic of the": "COG",
    "Korea, Republic of": "KOR",
    "Korea, Democratic People's Republic of": "PRK",
    "Russian Federation": "RUS",
    "Venezuela, Bolivarian Republic of": "VEN",
    "Bolivia, Plurinational State of": "BOL",
    "Iran, Islamic Republic of": "IRN",
    "Syrian Arab Republic": "SYR",
    "Egypt, Arab Republic of": "EGY",
    "Hong Kong": "HKG",
    "Hong Kong, China": "HKG",
    "Macao, China": "MAC",
    "Macao": "MAC",
    "Chinese Taipei": "TWN",
    "Taiwan": "TWN",
    "Czech Republic": "CZE",
    "Slovak Republic": "SVK",
    "Slovakia": "SVK",
    "North Macedonia": "MKD",
    "Eswatini": "SWZ",
    "Cabo Verde": "CPV",
    "Bahamas, The": "BHS",
    "Gambia, The": "GMB",
    "Lao People's Democratic Republic": "LAO",
    "Kyrgyz Republic": "KGZ",
    "Yemen, Republic of": "YEM",
    "Micronesia, Federated States of": "FSM",
    "Saint Kitts and Nevis": "KNA",
    "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT",
}


def _iso3_from_name(name: str) -> str | None:
    if name in NAME_FIXES:
        return NAME_FIXES[name]
    try:
        match = pycountry.countries.search_fuzzy(name)[0]
        return match.alpha_3
    except LookupError:
        return None


def _parse_accession_year(value: str | float | int) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    value_str = str(value)
    for fmt in ("%d %B %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value_str, fmt).year
        except ValueError:
            continue
    try:
        return int(value_str[-4:])
    except ValueError:
        return None


def _download_wto_list() -> pd.DataFrame | None:
    for url in WTO_LIST_URLS:
        try:
            resp = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (compatible; CodexBot/1.0)"},
            )
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            for table in tables:
                cols = [c.lower() for c in table.columns]
                if any("accession" in c or "membership" in c for c in cols):
                    return table
        except requests.RequestException:
            continue
    return None


def ingest_wto_events() -> Path:
    raw_path = RAW_DIR / "wto" / "wto_accessions.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        table = _download_wto_list()
        if table is None:
            raise FileNotFoundError(
                "WTO accession list not found. Place CSV at data/raw/wto/wto_accessions.csv."
            )

        cols = {c.lower(): c for c in table.columns}
        name_col = cols.get("member") or cols.get("members") or cols.get("member state") or list(table.columns)[0]
        date_col = None
        for c in table.columns:
            if "accession" in str(c).lower() or "membership" in str(c).lower():
                date_col = c
                break
        if date_col is None:
            raise ValueError("Could not identify accession date column in WTO table")

        df = table[[name_col, date_col]].copy()
        df = df.rename(columns={name_col: "country_name", date_col: "accession_date"})
        df["accession_year"] = df["accession_date"].apply(_parse_accession_year)
        df["iso3"] = df["country_name"].apply(_iso3_from_name)
        df = df.dropna(subset=["iso3", "accession_year"])
        df["accession_year"] = df["accession_year"].astype(int)
        df["article_xii"] = (df["accession_year"] > 1995).astype(int)
        df["wp_year"] = df["accession_year"].where(df["article_xii"] == 1, pd.NA) - 2
        df = df[["iso3", "country_name", "wp_year", "accession_year", "article_xii"]]
        df.to_csv(raw_path, index=False)

    required = {"iso3", "country_name", "wp_year", "accession_year", "article_xii"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"WTO accessions missing columns: {missing}")

    df["iso3"] = df["country_name"].apply(_iso3_from_name)
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    df["wp_year"] = pd.to_numeric(df["wp_year"], errors="coerce")
    df["accession_year"] = pd.to_numeric(df["accession_year"], errors="coerce")
    df["article_xii"] = pd.to_numeric(df["article_xii"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values(["iso3", "accession_year"]).drop_duplicates(subset=["iso3"], keep="last")

    out_path = PROCESSED_DIR / "wto_events.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    ingest_wto_events()
