"""Crisis data ingestion and quinquennial alignment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.country_utils import country_to_iso3, normalize_country_name


CRISIS_URL = (
    "https://static-content.springer.com/esm/art%3A10.1057%2Fs41308-020-00107-3/"
    "MediaObjects/41308_2020_107_MOESM1_ESM.xlsx"
)


@dataclass
class CrisisDataResult:
    events: pd.DataFrame
    panel: pd.DataFrame
    unmatched: pd.DataFrame


def _write_metadata(meta_path: Path, metadata: dict) -> None:
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def download_crisis_excel(cache_dir: Path | str) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "laeven_valencia_2020.xlsx"
    if dest.exists():
        return dest

    response = requests.get(CRISIS_URL, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)

    meta = {
        "url": CRISIS_URL,
        "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": dest.stat().st_size,
        "description": "Laeven & Valencia (2020) crisis years spreadsheet",
    }
    _write_metadata(dest.with_suffix(dest.suffix + ".meta"), meta)
    return dest


def _extract_years(value: object) -> List[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value)
    return [int(year) for year in re.findall(r"\d{4}", text)]


def _map_country_to_iso3(country: str, overrides: Optional[Dict[str, str]] = None) -> Optional[str]:
    return country_to_iso3(country, overrides=overrides)


def build_crisis_events(path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(path, sheet_name="Crisis Years", header=0)
    df = df.rename(columns={df.columns[0]: "country"})

    overrides = {
        "China, P.R.": "CHN",
        "China, P.R.: Hong Kong": "HKG",
    }

    records = []
    unmatched = []
    crisis_cols = {
        "systemic_banking": "Systemic Banking Crisis (starting date)",
        "currency": "Currency Crisis",
        "sovereign": "Sovereign Debt Crisis (year)",
        "sovereign_restructuring": "Sovereign Debt Restructuring (year)",
    }

    for _, row in df.iterrows():
        country = row["country"]
        if pd.isna(country):
            continue
        iso3 = _map_country_to_iso3(country, overrides=overrides)
        if iso3 is None:
            unmatched.append({"country": normalize_country_name(country)})
            continue
        for crisis_type, col in crisis_cols.items():
            years = _extract_years(row.get(col))
            for year in years:
                records.append(
                    {
                        "iso3": iso3,
                        "country": country,
                        "year": year,
                        "crisis_type": crisis_type,
                    }
                )

    events = pd.DataFrame.from_records(records)
    unmatched_df = pd.DataFrame.from_records(unmatched).drop_duplicates()
    return events, unmatched_df


def align_to_quinquennial(
    events: pd.DataFrame,
    start_year: int = 1970,
    end_year: int = 2020,
) -> pd.DataFrame:
    events = events.copy()
    remainder = events["year"] % 5
    events["year_quin"] = events["year"] + (5 - remainder)
    events.loc[remainder == 0, "year_quin"] = events.loc[remainder == 0, "year"]
    events = events[(events["year_quin"] >= start_year) & (events["year_quin"] <= end_year)]

    panel = (
        events.groupby(["iso3", "year_quin", "crisis_type"])
        .size()
        .reset_index(name="value")
    )
    panel["value"] = 1
    panel = panel.pivot_table(
        index=["iso3", "year_quin"],
        columns="crisis_type",
        values="value",
        fill_value=0,
    ).reset_index()
    panel.columns.name = None
    panel = panel.rename(columns={"year_quin": "year"})
    panel["any_crisis"] = (
        panel[[c for c in panel.columns if c not in ("iso3", "year")]].sum(axis=1) > 0
    ).astype(int)
    return panel


def build_crisis_panel(cache_dir: Path | str = "data/raw/crisis") -> CrisisDataResult:
    path = download_crisis_excel(cache_dir)
    events, unmatched = build_crisis_events(path)
    panel = align_to_quinquennial(events)
    return CrisisDataResult(events=events, panel=panel, unmatched=unmatched)
