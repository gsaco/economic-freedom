"""Macro data fetching utilities (World Bank + PWT)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests


@dataclass
class MacroFetchResult:
    wb_data: pd.DataFrame
    pwt_data: pd.DataFrame
    wb_sources: Dict[str, str]
    pwt_source: str


def _write_metadata(meta_path: Path, metadata: dict) -> None:
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def download_file(url: str, dest: Path, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Using cached download: {dest}")
        return dest
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with dest.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    meta = {
        "url": url,
        "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": dest.stat().st_size,
    }
    _write_metadata(dest.with_suffix(dest.suffix + ".json"), meta)
    print(f"Saved to: {dest}")
    return dest


def _fetch_wb_page(url: str, params: dict) -> Tuple[dict, List[dict]]:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Unexpected World Bank API response")
    return payload[0], payload[1]


def fetch_world_bank_indicator(
    indicator: str,
    cache_dir: Path | str,
    start_year: int = 1970,
    end_year: int = 2020,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{indicator}.json"

    if cache_path.exists():
        print(f"Using cached World Bank data: {cache_path}")
        data = json.loads(cache_path.read_text())
    else:
        base_url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        params = {
            "format": "json",
            "per_page": 20000,
            "page": 1,
        }
        meta, rows = _fetch_wb_page(base_url, params)
        pages = meta.get("pages", 1)
        all_rows = rows
        for page in range(2, pages + 1):
            params["page"] = page
            _, rows = _fetch_wb_page(base_url, params)
            all_rows.extend(rows)
        data = all_rows
        cache_path.write_text(json.dumps(data))
        meta_path = cache_path.with_suffix(".json.meta")
        _write_metadata(
            meta_path,
            {
                "url": base_url,
                "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "indicator": indicator,
                "rows": len(data),
            },
        )
        print(f"Saved World Bank data: {cache_path}")

    records = []
    for item in data:
        year = item.get("date")
        if year is None:
            continue
        try:
            year_int = int(year)
        except ValueError:
            continue
        if year_int < start_year or year_int > end_year:
            continue
        records.append(
            {
                "iso3": item.get("countryiso3code"),
                "country": item.get("country", {}).get("value"),
                "year": year_int,
                indicator: item.get("value"),
            }
        )

    df = pd.DataFrame.from_records(records)
    df[indicator] = pd.to_numeric(df[indicator], errors="coerce")
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    return df


def fetch_world_bank_indicators(
    indicators: Dict[str, str],
    cache_dir: Path | str = "data/raw/wb",
    start_year: int = 1970,
    end_year: int = 2020,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames = []
    for name, indicator in indicators.items():
        df = fetch_world_bank_indicator(indicator, cache_dir, start_year, end_year)
        df = df.rename(columns={indicator: name})
        frames.append(df)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["iso3", "country", "year"], how="outer")
    return merged, indicators


def fetch_pwt(
    cache_dir: Path | str = "data/raw/pwt",
) -> Tuple[pd.DataFrame, str]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = "https://www.rug.nl/ggdc/docs/pwt100.xlsx"
    dest = cache_dir / "pwt100.xlsx"
    download_file(url, dest)
    df = pd.read_excel(dest, sheet_name="Data")
    df = df.rename(columns={"countrycode": "iso3", "year": "year"})
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    return df, url


def prepare_pwt_subset(df: pd.DataFrame, start_year: int = 1970, end_year: int = 2020) -> pd.DataFrame:
    keep_cols = ["iso3", "country", "year", "rgdpe", "pop", "hc"]
    available = [col for col in keep_cols if col in df.columns]
    subset = df[available].copy()
    subset = subset[(subset["year"] >= start_year) & (subset["year"] <= end_year)]
    if "rgdpe" in subset.columns and "pop" in subset.columns:
        subset["pwt_gdppc"] = subset["rgdpe"] / subset["pop"]
    return subset
