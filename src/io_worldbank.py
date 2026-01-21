from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

WDI_BASE_URL = "https://api.worldbank.org/v2"
DEFAULT_PER_PAGE = 20000


def _request_json(url: str, params: dict | None = None, retries: int = 3) -> list:
    params = params or {}
    params.setdefault("format", "json")
    params.setdefault("per_page", DEFAULT_PER_PAGE)
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError("Unreachable retry guard")


def fetch_wdi_indicator(code: str, cache_dir: Path, force: bool = False) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{code}.parquet"
    if cache_path.exists() and not force:
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    url = f"{WDI_BASE_URL}/country/all/indicator/{code}"
    payload = _request_json(url)
    if len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"No data returned for WDI indicator {code}.")
    raw = pd.DataFrame(payload[1])
    df = raw[["countryiso3code", "date", "value"]].copy()
    df = df.rename(columns={"countryiso3code": "iso3c", "date": "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["iso3c", "year"])
    df["iso3c"] = df["iso3c"].astype(str).str.upper().str.strip()
    df = df[df["iso3c"].str.len() == 3]
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["iso3c", "year"])
    df.to_parquet(cache_path, index=False)
    return df


def fetch_indicator_metadata(code: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{code}_meta.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    url = f"{WDI_BASE_URL}/indicator/{code}"
    payload = _request_json(url)
    record = payload[1][0] if payload and payload[1] else {}
    meta = {
        "code": code,
        "name": record.get("name"),
        "source": record.get("source", {}).get("value"),
        "source_note": record.get("sourceNote"),
        "source_organization": record.get("sourceOrganization"),
        "unit": record.get("unit"),
    }
    cache_path.write_text(json.dumps(meta, indent=2))
    return meta


def fetch_country_metadata(cache_dir: Path, force: bool = False) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "country_metadata.parquet"
    if cache_path.exists() and not force:
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    url = f"{WDI_BASE_URL}/country"
    payload = _request_json(url)
    if len(payload) < 2 or payload[1] is None:
        raise RuntimeError("No country metadata returned from World Bank API.")
    raw = pd.DataFrame(payload[1])
    df = pd.DataFrame(
        {
            "iso3c": raw["id"],
            "country_name": raw["name"],
            "wb_region": raw["region"].apply(lambda item: item.get("value") if isinstance(item, dict) else None),
            "wb_income_group": raw["incomeLevel"].apply(
                lambda item: item.get("value") if isinstance(item, dict) else None
            ),
        }
    )
    df = df[df["iso3c"].str.len() == 3]
    df = df[df["wb_region"].notna() & (df["wb_region"] != "Aggregates")]
    df = df.sort_values("iso3c")
    df.to_parquet(cache_path, index=False)
    return df
