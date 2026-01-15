"""Regime classification from OWID grapher (V-Dem indices)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import requests


OWID_BASE = "https://ourworldindata.org/grapher"
REGIME_DATASETS = {
    "electoral_democracy": "electoral-democracy.csv",
    "liberal_democracy": "liberal-democracy.csv",
}


@dataclass
class RegimeDataResult:
    panel: pd.DataFrame


def _write_metadata(meta_path: Path, metadata: dict) -> None:
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def download_owid_dataset(name: str, cache_dir: Path | str) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = REGIME_DATASETS[name]
    dest = cache_dir / filename
    if dest.exists():
        return dest

    url = f"{OWID_BASE}/{filename}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    meta = {
        "url": url,
        "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": dest.stat().st_size,
        "dataset": name,
    }
    _write_metadata(dest.with_suffix(dest.suffix + ".meta"), meta)
    return dest


def _load_owid_grapher(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    value_cols = [col for col in df.columns if col not in ("Entity", "Code", "Year")]
    if not value_cols:
        raise ValueError(f"No value column found in {path}")
    df = df.rename(
        columns={
            "Entity": "country",
            "Code": "iso3",
            "Year": "year",
            value_cols[0]: value_name,
        }
    )
    df = df[df["iso3"].str.len() == 3]
    return df[["iso3", "year", value_name]]


def build_regime_panel(cache_dir: Path | str = "data/raw/regime") -> RegimeDataResult:
    frames = []
    for name, filename in REGIME_DATASETS.items():
        path = download_owid_dataset(name, cache_dir)
        frames.append(_load_owid_grapher(path, name))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["iso3", "year"], how="outer")

    merged = merged[(merged["year"] >= 1970) & (merged["year"] <= 2020)]
    merged = merged[merged["year"] % 5 == 0]

    merged["democracy"] = (merged["electoral_democracy"] >= 0.5).astype(float)
    merged["autocracy"] = 1.0 - merged["democracy"]

    return RegimeDataResult(panel=merged)
