"""SWIID inequality ingestion and quinquennial alignment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import zipfile

from src.country_utils import country_to_iso3, normalize_country_name


SWIID_PERSISTENT_ID = "doi:10.7910/DVN/LM4OWF"
SWIID_FILENAME = "swiid9_91.zip"


@dataclass
class InequalityDataResult:
    panel: pd.DataFrame
    unmatched: pd.DataFrame


def _write_metadata(meta_path: Path, metadata: dict) -> None:
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _get_swiid_file_id() -> int:
    url = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={SWIID_PERSISTENT_ID}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    files = response.json()["data"]["latestVersion"]["files"]
    for entry in files:
        data_file = entry["dataFile"]
        if data_file["filename"] == SWIID_FILENAME:
            return data_file["id"]
    raise ValueError(f"{SWIID_FILENAME} not found in SWIID dataverse.")


def download_swiid_zip(cache_dir: Path | str) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / SWIID_FILENAME
    if dest.exists():
        return dest

    file_id = _get_swiid_file_id()
    url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)

    meta = {
        "url": url,
        "persistent_id": SWIID_PERSISTENT_ID,
        "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": dest.stat().st_size,
    }
    _write_metadata(dest.with_suffix(dest.suffix + ".meta"), meta)
    return dest


def extract_swiid_dta(zip_path: Path) -> Path:
    dest = zip_path.with_suffix(".dta")
    if dest.exists():
        return dest

    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.endswith(".dta")]
        if not members:
            raise ValueError("No .dta file found in SWIID zip.")
        with zf.open(members[0]) as source, dest.open("wb") as target:
            target.write(source.read())
    return dest


def build_inequality_panel(cache_dir: Path | str = "data/raw/swiid") -> InequalityDataResult:
    zip_path = download_swiid_zip(cache_dir)
    dta_path = extract_swiid_dta(zip_path)

    df = pd.read_stata(dta_path, convert_categoricals=False)
    disp_cols = [col for col in df.columns if col.startswith("_") and col.endswith("gini_disp")]
    mkt_cols = [col for col in df.columns if col.startswith("_") and col.endswith("gini_mkt")]

    base = df[["country", "year", "gini_disp", "gini_mkt"]].copy()
    base["iso3"] = base["country"].apply(lambda x: country_to_iso3(x))

    unmatched = base[base["iso3"].isna()][["country"]].drop_duplicates()
    unmatched["country"] = unmatched["country"].apply(normalize_country_name)

    base["gini_net"] = base["gini_disp"]
    base["gini_market"] = base["gini_mkt"]
    if disp_cols:
        base["gini_net_sd"] = df[disp_cols].std(axis=1)
    if mkt_cols:
        base["gini_market_sd"] = df[mkt_cols].std(axis=1)

    panel = base[
        [
            "iso3",
            "year",
            "gini_net",
            "gini_market",
            "gini_net_sd",
            "gini_market_sd",
        ]
    ].copy()

    panel = panel[(panel["year"] >= 1970) & (panel["year"] <= 2020)]
    panel = panel[panel["year"] % 5 == 0]
    panel = panel[panel["iso3"].str.len() == 3]

    meta_path = dta_path.with_suffix(dta_path.suffix + ".meta")
    meta = {}
    zip_meta_path = zip_path.with_suffix(zip_path.suffix + ".meta")
    if zip_meta_path.exists():
        meta.update(json.loads(zip_meta_path.read_text()))
    meta.update(
        {
            "source_zip": zip_path.name,
            "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "variables": list(df.columns),
        }
    )
    _write_metadata(meta_path, meta)

    return InequalityDataResult(panel=panel, unmatched=unmatched)
