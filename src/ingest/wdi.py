from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from src.config import INTERIM_DIR, PROCESSED_DIR, QUINQUENNIAL_YEARS, RAW_DIR

WDI_INDICATORS = {
    "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp_const",  # GDP per capita, PPP (constant 2017 int $)
    "FP.CPI.TOTL.ZG": "inflation_cpi",
    "NE.GDI.FTOT.ZS": "inv_share_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_gdp",
    "FS.AST.PRVT.GD.ZS": "credit_gdp",
}


def _download_indicator(code: str, dest: Path) -> pd.DataFrame:
    if dest.exists():
        data = json.loads(dest.read_text())
        return pd.DataFrame(data[1])

    url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        dest.write_text(json.dumps(data))
        return pd.DataFrame(data[1])
    except requests.RequestException as exc:
        if dest.exists():
            data = json.loads(dest.read_text())
            return pd.DataFrame(data[1])
        raise RuntimeError(
            f"WDI download failed for {code}. Place cached JSON at {dest}."
        ) from exc


def ingest_wdi() -> Path:
    raw_dir = RAW_DIR / "macro"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    annual_frames = []
    for code, out_name in WDI_INDICATORS.items():
        dest = raw_dir / f"wdi_{code}.json"
        raw_df = _download_indicator(code, dest)

        df = raw_df[["countryiso3code", "date", "value"]].copy()
        df = df.rename(columns={"countryiso3code": "iso3", "date": "year", "value": out_name})
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["iso3", "year"])
        df["year"] = df["year"].astype(int)
        df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
        df = df[df["iso3"].str.len() == 3]
        annual_frames.append(df)

        df_q = df[df["year"].isin(QUINQUENNIAL_YEARS)].copy()
        frames.append(df_q)

    annual = annual_frames[0]
    for frame in annual_frames[1:]:
        annual = annual.merge(frame, on=["iso3", "year"], how="outer")
    annual_path = INTERIM_DIR / "wdi_annual.parquet"
    annual.to_parquet(annual_path, index=False)

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["iso3", "year"], how="outer")

    out_path = PROCESSED_DIR / "wdi_quinquennial.parquet"
    out.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    ingest_wdi()
