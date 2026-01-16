from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.config import PROCESSED_DIR, QUINQUENNIAL_YEARS, RAW_DIR

PWT_URLS = [
    "https://www.rug.nl/ggdc/docs/pwt1001.xlsx",
    "https://www.rug.nl/ggdc/docs/pwt1000.xlsx",
    "https://www.rug.nl/ggdc/docs/pwt90.xlsx",
]


def _download_pwt(dest: Path) -> bool:
    for url in PWT_URLS:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException:
            continue
    return False


def _identify_columns(df: pd.DataFrame) -> tuple[str, str, str | None, str | None]:
    cols = {c.lower(): c for c in df.columns}
    iso_col = cols.get("countrycode") or cols.get("ctrycode") or cols.get("isocode") or cols.get("iso3")
    year_col = cols.get("year")
    tfp_col = cols.get("rtfpna") or cols.get("ctfp") or cols.get("tfp")
    rgdp_col = cols.get("rgdpe") or cols.get("rgdpo")
    if iso_col is None or year_col is None:
        raise ValueError("PWT data missing required iso/year columns")
    return iso_col, year_col, tfp_col, rgdp_col


def ingest_pwt() -> Path:
    raw_dir = RAW_DIR / "macro"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest = raw_dir / "pwt.xlsx"
    if not dest.exists():
        downloaded = _download_pwt(dest)
    else:
        downloaded = True

    if not downloaded:
        candidates = list(raw_dir.glob("pwt*.xlsx")) + list(raw_dir.glob("pwt*.csv"))
        if candidates:
            dest = candidates[0]
            downloaded = True

    if not downloaded:
        raise FileNotFoundError(
            "PWT dataset not found. Place a PWT Excel/CSV file in data/raw/macro/."
        )

    if dest.suffix.lower() == ".csv":
        df = pd.read_csv(dest)
    else:
        xls = pd.ExcelFile(dest)
        sheet = "Data" if "Data" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(dest, sheet_name=sheet)

    iso_col, year_col, tfp_col, rgdp_col = _identify_columns(df)

    keep_cols = [iso_col, year_col]
    if tfp_col:
        keep_cols.append(tfp_col)
    if rgdp_col:
        keep_cols.append(rgdp_col)

    out = df[keep_cols].copy()
    out = out.rename(
        columns={
            iso_col: "iso3",
            year_col: "year",
            tfp_col: "tfp_level",
            rgdp_col: "rgdpe",
        }
    )

    out["iso3"] = out["iso3"].astype(str).str.upper().str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["iso3", "year"])
    out["year"] = out["year"].astype(int)
    out = out[out["year"].isin(QUINQUENNIAL_YEARS)]

    out_path = PROCESSED_DIR / "pwt_quinquennial.parquet"
    out.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    ingest_pwt()
