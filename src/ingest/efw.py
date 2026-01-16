from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.config import PROCESSED_DIR, QUINQUENNIAL_YEARS, RAW_DIR

EFW_URLS = [
    "https://www.fraserinstitute.org/sites/default/files/efw-2023-master-index-data-for-researchers.xlsx",
    "https://www.fraserinstitute.org/sites/default/files/efw-2022-master-index-data-for-researchers.xlsx",
    "https://www.fraserinstitute.org/sites/default/files/efw-2021-master-index-data-for-researchers.xlsx",
]


def _download_file(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return False
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException:
        return False


def _find_header_row(path: Path, sheet_name: str | int | None = None) -> int:
    preview = pd.read_excel(path, header=None, nrows=10, sheet_name=sheet_name)
    if isinstance(preview, dict):
        preview = list(preview.values())[0]
    for idx, row in preview.iterrows():
        values = [str(v).strip() for v in row.values]
        if "Year" in values and any("ISO" in v for v in values):
            return idx
    return 0


def _standardize_columns(columns: Iterable[str]) -> list[str]:
    cleaned = []
    for col in columns:
        col_str = str(col).strip()
        col_str = col_str.replace(" ", "_").replace("/", "_").replace("-", "_")
        col_str = col_str.replace("(", "").replace(")", "")
        col_str = col_str.replace("&", "and")
        cleaned.append(col_str.lower())
    return cleaned


def _select_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in df.columns:
        if col in candidates:
            return col
    return None


def ingest_efw() -> Path:
    raw_dir = RAW_DIR / "efw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest = raw_dir / "efw_master.xlsx"
    downloaded = False
    if not dest.exists():
        for url in EFW_URLS:
            if _download_file(url, dest):
                downloaded = True
                break
    else:
        downloaded = True

    if not downloaded:
        candidates = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.csv"))
        alt = Path("data/fraser.xlsx")
        if alt.exists():
            dest = alt
            downloaded = True
        elif candidates:
            dest = candidates[0]
            downloaded = True

    if not downloaded:
        raise FileNotFoundError(
            "EFW dataset not found. Place an EFW Excel/CSV file in data/raw/efw/."
        )

    if dest.suffix.lower() == ".csv":
        df = pd.read_csv(dest)
    else:
        header_row = _find_header_row(dest, sheet_name=0)
        df = pd.read_excel(dest, header=header_row, sheet_name=0)

    df.columns = _standardize_columns(df.columns)

    iso_col = _select_column(df, ["iso_code", "iso_code3", "iso3", "iso"])
    if iso_col is None and "iso_code" in df.columns:
        iso_col = "iso_code"
    year_col = _select_column(df, ["year"])

    overall_col = None
    for col in df.columns:
        if "economic_freedom_all_areas" in col or col == "efw":
            overall_col = col
            break

    area1_col = None
    area2_col = None
    area3_col = None
    area4_col = None
    area5_col = None
    for col in df.columns:
        if col.startswith("area_1") and "size_of_government" in col:
            area1_col = col
        if col.startswith("area_2") and "with_gender_adjustment" in col:
            area2_col = col
        if col.startswith("area_2") and "without_gender_adjustment" in col and area2_col is None:
            area2_col = col
        if col.startswith("area_3") and "sound_money" in col:
            area3_col = col
        if col.startswith("area_4") and "freedom_to_trade_internationally" in col:
            area4_col = col
        if col.startswith("area_5") and "regulation" in col:
            area5_col = col

    if iso_col is None or year_col is None or overall_col is None:
        raise ValueError("Unable to identify required columns in EFW dataset.")

    keep_cols = [iso_col, year_col, overall_col]
    for col in [area1_col, area2_col, area3_col, area4_col, area5_col]:
        if col:
            keep_cols.append(col)

    out = df[keep_cols].copy()
    out = out.rename(
        columns={
            iso_col: "iso3",
            year_col: "year",
            overall_col: "efw_overall",
            area1_col: "efw_area1",
            area2_col: "efw_area2",
            area3_col: "efw_area3",
            area4_col: "efw_area4",
            area5_col: "efw_area5",
        }
    )

    out["iso3"] = out["iso3"].astype(str).str.upper().str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["iso3", "year"])
    out["year"] = out["year"].astype(int)
    out = out[out["year"].isin(QUINQUENNIAL_YEARS)]

    out = out.drop_duplicates(subset=["iso3", "year"])
    out = out.sort_values(["iso3", "year"])

    out_path = PROCESSED_DIR / "efw_quinquennial.parquet"
    out.to_parquet(out_path, index=False)

    summary = out.groupby("year")["iso3"].nunique().reset_index(name="n_countries")
    summary_path = PROCESSED_DIR / "efw_coverage.json"
    summary_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2))

    return out_path


if __name__ == "__main__":
    ingest_efw()
