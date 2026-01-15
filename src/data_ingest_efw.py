"""EFW ingestion and cleaning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


@dataclass
class EfwIngestResult:
    df: pd.DataFrame
    crosswalk: pd.DataFrame
    source_sheet: str


def locate_fraser_file(root: Path | str) -> Path:
    root_path = Path(root)
    matches = sorted(root_path.rglob("fraser.xlsx"))
    if not matches:
        raise FileNotFoundError(
            f"fraser.xlsx not found under {root_path.resolve()}"
        )
    # If multiple matches, pick the first but expose all for logging upstream.
    return matches[0]


def _normalize_columns(cols: Iterable[str]) -> list[str]:
    normalized = []
    seen = {}
    for col in cols:
        clean = str(col).strip()
        # Ensure uniqueness if there are repeated column labels.
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}__{seen[clean]}"
        else:
            seen[clean] = 0
        normalized.append(clean)
    return normalized


def _load_panel_sheet(path: Path) -> Tuple[pd.DataFrame, str]:
    xl = pd.ExcelFile(path)
    if "EFW Panel Dataset" in xl.sheet_names:
        df = pd.read_excel(path, sheet_name="EFW Panel Dataset")
        return df, "EFW Panel Dataset"
    if "EFW Index 1970-2023" in xl.sheet_names:
        raw = pd.read_excel(path, sheet_name="EFW Index 1970-2023", header=2)
        header = raw.iloc[0].tolist()
        df = raw.iloc[1:].copy()
        df.columns = _normalize_columns(header)
        return df, "EFW Index 1970-2023"
    raise ValueError("No expected EFW sheet found in fraser.xlsx")


def clean_efw_panel(df: pd.DataFrame, source_sheet: str) -> pd.DataFrame:
    df = df.copy()
    if source_sheet == "EFW Index 1970-2023":
        # Map to the same column names used in the Panel Dataset if possible.
        rename_map = {
            "Year": "year",
            "ISO_Code": "iso3",
            "Countries": "country",
            "ECONOMIC FREEDOM ALL AREAS": "efw_summary",
            "Area 1 Size of Government": "efw_area1",
            "Area 2 Legal System and Property Rights": "efw_area2",
            "Area 3 Sound Money": "efw_area3",
            "Area 4 Freedom to Trade Internationally": "efw_area4",
            "Area 5 Regulation": "efw_area5",
            "World Bank Region": "wb_region",
            "World Bank Current Income Classification, 1990-Present": "wb_income_class",
        }
        df = df.rename(columns=rename_map)
    else:
        rename_map = {
            "Year": "year",
            "ISO_Code": "iso3",
            "Countries": "country",
            "Summary": "efw_summary",
            "Area 1": "efw_area1",
            "Area 2": "efw_area2",
            "Area 3": "efw_area3",
            "Area 4": "efw_area4",
            "Area 5": "efw_area5",
            "World Bank Region": "wb_region",
            "World Bank Current Income Classification, 1990-Present": "wb_income_class",
        }
        df = df.rename(columns=rename_map)

    keep_cols = [
        "year",
        "iso3",
        "country",
        "efw_summary",
        "efw_area1",
        "efw_area2",
        "efw_area3",
        "efw_area4",
        "efw_area5",
        "wb_region",
        "wb_income_class",
    ]
    available = [col for col in keep_cols if col in df.columns]
    df = df[available]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in [
        "efw_summary",
        "efw_area1",
        "efw_area2",
        "efw_area3",
        "efw_area4",
        "efw_area5",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df["country"] = df["country"].astype(str).str.strip()

    df = df.dropna(subset=["year", "iso3"])
    df = df[df["iso3"].str.len() == 3]

    return df


def to_quinquennial(df: pd.DataFrame, start_year: int = 1970, end_year: int = 2020) -> pd.DataFrame:
    quin_years = set(range(start_year, end_year + 1, 5))
    df = df[df["year"].isin(quin_years)].copy()
    return df


def compute_efw_changes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["iso3", "year"]).copy()
    for col in [
        "efw_summary",
        "efw_area1",
        "efw_area2",
        "efw_area3",
        "efw_area4",
        "efw_area5",
    ]:
        if col in df.columns:
            df[f"d_{col}"] = df.groupby("iso3")[col].diff()
    return df


def build_country_crosswalk(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["iso3", "country"]
    for extra in ["wb_region", "wb_income_class"]:
        if extra in df.columns:
            cols.append(extra)
    crosswalk = df[cols].drop_duplicates().sort_values(["iso3", "country"]).reset_index(drop=True)
    crosswalk["iso3_valid"] = crosswalk["iso3"].str.len() == 3
    return crosswalk


def ingest_efw(path: Path | str) -> EfwIngestResult:
    path = Path(path)
    df_raw, sheet = _load_panel_sheet(path)
    df_clean = clean_efw_panel(df_raw, sheet)
    df_clean = to_quinquennial(df_clean)
    df_clean = compute_efw_changes(df_clean)
    crosswalk = build_country_crosswalk(df_clean)
    return EfwIngestResult(df=df_clean, crosswalk=crosswalk, source_sheet=sheet)
