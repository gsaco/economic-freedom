from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.config import PROCESSED_DIR, RAW_DIR


def _find_iso_column(columns: list[str]) -> str | None:
    for col in columns:
        if col.lower() in {"iso3", "iso", "member_iso", "country_code", "ctrycode"}:
            return col
        if "iso" in col.lower() and "3" in col.lower():
            return col
    return None


def ingest_wto_acdb() -> Path:
    raw_dir = RAW_DIR / "wto_acdb"
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.xlsx"))
    if not candidates:
        out = pd.DataFrame(columns=["iso3", "accession_year", "commitment_count", "I_wto", "I_wto_pca"])
        out_path = PROCESSED_DIR / "wto_intensity_acdb.parquet"
        out.to_parquet(out_path, index=False)
        return out_path

    path = candidates[0]
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    iso_col = _find_iso_column(list(df.columns))
    if iso_col is None:
        raise ValueError("Unable to identify iso3 column in ACDB dataset")

    df = df.copy()
    df[iso_col] = df[iso_col].astype(str).str.upper().str.strip()

    year_col = None
    for col in df.columns:
        if col.lower() in {"accession_year", "year", "acc_year"}:
            year_col = col
            break

    numeric_cols = [c for c in df.columns if c not in {iso_col, year_col}]
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["commitment_count"] = numeric_df.sum(axis=1, skipna=True)

    if df["commitment_count"].std(ddof=0) == 0:
        df["I_wto"] = np.nan
    else:
        df["I_wto"] = (df["commitment_count"] - df["commitment_count"].mean()) / df["commitment_count"].std(ddof=0)

    if len(numeric_cols) >= 2 and numeric_df.notna().sum().sum() > 0:
        filled = numeric_df.fillna(0.0)
        pca = PCA(n_components=1)
        df["I_wto_pca"] = pca.fit_transform(filled)[:, 0]
    else:
        df["I_wto_pca"] = np.nan

    keep_cols = [iso_col]
    if year_col:
        keep_cols.append(year_col)
    keep_cols += ["commitment_count", "I_wto", "I_wto_pca"]

    out = df[keep_cols].rename(columns={iso_col: "iso3", year_col: "accession_year"})

    out_path = PROCESSED_DIR / "wto_intensity_acdb.parquet"
    out.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    ingest_wto_acdb()
