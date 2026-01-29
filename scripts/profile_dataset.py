from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "elections_final_clean_2000_tidy.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

PROFILE_JSON = REPORTS / "elections_profile_2000_tidy.json"
PROFILE_CSV = REPORTS / "elections_profile_2000_tidy.csv"


def infer_type(series: pd.Series, col: str) -> str:
    if col == "date":
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "string"


def summarize_numeric(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return {}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p1": float(s.quantile(0.01)),
        "p50": float(s.quantile(0.50)),
        "p99": float(s.quantile(0.99)),
    }


def summarize_categorical(series: pd.Series, top_n: int = 15) -> dict:
    s = series.dropna()
    if s.empty:
        return {"levels": [], "unique_count": 0}
    vc = s.value_counts(dropna=True)
    total = int(vc.sum())
    levels = []
    for val, cnt in vc.head(top_n).items():
        levels.append({
            "value": str(val),
            "count": int(cnt),
            "pct": float(cnt / total * 100),
        })
    return {
        "levels": levels,
        "unique_count": int(vc.size),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    n_rows, n_cols = df.shape

    profile = {
        "dataset": str(DATA_PATH),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": {},
    }

    rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        inferred = infer_type(series, col)
        missing_count = int(series.isna().sum())
        missing_pct = float(missing_count / n_rows * 100) if n_rows else 0.0
        nunique = int(series.nunique(dropna=True))
        examples = series.dropna().astype(str).unique().tolist()[:5]

        col_profile = {
            "dtype": dtype,
            "inferred_type": inferred,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "nunique": nunique,
            "examples": examples,
        }

        if inferred in {"int", "float"}:
            col_profile["numeric_stats"] = summarize_numeric(series)
        else:
            col_profile["categorical_stats"] = summarize_categorical(series)

        profile["columns"][col] = col_profile

        row = {
            "column": col,
            "dtype": dtype,
            "inferred_type": inferred,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "nunique": nunique,
        }
        if inferred in {"int", "float"}:
            stats = col_profile.get("numeric_stats", {})
            row.update({
                "min": stats.get("min"),
                "max": stats.get("max"),
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "p1": stats.get("p1"),
                "p50": stats.get("p50"),
                "p99": stats.get("p99"),
            })
        rows.append(row)

    PROFILE_JSON.write_text(json.dumps(profile, indent=2))
    pd.DataFrame(rows).to_csv(PROFILE_CSV, index=False)

    print(f"Rows: {n_rows}")
    print(f"Cols: {n_cols}")
    print(f"Wrote {PROFILE_JSON}")
    print(f"Wrote {PROFILE_CSV}")


if __name__ == "__main__":
    main()
