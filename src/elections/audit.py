from __future__ import annotations

from pathlib import Path
import pandas as pd


def audit_rowcount(
    df: pd.DataFrame,
    table_name: str,
    step_id: str,
    audit_path: Path,
    key_cols: list[str] | None = None,
    notes: str | None = None,
) -> None:
    if df is None:
        return
    metrics = {
        "step_id": step_id,
        "table_name": table_name,
        "n_rows": int(len(df)),
    }
    for col in ["iso3", "election_id", "record_id"]:
        if col in df.columns:
            metrics[f"n_unique_{col}"] = int(df[col].dropna().nunique())
    if "election_year" in df.columns:
        metrics["min_year"] = int(df["election_year"].min()) if df["election_year"].notna().any() else None
        metrics["max_year"] = int(df["election_year"].max()) if df["election_year"].notna().any() else None
    elif "year" in df.columns:
        metrics["min_year"] = int(df["year"].min()) if df["year"].notna().any() else None
        metrics["max_year"] = int(df["year"].max()) if df["year"].notna().any() else None

    for col in ["iso3", "election_date", "share_1", "party_1_id", "party_2_id"]:
        if col in df.columns:
            metrics[f"pct_missing_{col}"] = float(df[col].isna().mean())

    if key_cols:
        dupes = df.duplicated(key_cols, keep=False).sum()
        metrics["n_duplicate_keys"] = int(dupes)
    if notes:
        metrics["notes"] = notes

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([metrics])
    if audit_path.exists():
        existing = pd.read_csv(audit_path)
        combined = pd.concat([existing, out], ignore_index=True)
    else:
        combined = out
    combined.to_csv(audit_path, index=False)


def write_simple_audit(df: pd.DataFrame, path: Path, extra: dict | None = None) -> None:
    stats = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }
    if extra:
        stats.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(path, index=False)
