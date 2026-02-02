from __future__ import annotations

from pathlib import Path
import hashlib
import pandas as pd


def run_audit_path(audit_root: Path, run_id: str, filename: str) -> Path:
    return audit_root / "runs" / run_id / filename


def _rel_path(path: Path, root_dir: Path | None) -> str:
    if root_dir is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_output_manifest(processed_root: Path, out_path: Path, root_dir: Path | None = None, run_id: str | None = None) -> None:
    records = []
    for path in sorted(processed_root.rglob("*")):
        if path.is_file():
            record = {
                "relative_path": _rel_path(path, root_dir),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            if run_id is not None:
                record["run_id"] = run_id
            records.append(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)


def audit_rowcount(
    df: pd.DataFrame,
    table_name: str,
    step_id: str,
    audit_path: Path,
    key_cols: list[str] | None = None,
    notes: str | None = None,
    run_id: str | None = None,
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
    if run_id is not None:
        metrics["run_id"] = run_id

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([metrics])
    if audit_path.exists():
        existing = pd.read_csv(audit_path)
        combined = pd.concat([existing, out], ignore_index=True)
    else:
        combined = out
    combined.to_csv(audit_path, index=False)


def write_simple_audit(df: pd.DataFrame, path: Path, extra: dict | None = None, run_id: str | None = None) -> None:
    stats = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }
    if extra:
        stats.update(extra)
    if run_id is not None:
        stats["run_id"] = run_id
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(path, index=False)
