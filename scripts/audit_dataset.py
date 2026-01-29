from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from elections_audit import invariant_checks
from elections_core import load_iso_reference


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_metrics(df: pd.DataFrame, label: str) -> dict:
    metrics: dict[str, object] = {
        "label": label,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
    }
    for col in ["iso3", "year", "month", "office_type", "analysis_ready"]:
        if col in df.columns:
            metrics[f"{col}_missing_rate"] = float(df[col].isna().mean())
    if "iso3" in df.columns:
        metrics["iso3_unique"] = int(df["iso3"].nunique())
    if "year" in df.columns and df["year"].notna().any():
        metrics["year_min"] = int(df["year"].min())
        metrics["year_max"] = int(df["year"].max())
    if "share_1" in df.columns:
        metrics["share1_missing_rate"] = float(df["share_1"].isna().mean())
    if "share_2" in df.columns:
        metrics["share2_missing_rate"] = float(df["share_2"].isna().mean())
    if "margin_market_best" in df.columns:
        metrics["margin_best_missing_rate"] = float(df["margin_market_best"].isna().mean())
    key_base = [c for c in ["iso3", "office_type", "year", "month"] if c in df.columns]
    if key_base:
        metrics["dup_rows_iso_office_year_month"] = int(df.duplicated(key_base, keep=False).sum())
    if "election_id" in df.columns:
        metrics["dup_election_id"] = int(df["election_id"].duplicated().sum())
    return metrics


def subset_metrics(df: pd.DataFrame, label: str, min_year: int) -> dict:
    if "year" not in df.columns:
        return {"label": label, "rows": 0}
    sub = df[df["year"] >= min_year].copy()
    metrics = dataset_metrics(sub, label)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit processed elections datasets.")
    parser.add_argument("--final", default="data/processed/elections_final.parquet")
    parser.add_argument("--master", default="data/processed/elections_master.parquet")
    parser.add_argument("--out-metrics", default="reports/audit_metrics.json")
    parser.add_argument("--out-checksums", default="reports/audit_checksums.json")
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any invariant failures.")
    args = parser.parse_args()

    metrics_out = Path(args.out_metrics)
    checksums_out = Path(args.out_checksums)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    checksums_out.parent.mkdir(parents=True, exist_ok=True)

    metrics = {}
    checksums = {}

    iso_valid = None
    try:
        iso_valid = set(load_iso_reference()["iso3"])
    except Exception:
        iso_valid = None

    for label, path_str in [("final", args.final), ("master", args.master)]:
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        metrics[label] = dataset_metrics(df, label)
        metrics[f"{label}_{args.min_year}plus"] = subset_metrics(df, f"{label}_{args.min_year}plus", args.min_year)
        report = invariant_checks(df, iso3_valid=iso_valid)
        metrics[f"{label}_invariants"] = report.issues

        checksums[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    metrics_out.write_text(json.dumps(metrics, indent=2))
    checksums_out.write_text(json.dumps(checksums, indent=2))

    if args.strict:
        failures = 0
        for key, value in metrics.items():
            if key.endswith("_invariants") and isinstance(value, dict):
                failures += sum(int(v) for v in value.values() if isinstance(v, (int, float)))
        if failures:
            raise SystemExit(f"Invariant failures detected: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
