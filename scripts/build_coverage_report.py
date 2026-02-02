from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elections.config import load_config


def _missing_rate(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    return float(df[col].isna().mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build 2000+ coverage report for cleanroom outputs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.root)
    run_dir = cfg.paths.audit / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    full_path = cfg.paths.processed / "election_event_full.parquet"
    view_path = cfg.paths.processed / "election_event.parquet"
    path = full_path if full_path.exists() else view_path
    df = pd.read_parquet(path)

    df_2000 = df[df["election_year"] >= 2000]
    df_2000_2020 = df_2000[df_2000["election_year"] <= 2020]

    metrics = {
        "rows_total": float(len(df)),
        "rows_2000plus": float(len(df_2000)),
        "missing_iso3_2000plus": _missing_rate(df_2000, "iso3"),
        "missing_share_1_2000plus": _missing_rate(df_2000, "share_1"),
        "missing_party_1_id_2000plus": _missing_rate(df_2000, "party_1_id"),
        "missing_party_1_id_best_2000plus": _missing_rate(df_2000, "party_1_id_best"),
        "missing_ideo_std_1_2000plus": _missing_rate(df_2000, "ideo_std_1"),
        "missing_efw_baseline_2000plus": _missing_rate(df_2000, "efw_summary_baseline"),
        "missing_nelda_3_2000plus": _missing_rate(df_2000, "nelda3"),
        "pct_efw_backfilled_2000plus": float(df_2000.get("efw_baseline_backfilled_flag", pd.Series([False] * len(df_2000))).fillna(False).mean()) if len(df_2000) else 0.0,
    }

    if "nelda_match_status" in df_2000_2020.columns and len(df_2000_2020) > 0:
        matched = df_2000_2020["nelda_match_status"].isin(["matched", "matched_outside_tolerance"]).mean()
        metrics["nelda_match_rate_2000_2020"] = float(matched)

    metrics_df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
    metrics_df.to_csv(run_dir / "coverage_report_2000plus.csv", index=False)

    # simple markdown summary
    lines = ["# Coverage report (2000+)", "", f"Source: {path.name}", "", "| metric | value |", "| --- | ---: |"]
    for _, row in metrics_df.iterrows():
        lines.append(f"| {row['metric']} | {row['value']:.6f} |" if isinstance(row["value"], float) else f"| {row['metric']} | {row['value']} |")
    (run_dir / "coverage_report_2000plus.md").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
