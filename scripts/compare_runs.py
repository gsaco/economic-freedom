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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare coverage reports between two runs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--new", type=str, required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.root)
    base_path = cfg.paths.audit / "runs" / args.baseline / "coverage_report_2000plus.csv"
    new_path = cfg.paths.audit / "runs" / args.new / "coverage_report_2000plus.csv"

    if not base_path.exists() or not new_path.exists():
        missing = []
        if not base_path.exists():
            missing.append(str(base_path))
        if not new_path.exists():
            missing.append(str(new_path))
        raise FileNotFoundError(f"Missing coverage reports: {', '.join(missing)}")

    base = pd.read_csv(base_path)
    new = pd.read_csv(new_path)
    merged = base.merge(new, on="metric", how="outer", suffixes=("_baseline", "_new"))
    merged["delta"] = merged["value_new"] - merged["value_baseline"]

    out_path = cfg.paths.audit / f"compare_{args.baseline}_vs_{args.new}.md"
    lines = [f"# Coverage comparison {args.baseline} vs {args.new}", "", "| metric | baseline | new | delta |", "| --- | ---: | ---: | ---: |"]
    for _, row in merged.iterrows():
        b = row.get("value_baseline")
        n = row.get("value_new")
        d = row.get("delta")
        lines.append(f"| {row['metric']} | {b:.6f} | {n:.6f} | {d:.6f} |" if all(isinstance(x, float) for x in [b, n, d]) else f"| {row['metric']} | {b} | {n} | {d} |")
    out_path.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
