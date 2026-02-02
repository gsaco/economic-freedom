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
    parser = argparse.ArgumentParser(description="Build data dictionary for cleanroom full outputs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.root)
    run_dir = cfg.paths.audit / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    processed = cfg.paths.processed
    full_files = sorted(processed.glob("*_full.parquet"))
    if not full_files:
        raise FileNotFoundError("No *_full.parquet files found in data/processed")

    rows = []
    for path in full_files:
        table = path.stem
        df = pd.read_parquet(path)
        for col in df.columns:
            rows.append({
                "column_name": col,
                "table": table,
                "dtype": str(df[col].dtype),
                "meaning": "",
                "source(s)": "",
                "operationalization": "",
                "missingness_pct": float(df[col].isna().mean()),
                "caveats": "",
            })

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "data_dictionary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
