from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import TABLES_DIR


def _load_latest(prefix: str) -> pd.DataFrame:
    files = sorted(TABLES_DIR.glob(f"{prefix}_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def table_main_effects() -> Path:
    eu = _load_latest("estimates_eu")
    wto = _load_latest("estimates_wto")

    rows = []
    if not eu.empty:
        subset = eu[(eu["method"] == "sdid") & (eu["event_time"] == 0)]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "anchor": "EU",
                    "cohort": row["cohort"],
                    "outcome": row["outcome"],
                    "att": row["att"],
                    "se": row["se"],
                    "method": row["method"],
                }
            )

    if not wto.empty:
        subset = wto[(wto["method"] == "cs_did") & (wto["event_time"] == 0)]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "anchor": "WTO",
                    "cohort": row["cohort"],
                    "outcome": row["outcome"],
                    "att": row["att"],
                    "se": row["se"],
                    "method": row["method"],
                }
            )

    table = pd.DataFrame(rows)
    out_path = TABLES_DIR / "table_main_effects.csv"
    table.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    table_main_effects()
