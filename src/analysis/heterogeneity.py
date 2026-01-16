from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR, TABLES_DIR


def run_heterogeneity() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "panel_quinquennial.parquet")

    eu = panel[panel["EU_neg_share"] > 0]
    wto = panel[panel["WTO_neg_share"] > 0]

    rows = []
    if not eu.empty:
        for col in ["baseline_efw_tercile_eu", "baseline_income_tercile_eu"]:
            if col in eu.columns:
                summary = eu.groupby(col)["efw_delta5"].mean().reset_index()
                for _, row in summary.iterrows():
                    rows.append({"anchor": "EU", "split": col, "tercile": row[col], "efw_delta5": row["efw_delta5"]})

    if not wto.empty:
        for col in ["baseline_efw_tercile_wto", "baseline_income_tercile_wto"]:
            if col in wto.columns:
                summary = wto.groupby(col)["efw_delta5"].mean().reset_index()
                for _, row in summary.iterrows():
                    rows.append({"anchor": "WTO", "split": col, "tercile": row[col], "efw_delta5": row["efw_delta5"]})

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["anchor", "split", "tercile", "efw_delta5"])
    out.to_csv(TABLES_DIR / "heterogeneity_state_dependence.csv", index=False)


if __name__ == "__main__":
    run_heterogeneity()
