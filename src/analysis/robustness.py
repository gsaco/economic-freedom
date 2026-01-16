from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.estimators.wrappers import run_estimators


def run_robustness() -> None:
    panel_path = PROCESSED_DIR / "panel_quinquennial.parquet"
    panel = pd.read_parquet(panel_path)

    placebo = panel.sort_values(["iso3", "year"]).copy()
    placebo["EU_neg_share"] = placebo.groupby("iso3")["EU_neg_share"].shift(2).fillna(0)
    placebo["WTO_neg_share"] = placebo.groupby("iso3")["WTO_neg_share"].shift(2).fillna(0)

    placebo_path = PROCESSED_DIR / "panel_placebo_timing.parquet"
    placebo.to_parquet(placebo_path, index=False)

    core_outcomes = [
        "efw_overall",
        "efw_delta5",
        "reform_event_03",
        "reversal_event_03",
        "hazard_rev_03",
    ]
    run_estimators(
        placebo_path,
        tag="placebo_timing",
        methods=["sdid", "did_cs"],
        outcomes=core_outcomes,
        sdid_cohorts=["2004"],
    )


if __name__ == "__main__":
    run_robustness()
