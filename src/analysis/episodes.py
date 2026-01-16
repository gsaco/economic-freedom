from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from src.config import PROCESSED_DIR, TABLES_DIR
from src.estimators.wrappers import run_estimators


def _run_filtered(panel: pd.DataFrame, tag: str) -> None:
    path = PROCESSED_DIR / f"panel_{tag}.parquet"
    panel.to_parquet(path, index=False)
    core_outcomes = [
        "efw_overall",
        "efw_delta5",
        "reform_event_03",
        "reversal_event_03",
        "hazard_rev_03",
    ]
    run_estimators(
        path,
        tag=tag,
        methods=["sdid", "did_cs"],
        outcomes=core_outcomes,
        sdid_cohorts=["2004"],
    )


def run_episode_robustness() -> Path:
    panel_path = PROCESSED_DIR / "panel_quinquennial.parquet"
    panel = pd.read_parquet(panel_path)

    drop_1990s = panel[~panel["year"].isin([1990, 1995, 2000])]
    drop_2008 = panel[~panel["year"].isin([2005, 2010])]
    post_2000 = panel[panel["year"] >= 2000]

    _run_filtered(drop_1990s, "episodes_drop_1990s")
    _run_filtered(drop_2008, "episodes_drop_2008")
    _run_filtered(post_2000, "episodes_post_2000")

    panel = panel.copy()
    panel["crisis_1990s"] = panel["year"].isin([1990, 1995, 2000]).astype(int)
    panel["crisis_2008"] = panel["year"].isin([2005, 2010]).astype(int)
    panel["crisis_euro"] = panel["year"].isin([2010, 2015]).astype(int)

    outcome = "efw_delta5"
    panel = panel.dropna(subset=[outcome])

    model = smf.ols(
        f"{outcome} ~ EU_neg_share + WTO_neg_share + crisis_1990s + crisis_2008 + crisis_euro \
         + EU_neg_share:crisis_1990s + EU_neg_share:crisis_2008 + EU_neg_share:crisis_euro \
         + WTO_neg_share:crisis_1990s + WTO_neg_share:crisis_2008 + WTO_neg_share:crisis_euro \
         + C(iso3) + C(year)",
        data=panel,
    ).fit(cov_type="cluster", cov_kwds={"groups": panel["iso3"]})

    params = model.params.reset_index()
    params.columns = ["term", "estimate"]
    params["std_error"] = model.bse.values

    out_path = TABLES_DIR / "episode_interactions.csv"
    params.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    run_episode_robustness()
