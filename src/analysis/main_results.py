from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DIAGNOSTICS_DIR, PROCESSED_DIR
from src.estimators.wrappers import run_estimators
from src.viz.figures import generate_all_figures
from src.viz.tables import table_main_effects

OECD_ISO3 = {
    "AUS",
    "AUT",
    "BEL",
    "CAN",
    "CHE",
    "CHL",
    "COL",
    "CRI",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GBR",
    "GRC",
    "HUN",
    "IRL",
    "ISL",
    "ISR",
    "ITA",
    "JPN",
    "KOR",
    "LTU",
    "LUX",
    "LVA",
    "MEX",
    "NLD",
    "NOR",
    "NZL",
    "POL",
    "PRT",
    "SVK",
    "SVN",
    "SWE",
    "TUR",
    "USA",
}


def _load_diag(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def _load_latest_estimates(prefix: str, tag: str) -> pd.DataFrame:
    files = sorted((Path("outputs") / "tables").glob(f"{prefix}_{tag}_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def _sign(value: float) -> int:
    if pd.isna(value):
        return 0
    return 1 if value > 0 else -1 if value < 0 else 0


def _check_failures(tag: str) -> list[str]:
    failures = []

    did_diag = _load_diag(DIAGNOSTICS_DIR / f"did_wto_diag_{tag}.csv")
    sdid_diag = _load_diag(DIAGNOSTICS_DIR / f"sdid_eu_diag_{tag}.csv")

    if not did_diag.empty:
        core = did_diag[did_diag["outcome"].isin(["efw_overall", "efw_delta5"])]
        if not core.empty and (core["pretrend_p"] < 0.10).any():
            failures.append("WTO pretrend p-value < 0.10 for core outcomes")

    if not sdid_diag.empty:
        core = sdid_diag[sdid_diag["outcome"].isin(["efw_overall", "efw_delta5"])]
        if not core.empty and (core["rmspe_ratio"] > 2).any():
            failures.append("EU SDID pre-fit RMSPE ratio > 2")

    eu = _load_latest_estimates("estimates_eu", tag)
    if not eu.empty:
        sdid = eu[(eu["method"] == "sdid") & (eu["outcome"] == "efw_overall") & (eu["event_time"] == 0)]
        augs = eu[(eu["method"] == "augsynth") & (eu["outcome"] == "efw_overall")]
        gsc = eu[(eu["method"] == "gsc") & (eu["outcome"] == "efw_overall")]
        for alt in [augs, gsc]:
            if not sdid.empty and not alt.empty:
                if _sign(sdid["att"].iloc[0]) != _sign(alt["att"].iloc[0]):
                    failures.append("EU estimator sign flip between SDID and robustness")
                    break

    return failures


def _run_plan(panel_path: Path, tag: str) -> None:
    run_estimators(panel_path, tag=tag)
    generate_all_figures()
    table_main_effects()


def _apply_plan_b(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    if "EU_cand_share" in panel.columns:
        panel["EU_neg_share"] = panel["EU_cand_share"]
    if "WTO_post_share" in panel.columns:
        panel["WTO_neg_share"] = panel["WTO_post_share"]

    eu_treated = panel[panel["EU_neg_share"] > 0]["iso3"].unique().tolist()
    donors = list(OECD_ISO3)
    panel = panel[panel["iso3"].isin(set(eu_treated) | set(donors))]
    return panel


def run_main() -> dict[str, str]:
    panel_path = PROCESSED_DIR / "panel_quinquennial.parquet"
    _run_plan(panel_path, tag="baseline")

    failures = _check_failures(tag="baseline")
    plan_used = "A"

    if failures:
        panel = pd.read_parquet(panel_path)
        panel_b = _apply_plan_b(panel)
        panel_b_path = PROCESSED_DIR / "panel_quinquennial_planb.parquet"
        panel_b.to_parquet(panel_b_path, index=False)
        _run_plan(panel_b_path, tag="planb")
        plan_used = "B"

        failures_b = _check_failures(tag="planb")
        if failures_b:
            plan_used = "C"

    diagnostics = {
        "plan_used": plan_used,
        "failures": failures,
    }
    diag_path = DIAGNOSTICS_DIR / "plan_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2))

    return diagnostics


if __name__ == "__main__":
    run_main()
