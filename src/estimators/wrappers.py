from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DIAGNOSTICS_DIR, INTERIM_DIR, OUTPUTS_DIR, TABLES_DIR

R_SCRIPTS = {
    "sdid": Path(__file__).resolve().parent / "run_sdid.R",
    "did_cs": Path(__file__).resolve().parent / "run_did_cs.R",
    "augsynth": Path(__file__).resolve().parent / "run_augsynth.R",
    "gsc": Path(__file__).resolve().parent / "run_gsc.R",
}


def _run_r(script: Path, args: list[str]) -> None:
    cmd = ["Rscript", str(script)] + args
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"R script failed: {script}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def run_estimators(
    panel_path: Path,
    tag: str = "baseline",
    methods: list[str] | None = None,
    outcomes: list[str] | None = None,
    sdid_cohorts: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    timestamp = datetime.now().strftime("%Y%m%d")
    panel_csv = INTERIM_DIR / "panel_quinquennial.csv"
    panel = pd.read_parquet(panel_path)
    panel.to_csv(panel_csv, index=False)

    outputs: dict[str, pd.DataFrame] = {}

    if methods is None:
        methods = ["sdid", "did_cs", "augsynth"]
    outcome_arg = ",".join(outcomes) if outcomes else ""
    cohort_arg = ",".join(sdid_cohorts) if sdid_cohorts else ""

    sdid_out = TABLES_DIR / f"sdid_eu_{tag}.csv"
    sdid_diag = DIAGNOSTICS_DIR / f"sdid_eu_diag_{tag}.csv"
    if "sdid" in methods:
        _run_r(R_SCRIPTS["sdid"], [str(panel_csv), str(sdid_out), str(sdid_diag), outcome_arg, cohort_arg])

    did_out = TABLES_DIR / f"did_wto_{tag}.csv"
    did_diag = DIAGNOSTICS_DIR / f"did_wto_diag_{tag}.csv"
    if "did_cs" in methods:
        _run_r(R_SCRIPTS["did_cs"], [str(panel_csv), str(did_out), str(did_diag), outcome_arg])

    augs_out = TABLES_DIR / f"augsynth_eu_{tag}.csv"
    if "augsynth" in methods:
        _run_r(R_SCRIPTS["augsynth"], [str(panel_csv), str(augs_out), outcome_arg])

    gsc_out = TABLES_DIR / f"gsc_eu_{tag}.csv"
    if "gsc" in methods:
        _run_r(R_SCRIPTS["gsc"], [str(panel_csv), str(gsc_out)])

    sdid_df = pd.DataFrame()
    if "sdid" in methods and sdid_out.exists() and sdid_out.stat().st_size > 0:
        sdid_df = pd.read_csv(sdid_out)
        sdid_df["method"] = "sdid"
        outputs["eu_sdid"] = sdid_df

    augs_df = pd.DataFrame()
    if "augsynth" in methods and augs_out.exists() and augs_out.stat().st_size > 0:
        augs_df = pd.read_csv(augs_out)
        augs_df["method"] = "augsynth"
        outputs["eu_augsynth"] = augs_df

    gsc_df = pd.DataFrame()
    if "gsc" in methods and gsc_out.exists() and gsc_out.stat().st_size > 0:
        gsc_df = pd.read_csv(gsc_out)
        gsc_df["method"] = "gsc"
        outputs["eu_gsc"] = gsc_df

    did_df = pd.DataFrame()
    if "did_cs" in methods and did_out.exists() and did_out.stat().st_size > 0:
        did_df = pd.read_csv(did_out)
        did_df["method"] = "cs_did"
        outputs["wto_did"] = did_df

    eu_frames = [df for df in [sdid_df, augs_df, gsc_df] if not df.empty]
    eu_combined = pd.concat(eu_frames, ignore_index=True) if eu_frames else pd.DataFrame()
    wto_combined = did_df

    if not eu_combined.empty:
        eu_path = TABLES_DIR / f"estimates_eu_{tag}_{timestamp}.parquet"
        eu_combined.to_parquet(eu_path, index=False)

    if not wto_combined.empty:
        wto_path = TABLES_DIR / f"estimates_wto_{tag}_{timestamp}.parquet"
        wto_combined.to_parquet(wto_path, index=False)

    return outputs
