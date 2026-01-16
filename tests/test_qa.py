from pathlib import Path

import pandas as pd
import pytest

from src.build.qa import (
    check_baseline_no_leakage,
    check_delta5,
    check_exposure_bounds,
    check_quinquennial_years,
    check_unique_key,
)


def _load_panel():
    path = "data/processed/panel_quinquennial.parquet"
    if not Path(path).exists():
        pytest.skip("Panel data not built")
    return pd.read_parquet(path)


def test_quinquennial_years_only():
    df = _load_panel()
    check_quinquennial_years(df)


def test_unique_iso3_year_key():
    df = _load_panel()
    check_unique_key(df, ["iso3", "year"])


def test_exposure_shares_in_unit_interval():
    df = _load_panel()
    check_exposure_bounds(df, ["EU_neg_share", "EU_post_share", "WTO_neg_share", "WTO_post_share"])


def test_delta5_computation():
    df = _load_panel()
    if "efw_delta5" in df.columns:
        check_delta5(df, "efw_overall", "efw_delta5")


def test_reform_reversal_indicators_thresholds():
    df = _load_panel()
    for delta, label in [(0.3, "03"), (0.5, "05")]:
        reform = (df["efw_delta5"] >= delta).astype(int)
        reversal = (df["efw_delta5"] <= -delta).astype(int)
        assert reform.equals(df[f"reform_event_{label}"])
        assert reversal.equals(df[f"reversal_event_{label}"])


def test_baseline_state_no_leakage():
    df = _load_panel()
    check_baseline_no_leakage(df, "baseline_efw_eu", "EU_neg_share")
