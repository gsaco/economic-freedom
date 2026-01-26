from __future__ import annotations

from pathlib import Path

import pytest

from src.spec_search.runner import run_spec
from src.spec_search.specs import Spec, spec_from_dict, spec_id


def _minimal_spec() -> Spec:
    return spec_from_dict(
        {
            "name": "test_minimal",
            "sample": {
                "name": "positive",
                "running_var": "running_var_vote",
                "pooled": False,
            },
            "efw": {
                "mode": "prepost",
                "efw_col": "efw_summary",
                "pre_window": [-2, -1],
                "post_window": [1, 2],
                "min_pre_obs": 1,
                "min_post_obs": 1,
                "standardize": False,
                "winsorize": None,
            },
            "outcome": {
                "name": "log_gdp_cum",
                "base_col": "log_gdp_pc_const",
                "transform": "log_cum",
                "horizons": [0],
                "min_avg_obs": None,
            },
            "model": {
                "model_type": "lpiv",
                "controls": ["lag1_log_gdp_pc_const"],
                "window": 0.01,
                "windows_grid": [0.01],
                "select_window_by_balance": False,
                "cutoff": 0.0,
                "cluster": "iso3c",
                "year_fe": False,
                "year_col": "election_year",
                "kernel": "triangular",
                "order": 1,
            },
            "diagnostics": {
                "balance_vars": ["lag1_log_gdp_pc_const"],
                "balance_p": 0.15,
                "density_bw": 0.1,
                "pretrend_p": 0.1,
                "fstat_min": 5.0,
                "min_n": 5,
            },
        }
    )


def test_spec_id_deterministic():
    spec = _minimal_spec()
    first = spec_id(spec)
    second = spec_id(spec)
    assert first == second


def test_run_spec_smoke(tmp_path: Path):
    panel_path = Path("data/03_clean/panel_annual_atlas.parquet")
    sample_path = Path("data/04_analysis/close_elections_vote_margin.parquet")
    if not panel_path.exists() or not sample_path.exists():
        pytest.skip("Required data files missing")

    spec = _minimal_spec()
    result = run_spec(
        spec,
        output_root=tmp_path,
        panel_path=panel_path,
        sample_path=sample_path,
        allow_overwrite=True,
    )
    assert (result["spec_dir"] / "manifest.json").exists()
    assert (result["spec_dir"] / "tables" / "iv.csv").exists()
