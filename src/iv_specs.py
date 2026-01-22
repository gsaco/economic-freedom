from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.qc import assert_unique_key


@dataclass
class HorizonSpec:
    horizon: int
    gdp_growth_col: str
    inv_avg_col: str


def _get_value(panel_index: pd.DataFrame, iso3c: str, year: int, column: str) -> float:
    try:
        return panel_index.at[(iso3c, year), column]
    except KeyError:
        return np.nan


def _window_values(
    panel_index: pd.DataFrame,
    iso3c: str,
    years: list[int],
    column: str,
) -> list[float]:
    return [_get_value(panel_index, iso3c, year, column) for year in years]


def build_event_panel(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    iso_col: str = "iso3c",
    year_col: str = "year",
    event_year_col: str = "election_year",
    efw_col: str = "efw_summary",
    gdp_col: str = "gdp_pc_const",
    log_gdp_col: str = "log_gdp_pc_const",
    inv_col: str = "inv_share_gdp",
    efw_post_window: tuple[int, int] = (1, 3),
    horizons: tuple[int, ...] = (3, 5),
    lagged_cols: tuple[str, ...] = (
        "efw_summary",
        "log_gdp_pc_const",
        "inv_share_gdp",
        "inflation_cpi_ann_pct",
        "trade_open_gdp",
        "gov_cons_gdp",
        "pop_total",
    ),
    min_post_obs: int = 2,
) -> pd.DataFrame:
    panel = panel.copy()
    if log_gdp_col not in panel.columns and gdp_col in panel.columns:
        panel[log_gdp_col] = np.log(panel[gdp_col].where(panel[gdp_col] > 0))

    assert_unique_key(panel, [iso_col, year_col])
    panel_index = panel.set_index([iso_col, year_col])

    rows: list[dict] = []
    for _, event in events.iterrows():
        iso3c = event[iso_col]
        year = int(event[event_year_col])
        row: dict[str, float | int | str] = {}
        row["efw_pre"] = _get_value(panel_index, iso3c, year - 1, efw_col)

        post_years = list(range(year + efw_post_window[0], year + efw_post_window[1] + 1))
        post_vals = _window_values(panel_index, iso3c, post_years, efw_col)
        valid_post = [val for val in post_vals if pd.notna(val)]
        row["efw_post_1_3"] = np.nan
        if len(valid_post) >= min_post_obs:
            row["efw_post_1_3"] = float(np.nanmean(post_vals))

        row["log_gdp_tminus1"] = _get_value(panel_index, iso3c, year - 1, log_gdp_col)

        for horizon in horizons:
            gdp_future = _get_value(panel_index, iso3c, year + horizon, log_gdp_col)
            if pd.notna(gdp_future) and pd.notna(row["log_gdp_tminus1"]):
                row[f"gdp_growth_h{horizon}"] = gdp_future - row["log_gdp_tminus1"]
            else:
                row[f"gdp_growth_h{horizon}"] = np.nan

            inv_years = list(range(year + 1, year + horizon + 1))
            inv_vals = _window_values(panel_index, iso3c, inv_years, inv_col)
            inv_valid = [val for val in inv_vals if pd.notna(val)]
            if len(inv_valid) >= max(2, horizon // 2):
                row[f"inv_share_avg_h{horizon}"] = float(np.nanmean(inv_vals))
            else:
                row[f"inv_share_avg_h{horizon}"] = np.nan

        for col in lagged_cols:
            if col not in panel.columns and col != log_gdp_col:
                row[f"lag1_{col}"] = np.nan
                continue
            lag_col = col
            row[f"lag1_{col}"] = _get_value(panel_index, iso3c, year - 1, lag_col)

        rows.append(row)

    event_metrics = pd.DataFrame(rows, index=events.index)
    return pd.concat([events.reset_index(drop=True), event_metrics.reset_index(drop=True)], axis=1)

