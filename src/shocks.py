from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.qc import assert_unique_key


@dataclass
class ShockSpec:
    pre_window: tuple[int, int] = (-3, -1)
    post_window: tuple[int, int] = (1, 3)
    min_pre_obs: int = 2
    min_post_obs: int = 2


def _get_value(panel_index: pd.DataFrame, iso3c: str, year: int, column: str) -> float:
    try:
        return panel_index.at[(iso3c, year), column]
    except KeyError:
        return np.nan


def _window_values(panel_index: pd.DataFrame, iso3c: str, years: list[int], column: str) -> list[float]:
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
    gdp_growth_col: str = "gdp_growth_ann_pct",
    horizons: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    pretrend_horizons: tuple[int, ...] = (1, 3),
    lagged_cols: tuple[str, ...] = (
        "efw_summary",
        "log_gdp_pc_const",
        "inv_share_gdp",
        "inflation_cpi_ann_pct",
        "trade_open_gdp",
        "gov_cons_gdp",
        "pop_total",
    ),
    shock_spec: ShockSpec | None = None,
) -> pd.DataFrame:
    panel = panel.copy()
    if log_gdp_col not in panel.columns and gdp_col in panel.columns:
        panel[log_gdp_col] = np.log(panel[gdp_col].where(panel[gdp_col] > 0))

    assert_unique_key(panel, [iso_col, year_col])
    panel_index = panel.set_index([iso_col, year_col])
    shock_spec = shock_spec or ShockSpec()

    rows: list[dict] = []
    for _, event in events.iterrows():
        iso3c = event[iso_col]
        year = int(event[event_year_col])
        row: dict[str, float | int | str] = {}

        pre_years = list(range(year + shock_spec.pre_window[0], year + shock_spec.pre_window[1] + 1))
        post_years = list(range(year + shock_spec.post_window[0], year + shock_spec.post_window[1] + 1))
        pre_vals = _window_values(panel_index, iso3c, pre_years, efw_col)
        post_vals = _window_values(panel_index, iso3c, post_years, efw_col)

        pre_valid = [val for val in pre_vals if pd.notna(val)]
        post_valid = [val for val in post_vals if pd.notna(val)]
        row["efw_pre_3y"] = np.nan
        row["efw_post_1_3"] = np.nan
        if len(pre_valid) >= shock_spec.min_pre_obs:
            row["efw_pre_3y"] = float(np.nanmean(pre_vals))
        if len(post_valid) >= shock_spec.min_post_obs:
            row["efw_post_1_3"] = float(np.nanmean(post_vals))
        if pd.notna(row["efw_pre_3y"]) and pd.notna(row["efw_post_1_3"]):
            row["shock_efw"] = float(row["efw_post_1_3"] - row["efw_pre_3y"])
        else:
            row["shock_efw"] = np.nan

        row["log_gdp_tminus1"] = _get_value(panel_index, iso3c, year - 1, log_gdp_col)
        row["efw_tminus1"] = _get_value(panel_index, iso3c, year - 1, efw_col)

        for horizon in horizons:
            gdp_future = _get_value(panel_index, iso3c, year + horizon, log_gdp_col)
            if pd.notna(gdp_future) and pd.notna(row["log_gdp_tminus1"]):
                row[f"log_gdp_cum_h{horizon}"] = gdp_future - row["log_gdp_tminus1"]
            else:
                row[f"log_gdp_cum_h{horizon}"] = np.nan

            efw_future = _get_value(panel_index, iso3c, year + horizon, efw_col)
            if pd.notna(efw_future) and pd.notna(row["efw_tminus1"]):
                row[f"efw_path_h{horizon}"] = efw_future - row["efw_tminus1"]
            else:
                row[f"efw_path_h{horizon}"] = np.nan

            if gdp_growth_col in panel.columns:
                growth_years = list(range(year + 1, year + horizon + 1))
                growth_vals = _window_values(panel_index, iso3c, growth_years, gdp_growth_col)
                growth_valid = [val for val in growth_vals if pd.notna(val)]
                if len(growth_valid) >= max(1, horizon // 2):
                    row[f"gdp_growth_avg_h{horizon}"] = float(np.nanmean(growth_vals))
                else:
                    row[f"gdp_growth_avg_h{horizon}"] = np.nan

        for horizon in pretrend_horizons:
            gdp_prev = _get_value(panel_index, iso3c, year - 1, log_gdp_col)
            gdp_pre = _get_value(panel_index, iso3c, year - 1 - horizon, log_gdp_col)
            if pd.notna(gdp_prev) and pd.notna(gdp_pre):
                row[f"pretrend_h{horizon}"] = gdp_prev - gdp_pre
            else:
                row[f"pretrend_h{horizon}"] = np.nan

        for col in lagged_cols:
            lag_col = col
            if col not in panel.columns and col != log_gdp_col:
                row[f"lag1_{col}"] = np.nan
                continue
            row[f"lag1_{col}"] = _get_value(panel_index, iso3c, year - 1, lag_col)

        rows.append(row)

    event_metrics = pd.DataFrame(rows, index=events.index)
    return pd.concat([events.reset_index(drop=True), event_metrics.reset_index(drop=True)], axis=1)
