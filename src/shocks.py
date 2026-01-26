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
    efw_extra_cols: tuple[str, ...] | None = None,
    gdp_col: str = "gdp_pc_const",
    log_gdp_col: str = "log_gdp_pc_const",
    gdp_growth_col: str = "gdp_growth_ann_pct",
    inv_col: str = "inv_share_gdp",
    inflation_col: str = "inflation_cpi_ann_pct",
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
    inflation_thresholds: tuple[float, ...] = (20.0, 40.0),
    crisis_years: pd.DataFrame | None = None,
    shock_spec: ShockSpec | None = None,
) -> pd.DataFrame:
    panel = panel.copy()
    if log_gdp_col not in panel.columns and gdp_col in panel.columns:
        panel[log_gdp_col] = np.log(panel[gdp_col].where(panel[gdp_col] > 0))

    assert_unique_key(panel, [iso_col, year_col])
    panel_index = panel.set_index([iso_col, year_col])
    shock_spec = shock_spec or ShockSpec()

    rows: list[dict] = []
    crisis_lookup: dict[str, set[int]] = {}
    if crisis_years is not None and not crisis_years.empty:
        for iso3c, group in crisis_years.groupby(iso_col, dropna=True):
            crisis_lookup[str(iso3c)] = set(int(year) for year in group[year_col].dropna().astype(int))

    efw_cols = [efw_col] + list(efw_extra_cols or [])

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

            for col in efw_cols:
                if col not in panel.columns:
                    row[f"{col}_path_h{horizon}"] = np.nan
                    if col == efw_col:
                        row[f"efw_path_h{horizon}"] = np.nan
                    continue
                efw_future = _get_value(panel_index, iso3c, year + horizon, col)
                efw_base = _get_value(panel_index, iso3c, year - 1, col)
                if pd.notna(efw_future) and pd.notna(efw_base):
                    delta = efw_future - efw_base
                    row[f"{col}_path_h{horizon}"] = delta
                    if col == efw_col:
                        row[f"efw_path_h{horizon}"] = delta
                else:
                    row[f"{col}_path_h{horizon}"] = np.nan
                    if col == efw_col:
                        row[f"efw_path_h{horizon}"] = np.nan

            if gdp_growth_col in panel.columns:
                growth_years = list(range(year + 1, year + horizon + 1))
                growth_vals = _window_values(panel_index, iso3c, growth_years, gdp_growth_col)
                growth_valid = [val for val in growth_vals if pd.notna(val)]
                if len(growth_valid) >= max(1, horizon // 2):
                    row[f"gdp_growth_avg_h{horizon}"] = float(np.nanmean(growth_vals))
                else:
                    row[f"gdp_growth_avg_h{horizon}"] = np.nan

                tail_years = list(range(year, year + horizon + 1))
                tail_vals = _window_values(panel_index, iso3c, tail_years, gdp_growth_col)
                tail_valid = [val for val in tail_vals if pd.notna(val)]
                row[f"worst_growth_h{horizon}"] = float(np.nanmin(tail_valid)) if tail_valid else np.nan

            if inflation_col in panel.columns:
                infl_future = _get_value(panel_index, iso3c, year + horizon, inflation_col)
                infl_base = _get_value(panel_index, iso3c, year - 1, inflation_col)
                if pd.notna(infl_future) and pd.notna(infl_base):
                    row[f"inflation_path_h{horizon}"] = infl_future - infl_base
                else:
                    row[f"inflation_path_h{horizon}"] = np.nan

                infl_years = list(range(year, year + horizon + 1))
                infl_vals = _window_values(panel_index, iso3c, infl_years, inflation_col)
                infl_valid = [val for val in infl_vals if pd.notna(val)]
                for threshold in inflation_thresholds:
                    spike_col = f"infl_spike_{int(threshold)}_h{horizon}"
                    if infl_valid:
                        row[spike_col] = float(np.nanmax(infl_vals) >= threshold)
                    else:
                        row[spike_col] = np.nan

            if inv_col in panel.columns:
                inv_years = list(range(year + 1, year + horizon + 1))
                inv_vals = _window_values(panel_index, iso3c, inv_years, inv_col)
                inv_valid = [val for val in inv_vals if pd.notna(val)]
                if len(inv_valid) >= max(1, horizon // 2):
                    row[f"inv_share_avg_h{horizon}"] = float(np.nanmean(inv_vals))
                else:
                    row[f"inv_share_avg_h{horizon}"] = np.nan

            if log_gdp_col in panel.columns:
                dd_years = list(range(year, year + horizon + 1))
                dd_vals = _window_values(panel_index, iso3c, dd_years, log_gdp_col)
                dd_valid = [val for val in dd_vals if pd.notna(val)]
                if len(dd_valid) >= 2:
                    peak = -np.inf
                    max_drawdown = 0.0
                    for value in dd_valid:
                        peak = max(peak, value)
                        max_drawdown = max(max_drawdown, peak - value)
                    row[f"max_drawdown_h{horizon}"] = float(max_drawdown)
                else:
                    row[f"max_drawdown_h{horizon}"] = np.nan

            if crisis_lookup:
                crisis_years_iso = crisis_lookup.get(str(iso3c), set())
                if crisis_years_iso:
                    row[f"crisis_start_h{horizon}"] = float(
                        any((year + offset) in crisis_years_iso for offset in range(0, horizon + 1))
                    )
                else:
                    row[f"crisis_start_h{horizon}"] = np.nan

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
