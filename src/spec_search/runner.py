from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.lpiv import LPIVResult, first_stage, iv_estimate, reduced_form
from src.paths import ANALYSIS_DIR, CLEAN_DIR
from src.rd import RDEstimate, density_discontinuity, rd_estimate, rd_iv, select_bandwidth
from src.rd_localrand import balance_table, select_window_by_balance
from src.shocks import ShockSpec, build_event_panel
from src.spec_search.specs import Spec, flatten_dict, spec_id

DEFAULT_PRETRENDS = (1, 3)


class SpecRunError(RuntimeError):
    pass


def _git_sha(root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _data_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_quantile(series: pd.Series, q: float) -> float:
    series = series.dropna()
    if series.empty:
        return float("nan")
    return float(series.quantile(q))


def _winsorize(series: pd.Series, limits: tuple[float, float]) -> tuple[pd.Series, dict[str, float]]:
    low_q, high_q = limits
    lower = _safe_quantile(series, low_q)
    upper = _safe_quantile(series, high_q)
    clipped = series.clip(lower=lower, upper=upper)
    return clipped, {"lower": lower, "upper": upper}


def _infer_instrument(frame: pd.DataFrame, sample_name: str, running_var: str) -> tuple[pd.DataFrame, str]:
    frame = frame.copy()
    name_lower = sample_name.lower()
    if "positive" in name_lower:
        if "z_pos" not in frame.columns:
            frame["z_pos"] = (frame[running_var] > 0).astype(int)
        return frame, "z_pos"
    if "negative" in name_lower:
        if "z_neg" not in frame.columns:
            frame["z_neg"] = (frame[running_var] < 0).astype(int)
        return frame, "z_neg"
    if "pooled" in name_lower:
        if "z_pool" not in frame.columns:
            frame["z_pool"] = (frame[running_var] > 0).astype(int)
        return frame, "z_pool"
    if "z_pool" not in frame.columns:
        frame["z_pool"] = (frame[running_var] > 0).astype(int)
    return frame, "z_pool"


def _apply_sample_spec(sample: pd.DataFrame, spec: Spec) -> tuple[pd.DataFrame, str]:
    frame = sample.copy()
    sample_name = spec.sample.name
    if spec.sample.filter_query:
        frame = frame.query(spec.sample.filter_query).copy()

    if not spec.sample.pooled:
        if "positive" in sample_name.lower():
            frame = frame.loc[frame["incumbent_market"] == 0].copy()
        elif "negative" in sample_name.lower():
            frame = frame.loc[frame["incumbent_market"] == 1].copy()

    running_var = spec.sample.running_var
    if running_var not in frame.columns:
        raise SpecRunError(f"Running variable '{running_var}' not found in sample.")

    if spec.sample.instrument:
        instrument = spec.sample.instrument
        if instrument not in frame.columns:
            # Create instrument based on running var sign if missing
            frame[instrument] = (frame[running_var] > 0).astype(int)
        return frame, instrument

    frame, instrument = _infer_instrument(frame, sample_name, running_var)
    return frame, instrument


def _build_lagged_cols(controls: tuple[str, ...]) -> tuple[str, ...]:
    base = {
        "efw_summary",
        "log_gdp_pc_const",
        "inv_share_gdp",
        "inflation_cpi_ann_pct",
        "trade_open_gdp",
        "gov_cons_gdp",
        "pop_total",
    }
    for control in controls:
        if control.startswith("lag1_"):
            base.add(control.replace("lag1_", ""))
    return tuple(sorted(base))


def _compute_outcome_columns(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    outcome_name: str,
    base_col: str,
    transform: str,
    horizons: tuple[int, ...],
    min_avg_obs: int | None,
    event_year_col: str,
) -> pd.DataFrame:
    if transform in {"log_cum", "growth_avg"}:
        return events

    panel_index = panel.set_index(["iso3c", "year"])
    rows = []
    for _, event in events.iterrows():
        iso3c = event["iso3c"]
        year = int(event[event_year_col])
        row: dict[str, float] = {}
        for h in horizons:
            col_name = f"{outcome_name}_h{h}"
            if transform == "level":
                try:
                    val = panel_index.at[(iso3c, year + h), base_col]
                except KeyError:
                    val = np.nan
            elif transform == "change":
                try:
                    end_val = panel_index.at[(iso3c, year + h), base_col]
                    base_val = panel_index.at[(iso3c, year - 1), base_col]
                    val = end_val - base_val
                except KeyError:
                    val = np.nan
            elif transform == "avg":
                years = list(range(year + 1, year + h + 1))
                vals = []
                for yr in years:
                    try:
                        vals.append(panel_index.at[(iso3c, yr), base_col])
                    except KeyError:
                        vals.append(np.nan)
                valid = [v for v in vals if pd.notna(v)]
                min_obs = min_avg_obs or max(1, h // 2)
                if len(valid) >= min_obs:
                    val = float(np.nanmean(vals))
                else:
                    val = np.nan
            else:
                val = np.nan
            row[col_name] = val
        rows.append(row)
    extra = pd.DataFrame(rows, index=events.index)
    return pd.concat([events.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


def _add_shock_variants(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    efw_col: str,
    mode: str,
    standardize: bool,
    winsorize: tuple[float, float] | None,
    event_year_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel_index = panel.set_index(["iso3c", "year"])
    shocks = []
    for _, event in events.iterrows():
        iso3c = event["iso3c"]
        year = int(event[event_year_col])
        try:
            efw_t = panel_index.at[(iso3c, year), efw_col]
        except KeyError:
            efw_t = np.nan
        shocks.append(efw_t)
    events = events.copy()
    events["efw_t"] = shocks

    if mode == "delta":
        shock_raw = events["efw_t"] - events["efw_tminus1"]
    elif mode == "level":
        shock_raw = events["efw_tminus1"].copy()
    else:
        shock_raw = events["shock_efw"].copy()

    metadata: dict[str, Any] = {"winsor": None, "standardized": False}
    if winsorize:
        shock_raw, winsor_meta = _winsorize(shock_raw, winsorize)
        metadata["winsor"] = winsor_meta

    shock_final = shock_raw.copy()
    if standardize:
        mean = _coerce_float(shock_raw.mean())
        std = _coerce_float(shock_raw.std())
        if std and np.isfinite(std) and std > 0:
            shock_final = (shock_raw - mean) / std
            metadata["standardized"] = True
            metadata["standardize_mean"] = mean
            metadata["standardize_std"] = std
        else:
            shock_final = shock_raw * np.nan
            metadata["standardized"] = False
            metadata["standardize_mean"] = mean
            metadata["standardize_std"] = std

    events["shock_main"] = shock_final
    events["shock_raw"] = shock_raw
    return events, metadata


def _add_splits(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["shock_abs"] = events["shock_main"].abs()

    for label, q in [("shock_abs_tercile", 3), ("shock_abs_quartile", 4)]:
        try:
            events[label] = pd.qcut(events["shock_abs"], q=q, labels=False, duplicates="drop")
        except ValueError:
            events[label] = np.nan

    if "lr_distance_abs" in events.columns:
        median_lr = events["lr_distance_abs"].median()
        events["large_shift"] = (events["lr_distance_abs"] >= median_lr).astype(int)
    else:
        events["large_shift"] = np.nan

    if "lag1_efw_summary" in events.columns:
        median_efw = events["lag1_efw_summary"].median()
        events["low_efw"] = (events["lag1_efw_summary"] <= median_efw).astype(int)
    else:
        events["low_efw"] = np.nan

    if "lag1_log_gdp_pc_const" in events.columns:
        median_income = events["lag1_log_gdp_pc_const"].median()
        events["low_income"] = (events["lag1_log_gdp_pc_const"] <= median_income).astype(int)
    else:
        events["low_income"] = np.nan

    return events


def _select_window(
    frame: pd.DataFrame,
    running_var: str,
    diagnostics: dict[str, Any],
    *,
    balance_vars: tuple[str, ...],
    windows_grid: tuple[float, ...],
    balance_p: float,
    cluster: str | None,
    cutoff: float,
    fallback: float | None,
) -> tuple[float | None, pd.DataFrame]:
    try:
        choice, table = select_window_by_balance(
            frame,
            running_var,
            list(balance_vars),
            windows=list(windows_grid),
            p_threshold=balance_p,
            cluster=cluster,
            cutoff=cutoff,
        )
        diagnostics["balance_window_choice"] = asdict(choice)
        window = choice.window if choice.window is not None else fallback
        return window, table
    except Exception as exc:  # noqa: BLE001
        diagnostics["balance_window_choice_error"] = str(exc)
        return fallback, pd.DataFrame()


def _outcome_column(spec: Spec, horizon: int) -> str:
    if spec.outcome.transform == "log_cum":
        return f"log_gdp_cum_h{horizon}"
    if spec.outcome.transform == "growth_avg":
        return f"gdp_growth_avg_h{horizon}"
    return f"{spec.outcome.name}_h{horizon}"


def _filter_controls(frame: pd.DataFrame, controls: tuple[str, ...]) -> list[str]:
    return [control for control in controls if control in frame.columns]


def _empty_lpiv(outcome: str, window: float | None) -> LPIVResult:
    return LPIVResult(
        outcome=outcome,
        coef=float("nan"),
        se=float("nan"),
        pvalue=float("nan"),
        n_obs=0,
        n_left=0,
        n_right=0,
        window=float(window) if window is not None else float("nan"),
    )


def _empty_rd(outcome: str, bandwidth: float | None, order: int, cutoff: float) -> RDEstimate:
    return RDEstimate(
        outcome=outcome,
        coef=float("nan"),
        se=float("nan"),
        pvalue=float("nan"),
        n_obs=0,
        n_left=0,
        n_right=0,
        bandwidth=float(bandwidth) if bandwidth is not None else float("nan"),
        order=order,
        cutoff=cutoff,
        method="wls",
    )


def _compute_diagnostics(
    frame: pd.DataFrame,
    *,
    running_var: str,
    instrument_col: str,
    spec: Spec,
    window: float | None,
    cutoff: float,
    outcome_col: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    diagnostic_errors: list[str] = []
    density_bw = spec.diagnostics.density_bw
    if window and window > 0:
        density_bw = min(density_bw, window)
    try:
        density = density_discontinuity(frame[running_var], bandwidth=density_bw, cutoff=cutoff)
        density_pass = bool(np.isfinite(density["z_stat"]) and abs(density["z_stat"]) < 1.96)
    except Exception as exc:  # noqa: BLE001
        diagnostic_errors.append(f"density: {exc}")
        density = {
            "left_n": 0,
            "right_n": 0,
            "log_diff": float("nan"),
            "se": float("nan"),
            "z_stat": float("nan"),
            "bandwidth": density_bw,
        }
        density_pass = False
    diagnostics["density"] = {**density, "pass": density_pass}

    balance_vars = [var for var in spec.diagnostics.balance_vars if var in frame.columns]
    if window is not None and balance_vars:
        try:
            table = balance_table(
                frame,
                running_var,
                balance_vars,
                window=window,
                cutoff=cutoff,
                cluster=spec.model.cluster,
            )
            if not table.empty:
                min_p = float(table["pvalue"].min())
                balance_pass = bool((table["pvalue"] >= spec.diagnostics.balance_p).all())
            else:
                min_p = float("nan")
                balance_pass = False
        except Exception as exc:  # noqa: BLE001
            diagnostic_errors.append(f"balance: {exc}")
            table = pd.DataFrame()
            min_p = float("nan")
            balance_pass = False
    else:
        table = pd.DataFrame()
        min_p = float("nan")
        balance_pass = False
    diagnostics["balance"] = {
        "min_pvalue": min_p,
        "pass": balance_pass,
        "table_rows": len(table),
    }

    pretrend_pass = None
    pretrend_stats: dict[str, Any] = {}
    if spec.model.model_type == "lpiv":
        controls = _filter_controls(frame, spec.model.controls)
        for h in DEFAULT_PRETRENDS:
            col = f"pretrend_h{h}"
            if col not in frame.columns:
                continue
            try:
                rf = reduced_form(
                    frame,
                    outcome_col=col,
                    running_col=running_var,
                    instrument_col=instrument_col,
                    controls=controls,
                    window=window or spec.model.window or 0.03,
                    cutoff=cutoff,
                    cluster=spec.model.cluster,
                    year_fe=spec.model.year_fe,
                    year_col=spec.model.year_col,
                )
                pretrend_stats[f"pretrend_h{h}"] = {
                    "coef": rf.coef,
                    "se": rf.se,
                    "pvalue": rf.pvalue,
                    "n_obs": rf.n_obs,
                }
            except Exception as exc:  # noqa: BLE001
                diagnostic_errors.append(f"pretrend_h{h}: {exc}")
                pretrend_stats[f"pretrend_h{h}"] = {
                    "coef": float("nan"),
                    "se": float("nan"),
                    "pvalue": float("nan"),
                    "n_obs": 0,
                }
        if pretrend_stats:
            pretrend_pass = all(stat["pvalue"] >= spec.diagnostics.pretrend_p for stat in pretrend_stats.values())
    elif spec.model.model_type == "rd_iv":
        for h in DEFAULT_PRETRENDS:
            col = f"pretrend_h{h}"
            if col not in frame.columns:
                continue
            try:
                est = rd_estimate(
                    frame,
                    outcome=col,
                    running=running_var,
                    cutoff=cutoff,
                    bandwidth=window,
                    kernel=spec.model.kernel,
                    order=spec.model.order,
                    cluster=spec.model.cluster,
                )
                pretrend_stats[f"pretrend_h{h}"] = {
                    "coef": est.coef,
                    "se": est.se,
                    "pvalue": est.pvalue,
                    "n_obs": est.n_obs,
                }
            except Exception as exc:  # noqa: BLE001
                diagnostic_errors.append(f"pretrend_h{h}: {exc}")
                pretrend_stats[f"pretrend_h{h}"] = {
                    "coef": float("nan"),
                    "se": float("nan"),
                    "pvalue": float("nan"),
                    "n_obs": 0,
                }
        if pretrend_stats:
            pretrend_pass = all(stat["pvalue"] >= spec.diagnostics.pretrend_p for stat in pretrend_stats.values())

    diagnostics["pretrend"] = {"pass": pretrend_pass, **pretrend_stats}

    # First-stage strength
    if spec.model.model_type == "lpiv":
        controls = _filter_controls(frame, spec.model.controls)
        try:
            fs = first_stage(
                frame,
                outcome_col="shock_main",
                running_col=running_var,
                instrument_col=instrument_col,
                controls=controls,
                window=window or spec.model.window or 0.03,
                cutoff=cutoff,
                cluster=spec.model.cluster,
                year_fe=spec.model.year_fe,
                year_col=spec.model.year_col,
            )
            t_stat = fs.coef / fs.se if fs.se and np.isfinite(fs.se) else float("nan")
            f_stat = t_stat**2 if np.isfinite(t_stat) else float("nan")
            fs_pass = bool(np.isfinite(f_stat) and f_stat >= spec.diagnostics.fstat_min)
            diagnostics["first_stage"] = {
                "coef": fs.coef,
                "se": fs.se,
                "t_stat": t_stat,
                "f_stat": f_stat,
                "pass": fs_pass,
                "n_obs": fs.n_obs,
                "n_left": fs.n_left,
                "n_right": fs.n_right,
            }
        except Exception as exc:  # noqa: BLE001
            diagnostic_errors.append(f"first_stage: {exc}")
            diagnostics["first_stage"] = {
                "coef": float("nan"),
                "se": float("nan"),
                "t_stat": float("nan"),
                "f_stat": float("nan"),
                "pass": False,
                "n_obs": 0,
                "n_left": 0,
                "n_right": 0,
            }
    else:
        try:
            fs = rd_estimate(
                frame,
                outcome="shock_main",
                running=running_var,
                cutoff=cutoff,
                bandwidth=window,
                kernel=spec.model.kernel,
                order=spec.model.order,
                cluster=spec.model.cluster,
            )
            t_stat = fs.coef / fs.se if fs.se and np.isfinite(fs.se) else float("nan")
            f_stat = t_stat**2 if np.isfinite(t_stat) else float("nan")
            fs_pass = bool(np.isfinite(f_stat) and f_stat >= spec.diagnostics.fstat_min)
            diagnostics["first_stage"] = {
                "coef": fs.coef,
                "se": fs.se,
                "t_stat": t_stat,
                "f_stat": f_stat,
                "pass": fs_pass,
                "n_obs": fs.n_obs,
                "n_left": fs.n_left,
                "n_right": fs.n_right,
            }
        except Exception as exc:  # noqa: BLE001
            diagnostic_errors.append(f"first_stage: {exc}")
            diagnostics["first_stage"] = {
                "coef": float("nan"),
                "se": float("nan"),
                "t_stat": float("nan"),
                "f_stat": float("nan"),
                "pass": False,
                "n_obs": 0,
                "n_left": 0,
                "n_right": 0,
            }

    # Sample adequacy
    if spec.model.model_type == "lpiv":
        frame_window = frame.copy()
        frame_window["m"] = frame_window[running_var] - cutoff
        if window is not None:
            frame_window = frame_window.loc[frame_window["m"].abs() <= window]
    else:
        frame_window = frame.copy()
        frame_window["m"] = frame_window[running_var] - cutoff
        if window is not None:
            frame_window = frame_window.loc[frame_window["m"].abs() <= window]

    n_left = int((frame_window["m"] < 0).sum())
    n_right = int((frame_window["m"] >= 0).sum())
    sample_pass = bool(n_left >= spec.diagnostics.min_n and n_right >= spec.diagnostics.min_n)
    diagnostics["sample"] = {"n_left": n_left, "n_right": n_right, "pass": sample_pass}

    # Keep reference outcome column for logging
    diagnostics["outcome_col"] = outcome_col
    diagnostics["errors"] = diagnostic_errors

    return diagnostics


def run_spec(
    spec: Spec,
    *,
    output_root: Path,
    panel_path: Path | None = None,
    sample_path: Path | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    panel_path = panel_path or (CLEAN_DIR / "panel_annual_atlas.parquet")
    sample_path = sample_path or (ANALYSIS_DIR / "close_elections_vote_margin.parquet")

    panel = pd.read_parquet(panel_path)
    sample = pd.read_parquet(sample_path)

    sample_frame, instrument_col = _apply_sample_spec(sample, spec)

    # Build event panel
    shock_spec = ShockSpec(
        pre_window=tuple(spec.efw.pre_window),
        post_window=tuple(spec.efw.post_window),
        min_pre_obs=spec.efw.min_pre_obs,
        min_post_obs=spec.efw.min_post_obs,
    )
    horizons = tuple(spec.outcome.horizons)
    lagged_cols = _build_lagged_cols(spec.model.controls)

    events = build_event_panel(
        panel,
        sample_frame,
        event_year_col="election_year",
        efw_col=spec.efw.efw_col,
        horizons=horizons,
        pretrend_horizons=DEFAULT_PRETRENDS,
        lagged_cols=lagged_cols,
        shock_spec=shock_spec,
    )

    # Add outcome columns for non-default transforms
    events = _compute_outcome_columns(
        panel,
        events,
        outcome_name=spec.outcome.name,
        base_col=spec.outcome.base_col,
        transform=spec.outcome.transform,
        horizons=horizons,
        min_avg_obs=spec.outcome.min_avg_obs,
        event_year_col="election_year",
    )

    # Add shock variants & splits
    events, shock_meta = _add_shock_variants(
        panel,
        events,
        efw_col=spec.efw.efw_col,
        mode=spec.efw.mode,
        standardize=spec.efw.standardize,
        winsorize=spec.efw.winsorize,
        event_year_col="election_year",
    )
    events = _add_splits(events)

    # Apply group filter if requested
    if spec.group:
        if spec.group.column not in events.columns:
            raise SpecRunError(f"Group column '{spec.group.column}' not available.")
        events = events.loc[events[spec.group.column] == spec.group.value].copy()

    # Determine window
    diagnostics: dict[str, Any] = {}
    window = spec.model.window
    balance_table_df = pd.DataFrame()
    if spec.model.select_window_by_balance:
        window, balance_table_df = _select_window(
            events,
            spec.sample.running_var,
            diagnostics,
            balance_vars=spec.diagnostics.balance_vars,
            windows_grid=spec.model.windows_grid,
            balance_p=spec.diagnostics.balance_p,
            cluster=spec.model.cluster,
            cutoff=spec.model.cutoff,
            fallback=window,
        )

    # Estimation
    fs_rows: list[dict[str, Any]] = []
    rf_rows: list[dict[str, Any]] = []
    iv_rows: list[dict[str, Any]] = []
    estimation_errors: list[str] = []

    running_var = spec.sample.running_var
    if spec.model.model_type == "rd_iv" and "negative" in spec.sample.name.lower():
        # Flip running variable so cutoff sign aligns with instrument
        events["_running_flip"] = -events[running_var]
        running_var = "_running_flip"

    for h in horizons:
        outcome_col = _outcome_column(spec, h)
        if outcome_col not in events.columns:
            continue
        if spec.model.model_type == "lpiv":
            controls = _filter_controls(events, spec.model.controls)
            try:
                fs = first_stage(
                    events,
                    outcome_col="shock_main",
                    running_col=running_var,
                    instrument_col=instrument_col,
                    controls=controls,
                    window=window or 0.03,
                    cutoff=spec.model.cutoff,
                    cluster=spec.model.cluster,
                    year_fe=spec.model.year_fe,
                    year_col=spec.model.year_col,
                )
            except Exception as exc:  # noqa: BLE001
                estimation_errors.append(f"first_stage_h{h}: {exc}")
                fs = _empty_lpiv("shock_main", window or 0.03)
            try:
                rf = reduced_form(
                    events,
                    outcome_col=outcome_col,
                    running_col=running_var,
                    instrument_col=instrument_col,
                    controls=controls,
                    window=window or 0.03,
                    cutoff=spec.model.cutoff,
                    cluster=spec.model.cluster,
                    year_fe=spec.model.year_fe,
                    year_col=spec.model.year_col,
                )
            except Exception as exc:  # noqa: BLE001
                estimation_errors.append(f"reduced_form_h{h}: {exc}")
                rf = _empty_lpiv(outcome_col, window or 0.03)
            try:
                iv = iv_estimate(
                    events,
                    outcome_col=outcome_col,
                    endog_col="shock_main",
                    running_col=running_var,
                    instrument_col=instrument_col,
                    controls=controls,
                    window=window or 0.03,
                    cutoff=spec.model.cutoff,
                    cluster=spec.model.cluster,
                    year_fe=spec.model.year_fe,
                    year_col=spec.model.year_col,
                )
            except Exception as exc:  # noqa: BLE001
                estimation_errors.append(f"iv_h{h}: {exc}")
                iv = _empty_lpiv(outcome_col, window or 0.03)
            fs_rows.append({"horizon": h, **asdict(fs)})
            rf_rows.append({"horizon": h, **asdict(rf)})
            iv_rows.append({"horizon": h, **asdict(iv)})
        else:
            bw = window
            if bw is None:
                bw = select_bandwidth(events[running_var])
            try:
                fs = rd_estimate(
                    events,
                    outcome="shock_main",
                    running=running_var,
                    cutoff=spec.model.cutoff,
                    bandwidth=bw,
                    kernel=spec.model.kernel,
                    order=spec.model.order,
                    cluster=spec.model.cluster,
                )
            except Exception as exc:  # noqa: BLE001
                estimation_errors.append(f"rd_first_stage_h{h}: {exc}")
                fs = _empty_rd("shock_main", bw, spec.model.order, spec.model.cutoff)
            try:
                rf = rd_estimate(
                    events,
                    outcome=outcome_col,
                    running=running_var,
                    cutoff=spec.model.cutoff,
                    bandwidth=bw,
                    kernel=spec.model.kernel,
                    order=spec.model.order,
                    cluster=spec.model.cluster,
                )
            except Exception as exc:  # noqa: BLE001
                estimation_errors.append(f"rd_reduced_form_h{h}: {exc}")
                rf = _empty_rd(outcome_col, bw, spec.model.order, spec.model.cutoff)
            try:
                iv = rd_iv(
                    events,
                    outcome=outcome_col,
                    running=running_var,
                    endogenous="shock_main",
                    cutoff=spec.model.cutoff,
                    bandwidth=bw,
                    kernel=spec.model.kernel,
                    order=spec.model.order,
                    cluster=spec.model.cluster,
                )
            except Exception as exc:  # noqa: BLE001
                estimation_errors.append(f"rd_iv_h{h}: {exc}")
                iv = _empty_rd(outcome_col, bw, spec.model.order, spec.model.cutoff)
            fs_rows.append({"horizon": h, **asdict(fs)})
            rf_rows.append({"horizon": h, **asdict(rf)})
            iv_rows.append({"horizon": h, **asdict(iv)})

    fs_df = pd.DataFrame(fs_rows)
    rf_df = pd.DataFrame(rf_rows)
    iv_df = pd.DataFrame(iv_rows)

    # Diagnostics
    ref_outcome_col = _outcome_column(spec, horizons[0]) if horizons else ""
    diagnostics.update(
        _compute_diagnostics(
            events,
            running_var=running_var,
            instrument_col=instrument_col,
            spec=spec,
            window=window or spec.model.window,
            cutoff=spec.model.cutoff,
            outcome_col=ref_outcome_col,
        )
    )

    # Economic significance (use horizon 3 if available else last horizon)
    horizon_main = 3 if 3 in horizons else (horizons[-1] if horizons else None)
    econ = {}
    if horizon_main is not None:
        row = iv_df.loc[iv_df["horizon"] == horizon_main]
        if not row.empty:
            coef = float(row["coef"].iloc[0])
            window_used = float(row["window"].iloc[0]) if "window" in row.columns else float("nan")
            subset = events.copy()
            subset["m"] = subset[running_var] - spec.model.cutoff
            if window_used and np.isfinite(window_used):
                subset = subset.loc[subset["m"].abs() <= window_used]
            shock_sd = float(subset["shock_main"].std()) if not subset.empty else float("nan")
            out_col = _outcome_column(spec, horizon_main)
            outcome_mean = float(subset[out_col].mean()) if out_col in subset.columns else float("nan")
            effect_sd = coef * shock_sd if np.isfinite(shock_sd) else float("nan")
            if spec.outcome.transform == "log_cum":
                effect_pct = 100 * effect_sd if np.isfinite(effect_sd) else float("nan")
            else:
                effect_pct = effect_sd
            effect_pct_of_mean = effect_sd / outcome_mean if np.isfinite(outcome_mean) and outcome_mean != 0 else float("nan")
            econ = {
                "horizon": horizon_main,
                "coef": coef,
                "shock_sd": shock_sd,
                "outcome_mean": outcome_mean,
                "effect_sd": effect_sd,
                "effect_pct": effect_pct,
                "effect_pct_of_mean": effect_pct_of_mean,
            }

    # Output handling
    spec_identifier = spec_id(spec)
    spec_dir = output_root / spec_identifier
    if spec_dir.exists() and not allow_overwrite:
        raise SpecRunError(f"Spec output already exists: {spec_dir}")
    spec_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = spec_dir / "tables"
    figures_dir = spec_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fs_path = tables_dir / "first_stage.csv"
    rf_path = tables_dir / "reduced_form.csv"
    iv_path = tables_dir / "iv.csv"
    fs_df.to_csv(fs_path, index=False)
    rf_df.to_csv(rf_path, index=False)
    iv_df.to_csv(iv_path, index=False)

    if not balance_table_df.empty:
        balance_table_df.to_csv(tables_dir / "balance_table.csv", index=False)

    spec_payload = spec.to_dict()
    manifest = {
        "spec_id": spec_identifier,
        "spec": spec_payload,
        "shock_metadata": shock_meta,
        "git_sha": _git_sha(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "panel": _data_manifest(panel_path),
            "sample": _data_manifest(sample_path),
        },
        "outputs": {
            "tables": [str(fs_path), str(rf_path), str(iv_path)],
            "figures": [str(path) for path in figures_dir.glob("*")],
        },
        "diagnostics": diagnostics,
        "economic_significance": econ,
        "estimation_errors": estimation_errors,
    }

    manifest_path = spec_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    summary = {
        "spec_id": spec_identifier,
        "name": spec.name,
        "sample": spec.sample.name,
        "running_var": spec.sample.running_var,
        "instrument": instrument_col,
        "outcome": spec.outcome.name,
        "transform": spec.outcome.transform,
        "model": spec.model.model_type,
        "window": window,
        "cutoff": spec.model.cutoff,
        "horizon_main": econ.get("horizon"),
        "coef_main": econ.get("coef"),
        "effect_sd": econ.get("effect_sd"),
        "effect_pct": econ.get("effect_pct"),
        "effect_pct_of_mean": econ.get("effect_pct_of_mean"),
        "shock_sd": econ.get("shock_sd"),
        "outcome_mean": econ.get("outcome_mean"),
        "density_z": diagnostics.get("density", {}).get("z_stat"),
        "density_pass": diagnostics.get("density", {}).get("pass"),
        "balance_min_p": diagnostics.get("balance", {}).get("min_pvalue"),
        "balance_pass": diagnostics.get("balance", {}).get("pass"),
        "pretrend_pass": diagnostics.get("pretrend", {}).get("pass"),
        "first_stage_f": diagnostics.get("first_stage", {}).get("f_stat"),
        "first_stage_pass": diagnostics.get("first_stage", {}).get("pass"),
        "n_left": diagnostics.get("sample", {}).get("n_left"),
        "n_right": diagnostics.get("sample", {}).get("n_right"),
        "sample_pass": diagnostics.get("sample", {}).get("pass"),
        "estimation_errors": len(estimation_errors),
    }

    return {
        "summary": summary,
        "spec_id": spec_identifier,
        "manifest_path": manifest_path,
        "spec_dir": spec_dir,
    }


def append_results(
    rows: list[dict[str, Any]],
    *,
    output_root: Path,
    file_prefix: str = "spec_results",
) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    csv_path = output_root / f"{file_prefix}.csv"
    parquet_path = output_root / f"{file_prefix}.parquet"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
    if "spec_id" in df.columns:
        df = df.drop_duplicates(subset=["spec_id"], keep="last")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def write_spec_matrix(specs: list[Spec], *, output_root: Path) -> Path:
    rows = []
    for spec in specs:
        payload = spec.to_dict()
        flat = flatten_dict(payload)
        flat["spec_id"] = spec_id(spec)
        rows.append(flat)
    df = pd.DataFrame(rows)
    out_path = output_root / "spec_matrix.csv"
    df.to_csv(out_path, index=False)
    return out_path


def run_grid(
    specs: list[Spec],
    *,
    output_root: Path,
    panel_path: Path | None = None,
    sample_path: Path | None = None,
    allow_overwrite: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    run_log_path = output_root / "notes" / "run_log.jsonl"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, spec in enumerate(specs):
        if limit is not None and idx >= limit:
            break
        start = time.time()
        status = "success"
        error = ""
        result: dict[str, Any] | None = None
        try:
            result = run_spec(
                spec,
                output_root=output_root,
                panel_path=panel_path,
                sample_path=sample_path,
                allow_overwrite=allow_overwrite,
            )
            summaries.append(result["summary"])
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)
        finally:
            entry = {
                "spec_id": spec_id(spec),
                "name": spec.name,
                "status": status,
                "error": error,
                "duration_sec": round(time.time() - start, 3),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            with run_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
    return summaries
