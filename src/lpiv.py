from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS


@dataclass
class LPIVResult:
    outcome: str
    coef: float
    se: float
    pvalue: float
    n_obs: int
    n_left: int
    n_right: int
    window: float


def _prepare_frame(
    df: pd.DataFrame,
    running_col: str,
    instrument_col: str,
    *,
    window: float,
    cutoff: float = 0.0,
) -> pd.DataFrame:
    frame = df[df[running_col].notna() & df[instrument_col].notna()].copy()
    frame["m"] = frame[running_col] - cutoff
    frame = frame.loc[frame["m"].abs() <= window].copy()
    frame["mz"] = frame["m"] * frame[instrument_col]
    frame["z"] = frame[instrument_col].astype(float)
    return frame


def _add_year_fe(frame: pd.DataFrame, year_col: str) -> pd.DataFrame:
    dummies = pd.get_dummies(frame[year_col].astype(int), prefix="year", drop_first=True)
    return pd.concat([frame, dummies], axis=1)


def _build_exog(
    frame: pd.DataFrame,
    *,
    controls: list[str] | None = None,
    year_fe: bool = False,
    year_col: str | None = None,
) -> list[str]:
    exog_cols = ["m", "mz"]
    if controls:
        exog_cols += controls
    if year_fe:
        if not year_col:
            raise ValueError("year_col must be provided when year_fe=True")
        year_cols = [col for col in frame.columns if col.startswith("year_")]
        exog_cols += year_cols
    return exog_cols


def first_stage(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    running_col: str,
    instrument_col: str,
    controls: list[str] | None,
    window: float,
    cutoff: float = 0.0,
    cluster: str | None = None,
    year_fe: bool = False,
    year_col: str | None = None,
) -> LPIVResult:
    frame = _prepare_frame(df, running_col, instrument_col, window=window, cutoff=cutoff)
    if year_fe:
        frame = _add_year_fe(frame, year_col)
    needed = [outcome_col] + (controls or [])
    frame = frame.dropna(subset=needed).copy()
    if frame.empty:
        return LPIVResult(outcome_col, np.nan, np.nan, np.nan, 0, 0, 0, window)

    exog_cols = ["z"] + _build_exog(frame, controls=controls, year_fe=year_fe, year_col=year_col)
    X = sm.add_constant(frame[exog_cols])
    model = sm.OLS(frame[outcome_col], X)
    if cluster and cluster in frame.columns:
        groups = frame[cluster]
        if groups.nunique(dropna=True) < 2 or len(frame) <= X.shape[1]:
            result = model.fit(cov_type="HC1")
        else:
            try:
                result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            except (ValueError, ZeroDivisionError):
                result = model.fit(cov_type="HC1")
    else:
        result = model.fit(cov_type="HC1")

    n_left = int((frame["m"] < 0).sum())
    n_right = int((frame["m"] >= 0).sum())
    return LPIVResult(
        outcome=outcome_col,
        coef=float(result.params["z"]),
        se=float(result.bse["z"]),
        pvalue=float(result.pvalues["z"]),
        n_obs=int(result.nobs),
        n_left=n_left,
        n_right=n_right,
        window=float(window),
    )


def reduced_form(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    running_col: str,
    instrument_col: str,
    controls: list[str] | None,
    window: float,
    cutoff: float = 0.0,
    cluster: str | None = None,
    year_fe: bool = False,
    year_col: str | None = None,
) -> LPIVResult:
    frame = _prepare_frame(df, running_col, instrument_col, window=window, cutoff=cutoff)
    if year_fe:
        frame = _add_year_fe(frame, year_col)
    needed = [outcome_col] + (controls or [])
    frame = frame.dropna(subset=needed).copy()
    if frame.empty:
        return LPIVResult(outcome_col, np.nan, np.nan, np.nan, 0, 0, 0, window)

    exog_cols = ["z"] + _build_exog(frame, controls=controls, year_fe=year_fe, year_col=year_col)
    X = sm.add_constant(frame[exog_cols])
    model = sm.OLS(frame[outcome_col], X)
    if cluster and cluster in frame.columns:
        groups = frame[cluster]
        if groups.nunique(dropna=True) < 2 or len(frame) <= X.shape[1]:
            result = model.fit(cov_type="HC1")
        else:
            try:
                result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            except (ValueError, ZeroDivisionError):
                result = model.fit(cov_type="HC1")
    else:
        result = model.fit(cov_type="HC1")

    n_left = int((frame["m"] < 0).sum())
    n_right = int((frame["m"] >= 0).sum())
    return LPIVResult(
        outcome=outcome_col,
        coef=float(result.params["z"]),
        se=float(result.bse["z"]),
        pvalue=float(result.pvalues["z"]),
        n_obs=int(result.nobs),
        n_left=n_left,
        n_right=n_right,
        window=float(window),
    )


def iv_estimate(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    endog_col: str,
    running_col: str,
    instrument_col: str,
    controls: list[str] | None,
    window: float,
    cutoff: float = 0.0,
    cluster: str | None = None,
    year_fe: bool = False,
    year_col: str | None = None,
) -> LPIVResult:
    frame = _prepare_frame(df, running_col, instrument_col, window=window, cutoff=cutoff)
    if year_fe:
        frame = _add_year_fe(frame, year_col)
    needed = [outcome_col, endog_col] + (controls or [])
    frame = frame.dropna(subset=needed).copy()
    if frame.empty:
        return LPIVResult(outcome_col, np.nan, np.nan, np.nan, 0, 0, 0, window)

    exog_cols = _build_exog(frame, controls=controls, year_fe=year_fe, year_col=year_col)
    exog = sm.add_constant(frame[exog_cols])
    endog = frame[[endog_col]]
    instr = frame[["z"]]

    model = IV2SLS(frame[outcome_col], exog, endog, instr)
    if cluster and cluster in frame.columns:
        result = model.fit(cov_type="clustered", clusters=frame[cluster])
    else:
        result = model.fit(cov_type="robust")

    n_left = int((frame["m"] < 0).sum())
    n_right = int((frame["m"] >= 0).sum())
    return LPIVResult(
        outcome=outcome_col,
        coef=float(result.params[endog_col]),
        se=float(result.std_errors[endog_col]),
        pvalue=float(result.pvalues[endog_col]),
        n_obs=int(result.nobs),
        n_left=n_left,
        n_right=n_right,
        window=float(window),
    )
