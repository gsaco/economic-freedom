from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS


@dataclass
class RDEstimate:
    outcome: str
    coef: float
    se: float
    pvalue: float
    n_obs: int
    n_left: int
    n_right: int
    bandwidth: float
    order: int
    cutoff: float
    method: str


def select_bandwidth(series: pd.Series, quantile: float = 0.3, max_bw: float | None = None) -> float:
    abs_vals = series.abs().dropna()
    if abs_vals.empty:
        return np.nan
    bw = float(abs_vals.quantile(quantile))
    if max_bw is not None:
        bw = min(bw, max_bw)
    return bw


def triangular_weights(running: pd.Series, bandwidth: float) -> pd.Series:
    weights = 1 - (running.abs() / bandwidth)
    return weights.clip(lower=0)


def _prepare_rd_frame(
    df: pd.DataFrame,
    running: str,
    *,
    cutoff: float,
    bandwidth: float,
    kernel: str,
) -> pd.DataFrame:
    frame = df.copy()
    frame = frame[frame[running].notna()].copy()
    frame["m"] = frame[running] - cutoff
    frame = frame[frame["m"].abs() <= bandwidth].copy()
    frame["z"] = (frame["m"] >= 0).astype(int)

    if kernel == "triangular":
        frame["w"] = triangular_weights(frame["m"], bandwidth)
    else:
        frame["w"] = 1.0
    return frame


def rd_estimate(
    df: pd.DataFrame,
    outcome: str,
    running: str,
    *,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    kernel: str = "triangular",
    order: int = 1,
    cluster: str | None = None,
    method: str = "cct",
    bwselect: str = "mserd",
) -> RDEstimate:
    if method == "cct":
        try:
            import rdrobust  # type: ignore
        except Exception:
            method = "wls"

    if method == "cct":
        frame = df[[outcome, running]].copy()
        if cluster and cluster in df.columns:
            frame[cluster] = df[cluster]
        frame = frame.dropna(subset=[outcome, running]).copy()
        if frame.empty:
            raise ValueError("No valid observations for RD estimation.")

        kernel_map = {"triangular": "tri", "uniform": "uni", "epanechnikov": "epa"}
        kernel_opt = kernel_map.get(kernel, kernel)
        h_arg = bandwidth if bandwidth is not None else None
        res = rdrobust.rdrobust(
            frame[outcome].to_numpy(),
            frame[running].to_numpy(),
            c=cutoff,
            p=order,
            kernel=kernel_opt,
            bwselect=bwselect if h_arg is None else "mserd",
            h=h_arg,
            cluster=frame[cluster].to_numpy() if cluster and cluster in frame.columns else None,
        )

        coef = float(res.coef.loc["Robust", "Coeff"])
        se = float(res.se.loc["Robust", "Std. Err."])
        pvalue = float(res.pv.loc["Robust", "P>|t|"])
        n_left = int(res.N_h[0])
        n_right = int(res.N_h[1])
        bw = float(np.nanmean(res.bws.loc["h"]))

        return RDEstimate(
            outcome=outcome,
            coef=coef,
            se=se,
            pvalue=pvalue,
            n_obs=n_left + n_right,
            n_left=n_left,
            n_right=n_right,
            bandwidth=bw,
            order=order,
            cutoff=cutoff,
            method="cct",
        )

    if bandwidth is None:
        bandwidth = select_bandwidth(df[running])
    if bandwidth is None or np.isnan(bandwidth):
        raise ValueError("Unable to select bandwidth for RD estimation.")

    frame = _prepare_rd_frame(df, running, cutoff=cutoff, bandwidth=bandwidth, kernel=kernel)
    frame["mz"] = frame["m"] * frame["z"]
    exog_cols = ["z", "m", "mz"]
    if order >= 2:
        frame["m2"] = frame["m"] ** 2
        frame["m2z"] = frame["m2"] * frame["z"]
        exog_cols += ["m2", "m2z"]

    X = sm.add_constant(frame[exog_cols])
    model = sm.WLS(frame[outcome], X, weights=frame["w"])
    if cluster and cluster in frame.columns:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster]})
    else:
        result = model.fit(cov_type="HC1")

    coef = float(result.params["z"])
    se = float(result.bse["z"])
    pvalue = float(result.pvalues["z"])
    n_left = int((frame["m"] < 0).sum())
    n_right = int((frame["m"] >= 0).sum())

    return RDEstimate(
        outcome=outcome,
        coef=coef,
        se=se,
        pvalue=pvalue,
        n_obs=len(frame),
        n_left=n_left,
        n_right=n_right,
        bandwidth=float(bandwidth),
        order=order,
        cutoff=cutoff,
        method="wls",
    )


def rd_iv(
    df: pd.DataFrame,
    outcome: str,
    running: str,
    endogenous: str,
    *,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    kernel: str = "triangular",
    order: int = 1,
    cluster: str | None = None,
) -> RDEstimate:
    if bandwidth is None:
        bandwidth = select_bandwidth(df[running])
    if bandwidth is None or np.isnan(bandwidth):
        raise ValueError("Unable to select bandwidth for RD-IV estimation.")

    frame = _prepare_rd_frame(df, running, cutoff=cutoff, bandwidth=bandwidth, kernel=kernel)
    frame["mz"] = frame["m"] * frame["z"]
    exog_cols = ["m", "mz"]
    if order >= 2:
        frame["m2"] = frame["m"] ** 2
        frame["m2z"] = frame["m2"] * frame["z"]
        exog_cols += ["m2", "m2z"]

    exog = sm.add_constant(frame[exog_cols])
    endog = frame[[endogenous]]
    instr = frame[["z"]]

    model = IV2SLS(frame[outcome], exog, endog, instr, weights=frame["w"])
    if cluster and cluster in frame.columns:
        result = model.fit(cov_type="clustered", clusters=frame[cluster])
    else:
        result = model.fit(cov_type="robust")

    coef = float(result.params[endogenous])
    se = float(result.std_errors[endogenous])
    pvalue = float(result.pvalues[endogenous])
    n_left = int((frame["m"] < 0).sum())
    n_right = int((frame["m"] >= 0).sum())

    return RDEstimate(
        outcome=outcome,
        coef=coef,
        se=se,
        pvalue=pvalue,
        n_obs=len(frame),
        n_left=n_left,
        n_right=n_right,
        bandwidth=float(bandwidth),
        order=order,
        cutoff=cutoff,
        method="2sls",
    )


def rd_binned_means(
    df: pd.DataFrame,
    outcome: str,
    running: str,
    *,
    cutoff: float = 0.0,
    bins: int = 20,
    bandwidth: float | None = None,
) -> pd.DataFrame:
    if bandwidth is None:
        bandwidth = select_bandwidth(df[running])
    frame = df[df[running].notna()].copy()
    frame["m"] = frame[running] - cutoff
    frame = frame[frame["m"].abs() <= bandwidth].copy()
    left = frame[frame["m"] < 0].copy()
    right = frame[frame["m"] >= 0].copy()

    def _bin_side(side: pd.DataFrame) -> pd.DataFrame:
        if side.empty:
            return pd.DataFrame(columns=["bin_center", outcome])
        bins_edges = np.linspace(side["m"].min(), side["m"].max(), bins + 1)
        side["bin"] = pd.cut(side["m"], bins=bins_edges, include_lowest=True)
        binned = side.groupby("bin", observed=False).agg(
            bin_center=("m", "mean"),
            mean_outcome=(outcome, "mean"),
        )
        return binned.reset_index(drop=True)

    left_bins = _bin_side(left)
    right_bins = _bin_side(right)
    return pd.concat([left_bins, right_bins], ignore_index=True)


def density_discontinuity(
    series: pd.Series,
    *,
    cutoff: float = 0.0,
    bandwidth: float = 0.1,
) -> dict[str, float]:
    series = series.dropna()
    left = series[(series >= cutoff - bandwidth) & (series < cutoff)]
    right = series[(series >= cutoff) & (series <= cutoff + bandwidth)]
    left_n = len(left)
    right_n = len(right)
    left_density = left_n / bandwidth if bandwidth > 0 else np.nan
    right_density = right_n / bandwidth if bandwidth > 0 else np.nan
    if left_n > 0 and right_n > 0:
        log_diff = np.log(right_density) - np.log(left_density)
        se = np.sqrt(1 / left_n + 1 / right_n)
        z_stat = log_diff / se
    else:
        log_diff = np.nan
        se = np.nan
        z_stat = np.nan
    return {
        "left_n": left_n,
        "right_n": right_n,
        "log_diff": float(log_diff) if np.isfinite(log_diff) else np.nan,
        "se": float(se) if np.isfinite(se) else np.nan,
        "z_stat": float(z_stat) if np.isfinite(z_stat) else np.nan,
        "bandwidth": bandwidth,
    }
