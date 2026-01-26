from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass
class WindowChoice:
    window: float | None
    p_threshold: float
    windows_tested: list[float]


def subset_window(
    df: pd.DataFrame,
    running_col: str,
    *,
    window: float,
    cutoff: float = 0.0,
) -> pd.DataFrame:
    frame = df[df[running_col].notna()].copy()
    frame["m"] = frame[running_col] - cutoff
    return frame.loc[frame["m"].abs() <= window].copy()


def balance_table(
    df: pd.DataFrame,
    running_col: str,
    covariates: list[str],
    *,
    window: float,
    cutoff: float = 0.0,
    cluster: str | None = None,
) -> pd.DataFrame:
    frame = subset_window(df, running_col, window=window, cutoff=cutoff)
    frame["treat"] = (frame["m"] >= 0).astype(int)

    rows = []
    for cov in covariates:
        if cov not in frame.columns:
            continue
        sample = frame.dropna(subset=[cov]).copy()
        if sample.empty:
            continue
        X = sm.add_constant(sample[["treat"]])
        model = sm.OLS(sample[cov], X)
        if cluster and cluster in sample.columns:
            groups = sample[cluster]
            if groups.nunique(dropna=True) < 2:
                result = model.fit(cov_type="HC1")
            else:
                try:
                    result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
                except ValueError:
                    result = model.fit(cov_type="HC1")
        else:
            result = model.fit(cov_type="HC1")
        rows.append(
            {
                "covariate": cov,
                "coef": float(result.params["treat"]),
                "se": float(result.bse["treat"]),
                "pvalue": float(result.pvalues["treat"]),
                "n_obs": int(result.nobs),
                "window": float(window),
            }
        )
    return pd.DataFrame(rows)


def select_window_by_balance(
    df: pd.DataFrame,
    running_col: str,
    covariates: list[str],
    *,
    windows: list[float],
    cutoff: float = 0.0,
    p_threshold: float = 0.15,
    cluster: str | None = None,
) -> tuple[WindowChoice, pd.DataFrame]:
    tables = []
    chosen: float | None = None
    for window in sorted(windows):
        table = balance_table(
            df,
            running_col,
            covariates,
            window=window,
            cutoff=cutoff,
            cluster=cluster,
        )
        if not table.empty:
            tables.append(table)
            if (table["pvalue"] >= p_threshold).all():
                chosen = window
    choice = WindowChoice(window=chosen, p_threshold=p_threshold, windows_tested=sorted(windows))
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return choice, combined
