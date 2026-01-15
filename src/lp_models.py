"""Local projection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


@dataclass
class LPResult:
    horizon: int
    model: PanelOLS
    fit: object


def prepare_lp_data(df: pd.DataFrame, entity_col: str, time_col: str) -> pd.DataFrame:
    df = df.sort_values([entity_col, time_col]).copy()
    df = df.set_index([entity_col, time_col])
    return df


def run_local_projections(
    df: pd.DataFrame,
    outcome_col: str,
    shock_cols: List[str],
    control_cols: List[str],
    horizons: Iterable[int],
    entity_col: str = "iso3",
    time_col: str = "year",
    add_time_fe: bool = True,
) -> List[LPResult]:
    panel = prepare_lp_data(df, entity_col, time_col)
    results: List[LPResult] = []

    for h in horizons:
        data = panel.copy()
        data["lp_outcome"] = data.groupby(level=0)[outcome_col].shift(-h)
        needed = ["lp_outcome"] + shock_cols + control_cols
        data = data.dropna(subset=needed)
        exog = data[shock_cols + control_cols]
        model = PanelOLS(
            data["lp_outcome"],
            exog,
            entity_effects=True,
            time_effects=add_time_fe,
        )
        fit = model.fit(cov_type="clustered", cluster_entity=True)
        results.append(LPResult(horizon=h, model=model, fit=fit))
    return results


def summarize_lp_results(results: List[LPResult], shock_cols: List[str]) -> pd.DataFrame:
    records = []
    for res in results:
        for term in shock_cols:
            if term not in res.fit.params:
                continue
            records.append(
                {
                    "horizon": res.horizon,
                    "term": term,
                    "coef": res.fit.params[term],
                    "std_err": res.fit.std_errors[term],
                    "t_stat": res.fit.tstats[term],
                    "p_value": res.fit.pvalues[term],
                    "nobs": res.fit.nobs,
                }
            )
    return pd.DataFrame.from_records(records)
