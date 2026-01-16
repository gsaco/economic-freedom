# %% [markdown]
# # 06. Estimate stacked event-study LP/DiD

# %%
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for _ in range(6):
        if (current / "src").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return start.resolve()


ROOT = find_repo_root(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stacked_event import build_event_dummies
from src.viz import plot_irf, set_plot_style

DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUT_TABLES = OUTPUTS / "tables"
OUTPUT_FIGS = OUTPUTS / "figures"

for path in [OUTPUTS, OUTPUT_TABLES, OUTPUT_FIGS]:
    path.mkdir(parents=True, exist_ok=True)

set_plot_style()

# %%
panel = pd.read_csv(DATA_PROC / "panel_master_quinquennial_1970_2020.csv")
collapse_threshold = panel["gdppc_growth_5y"].quantile(0.1)

stacked_pos = pd.read_csv(DATA_PROC / "stacked_event_pos.csv")
stacked_neg = pd.read_csv(DATA_PROC / "stacked_event_neg.csv")


def add_controls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["gdppc_wb"] <= 0, "gdppc_wb"] = np.nan
    df["gdppc_log"] = np.log(df["gdppc_wb"])
    df["lag_gdppc_log"] = df.groupby("entity_id")["gdppc_log"].shift(1)
    df["lag_gdppc_growth_5y"] = df.groupby("entity_id")["gdppc_growth_5y"].shift(1)
    df["lag_efw_summary"] = df.groupby("entity_id")["efw_summary"].shift(1)
    df["growth_collapse"] = (df["gdppc_growth_5y"] <= collapse_threshold).astype(float)
    return df


def estimate_stacked_event(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    omit: int = -1,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df, event_cols = build_event_dummies(df, "event_time", "treated", omit=omit)
    needed = [outcome, "entity_id", "year"] + event_cols + controls
    df = df.dropna(subset=needed)
    if df.empty:
        return pd.DataFrame()
    panel = df.set_index(["entity_id", "year"])
    exog = panel[event_cols + controls]
    # Drop zero-variance columns to avoid singular designs.
    variance = exog.var()
    exog = exog.loc[:, variance > 0]
    event_cols = [col for col in event_cols if col in exog.columns]
    if not event_cols:
        return pd.DataFrame()
    model = PanelOLS(
        panel[outcome],
        exog,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False,
    )
    try:
        fit = model.fit(cov_type="clustered", cluster_entity=True)
        cov_label = "clustered"
    except np.linalg.LinAlgError:
        fit = model.fit(cov_type="robust")
        cov_label = "robust_fallback"

    records = []
    for col in event_cols:
        if col not in fit.params.index:
            continue
        event_time = int(col.replace("event_", ""))
        try:
            std_err = fit.std_errors[col]
            p_value = fit.pvalues[col]
        except np.linalg.LinAlgError:
            std_err = np.nan
            p_value = np.nan
        records.append(
            {
                "event_time": event_time,
                "horizon": event_time * 5,
                "term": "treated",
                "coef": fit.params[col],
                "std_err": std_err,
                "p_value": p_value,
                "nobs": fit.nobs,
                "cov_type": cov_label,
            }
        )
    return pd.DataFrame.from_records(records)


def run_and_save(df: pd.DataFrame, label: str) -> None:
    df = add_controls(df)
    controls = [
        "lag_gdppc_growth_5y",
        "lag_gdppc_log",
        "lag_efw_summary",
        "trade_gdp",
        "inflation_cpi",
        "gcf_gdp",
        "gov_consumption_gdp",
    ]
    controls = [c for c in controls if c in df.columns]

    for outcome in ["gdppc_growth_5y", "growth_collapse"]:
        res = estimate_stacked_event(df, outcome, controls)
        if res.empty:
            continue
        table_path = OUTPUT_TABLES / f"stacked_event_irf_{label}_{outcome}.csv"
        res.to_csv(table_path, index=False)
        plot_irf(
            res,
            f"Stacked event study ({label}) - {outcome}",
            OUTPUT_FIGS / f"stacked_event_irf_{label}_{outcome}.png",
        )


run_and_save(stacked_pos, "pos")
run_and_save(stacked_neg, "neg")
