# %% [markdown]
# # LP reduced-form (fallback if IV weak)
# Estimate dynamic winner effects using local projections without IV interpretation.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import ANALYSIS_DIR, PAPER_TABLES_DIR
from src.viz_style import set_style

# %%
set_style()

panel_path = ANALYSIS_DIR / "rd_event_panel.parquet"
if not panel_path.exists():
    raise FileNotFoundError("Missing event panel. Run 13_construct_efw_shocks_and_outcomes first.")

panel = pd.read_parquet(panel_path)

# %%
HORIZONS = [1, 2, 4]
OUTCOMES = {
    "log_gdp_cum": "log_gdp_cum_h{h}",
    "inv_share_avg": "inv_share_avg_h{h}",
    "inflation_path": "inflation_path_h{h}",
}

controls = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
]

def safe_ols(frame: pd.DataFrame, y: str, x: list[str], cluster: str | None = None):
    needed = [y] + x
    frame = frame.dropna(subset=needed).copy()
    if frame.empty:
        return np.nan, np.nan, np.nan, 0
    X = sm.add_constant(frame[x])
    model = sm.OLS(frame[y], X)
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
    coef = float(result.params.get("winner_market", np.nan))
    se = float(result.bse.get("winner_market", np.nan))
    pval = float(result.pvalues.get("winner_market", np.nan))
    return coef, se, pval, int(result.nobs)

# %%
rows = []
for outcome, template in OUTCOMES.items():
    for h in HORIZONS:
        y_col = template.format(h=h)
        coef, se, pval, n_obs = safe_ols(
            panel,
            y=y_col,
            x=["winner_market"] + controls,
            cluster="iso3c",
        )
        rows.append(
            {
                "outcome": outcome,
                "horizon": h,
                "coef": coef,
                "se": se,
                "pvalue": pval,
                "n_obs": n_obs,
                "method": "lp_ols",
            }
        )

lp_df = pd.DataFrame(rows)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
output_path = PAPER_TABLES_DIR / "lp_reduced_form.csv"
lp_df.to_csv(output_path, index=False)

# %%
display(lp_df.style.set_caption("LP reduced-form (winner effects)"))

# %% [markdown]
# ## Interpretation
# Local projection estimates summarize dynamic winner effects without IV interpretation,
# providing a descriptive fallback when RD‑IV is weak.
