# %% [markdown]
# # RD-IV main results (Wald ratio)
# Compute local IV effects as the ratio of RD reduced-form to RD first-stage.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import PAPER_TABLES_DIR
from src.viz_style import set_style

# %%
set_style()

first_stage_path = PAPER_TABLES_DIR / "rd_first_stage.csv"
reduced_form_path = PAPER_TABLES_DIR / "rd_reduced_form.csv"

for path in [first_stage_path, reduced_form_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

first_stage = pd.read_csv(first_stage_path)
reduced_form = pd.read_csv(reduced_form_path)

PRIMARY_HORIZONS = [1, 2, 4]

rows = []
for outcome in ["log_gdp_cum", "inv_share_avg", "inflation_path"]:
    for h in PRIMARY_HORIZONS:
        tau_e = first_stage.loc[
            (first_stage["outcome"] == "efw_summary") & (first_stage["horizon"] == h)
        ]
        tau_y = reduced_form.loc[(reduced_form["outcome"] == outcome) & (reduced_form["horizon"] == h)]
        if tau_e.empty or tau_y.empty:
            continue

        tau_e_val = float(tau_e["coef"].iloc[0])
        tau_e_se = float(tau_e["se"].iloc[0])
        tau_y_val = float(tau_y["coef"].iloc[0])
        tau_y_se = float(tau_y["se"].iloc[0])

        if not np.isfinite(tau_e_val) or tau_e_val == 0:
            continue

        beta = tau_y_val / tau_e_val
        # Delta method (ignores covariance)
        se_beta = np.sqrt((tau_y_se / tau_e_val) ** 2 + (tau_y_val * tau_e_se / (tau_e_val**2)) ** 2)

        rows.append(
            {
                "outcome": outcome,
                "horizon": h,
                "beta": float(beta),
                "se": float(se_beta),
                "tau_e": tau_e_val,
                "tau_y": tau_y_val,
                "n_left": int(tau_e["n_left"].iloc[0]) if "n_left" in tau_e.columns else np.nan,
                "n_right": int(tau_e["n_right"].iloc[0]) if "n_right" in tau_e.columns else np.nan,
            }
        )

iv_df = pd.DataFrame(rows)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
iv_path = PAPER_TABLES_DIR / "rd_iv_main.csv"
iv_df.to_csv(iv_path, index=False)

display(iv_df.style.set_caption("RD-IV (Wald) estimates"))

# %% [markdown]
# ## Interpretation
# The IV estimates use the Wald ratio of RD reduced-form to first-stage
# discontinuities at the cutoff (primary horizons h=1,2,4).
