# %% [markdown]
# # RD validity checks (vote margin)
# McCrary-style density and covariate balance checks with local-randomization windows.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import ANALYSIS_DIR, PAPER_FIGURES_DIR, PAPER_LOGS_DIR, PAPER_TABLES_DIR
from src.rd import density_discontinuity, select_bandwidth
from src.rd_localrand import select_window_by_balance
from src.viz_style import savefig, set_style

# %%
set_style()

pos_path = ANALYSIS_DIR / "rd_event_panel_pos.parquet"
neg_path = ANALYSIS_DIR / "rd_event_panel_neg.parquet"

for path in [pos_path, neg_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing event panel: {path}")

panel_pos = pd.read_parquet(pos_path)
panel_neg = pd.read_parquet(neg_path)

RUNNING = "running_var_vote"

# %%
# Density plots and discontinuity stats

def density_block(frame: pd.DataFrame, label: str) -> dict:
    bandwidth = select_bandwidth(frame[RUNNING], quantile=0.3, max_bw=0.1)
    if pd.isna(bandwidth):
        bandwidth = 0.05

    subset = frame[frame[RUNNING].abs() <= bandwidth]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(subset[RUNNING], bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Running variable density near cutoff ({label})")
    ax.set_xlabel("Vote share margin (market bloc)")
    ax.set_ylabel("Count")
    savefig(fig, PAPER_FIGURES_DIR / "rd_validity" / f"rd_density_{label}")
    plt.close(fig)

    stats = density_discontinuity(frame[RUNNING], bandwidth=bandwidth)
    stats["sample"] = label
    return stats


density_pos = density_block(panel_pos, "pos")
density_neg = density_block(panel_neg, "neg")

density_df = pd.DataFrame([density_pos, density_neg])
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
density_path = PAPER_TABLES_DIR / "rd_density_vote_margin.csv"
density_df.to_csv(density_path, index=False)

display(density_df.style.set_caption("Density discontinuity (vote margin)"))

# %%
# Balance checks by window
balance_vars = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
    "lag1_inv_share_gdp",
]

windows = [0.01, 0.02, 0.03, 0.04, 0.05]

choice_pos, table_pos = select_window_by_balance(
    panel_pos,
    RUNNING,
    balance_vars,
    windows=windows,
    p_threshold=0.15,
    cluster="iso3c",
)

choice_neg, table_neg = select_window_by_balance(
    panel_neg,
    RUNNING,
    balance_vars,
    windows=windows,
    p_threshold=0.15,
    cluster="iso3c",
)

PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
valid_pos_path = PAPER_TABLES_DIR / "rd_validity_vote_margin_pos.csv"
valid_neg_path = PAPER_TABLES_DIR / "rd_validity_vote_margin_neg.csv"

table_pos.to_csv(valid_pos_path, index=False)
table_neg.to_csv(valid_neg_path, index=False)

display(table_pos.style.set_caption("Balance table (positive shocks)"))
display(table_neg.style.set_caption("Balance table (negative shocks)"))

# %%
# Record window choice
window_choice = {
    "positive": {
        "window": choice_pos.window,
        "p_threshold": choice_pos.p_threshold,
        "windows_tested": choice_pos.windows_tested,
    },
    "negative": {
        "window": choice_neg.window,
        "p_threshold": choice_neg.p_threshold,
        "windows_tested": choice_neg.windows_tested,
    },
}

PAPER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
choice_path = PAPER_LOGS_DIR / "rd_window_choice.json"
choice_path.write_text(json.dumps(window_choice, indent=2))

# %% [markdown]
# ## Interpretation
# Density continuity and covariate balance are prerequisites for the local
# randomization RD design. The selected window(s) define the main estimation
# sample for LP-IV impulse responses.
