# %% [markdown]
# # Asymmetry tests
# Compare positive vs negative IRFs using pre-registered metrics and bootstrap inference.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lpiv import iv_estimate
from src.paths import ANALYSIS_DIR, PAPER_FIGURES_DIR, PAPER_LOGS_DIR, PAPER_TABLES_DIR
from src.viz_style import savefig, set_style

# %%
set_style()

iv_pos_path = PAPER_TABLES_DIR / "irf_iv_pos.csv"
iv_neg_path = PAPER_TABLES_DIR / "irf_iv_neg.csv"

pos_panel_path = ANALYSIS_DIR / "rd_event_panel_pos.parquet"
neg_panel_path = ANALYSIS_DIR / "rd_event_panel_neg.parquet"

for path in [iv_pos_path, iv_neg_path, pos_panel_path, neg_panel_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

iv_pos = pd.read_csv(iv_pos_path)
iv_neg = pd.read_csv(iv_neg_path)

panel_pos = pd.read_parquet(pos_panel_path)
panel_neg = pd.read_parquet(neg_panel_path)

window_choice_path = PAPER_LOGS_DIR / "rd_window_choice.json"
window_choice = json.loads(window_choice_path.read_text()) if window_choice_path.exists() else {}

window_pos = window_choice.get("positive", {}).get("window") or 0.03
window_neg = window_choice.get("negative", {}).get("window") or 0.03

RUNNING = "running_var_vote"
controls = [
    "lag1_log_gdp_pc_const",
    "lag1_trade_open_gdp",
    "lag1_inflation_cpi_ann_pct",
    "lag1_efw_summary",
]

# %%
# Metrics from point estimates

def metric_value(df: pd.DataFrame, horizon: int) -> float:
    row = df.loc[df["horizon"] == horizon]
    if row.empty:
        return np.nan
    return float(row["coef"].iloc[0])


early_pos = metric_value(iv_pos, 1)
early_neg = metric_value(iv_neg, 1)

intensity_pos = iv_pos["coef"].abs().max()
intensity_neg = iv_neg["coef"].abs().max()

persist_pos = iv_pos.loc[iv_pos["horizon"].between(3, 5), "coef"].mean()
persist_neg = iv_neg.loc[iv_neg["horizon"].between(3, 5), "coef"].mean()

metrics_df = pd.DataFrame(
    [
        {
            "metric": "early_h1",
            "positive": early_pos,
            "negative": early_neg,
            "diff_neg_minus_pos": early_neg - early_pos,
        },
        {
            "metric": "intensity_peak_abs",
            "positive": intensity_pos,
            "negative": intensity_neg,
            "diff_neg_minus_pos": intensity_neg - intensity_pos,
        },
        {
            "metric": "persistence_avg_h3_h5",
            "positive": persist_pos,
            "negative": persist_neg,
            "diff_neg_minus_pos": persist_neg - persist_pos,
        },
    ]
)

# %%
# Bootstrap by country for selected horizons
np.random.seed(42)
bootstrap_horizons = [1, 3, 5]
B = 200

countries_pos = panel_pos["iso3c"].dropna().unique()
countries_neg = panel_neg["iso3c"].dropna().unique()


def resample_by_country(frame: pd.DataFrame, countries: np.ndarray) -> pd.DataFrame:
    draw = np.random.choice(countries, size=len(countries), replace=True)
    blocks = [frame.loc[frame["iso3c"] == c] for c in draw]
    return pd.concat(blocks, ignore_index=True)


boot_rows = []
for h in bootstrap_horizons:
    diffs = []
    for _ in range(B):
        boot_pos = resample_by_country(panel_pos, countries_pos)
        boot_neg = resample_by_country(panel_neg, countries_neg)

        try:
            iv_pos_b = iv_estimate(
                boot_pos,
                outcome_col=f"log_gdp_cum_h{h}",
                endog_col="shock_efw",
                running_col=RUNNING,
                instrument_col="z_pos",
                controls=controls,
                window=window_pos,
                cluster="iso3c",
            )
            iv_neg_b = iv_estimate(
                boot_neg,
                outcome_col=f"log_gdp_cum_h{h}",
                endog_col="shock_efw",
                running_col=RUNNING,
                instrument_col="z_neg",
                controls=controls,
                window=window_neg,
                cluster="iso3c",
            )
        except ValueError:
            continue

        if np.isnan(iv_pos_b.coef) or np.isnan(iv_neg_b.coef):
            continue
        diffs.append(iv_neg_b.coef - iv_pos_b.coef)

    if diffs:
        diff_arr = np.array(diffs)
        ci_low, ci_high = np.percentile(diff_arr, [2.5, 97.5])
        p_value = 2 * min((diff_arr <= 0).mean(), (diff_arr >= 0).mean())
    else:
        diff_arr = np.array([])
        ci_low, ci_high, p_value = np.nan, np.nan, np.nan

    boot_rows.append(
        {
            "metric": f"bootstrap_diff_h{h}",
            "positive": metric_value(iv_pos, h),
            "negative": metric_value(iv_neg, h),
            "diff_neg_minus_pos": metric_value(iv_neg, h) - metric_value(iv_pos, h),
            "boot_ci_low": ci_low,
            "boot_ci_high": ci_high,
            "boot_pvalue": p_value,
            "boot_draws": len(diff_arr),
        }
    )

boot_df = pd.DataFrame(boot_rows)

# %%
# Combine outputs
output_df = pd.concat([metrics_df, boot_df], ignore_index=True)

PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
output_path = PAPER_TABLES_DIR / "asymmetry_tests.csv"
output_df.to_csv(output_path, index=False)

# %%
# Plot metrics
fig, ax = plt.subplots(figsize=(6, 3))
metric_labels = ["Early (h=1)", "Peak |IRF|", "Persistence (h=3..5)"]
index = np.arange(len(metric_labels))
bar_width = 0.35

ax.bar(index - bar_width / 2, [early_pos, intensity_pos, persist_pos], bar_width, label="Positive", color="#2A9D8F")
ax.bar(index + bar_width / 2, [early_neg, intensity_neg, persist_neg], bar_width, label="Negative", color="#E76F51")
ax.set_xticks(index)
ax.set_xticklabels(metric_labels)
ax.set_ylabel("Effect size")
ax.set_title("Asymmetry metrics")
ax.legend(frameon=False)

savefig(fig, PAPER_FIGURES_DIR / "asymmetry" / "peak_persistence_bars")
plt.close(fig)

# %%
display(output_df.style.set_caption("Asymmetry metrics and bootstrap tests"))

# %% [markdown]
# ## Interpretation
# The asymmetry metrics summarize speed, intensity, and persistence, while
# bootstrap differences test whether negative shocks are statistically larger
# than positive ones at key horizons.
