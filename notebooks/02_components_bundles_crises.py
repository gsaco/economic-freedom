# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EFW components, shocks, and decomposition (quinquennial 1970–2020)
#
# Build a component-rich EFW panel, construct shock definitions, and decompose
# large EFW summary changes into area contributions.

# %%
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

from src.components import (
    add_shock_definitions,
    build_efw_component_panel,
    compute_shock_decomposition,
    get_core_component_cols,
)
from src.crisis_data import build_crisis_panel
from src.data_fetch_macro import fetch_world_bank_indicators, fetch_pwt, prepare_pwt_subset
from src.data_ingest_efw import locate_fraser_file
from src.inequality_data import build_inequality_panel
from src.regime_data import build_regime_panel
from src.build_panel import build_panel, validate_panel
from src.viz import save_figure, set_plot_style

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

for path in [DATA_RAW, DATA_PROC, OUTPUTS, FIGURES, TABLES]:
    path.mkdir(parents=True, exist_ok=True)

set_plot_style()

# %% [markdown]
# ## 1. Load EFW components and compute quinquennial changes

# %%
fraser_path = locate_fraser_file(ROOT)
result = build_efw_component_panel(fraser_path, start_year=1970, end_year=2020)
efw = result.df.copy()

if not result.iso3_issues.empty:
    iso3_path = DATA_PROC / "efw_iso3_issues.csv"
    result.iso3_issues.to_csv(iso3_path, index=False)
    print("ISO3 issues logged:", iso3_path)

efw.to_csv(DATA_PROC / "efw_quinquennial.csv", index=False)
print("EFW components saved:", DATA_PROC / "efw_quinquennial.csv")

# %% [markdown]
# ## 2. Shock definitions (quantiles, SD, absolute thresholds)

# %%
core_cols = get_core_component_cols(efw)
change_cols = [f"d_{col}" for col in core_cols if f"d_{col}" in efw.columns]

efw, thresholds_global, thresholds_by_year = add_shock_definitions(
    efw,
    change_cols=change_cols,
    quantiles=(0.9, 0.85),
    sd_thresholds=(1.0, 1.5),
    abs_quantile=0.9,
)

thresholds_global.to_csv(TABLES / "efw_shock_thresholds_global.csv", index=False)
thresholds_by_year.to_csv(TABLES / "efw_shock_thresholds_by_year.csv", index=False)
print("Shock thresholds saved:", TABLES / "efw_shock_thresholds_global.csv")

shock_cols = [col for col in efw.columns if col.startswith("shock_pos_") or col.startswith("shock_neg_")]
shock_freq = efw[shock_cols].mean().sort_values(ascending=False).reset_index()
shock_freq.columns = ["shock_definition", "frequency"]
shock_freq.to_csv(TABLES / "efw_shock_frequencies_core.csv", index=False)
print("Shock frequencies saved:", TABLES / "efw_shock_frequencies_core.csv")

efw.to_csv(DATA_PROC / "efw_quinquennial.csv", index=False)

# %% [markdown]
# ## 3. Shock decomposition for large summary changes

# %%
summary_change = "d_efw_summary"
pos_abs = f"shock_pos_{summary_change}_abs"
neg_abs = f"shock_neg_{summary_change}_abs"
efw["shock_abs_d_efw_summary"] = efw[pos_abs] | efw[neg_abs]

area_change_cols = [f"d_efw_area{k}" for k in range(1, 6) if f"d_efw_area{k}" in efw.columns]

decomp = compute_shock_decomposition(
    efw,
    summary_change_col=summary_change,
    area_change_cols=area_change_cols,
    shock_flag_col="shock_abs_d_efw_summary",
)
decomp.to_csv(TABLES / "efw_shock_decomposition.csv", index=False)
print("Shock decomposition saved:", TABLES / "efw_shock_decomposition.csv")

top_area = decomp[decomp["rank_abs"] == 1].copy()
top_area.to_csv(TABLES / "efw_shock_decomposition_top_area.csv", index=False)
print("Top-area decomposition saved:", TABLES / "efw_shock_decomposition_top_area.csv")

# %% [markdown]
# ## 4. External data: macro, crises, regime, inequality

# %%
wb_indicators = {
    "gdppc_wb": "NY.GDP.PCAP.KD",
    "gdppc_growth_wb": "NY.GDP.PCAP.KD.ZG",
    "population": "SP.POP.TOTL",
    "gcf_gdp": "NE.GDI.FTOT.ZS",
    "trade_gdp": "NE.TRD.GNFS.ZS",
    "inflation_cpi": "FP.CPI.TOTL.ZG",
    "gov_consumption_gdp": "NE.CON.GOVT.ZS",
}

wb_raw, wb_sources = fetch_world_bank_indicators(
    wb_indicators,
    cache_dir=DATA_RAW / "wb",
    start_year=1970,
    end_year=2020,
)

pwt_raw, pwt_url = fetch_pwt(cache_dir=DATA_RAW / "pwt")
pwt_subset = prepare_pwt_subset(pwt_raw, start_year=1970, end_year=2020)

crisis = build_crisis_panel(cache_dir=DATA_RAW / "crisis")
regime = build_regime_panel(cache_dir=DATA_RAW / "regime")
inequality = build_inequality_panel(cache_dir=DATA_RAW / "swiid")

if not crisis.unmatched.empty:
    crisis.unmatched.to_csv(TABLES / "crisis_unmatched_countries.csv", index=False)
    print("Crisis unmatched countries saved:", TABLES / "crisis_unmatched_countries.csv")

if not inequality.unmatched.empty:
    inequality.unmatched.to_csv(TABLES / "inequality_unmatched_countries.csv", index=False)
    print("Inequality unmatched countries saved:", TABLES / "inequality_unmatched_countries.csv")

# %%
quin_years = set(range(1970, 2021, 5))
wb = wb_raw[wb_raw["year"].isin(quin_years)].copy()
wb = wb[wb["iso3"].str.len() == 3]
wb = wb.rename(columns={"country": "country_wb"})

pwt = pwt_subset[pwt_subset["year"].isin(quin_years)].copy()
pwt = pwt.rename(columns={"country": "country_pwt"})

macro_keep = ["iso3", "year", "country_wb"] + list(wb_indicators.keys())
macro_df = wb[macro_keep]

pwt_keep = [
    "iso3",
    "year",
    "country_pwt",
    "pwt_gdppc",
    "pwt_tfp",
    "pwt_capital",
    "pwt_inv_share",
    "hc",
]
pwt_df = pwt[[col for col in pwt_keep if col in pwt.columns]]

panel, merge_reports = build_panel(efw, macro_df, pwt_df)
for report in merge_reports:
    print(report)

panel = panel.merge(crisis.panel, on=["iso3", "year"], how="left")
panel = panel.merge(regime.panel, on=["iso3", "year"], how="left")
panel = panel.merge(inequality.panel, on=["iso3", "year"], how="left")
crisis_cols = [col for col in crisis.panel.columns if col not in ("iso3", "year")]
if crisis_cols:
    panel[crisis_cols] = panel[crisis_cols].fillna(0)
validate_panel(panel)

# %%
panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
panel.loc[panel["gdppc_wb"] <= 0, "gdppc_wb"] = np.nan
panel.loc[panel["pwt_gdppc"] <= 0, "pwt_gdppc"] = np.nan

panel["gdppc_log"] = np.log(panel["gdppc_wb"])
panel["gdppc_growth_5y"] = panel.groupby("iso3")["gdppc_log"].diff()

if "pwt_tfp" in panel.columns:
    panel.loc[panel["pwt_tfp"] <= 0, "pwt_tfp"] = np.nan
    panel["pwt_tfp_log"] = np.log(panel["pwt_tfp"])
    panel["pwt_tfp_growth_5y"] = panel.groupby("iso3")["pwt_tfp_log"].diff()

if "pwt_capital" in panel.columns:
    panel.loc[panel["pwt_capital"] <= 0, "pwt_capital"] = np.nan
    panel["pwt_capital_log"] = np.log(panel["pwt_capital"])
    panel["pwt_capital_growth_5y"] = panel.groupby("iso3")["pwt_capital_log"].diff()

if "pwt_inv_share" in panel.columns:
    panel["pwt_inv_share_change"] = panel.groupby("iso3")["pwt_inv_share"].diff()

panel.to_csv(DATA_PROC / "panel_master_quinquennial_1970_2020.csv", index=False)
print("Master panel saved:", DATA_PROC / "panel_master_quinquennial_1970_2020.csv")

# %% [markdown]
# ## 5. Coverage audit and GDP consistency checks

# %%
audit_vars = [
    "gdppc_wb",
    "pwt_gdppc",
    "gdppc_growth_5y",
    "pwt_tfp_growth_5y",
    "pwt_capital_growth_5y",
    "pwt_inv_share",
    "gini_net",
    "gini_market",
    "electoral_democracy",
    "liberal_democracy",
]
audit_vars = [col for col in audit_vars if col in panel.columns]

coverage = panel.groupby("year")[audit_vars].apply(lambda x: x.notna().sum())
coverage.to_csv(TABLES / "master_coverage_by_year.csv")
print("Coverage table saved:", TABLES / "master_coverage_by_year.csv")

country_counts = panel.groupby("year")["iso3"].nunique()
missing_frac = 1 - (coverage.T / country_counts)

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(missing_frac, ax=ax, cmap="Reds", cbar_kws={"label": "Missing fraction"})
ax.set_title("Missingness by variable and year (quinquennial)")
save_figure(fig, FIGURES / "master_missingness_heatmap.png")
plt.close(fig)

# %%
compare = panel.dropna(subset=["gdppc_wb", "pwt_gdppc"]).copy()
compare["log_ratio"] = np.log(compare["gdppc_wb"] / compare["pwt_gdppc"])

regime_label = compare["democracy"].map({1.0: "democracy", 0.0: "autocracy"})
compare["regime_type"] = regime_label.fillna("unknown")

corr_by_regime = compare.groupby("regime_type").apply(
    lambda x: x[["gdppc_wb", "pwt_gdppc"]].corr().iloc[0, 1]
)
corr_by_regime = corr_by_regime.reset_index(name="corr_gdppc_wb_pwt")

logratio_by_regime = (
    compare.groupby("regime_type")["log_ratio"]
    .agg(["mean", "median", "std", "count"])
    .reset_index()
)

corr_by_regime.to_csv(TABLES / "gdppc_wb_pwt_corr_by_regime.csv", index=False)
logratio_by_regime.to_csv(TABLES / "gdppc_wb_pwt_logratio_by_regime.csv", index=False)
print("GDP consistency checks saved:", TABLES / "gdppc_wb_pwt_corr_by_regime.csv")
