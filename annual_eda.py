# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Annual EDA: Economic Freedom + Macro Panel (2000-Present)
#
# **Focus**: Annual-frequency analysis of the EFW dataset merged with World Bank macro indicators.
#
# This notebook avoids mixed-frequency bias by focusing on the annual EFW era (2000+).
#
# **Key outputs**:
# - Sample definition dashboard
# - Time trends (EFW + macro)
# - Correlations and scatter plots
# - Within-country changes
# - Outlier/extreme regime tables

# %% [markdown]
# ## 0. Configuration

# %%
import sys
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# %matplotlib inline

# warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

from src import downloaders, harmonize, merge_efw, quality, plots

# Configuration
EFW_PATH = "data/fraser.xlsx"
START_YEAR = 2000  # Annual EFW era

# Core macro variables for analysis (filled series where possible)
CORE_MACRO = [
    'gdppc_filled',
    'NY.GDP.MKTP.KD.ZG',
    'FP.CPI.TOTL.ZG',
    'trade_openness_filled',
    'investment_gfcf_filled',
    'govt_consumption_filled',
]

# Additional macro series for richer diagnostics
EXTENDED_MACRO = [
    'SP.POP.TOTL',
    'SL.UEM.TOTL.ZS',
    'FS.AST.PRVT.GD.ZS',
    'BX.KLT.DINV.WD.GD.ZS',
    'BN.CAB.XOKA.GD.ZS',
    'GC.TAX.TOTL.GD.ZS',
    'GC.DOD.TOTL.GD.ZS',
    'FM.LBL.BMNY.GD.ZS',
    'kaopen',
    'kof_globalization',
    'wgi_control_corruption',
    'wgi_govt_effectiveness',
    'wgi_political_stability',
    'wgi_regulatory_quality',
    'wgi_rule_of_law',
    'wgi_voice_accountability',
    'gdppc_pwt_scaled',
    'trade_openness_pwt',
    'investment_share_pwt',
    'govt_consumption_share_pwt',
    'human_capital',
    'SE.SEC.ENRR',        # School enrollment, secondary
    'SI.POV.GINI',        # Gini index
    'SP.DYN.LE00.IN',     # Life expectancy
    'NY.GDP.TOTL.RT.ZS',  # Natural resources rents
    'SL.TLF.CACT.ZS',     # Labor force participation
    'gdppc_maddison',     # Maddison Project GDP
    'years_schooling',    # Barro-Lee Years of Schooling
]

ANALYSIS_MACRO = CORE_MACRO + EXTENDED_MACRO

# Output directories
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

plots.setup_plot_style()
print("Configuration loaded.")
print(f"Analysis period: {START_YEAR}-present")

# %% [markdown]
# ## 1. Data Loading
# ### 1.1 Download Macro Data (WDI + secondary sources)

# %%
# Download macro data (cached)
print("=" * 60)
print("STEP 1: DOWNLOAD MACRO DATA")
print("=" * 60)

df_macro_long = downloaders.download_all_sources(
    force_download=False,
    include_secondary=True,
    verbose=True
)

# %% [markdown]
# ### 1.2 Load EFW Data

# %%
# Load EFW data
print("\n" + "=" * 60)
print("STEP 2: LOAD EFW DATA")
print("=" * 60)

df_efw = merge_efw.load_efw(EFW_PATH, verbose=True)

# Filter to annual era
df_efw_annual = merge_efw.get_efw_annual(df_efw, start_year=START_YEAR)
print(f"\nAnnual EFW (>= {START_YEAR}): {len(df_efw_annual):,} obs")

# %% [markdown]
# ## 2. Data Harmonization
# ### 2.1 Build ISO3 Concordance

# %%
print("\n" + "=" * 60)
print("STEP 3: HARMONIZE DATA")
print("=" * 60)

# Build concordance
name_to_iso3, concordance_df = harmonize.build_iso3_concordance(
    df_efw, 
    efw_iso_col='iso3',  # Already renamed
    efw_name_col='Countries',
    verbose=True
)

# Save concordance
harmonize.save_concordance(concordance_df)

# %% [markdown]
# ### 2.2 Standardize and Reshape Macro Data

# %%
# Apply concordance for non-ISO3 labels before standardization
df_macro_long['iso3'] = df_macro_long['iso3'].astype(str).str.strip()
df_macro_long['iso3'] = df_macro_long['iso3'].map(name_to_iso3).fillna(df_macro_long['iso3'])

# Standardize ISO3 codes in macro data
df_macro_clean = harmonize.standardize_iso3(df_macro_long, verbose=True)

# Filter to annual period
df_macro_annual = df_macro_clean[df_macro_clean['year'] >= START_YEAR].copy()
print(f"Macro annual (>= {START_YEAR}): {len(df_macro_annual):,} obs")

# Reshape to wide
df_macro_wide = harmonize.long_to_wide(df_macro_annual, verbose=True)

def _to_pct(series):
    if series is None:
        return None
    non_na = series.dropna()
    if non_na.empty:
        return series
    if non_na.median() <= 1.5:
        return series * 100.0
    return series

def _coalesce(primary, fallback):
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return primary.fillna(fallback)

# PWT-derived series to improve coverage
if 'csh_i' in df_macro_wide.columns:
    df_macro_wide['investment_share_pwt'] = _to_pct(df_macro_wide['csh_i'])
if 'csh_g' in df_macro_wide.columns:
    df_macro_wide['govt_consumption_share_pwt'] = _to_pct(df_macro_wide['csh_g'])
if 'csh_x' in df_macro_wide.columns and 'csh_m' in df_macro_wide.columns:
    exports = _to_pct(df_macro_wide['csh_x'])
    imports = _to_pct(df_macro_wide['csh_m'])
    df_macro_wide['trade_openness_pwt'] = exports + imports

# Scale PWT GDP per capita to WDI units by country median ratio
if 'NY.GDP.PCAP.KD' in df_macro_wide.columns and 'gdppc_pwt' in df_macro_wide.columns:
    ratio = df_macro_wide['NY.GDP.PCAP.KD'] / df_macro_wide['gdppc_pwt'].replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio_by_iso = ratio.groupby(df_macro_wide['iso3']).median()
    df_macro_wide['gdppc_pwt_scaled'] = df_macro_wide['gdppc_pwt'] * df_macro_wide['iso3'].map(ratio_by_iso)

# Filled core series
df_macro_wide['gdppc_filled'] = _coalesce(
    df_macro_wide.get('NY.GDP.PCAP.KD'),
    df_macro_wide.get('gdppc_pwt_scaled')
)
df_macro_wide['trade_openness_filled'] = _coalesce(
    df_macro_wide.get('NE.TRD.GNFS.ZS'),
    df_macro_wide.get('trade_openness_pwt')
)
df_macro_wide['investment_gfcf_filled'] = _coalesce(
    df_macro_wide.get('NE.GDI.FTOT.ZS'),
    df_macro_wide.get('investment_share_pwt')
)
df_macro_wide['govt_consumption_filled'] = _coalesce(
    df_macro_wide.get('NE.CON.GOVT.ZS'),
    df_macro_wide.get('govt_consumption_share_pwt')
)

# Save intermediate
harmonize.save_long_panel(df_macro_annual, 'macro_annual_long')
harmonize.save_wide_panel(df_macro_wide, 'macro_annual_wide')

# %% [markdown]
# ### 2.3 Coverage Analysis

# %%
# Compute coverage statistics
print("\n" + "=" * 60)
print("MACRO VARIABLE COVERAGE")
print("=" * 60)

coverage = harmonize.compute_coverage_stats(df_macro_wide, ANALYSIS_MACRO, verbose=True)
quality.save_table(coverage, 'macro_coverage_annual')

# Decade coverage
decade_cov = harmonize.compute_decade_coverage(df_macro_wide, ANALYSIS_MACRO)
print("\nCoverage by Decade (%):")
print(decade_cov.round(1).to_string())
quality.save_table(decade_cov.reset_index(), 'macro_decade_coverage')

# %% [markdown]
# ## 3. Merge EFW + Macro

# %%
print("\n" + "=" * 60)
print("STEP 4: MERGE EFW + MACRO")
print("=" * 60)

df_merged, overlap_stats = merge_efw.merge_efw_macro(
    df_efw_annual, 
    df_macro_wide,
    how='outer',
    verbose=True
)

# Validate uniqueness
quality.validate_unique_keys(df_merged, ['iso3', 'year'])

# Save merged panel
merge_efw.save_merged_panel(df_merged, 'annual_merged')

# %% [markdown]
# ### 3.1 Sample Definition Dashboard

# %%
print("\n" + "=" * 60)
print("SAMPLE DEFINITION DASHBOARD")
print("=" * 60)

samples = merge_efw.create_analysis_samples(df_merged, CORE_MACRO, verbose=True)
quality.print_sample_dashboard(samples)

# Use EFW + any macro as primary sample
df_analysis = samples['efw_any_macro'].copy()
print(f"\n→ Primary analysis sample: {len(df_analysis):,} observations")

# %% [markdown]
# ## 4. Descriptive Statistics

# %%
from src.merge_efw import EFW_COL, AREA_COLS, AREA_LABELS

# Define analysis columns
efw_cols = [EFW_COL] + [c for c in AREA_COLS if c in df_analysis.columns]
macro_cols_present = [c for c in ANALYSIS_MACRO if c in df_analysis.columns]
all_analysis_cols = efw_cols + macro_cols_present

# Compute descriptive stats
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

desc = df_analysis[all_analysis_cols].describe().T
desc['missing%'] = df_analysis[all_analysis_cols].isnull().mean() * 100
print(desc.round(2).to_string())
quality.save_table(desc.reset_index(), 'descriptive_stats_annual')

# %% [markdown]
# ## 5. Time Trends
# ### 5.1 EFW Trends

# %%
print("\n" + "=" * 60)
print("TIME TRENDS")
print("=" * 60)

# EFW overall trend
fig = plots.plot_time_trend(
    df_analysis, EFW_COL, 
    title=f'Economic Freedom Trends ({START_YEAR}-Present)',
    sample_label=f'N={len(df_analysis):,}, {df_analysis["iso3"].nunique()} countries'
)
plots.savefig(fig, 'annual_efw_trend.png')
plt.show()

# %%
# EFW area trends
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

area_cols_present = [c for c in AREA_COLS if c in df_analysis.columns]
for i, (col, label) in enumerate(zip(area_cols_present, AREA_LABELS)):
    ax = axes[i]
    yearly = df_analysis.groupby('year')[col].agg(['mean', 'median'])
    ax.plot(yearly.index, yearly['mean'], 'o-', label='Mean')
    ax.plot(yearly.index, yearly['median'], 's--', label='Median')
    ax.set_title(label)
    ax.set_ylim(4, 9)
    ax.legend(fontsize=8)

axes[-1].axis('off')  # Hide extra subplot
plt.suptitle(f'EFW Area Trends ({START_YEAR}-Present)', fontsize=12)
plt.tight_layout()
plots.savefig(plt.gcf(), 'annual_efw_areas.png')
plt.show()

# %% [markdown]
# ### 5.2 Macro Trends

# %%
# Key macro trends
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

macro_labels = {
    'gdppc_filled': 'GDP per capita (WDI/PWT scaled)',
    'NY.GDP.MKTP.KD.ZG': 'GDP Growth (%)',
    'FP.CPI.TOTL.ZG': 'Inflation CPI (%)',
    'trade_openness_filled': 'Trade (% GDP)',
    'investment_gfcf_filled': 'Investment (% GDP)',
    'govt_consumption_filled': 'Govt Consumption (% GDP)'
}

for i, col in enumerate(CORE_MACRO):
    if col not in df_analysis.columns:
        continue
    ax = axes[i]
    yearly = df_analysis.groupby('year')[col].agg(['mean', 'median'])
    ax.plot(yearly.index, yearly['mean'], 'o-', color='#3498db', label='Mean')
    ax.plot(yearly.index, yearly['median'], 's--', color='#27ae60', label='Median')
    ax.set_title(macro_labels.get(col, col[:20]))
    ax.legend(fontsize=8)

plt.suptitle(f'Macro Indicator Trends ({START_YEAR}-Present)', fontsize=12)
plt.tight_layout()
plots.savefig(plt.gcf(), 'annual_macro_trends.png')
plt.show()

# %% [markdown]
# ## 6. Correlations and Associations
# ### 6.1 Correlation Matrix

# %%
print("\n" + "=" * 60)
print("CORRELATIONS")
print("=" * 60)

# Create log GDP per capita
if 'gdppc_filled' in df_analysis.columns:
    df_analysis['log_gdppc'] = np.log(df_analysis['gdppc_filled'].replace(0, np.nan))

# Correlation matrix
corr_cols = [EFW_COL, 'log_gdppc', 'NY.GDP.MKTP.KD.ZG', 'FP.CPI.TOTL.ZG',
             'trade_openness_filled', 'investment_gfcf_filled']
corr_cols = [c for c in corr_cols if c in df_analysis.columns]

fig = plots.plot_correlation_matrix(
    df_analysis, corr_cols,
    title=f'Correlation Matrix: EFW and Macro ({START_YEAR}-Present)'
)
plots.savefig(fig, 'annual_correlation_matrix.png')
plt.show()

# Print correlations with EFW
print("\nCorrelations with EFW Overall:")
for col in corr_cols:
    if col != EFW_COL and col in df_analysis.columns:
        corr = df_analysis[[EFW_COL, col]].dropna().corr().iloc[0, 1]
        print(f"  {col[:30]:30s}: {corr:+.3f}")

# %% [markdown]
# ### 6.2 Key Scatter Plots

# %%
# EFW vs Log GDP per capita
if 'log_gdppc' in df_analysis.columns:
    fig = plots.plot_scatter(
        df_analysis, 'log_gdppc', EFW_COL,
        title='Economic Freedom vs Income Level',
        add_trend=True
    )
    plt.xlabel('Log GDP per capita (constant USD)')
    plt.ylabel('EFW Overall Score')
    plots.savefig(fig, 'annual_scatter_efw_gdp.png')
    plt.show()

# %%
# EFW vs Inflation
if 'FP.CPI.TOTL.ZG' in df_analysis.columns:
    # Cap extreme inflation for visualization
    df_plot = df_analysis.copy()
    df_plot['inflation_capped'] = df_plot['FP.CPI.TOTL.ZG'].clip(-10, 100)
    
    fig = plots.plot_scatter(
        df_plot, 'inflation_capped', EFW_COL,
        title='Economic Freedom vs Inflation',
        add_trend=True
    )
    plt.xlabel('Inflation CPI (%, capped at 100)')
    plt.ylabel('EFW Overall Score')
    plots.savefig(fig, 'annual_scatter_efw_inflation.png')
    plt.show()

# %%
# Trade area vs Trade openness
trade_area = 'Area 4 Freedom to trade internationally'
if trade_area in df_analysis.columns and 'trade_openness_filled' in df_analysis.columns:
    fig = plots.plot_scatter(
        df_analysis, 'trade_openness_filled', trade_area,
        title='EFW Trade Freedom vs Macro Trade Openness',
        add_trend=True
    )
    plt.xlabel('Trade (% of GDP)')
    plt.ylabel('EFW Trade Freedom Area')
    plots.savefig(fig, 'annual_scatter_trade.png')
    plt.show()

# %% [markdown]
# ## 7. Within-Country Changes

# %%
print("\n" + "=" * 60)
print("WITHIN-COUNTRY CHANGES")
print("=" * 60)

# Compute 5-year changes
df_sorted = df_analysis.sort_values(['iso3', 'year'])
df_sorted['efw_lag5'] = df_sorted.groupby('iso3')[EFW_COL].shift(5)
df_sorted['efw_change5'] = df_sorted[EFW_COL] - df_sorted['efw_lag5']

if 'NY.GDP.MKTP.KD.ZG' in df_sorted.columns:
    # Average growth over 5 years
    df_sorted['growth_avg5'] = df_sorted.groupby('iso3')['NY.GDP.MKTP.KD.ZG'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )

# Filter to rows with valid changes
df_changes = df_sorted[df_sorted['efw_change5'].notna()].copy()
print(f"Observations with 5-year EFW changes: {len(df_changes):,}")

# %%
# Plot: EFW change vs GDP growth
if 'growth_avg5' in df_changes.columns:
    fig = plots.plot_scatter(
        df_changes, 'growth_avg5', 'efw_change5',
        title='5-Year EFW Change vs Average GDP Growth',
        add_trend=True
    )
    plt.xlabel('Average GDP Growth (%, 5-year)')
    plt.ylabel('5-Year Change in EFW')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plots.savefig(fig, 'annual_change_efw_growth.png')
    plt.show()
    
    # Correlation
    corr = df_changes[['efw_change5', 'growth_avg5']].dropna().corr().iloc[0, 1]
    print(f"Correlation (ΔEFW vs Avg Growth): {corr:+.3f}")

# %% [markdown]
# ## 8. Outliers and Extreme Regimes

# %%
print("\n" + "=" * 60)
print("OUTLIERS AND EXTREME REGIMES")
print("=" * 60)

# High inflation episodes
if 'FP.CPI.TOTL.ZG' in df_analysis.columns:
    high_inflation = df_analysis[df_analysis['FP.CPI.TOTL.ZG'] > 50].copy()
    if len(high_inflation) > 0:
        print(f"\nHigh inflation episodes (CPI > 50%): {len(high_inflation)}")
        high_inf_summary = high_inflation.groupby('iso3').agg({
            'year': ['min', 'max', 'count'],
            'FP.CPI.TOTL.ZG': 'max',
            EFW_COL: 'mean'
        })
        high_inf_summary.columns = ['year_min', 'year_max', 'n_years', 'max_inflation', 'mean_efw']
        print(high_inf_summary.sort_values('max_inflation', ascending=False).head(10))
        quality.save_table(high_inf_summary.reset_index(), 'high_inflation_episodes')

# %%
# Growth collapses
if 'NY.GDP.MKTP.KD.ZG' in df_analysis.columns:
    growth_collapse = df_analysis[df_analysis['NY.GDP.MKTP.KD.ZG'] < -10].copy()
    if len(growth_collapse) > 0:
        print(f"\nGrowth collapses (GDP growth < -10%): {len(growth_collapse)}")
        collapse_summary = growth_collapse.groupby('iso3').agg({
            'year': 'count',
            'NY.GDP.MKTP.KD.ZG': 'min',
            EFW_COL: 'mean'
        })
        collapse_summary.columns = ['n_episodes', 'worst_growth', 'mean_efw']
        print(collapse_summary.sort_values('worst_growth').head(10))
        quality.save_table(collapse_summary.reset_index(), 'growth_collapse_episodes')

# %% [markdown]
# ## 9. Missingness Dashboard

# %%
# Missingness by variable
print("\n" + "=" * 60)
print("MISSINGNESS ANALYSIS")
print("=" * 60)

miss_table = quality.compute_missingness_table(df_merged, all_analysis_cols)
print(miss_table.to_string(index=False))
quality.save_table(miss_table, 'missingness_annual')

# %%
# Missingness heatmap
fig = plots.plot_missingness_heatmap(
    df_merged, 
    CORE_MACRO,
    title=f'Macro Variable Missingness by Year ({START_YEAR}-Present)'
)
plots.savefig(fig, 'annual_missingness_heatmap.png')
plt.show()

# %% [markdown]
# ## 10. Summary

# %%
print("\n" + "=" * 70)
print("ANNUAL EDA COMPLETE")
print("=" * 70)

efw_gdppc_corr = df_analysis[[EFW_COL, 'log_gdppc']].dropna().corr().iloc[0,1] if 'log_gdppc' in df_analysis.columns else float('nan')

print(f"""
Analysis Period: {START_YEAR}-{df_analysis['year'].max()}
Primary Sample: {len(df_analysis):,} country-year observations
Countries: {df_analysis['iso3'].nunique()}

Key Findings:
- EFW overall mean: {df_analysis[EFW_COL].mean():.2f} (SD: {df_analysis[EFW_COL].std():.2f})
- Correlation EFW vs Log GDP/capita: {efw_gdppc_corr:.3f}

Outputs saved to:
- Figures: outputs/figures/
- Tables: outputs/tables/
""")
