# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (dml)
#     language: python
#     name: dml
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.11.14
# ---

# %% [markdown]
# # 03. Build Quinquennial Panel
#
# This notebook merges Fraser EFW data with World Bank outcomes to create 
# the master quinquennial panel for analysis.
#
# **Inputs:**
# - `data/02_intermediate/efw_panel_raw.parquet`
# - `data/02_intermediate/worldbank_raw.parquet`
#
# **Outputs:**
# - `data/03_clean/panel_quinquennial.parquet`
#
# **Key steps:**
# 1. Load and merge EFW and World Bank data
# 2. Validate merge quality
# 3. Create predetermined controls (lagged values)
# 4. Create region/income group indicators
# 5. Final validation and output

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in dir() else Path.cwd().parent
DATA_INTERMEDIATE = PROJECT_ROOT / 'data/02_intermediate'
DATA_CLEAN = PROJECT_ROOT / 'data/03_clean'

# Ensure output directory exists
DATA_CLEAN.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Load Input Data

# %%
# Load EFW data
efw_path = DATA_INTERMEDIATE / 'efw_panel_raw.parquet'
assert efw_path.exists(), f"EFW data not found: {efw_path}"
df_efw = pd.read_parquet(efw_path)
print(f"EFW data: {len(df_efw)} observations, {df_efw['iso3c'].nunique()} countries")

# Load World Bank data
wb_path = DATA_INTERMEDIATE / 'worldbank_raw.parquet'
assert wb_path.exists(), f"World Bank data not found: {wb_path}"
df_wb = pd.read_parquet(wb_path)
print(f"World Bank data: {len(df_wb)} observations, {df_wb['iso3c'].nunique()} countries")

# Check year overlap
efw_years = set(df_efw['year'].unique())
wb_years = set(df_wb['year'].unique())
common_years = sorted(efw_years & wb_years)
print(f"\nCommon quinquennial years: {common_years}")

# %% [markdown]
# ## 2. Merge Datasets

# %%
# Select columns from each dataset
efw_cols = [
    'iso3c', 'country_name', 'year',
    'efw_aggregate', 'efw_area1', 'efw_area2', 'efw_area3', 'efw_area4', 'efw_area5',
    'd_efw_aggregate', 'd_efw_area1', 'd_efw_area2', 'd_efw_area3', 'd_efw_area4', 'd_efw_area5',
    'efw_sd', 'wb_region', 'wb_income'
]

wb_cols = [
    'iso3c', 'year',
    'gdp_pc_constant', 'gdp_pc_ppp', 'population',
    'ln_gdp_pc', 'growth'
]

# Filter to desired columns (check which exist)
efw_cols_exist = [c for c in efw_cols if c in df_efw.columns]
wb_cols_exist = [c for c in wb_cols if c in df_wb.columns]

df_efw_subset = df_efw[efw_cols_exist].copy()
df_wb_subset = df_wb[wb_cols_exist].copy()

# Merge on country-year
df_panel = df_efw_subset.merge(
    df_wb_subset,
    on=['iso3c', 'year'],
    how='left',
    validate='one_to_one'
)

print(f"Merged panel: {len(df_panel)} observations")
print(f"Countries: {df_panel['iso3c'].nunique()}")

# ASSERTION: No duplicate keys after merge
assert not df_panel.duplicated(subset=['iso3c', 'year']).any(), \
    "CRITICAL: Duplicate country-year keys after merge!"

# %% [markdown]
# ## 3. Merge Quality Assessment

# %%
# Check merge success rates
n_total = len(df_panel)
n_gdp = df_panel['gdp_pc_constant'].notna().sum()
n_growth = df_panel['growth'].notna().sum()

print("Merge quality:")
print(f"  Total observations: {n_total}")
print(f"  With GDP per capita: {n_gdp} ({100*n_gdp/n_total:.1f}%)")
print(f"  With growth: {n_growth} ({100*n_growth/n_total:.1f}%)")

# Coverage by year
print("\nMerge by year:")
for year in sorted(df_panel['year'].unique()):
    yr_data = df_panel[df_panel['year'] == year]
    n_efw = len(yr_data)
    n_gdp_yr = yr_data['gdp_pc_constant'].notna().sum()
    print(f"  {year}: {n_efw} countries with EFW, {n_gdp_yr} with GDP ({100*n_gdp_yr/n_efw:.0f}%)")

# Identify unmatched countries
unmatched = df_panel[df_panel['gdp_pc_constant'].isna()]['iso3c'].unique()
if len(unmatched) > 0:
    print(f"\nCountries without World Bank match (sample): {list(unmatched[:10])}")

# %% [markdown]
# ## 4. Create Predetermined Controls (Lagged Variables)
#
# **CRITICAL TIMING RULE:** Controls must be predetermined (lagged) to avoid
# post-treatment bias. We use $t-1$ values (prior quinquennial period).

# %%
# Sort for lagging
df_panel = df_panel.sort_values(['iso3c', 'year'])

# Lagged outcome variables (predetermined)
df_panel['ln_gdp_pc_lag1'] = df_panel.groupby('iso3c')['ln_gdp_pc'].shift(1)
df_panel['growth_lag1'] = df_panel.groupby('iso3c')['growth'].shift(1)

# Lagged EFW levels (predetermined)
df_panel['efw_aggregate_lag1'] = df_panel.groupby('iso3c')['efw_aggregate'].shift(1)

# Lagged EFW areas (for bundle analysis)
for area in range(1, 6):
    col = f'efw_area{area}'
    df_panel[f'{col}_lag1'] = df_panel.groupby('iso3c')[col].shift(1)

# Verify lags are properly computed
df_panel['year_lag1'] = df_panel.groupby('iso3c')['year'].shift(1)
df_panel['lag_diff'] = df_panel['year'] - df_panel['year_lag1']

non_missing_lags = df_panel['lag_diff'].dropna()
assert (non_missing_lags == 5).all(), "Lag differences are not all 5 years"
print("✓ All lagged variables use proper 5-year lags")

# Clean up
df_panel = df_panel.drop(columns=['year_lag1', 'lag_diff'])

# %% [markdown]
# ## 5. Create Lead Variables (for Maintenance Rule Classification ONLY)
#
# **CRITICAL:** These leads are used ONLY for classifying sustained reforms.
# They must NEVER be used as controls or in outcome construction.

# %%
# Lead EFW values for maintenance rule classification
df_panel['efw_aggregate_lead1'] = df_panel.groupby('iso3c')['efw_aggregate'].shift(-1)
df_panel['efw_aggregate_lead2'] = df_panel.groupby('iso3c')['efw_aggregate'].shift(-2)

print("✓ Created lead variables for maintenance rule classification")
print("  WARNING: These are for classification ONLY, not for controls/outcomes")

# %% [markdown]
# ## 6. Create Region and Income Indicators

# %%
# Clean region and income classification
df_panel['region'] = df_panel['wb_region'].fillna('Unknown')
df_panel['income_group'] = df_panel['wb_income'].fillna('Unknown')

# Create region dummies for region×time FE
regions = df_panel['region'].unique()
print(f"Regions: {list(regions)}")

# Income group summary
print("\nIncome group distribution:")
print(df_panel.groupby('income_group')['iso3c'].nunique())

# %% [markdown]
# ## 7. Create Initial Condition Variables (for State Dependence)

# %%
# Initial log GDP per capita (first available observation per country)
initial_gdp = df_panel.groupby('iso3c')['ln_gdp_pc'].transform('first')
df_panel['ln_gdp_pc_initial'] = initial_gdp

# Initial EFW (first available observation per country)
initial_efw = df_panel.groupby('iso3c')['efw_aggregate'].transform('first')
df_panel['efw_initial'] = initial_efw

# Create quartile bins for state dependence analysis
df_panel['gdp_quartile'] = pd.qcut(
    df_panel['ln_gdp_pc_lag1'].dropna(), 
    q=4, 
    labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
).reindex(df_panel.index)

df_panel['efw_quartile'] = pd.qcut(
    df_panel['efw_aggregate_lag1'].dropna(),
    q=4,
    labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
).reindex(df_panel.index)

print("✓ Created initial condition and quartile variables for state dependence")

# %% [markdown]
# ## 8. Summary Statistics

# %%
# Key variable summary
key_vars = [
    'efw_aggregate', 'd_efw_aggregate',
    'ln_gdp_pc', 'growth',
    'gdp_pc_constant', 'population'
]

existing_vars = [v for v in key_vars if v in df_panel.columns]

print("Summary Statistics - Quinquennial Panel")
print("="*60)
print(df_panel[existing_vars].describe().round(3))

# EFW areas summary
print("\nEFW Areas Summary:")
efw_area_vars = [f'efw_area{i}' for i in range(1, 6)]
print(df_panel[efw_area_vars].describe().round(2))

# Change variables summary
print("\nEFW Change Variables (Δ):")
d_efw_vars = [f'd_efw_area{i}' for i in range(1, 6)] + ['d_efw_aggregate']
existing_d_vars = [v for v in d_efw_vars if v in df_panel.columns]
print(df_panel[existing_d_vars].describe().round(3))

# %% [markdown]
# ## 9. Final Validation

# %%
print("\n" + "="*60)
print("FINAL VALIDATION CHECKS")
print("="*60)

# 1. No duplicate keys
assert not df_panel.duplicated(subset=['iso3c', 'year']).any(), "Duplicate keys found"
print("✓ No duplicate country-year keys")

# 2. All quinquennial years
QUINQUENNIAL_YEARS = [1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
assert df_panel['year'].isin(QUINQUENNIAL_YEARS).all(), "Non-quinquennial years found"
print("✓ All years are quinquennial")

# 3. Sufficient panel size
n_obs = len(df_panel)
n_countries = df_panel['iso3c'].nunique()
n_years = df_panel['year'].nunique()
assert n_obs > 1000, "Too few observations"
assert n_countries > 100, "Too few countries"
print(f"✓ Panel size: {n_obs} obs, {n_countries} countries, {n_years} years")

# 4. Outcome coverage
n_with_outcome = df_panel['growth'].notna().sum()
assert n_with_outcome > 800, "Too few observations with growth data"
print(f"✓ Observations with growth outcome: {n_with_outcome}")

# 5. EFW change coverage
n_with_change = df_panel['d_efw_aggregate'].notna().sum()
print(f"✓ Observations with EFW changes: {n_with_change}")

# %% [markdown]
# ## 10. Save Output

# %%
# Save final panel
output_path = DATA_CLEAN / 'panel_quinquennial.parquet'
df_panel.to_parquet(output_path, index=False)
print(f"\n✓ Saved quinquennial panel to {output_path}")
print(f"  Observations: {len(df_panel)}")
print(f"  Countries: {df_panel['iso3c'].nunique()}")
print(f"  Variables: {len(df_panel.columns)}")

# Save variable documentation
var_docs = {
    'identifiers': ['iso3c', 'country_name', 'year'],
    'efw_levels': ['efw_aggregate'] + [f'efw_area{i}' for i in range(1, 6)] + ['efw_sd'],
    'efw_changes': ['d_efw_aggregate'] + [f'd_efw_area{i}' for i in range(1, 6)],
    'outcomes': ['gdp_pc_constant', 'gdp_pc_ppp', 'ln_gdp_pc', 'growth', 'population'],
    'lagged_controls': ['ln_gdp_pc_lag1', 'growth_lag1', 'efw_aggregate_lag1'] + 
                       [f'efw_area{i}_lag1' for i in range(1, 6)],
    'leads_for_classification': ['efw_aggregate_lead1', 'efw_aggregate_lead2'],
    'state_dependence': ['ln_gdp_pc_initial', 'efw_initial', 'gdp_quartile', 'efw_quartile'],
    'groups': ['region', 'income_group', 'wb_region', 'wb_income'],
}

doc_path = DATA_CLEAN / 'panel_documentation.json'
with open(doc_path, 'w') as f:
    json.dump(var_docs, f, indent=2)
print(f"✓ Saved variable documentation to {doc_path}")

# %% [markdown]
# ## 11. Column Summary

# %%
print("\n" + "="*60)
print("COLUMN SUMMARY")
print("="*60)
print(f"\nTotal columns: {len(df_panel.columns)}")
for category, cols in var_docs.items():
    existing = [c for c in cols if c in df_panel.columns]
    print(f"\n{category}:")
    for col in existing:
        n_valid = df_panel[col].notna().sum() if df_panel[col].dtype != 'object' else len(df_panel[df_panel[col].notna()])
        print(f"  {col}: {n_valid} non-missing")

print("\n" + "="*60)
print("03_BUILD_QUINQUENNIAL_PANEL COMPLETE")
print("="*60)
