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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01. Ingest Fraser EFW Data
#
# This notebook ingests the Fraser Institute Economic Freedom of the World (EFW) 
# data from the local Excel file `fraser.xlsx`.
#
# **Inputs:** `data/01_raw/fraser.xlsx`
# **Outputs:** `data/02_intermediate/efw_panel_raw.parquet`
#
# **Key steps:**
# 1. Load raw EFW panel data
# 2. Standardize column names and ISO codes
# 3. Filter to quinquennial years
# 4. Validate data integrity
# 5. Save processed panel

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in dir() else Path.cwd().parent
DATA_RAW = PROJECT_ROOT / 'data/01_raw'
DATA_INTERMEDIATE = PROJECT_ROOT / 'data/02_intermediate'

print(f"Project root: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Load Raw Fraser Data

# %%
fraser_path = DATA_RAW / 'fraser.xlsx'
assert fraser_path.exists(), f"Fraser data not found: {fraser_path}"

# Read the panel dataset sheet
df_raw = pd.read_excel(fraser_path, sheet_name='EFW Panel Dataset')

print(f"Raw data shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print(f"\nYear range: {df_raw['Year'].min()} - {df_raw['Year'].max()}")
print(f"Countries: {df_raw['ISO_Code'].nunique()}")

# %% [markdown]
# ## 2. Standardize Column Names

# %%
# Rename columns for consistency
column_mapping = {
    'ISO_Code': 'iso3c',
    'Countries': 'country_name',
    'Year': 'year',
    'Summary': 'efw_aggregate',
    'Area 1': 'efw_area1',  # Size of Government
    'Area 2': 'efw_area2',  # Legal System & Property Rights
    'Area 3': 'efw_area3',  # Sound Money
    'Area 4': 'efw_area4',  # Freedom to Trade Internationally
    'Area 5': 'efw_area5',  # Regulation
    'Standard Deviation of the 5 EFW Areas': 'efw_sd',
    'World Bank Region': 'wb_region',
    'World Bank Current Income Classification, 1990-Present': 'wb_income',
}

df = df_raw.rename(columns=column_mapping)

# Area descriptions for reference
EFW_AREAS = {
    'efw_area1': 'Size of Government',
    'efw_area2': 'Legal System & Property Rights',
    'efw_area3': 'Sound Money',
    'efw_area4': 'Freedom to Trade Internationally',
    'efw_area5': 'Regulation',
}

print("Standardized columns:")
for col in df.columns:
    print(f"  - {col}")

# %% [markdown]
# ## 3. Data Quality Checks
#
# **CRITICAL:** Check for hidden missing values (spaces, empty strings, "NA", etc.)

# %%
# === COMPREHENSIVE DATA QUALITY CHECKS ===

print("="*60)
print("COMPREHENSIVE DATA QUALITY CHECKS")
print("="*60)

# 3.1 Check data types
print("\n3.1 Data Types:")
print(df.dtypes)

# 3.2 Check for hidden missing values in string columns
print("\n3.2 Checking for Hidden Missing Values...")

string_cols = ['iso3c', 'country_name', 'wb_region', 'wb_income']
for col in string_cols:
    if col in df.columns:
        # Count various encodings of "missing"
        empty_string = (df[col] == '').sum()
        just_spaces = df[col].astype(str).str.strip().eq('').sum() - empty_string
        na_strings = df[col].astype(str).str.lower().isin(['na', 'n/a', 'nan', 'null', '.', '-', '--']).sum()
        actual_nan = df[col].isna().sum()
        
        print(f"  {col}:")
        print(f"    - NaN/None: {actual_nan}")
        print(f"    - Empty string: {empty_string}")
        print(f"    - Whitespace only: {just_spaces}")
        print(f"    - NA-like strings: {na_strings}")

# 3.3 Check numeric columns for hidden issues
print("\n3.3 Checking Numeric Columns for Issues...")

numeric_cols = ['efw_aggregate', 'efw_area1', 'efw_area2', 'efw_area3', 'efw_area4', 'efw_area5', 'efw_sd']
for col in numeric_cols:
    if col in df.columns:
        # Check if any values were read as strings
        non_numeric = 0
        try:
            # Convert to numeric, marking invalid as NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            converted = pd.to_numeric(df[col], errors='coerce')
            non_numeric = (converted.isna() & df[col].notna()).sum() if col in df_raw.columns else 0
        except:
            non_numeric = 0
        
        actual_nan = df[col].isna().sum()
        
        # Check for sentinel values that might indicate missing
        if df[col].dtype in ['float64', 'int64']:
            zeros = (df[col] == 0).sum()
            negative = (df[col] < 0).sum()
            above_10 = (df[col] > 10).sum()
            
            print(f"  {col}:")
            print(f"    - NaN/None: {actual_nan}")
            print(f"    - Zeros (possible sentinel): {zeros}")
            print(f"    - Negative values: {negative}")
            print(f"    - Values > 10: {above_10}")

# 3.4 Check ISO codes
print("\n3.4 ISO Code Validation:")
missing_iso = df['iso3c'].isna().sum()
print(f"  Missing ISO codes: {missing_iso}")

# Check for invalid ISO codes (should be 3 uppercase letters)
valid_iso_pattern = df['iso3c'].str.match(r'^[A-Z]{3}$', na=False)
invalid_iso = (~valid_iso_pattern).sum()
if invalid_iso > 0:
    bad_isos = df[~valid_iso_pattern]['iso3c'].unique()[:10]
    print(f"  ⚠ Invalid ISO codes ({invalid_iso}): {list(bad_isos)}")
else:
    print(f"  ✓ All ISO codes are valid 3-letter codes")

# 3.5 Check for duplicates
print("\n3.5 Duplicate Check:")
df['country_year'] = df['iso3c'].astype(str) + '_' + df['year'].astype(str)
duplicates = df.duplicated(subset=['iso3c', 'year'], keep=False)
print(f"  Duplicate country-year entries: {duplicates.sum()}")

# ASSERTION: No duplicate country-year keys
assert not df.duplicated(subset=['iso3c', 'year']).any(), \
    "CRITICAL: Duplicate country-year keys found!"

# 3.6 EFW value ranges
print("\n3.6 EFW Value Ranges (should be 0-10):")
efw_cols = ['efw_aggregate', 'efw_area1', 'efw_area2', 'efw_area3', 'efw_area4', 'efw_area5']
for col in efw_cols:
    valid = df[col].dropna()
    if len(valid) > 0:
        print(f"  {col}: min={valid.min():.2f}, max={valid.max():.2f}, missing={df[col].isna().sum()}, n={len(valid)}")
        # ASSERTION: EFW values should be in [0, 10] range (with some tolerance)
        assert valid.min() >= 0, f"{col} has negative values"
        assert valid.max() <= 11, f"{col} has values > 11"  # Allow small tolerance
    else:
        print(f"  {col}: ALL MISSING!")

# 3.7 Coverage summary
print("\n3.7 Data Completeness Summary:")
total_obs = len(df)
for col in efw_cols:
    n_valid = df[col].notna().sum()
    pct = 100 * n_valid / total_obs
    print(f"  {col}: {n_valid}/{total_obs} ({pct:.1f}%)")

print("\n" + "="*60)

# %%
# Coverage by year - ACTUAL DATA AVAILABILITY (not just row counts!)
print("\n" + "="*60)
print("ACTUAL EFW DATA AVAILABILITY BY YEAR")
print("(Row count ≠ data completeness!)")
print("="*60)

print("\nEFW Aggregate availability:")
for year in sorted(df['year'].unique()):
    if year % 5 == 0:  # Only quinquennial
        year_data = df[df['year'] == year]
        n_total = len(year_data)
        n_valid = year_data['efw_aggregate'].notna().sum()
        pct = 100 * n_valid / n_total
        status = "✓" if pct >= 90 else "⚠" if pct >= 70 else "✗"
        print(f"  {status} {year}: {n_valid:3d}/{n_total:3d} ({pct:5.1f}%) countries with actual EFW data")

print("\n⚠ WARNING: Early years (1970-1995) have significant missing data!")
print("  Analysis should account for this unbalanced panel.")

# %% [markdown]
# ## 4. Filter to Quinquennial Grid

# %%
# Define quinquennial years
QUINQUENNIAL_YEARS = [1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

# Filter to quinquennial years only
df_quin = df[df['year'].isin(QUINQUENNIAL_YEARS)].copy()

print(f"Original observations: {len(df)}")
print(f"Quinquennial observations: {len(df_quin)}")
print(f"Quinquennial years: {sorted(df_quin['year'].unique())}")

# Coverage check - ACTUAL DATA, not just row counts
print(f"\nQuinquennial EFW DATA availability (NOT row counts!):")
print("-" * 50)
for year in QUINQUENNIAL_YEARS:
    year_data = df_quin[df_quin['year'] == year]
    n_rows = len(year_data)
    n_with_efw = year_data['efw_aggregate'].notna().sum()
    pct = 100 * n_with_efw / n_rows if n_rows > 0 else 0
    status = "✓" if pct >= 90 else "⚠" if pct >= 70 else "✗"
    print(f"  {status} {year}: {n_with_efw:3d} countries with EFW data (out of {n_rows} rows, {pct:.1f}%)")

# Summary stats
total_rows = len(df_quin)
total_with_efw = df_quin['efw_aggregate'].notna().sum()
print(f"\nSUMMARY:")
print(f"  Total rows in quinquennial panel: {total_rows}")
print(f"  Rows WITH EFW aggregate data: {total_with_efw} ({100*total_with_efw/total_rows:.1f}%)")
print(f"  Rows MISSING EFW aggregate: {total_rows - total_with_efw} ({100*(total_rows-total_with_efw)/total_rows:.1f}%)")

# %% [markdown]
# ## 5. Calculate Quinquennial Changes

# %%
# Sort for proper differencing
df_quin = df_quin.sort_values(['iso3c', 'year'])

# Calculate changes for all EFW variables
change_cols = []
for col in efw_cols:
    change_col = f'd_{col}'
    df_quin[change_col] = df_quin.groupby('iso3c')[col].diff()
    change_cols.append(change_col)

# Verify timing: changes should be quinquennial (5-year spans)
df_quin['year_lag'] = df_quin.groupby('iso3c')['year'].shift(1)
df_quin['year_diff'] = df_quin['year'] - df_quin['year_lag']

# Count valid changes (5-year spans)
valid_changes = df_quin['year_diff'] == 5
print(f"\nValid 5-year changes: {valid_changes.sum()} out of {len(df_quin)} observations")
print(f"Missing changes (first obs per country): {df_quin['year_diff'].isna().sum()}")

# ASSERTION: All non-missing year differences should be 5
non_missing = df_quin['year_diff'].dropna()
assert (non_missing == 5).all(), \
    f"CRITICAL: Non-quinquennial year differences found: {non_missing[non_missing != 5].unique()}"

# Clean up
df_quin = df_quin.drop(columns=['country_year', 'year_lag', 'year_diff'])

# %% [markdown]
# ## 6. Summary Statistics

# %%
# Summary statistics for EFW levels and changes
print("EFW Summary Statistics (Quinquennial Panel):")
print("\nLevels:")
print(df_quin[efw_cols].describe().round(2))

print("\nChanges (Δ):")
print(df_quin[change_cols].describe().round(2))

# Distribution of changes
print("\nDistribution of aggregate EFW changes:")
d_efw = df_quin['d_efw_aggregate'].dropna()
print(f"  Mean: {d_efw.mean():.3f}")
print(f"  Std:  {d_efw.std():.3f}")
print(f"  Negative: {(d_efw < 0).sum()} ({100*(d_efw < 0).mean():.1f}%)")
print(f"  Positive: {(d_efw > 0).sum()} ({100*(d_efw > 0).mean():.1f}%)")
print(f"  |Δ| ≥ 1.0: {(d_efw.abs() >= 1.0).sum()}")
print(f"  |Δ| ≥ 1.5: {(d_efw.abs() >= 1.5).sum()}")
print(f"  |Δ| ≥ 2.0: {(d_efw.abs() >= 2.0).sum()}")

# %% [markdown]
# ## 7. Save Processed Data

# %%
# Ensure output directory exists
DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)

# Save as parquet for efficiency
output_path = DATA_INTERMEDIATE / 'efw_panel_raw.parquet'
df_quin.to_parquet(output_path, index=False)

print(f"\n✓ Saved quinquennial EFW panel to {output_path}")
print(f"  Observations: {len(df_quin)}")
print(f"  Countries: {df_quin['iso3c'].nunique()}")
print(f"  Years: {sorted(df_quin['year'].unique())}")

# Also save column metadata
metadata = {
    'efw_areas': EFW_AREAS,
    'level_cols': efw_cols,
    'change_cols': change_cols,
    'quinquennial_years': QUINQUENNIAL_YEARS,
    'source': 'fraser.xlsx - EFW Panel Dataset',
}

import json
metadata_path = DATA_INTERMEDIATE / 'efw_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved metadata to {metadata_path}")

# %% [markdown]
# ## 8. Final Assertions

# %%
# Final validation checks
print("\n" + "="*60)
print("FINAL VALIDATION CHECKS")
print("="*60)

# 1. No duplicate keys
assert not df_quin.duplicated(subset=['iso3c', 'year']).any(), "Duplicate keys found"
print("✓ No duplicate country-year keys")

# 2. All years are quinquennial
assert df_quin['year'].isin(QUINQUENNIAL_YEARS).all(), "Non-quinquennial years found"
print("✓ All years are quinquennial")

# 3. EFW values in valid range
for col in efw_cols:
    valid = df_quin[col].dropna()
    assert valid.between(0, 10.1).all(), f"{col} has out-of-range values"
print("✓ EFW values in valid range [0, 10]")

# 4. Reasonable number of observations
assert len(df_quin) > 1000, "Too few observations"
assert df_quin['iso3c'].nunique() > 100, "Too few countries"
print(f"✓ Panel size: {len(df_quin)} obs, {df_quin['iso3c'].nunique()} countries")

print("\n" + "="*60)
print("01_INGEST_FRASER COMPLETE")
print("="*60)
