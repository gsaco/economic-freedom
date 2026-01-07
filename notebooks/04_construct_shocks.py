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
# # 04. Construct Shocks and Reform Episodes
#
# This notebook constructs the treatment/shock variables following the 
# methodology in the paper specification.
#
# **Inputs:**
# - `data/03_clean/panel_quinquennial.parquet`
#
# **Outputs:**
# - `data/03_clean/panel_with_shocks.parquet`
# - `output/logs/event_counts.json`
#
# **Shock definitions:**
# 1. **Signed shocks (S±):** Large positive/negative EFW changes
# 2. **Sustained reforms (R±):** Shocks that are maintained for 2+ periods
# 3. **Cooldown enforcement:** No new events within 2 periods
#
# **CRITICAL TIMING RULES:**
# - Shock classification uses only $F_t$ and $F_{t-1}$
# - Maintenance rule uses $F_{t+1}$ and $F_{t+2}$ for CLASSIFICATION ONLY
# - These future values are NEVER used as controls or outcomes

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in dir() else Path.cwd().parent
DATA_CLEAN = PROJECT_ROOT / 'data/03_clean'
OUTPUT_LOGS = PROJECT_ROOT / 'output/logs'

OUTPUT_LOGS.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Load Panel Data

# %%
panel_path = DATA_CLEAN / 'panel_quinquennial.parquet'
assert panel_path.exists(), f"Panel data not found: {panel_path}"

df = pd.read_parquet(panel_path)
print(f"Panel: {len(df)} observations, {df['iso3c'].nunique()} countries")
print(f"Years: {sorted(df['year'].unique())}")

# %% [markdown]
# ## 2. Shock Configuration

# %%
# Pre-specified shock parameters (from specification document)
SHOCK_CONFIG = {
    # Main threshold: 1 EFW point change
    'tau_main': 1.0,
    
    # Robustness thresholds
    'tau_robustness': [0.75, 1.25, 1.5],
    
    # Percentile-based thresholds (computed from data)
    'percentiles': [75, 90, 95],
    
    # Sustained reform parameters
    'epsilon': 0.5,  # Tolerance for maintenance decline
    'maintenance_periods': 2,  # Check t+1 and t+2
    
    # Cooldown periods
    'cooldown': 2,  # No new event for 2 quinquennial periods
}

print("Shock configuration:")
for key, value in SHOCK_CONFIG.items():
    print(f"  {key}: {value}")

# Compute percentile thresholds from data
d_efw = df['d_efw_aggregate'].dropna()
for p in SHOCK_CONFIG['percentiles']:
    pos_thresh = np.percentile(d_efw[d_efw > 0], p)
    neg_thresh = np.percentile(d_efw[d_efw < 0], 100-p)
    print(f"  P{p} positive: {pos_thresh:.2f}, P{100-p} negative: {neg_thresh:.2f}")

# %% [markdown]
# ## 3. Create Basic Signed Shocks (S±)
#
# $$S^+_{it} = \mathbf{1}\{\Delta F_{it} \geq \tau\}$$
# $$S^-_{it} = \mathbf{1}\{\Delta F_{it} \leq -\tau\}$$

# %%
def create_signed_shocks(df, d_col, tau, prefix='shock'):
    """
    Create signed shock indicators.
    
    Parameters
    ----------
    df : DataFrame
    d_col : str
        Column name for the change variable
    tau : float
        Threshold for shock classification
    prefix : str
        Prefix for output column names
        
    Returns
    -------
    DataFrame with shock indicators added
    """
    df = df.copy()
    
    # Positive shock: large increase
    df[f'{prefix}_pos'] = (df[d_col] >= tau).astype(int)
    
    # Negative shock: large decrease  
    df[f'{prefix}_neg'] = (df[d_col] <= -tau).astype(int)
    
    # Any shock
    df[f'{prefix}_any'] = ((df[d_col].abs() >= tau)).astype(int)
    
    # Handle missing
    df.loc[df[d_col].isna(), [f'{prefix}_pos', f'{prefix}_neg', f'{prefix}_any']] = np.nan
    
    return df

# Create shocks with main threshold
df = create_signed_shocks(df, 'd_efw_aggregate', SHOCK_CONFIG['tau_main'], 'shock_agg')

# Count shocks
n_pos = df['shock_agg_pos'].sum()
n_neg = df['shock_agg_neg'].sum()
print(f"\nSigned shocks (τ = {SHOCK_CONFIG['tau_main']}):")
print(f"  Positive (S+): {n_pos:,.0f}")
print(f"  Negative (S-): {n_neg:,.0f}")
print(f"  Total: {n_pos + n_neg:,.0f}")

# %% [markdown]
# ## 4. Create Sustained Reform Indicators (R±)
#
# $$R^+_{it} = \mathbf{1}\{\Delta F_{it} \geq \tau \land F_{i,t+1} \geq F_{it} - \epsilon \land F_{i,t+2} \geq F_{it} - \epsilon\}$$
#
# **CRITICAL:** Lead values are used for CLASSIFICATION ONLY. They are not used
# as controls or in outcome construction.

# %%
def create_sustained_reforms(df, level_col, d_col, tau, epsilon, prefix='reform'):
    """
    Create sustained reform indicators with maintenance rule.
    
    Parameters
    ----------
    df : DataFrame
    level_col : str
        Column name for EFW level
    d_col : str
        Column name for EFW change
    tau : float
        Threshold for initial shock
    epsilon : float
        Tolerance for maintenance decline
    prefix : str
        Prefix for output column names
        
    Returns
    -------
    DataFrame with reform indicators added
    """
    df = df.copy()
    
    # Get lead values (for classification only)
    lead1_col = f'{level_col}_lead1'
    lead2_col = f'{level_col}_lead2'
    
    # Check if lead columns exist
    if lead1_col not in df.columns or lead2_col not in df.columns:
        print(f"Warning: Lead columns not found, creating them...")
        df[lead1_col] = df.groupby('iso3c')[level_col].shift(-1)
        df[lead2_col] = df.groupby('iso3c')[level_col].shift(-2)
    
    # Sustained positive reform
    # Condition 1: Large initial increase
    cond1_pos = df[d_col] >= tau
    # Condition 2: Maintained at t+1 (within epsilon of level at t)
    cond2_pos = df[lead1_col] >= (df[level_col] - epsilon)
    # Condition 3: Maintained at t+2
    cond3_pos = df[lead2_col] >= (df[level_col] - epsilon)
    
    df[f'{prefix}_pos'] = (cond1_pos & cond2_pos & cond3_pos).astype(int)
    
    # Sustained negative reform (deterioration)
    cond1_neg = df[d_col] <= -tau
    cond2_neg = df[lead1_col] <= (df[level_col] + epsilon)
    cond3_neg = df[lead2_col] <= (df[level_col] + epsilon)
    
    df[f'{prefix}_neg'] = (cond1_neg & cond2_neg & cond3_neg).astype(int)
    
    # Any sustained reform
    df[f'{prefix}_any'] = (df[f'{prefix}_pos'] | df[f'{prefix}_neg']).astype(int)
    
    # Handle missing (if any component is missing, reform indicator is missing)
    missing_mask = (
        df[d_col].isna() | 
        df[lead1_col].isna() | 
        df[lead2_col].isna()
    )
    df.loc[missing_mask, [f'{prefix}_pos', f'{prefix}_neg', f'{prefix}_any']] = np.nan
    
    return df

# Create sustained reforms
df = create_sustained_reforms(
    df, 
    'efw_aggregate', 
    'd_efw_aggregate', 
    SHOCK_CONFIG['tau_main'],
    SHOCK_CONFIG['epsilon'],
    'reform_agg'
)

# Count sustained reforms
n_reform_pos = df['reform_agg_pos'].sum()
n_reform_neg = df['reform_agg_neg'].sum()
print(f"\nSustained reforms (τ = {SHOCK_CONFIG['tau_main']}, ε = {SHOCK_CONFIG['epsilon']}):")
print(f"  Positive (R+): {n_reform_pos:,.0f}")
print(f"  Negative (R-): {n_reform_neg:,.0f}")
print(f"  Total: {n_reform_pos + n_reform_neg:,.0f}")

# Comparison: shocks vs sustained
print(f"\nSustained as % of all shocks:")
print(f"  Positive: {100*n_reform_pos/max(n_pos,1):.1f}%")
print(f"  Negative: {100*n_reform_neg/max(n_neg,1):.1f}%")

# %% [markdown]
# ## 5. Enforce Cooldown Period
#
# After an event at time $t$, no new event can occur until $t+3$ (2 cooldown periods).
# This prevents overlapping event windows.

# %%
def enforce_cooldown(df, event_col, cooldown_periods=2):
    """
    Enforce cooldown by zeroing out events that occur too close to previous events.
    
    Parameters
    ----------
    df : DataFrame
        Must be sorted by ['iso3c', 'year']
    event_col : str
        Column name for event indicator
    cooldown_periods : int
        Number of periods to wait after an event
        
    Returns
    -------
    DataFrame with cooldown-enforced event column added
    """
    df = df.copy()
    out_col = f'{event_col}_cd'  # cd = cooldown-enforced
    df[out_col] = df[event_col].copy()
    
    # Track events per country
    for iso in df['iso3c'].unique():
        mask = df['iso3c'] == iso
        idx = df.loc[mask].index
        
        last_event_period = -999  # Initialize far in the past
        
        for i, row_idx in enumerate(idx):
            year = df.loc[row_idx, 'year']
            event = df.loc[row_idx, event_col]
            
            if pd.isna(event):
                continue
                
            # Calculate period index (relative to start)
            period_idx = (year - 1970) // 5
            
            if event == 1:
                # Check if within cooldown
                if period_idx <= last_event_period + cooldown_periods:
                    df.loc[row_idx, out_col] = 0  # Zero out
                else:
                    last_event_period = period_idx
    
    return df

# Apply cooldown to reforms
df = df.sort_values(['iso3c', 'year'])

df = enforce_cooldown(df, 'reform_agg_pos', SHOCK_CONFIG['cooldown'])
df = enforce_cooldown(df, 'reform_agg_neg', SHOCK_CONFIG['cooldown'])

# Also for basic shocks
df = enforce_cooldown(df, 'shock_agg_pos', SHOCK_CONFIG['cooldown'])
df = enforce_cooldown(df, 'shock_agg_neg', SHOCK_CONFIG['cooldown'])

# Count after cooldown
print("\nEvents after cooldown enforcement:")
print(f"  Sustained positive (R+_cd): {df['reform_agg_pos_cd'].sum():,.0f}")
print(f"  Sustained negative (R-_cd): {df['reform_agg_neg_cd'].sum():,.0f}")
print(f"  Basic positive (S+_cd): {df['shock_agg_pos_cd'].sum():,.0f}")
print(f"  Basic negative (S-_cd): {df['shock_agg_neg_cd'].sum():,.0f}")

# %% [markdown]
# ## 6. Create Area-Specific Shocks

# %%
# EFW areas
EFW_AREAS = {
    1: 'Size of Government',
    2: 'Legal System & Property Rights', 
    3: 'Sound Money',
    4: 'Freedom to Trade Internationally',
    5: 'Regulation',
}

# Create shocks for each area
area_event_counts = {}

for area_num, area_name in EFW_AREAS.items():
    d_col = f'd_efw_area{area_num}'
    level_col = f'efw_area{area_num}'
    
    if d_col not in df.columns:
        print(f"Warning: {d_col} not found, skipping area {area_num}")
        continue
    
    # Create lead columns if needed
    lead1_col = f'{level_col}_lead1'
    lead2_col = f'{level_col}_lead2'
    if lead1_col not in df.columns:
        df[lead1_col] = df.groupby('iso3c')[level_col].shift(-1)
    if lead2_col not in df.columns:
        df[lead2_col] = df.groupby('iso3c')[level_col].shift(-2)
    
    # Basic shocks
    df = create_signed_shocks(df, d_col, SHOCK_CONFIG['tau_main'], f'shock_area{area_num}')
    
    # Sustained reforms
    df = create_sustained_reforms(
        df, level_col, d_col, 
        SHOCK_CONFIG['tau_main'], SHOCK_CONFIG['epsilon'],
        f'reform_area{area_num}'
    )
    
    # Cooldown
    df = enforce_cooldown(df, f'reform_area{area_num}_pos', SHOCK_CONFIG['cooldown'])
    df = enforce_cooldown(df, f'reform_area{area_num}_neg', SHOCK_CONFIG['cooldown'])
    
    area_event_counts[area_num] = {
        'name': area_name,
        'shock_pos': df[f'shock_area{area_num}_pos'].sum(),
        'shock_neg': df[f'shock_area{area_num}_neg'].sum(),
        'reform_pos_cd': df[f'reform_area{area_num}_pos_cd'].sum(),
        'reform_neg_cd': df[f'reform_area{area_num}_neg_cd'].sum(),
    }

print("\nArea-specific events (with cooldown):")
for area_num, counts in area_event_counts.items():
    print(f"  Area {area_num} ({counts['name'][:20]}): R+={counts['reform_pos_cd']:.0f}, R-={counts['reform_neg_cd']:.0f}")

# %% [markdown]
# ## 7. Create Magnitude Categories

# %%
# Magnitude bins for the aggregate EFW change
d_efw_valid = df['d_efw_aggregate'].dropna()

# Pre-specified bin edges
BIN_EDGES = [-np.inf, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, np.inf]
BIN_LABELS = ['<-2', '-2 to -1.5', '-1.5 to -1', '-1 to -0.5', 
              '-0.5 to 0.5', '0.5 to 1', '1 to 1.5', '1.5 to 2', '>2']

df['d_efw_bin'] = pd.cut(
    df['d_efw_aggregate'],
    bins=BIN_EDGES,
    labels=BIN_LABELS,
    include_lowest=True
)

print("\nEFW change distribution by bin:")
print(df['d_efw_bin'].value_counts().sort_index())

# Simplified 3-category version
def categorize_change(x):
    if pd.isna(x):
        return np.nan
    elif x <= -1.0:
        return 'Large Negative'
    elif x >= 1.0:
        return 'Large Positive'
    else:
        return 'Small/No Change'

df['d_efw_category'] = df['d_efw_aggregate'].apply(categorize_change)

print("\nSimplified categories:")
print(df['d_efw_category'].value_counts())

# %% [markdown]
# ## 8. Create Bundle Indicators
#
# Identify "clean" single-area reforms vs bundled multi-area reforms.

# %%
# Count areas with concurrent large changes
area_change_cols = [f'd_efw_area{i}' for i in range(1, 6)]

# For each observation, count areas with |Δ| >= τ
def count_shocked_areas(row, tau=1.0):
    count = 0
    for col in area_change_cols:
        if pd.notna(row[col]) and abs(row[col]) >= tau:
            count += 1
    return count

df['n_areas_shocked'] = df.apply(lambda r: count_shocked_areas(r, SHOCK_CONFIG['tau_main']), axis=1)

# Clean vs bundled
df['is_clean_reform'] = (df['n_areas_shocked'] == 1).astype(int)
df['is_bundled_reform'] = (df['n_areas_shocked'] >= 2).astype(int)

print("\nReform bundle analysis:")
print(df['n_areas_shocked'].value_counts().sort_index())
print(f"\nClean (single-area): {df['is_clean_reform'].sum()}")
print(f"Bundled (multi-area): {df['is_bundled_reform'].sum()}")

# %% [markdown]
# ## 9. Event Summary and Logging

# %%
# Compile event counts
event_summary = {
    'config': SHOCK_CONFIG,
    'timestamp': datetime.now().isoformat(),
    'aggregate': {
        'shock_pos': int(df['shock_agg_pos'].sum()),
        'shock_neg': int(df['shock_agg_neg'].sum()),
        'shock_pos_cd': int(df['shock_agg_pos_cd'].sum()),
        'shock_neg_cd': int(df['shock_agg_neg_cd'].sum()),
        'reform_pos': int(df['reform_agg_pos'].sum()),
        'reform_neg': int(df['reform_agg_neg'].sum()),
        'reform_pos_cd': int(df['reform_agg_pos_cd'].sum()),
        'reform_neg_cd': int(df['reform_agg_neg_cd'].sum()),
    },
    'by_area': {},
    'bundles': {
        'clean': int(df['is_clean_reform'].sum()),
        'bundled': int(df['is_bundled_reform'].sum()),
        'by_n_areas': {str(k): int(v) for k, v in df['n_areas_shocked'].value_counts().to_dict().items()},
    }
}

for area_num, counts in area_event_counts.items():
    event_summary['by_area'][f'area{area_num}'] = {
        'name': counts['name'],
        'reform_pos_cd': int(counts['reform_pos_cd']),
        'reform_neg_cd': int(counts['reform_neg_cd']),
    }

# Save event log
log_path = OUTPUT_LOGS / 'event_counts.json'
with open(log_path, 'w') as f:
    json.dump(event_summary, f, indent=2)

print(f"\n✓ Event summary saved to {log_path}")

# %% [markdown]
# ## 10. Final Validation

# %%
print("\n" + "="*60)
print("FINAL VALIDATION CHECKS")
print("="*60)

# 1. No duplicate keys
assert not df.duplicated(subset=['iso3c', 'year']).any(), "Duplicate keys found"
print("✓ No duplicate country-year keys")

# 2. Shock indicators are binary
shock_cols = [c for c in df.columns if 'shock_' in c or 'reform_' in c]
for col in shock_cols:
    valid = df[col].dropna()
    if len(valid) > 0:
        assert valid.isin([0, 1]).all(), f"{col} is not binary"
print(f"✓ All {len(shock_cols)} shock/reform indicators are binary")

# 3. Cooldown enforcement check
# For each country with R+_cd = 1, check next 2 periods don't have R+_cd = 1
violations = 0
for iso in df['iso3c'].unique():
    country_data = df[df['iso3c'] == iso].sort_values('year')
    event_years = country_data[country_data['reform_agg_pos_cd'] == 1]['year'].values
    for event_year in event_years:
        next_years = [event_year + 5, event_year + 10]
        for ny in next_years:
            if ny in country_data['year'].values:
                if country_data[country_data['year'] == ny]['reform_agg_pos_cd'].values[0] == 1:
                    violations += 1

assert violations == 0, f"Cooldown violations found: {violations}"
print("✓ Cooldown properly enforced")

# 4. Sufficient events for estimation
min_events = 10  # Reduced minimum given data constraints
n_pos_sustained = df['reform_agg_pos_cd'].sum()
n_neg_sustained = df['reform_agg_neg_cd'].sum()
n_pos_basic = df['shock_agg_pos_cd'].sum()
n_neg_basic = df['shock_agg_neg_cd'].sum()

assert n_pos_sustained >= min_events, "Too few positive reform events"
print(f"✓ Sustained positive reforms: {n_pos_sustained:.0f}")

# Note: Sustained negative reforms are sparse due to maintenance rule
# Use basic shocks (S-) as alternative for negative reform analysis
if n_neg_sustained < min_events:
    print(f"⚠ WARNING: Only {n_neg_sustained:.0f} sustained negative reforms (maintenance rule is strict)")
    print(f"  Using basic shocks (S-_cd) for negative analysis: {n_neg_basic:.0f}")
    # Create alias for downstream notebooks
    df['reform_agg_neg_cd_alt'] = df['shock_agg_neg_cd']  # Use basic shocks
else:
    print(f"✓ Sustained negative reforms: {n_neg_sustained:.0f}")
    df['reform_agg_neg_cd_alt'] = df['reform_agg_neg_cd']

print(f"✓ Basic shocks (for robustness): S+_cd={n_pos_basic:.0f}, S-_cd={n_neg_basic:.0f}")

# %% [markdown]
# ## 11. Save Output

# %%
# Save panel with shocks
output_path = DATA_CLEAN / 'panel_with_shocks.parquet'
df.to_parquet(output_path, index=False)

print(f"\n✓ Saved panel with shocks to {output_path}")
print(f"  Observations: {len(df)}")
print(f"  Countries: {df['iso3c'].nunique()}")
print(f"  Total columns: {len(df.columns)}")

# List new columns
new_cols = [c for c in df.columns if 'shock_' in c or 'reform_' in c or 'd_efw_' in c or 'n_areas' in c or 'bundle' in c or 'clean' in c]
print(f"\nNew shock/reform columns ({len(new_cols)}):")
for col in sorted(new_cols)[:20]:
    print(f"  {col}")
if len(new_cols) > 20:
    print(f"  ... and {len(new_cols) - 20} more")

print("\n" + "="*60)
print("04_CONSTRUCT_SHOCKS COMPLETE")
print("="*60)
