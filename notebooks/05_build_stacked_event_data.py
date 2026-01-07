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
# # 05. Build Stacked Event-Study Dataset
#
# This notebook constructs the stacked event-study dataset for LP-DiD estimation.
# Following Cengiz et al. (2019) and Baker et al. (2022), we create cohort-specific
# datasets that avoid TWFE contamination from staggered adoption.
#
# **Inputs:**
# - `data/03_clean/panel_with_shocks.parquet`
#
# **Outputs:**
# - `data/03_clean/stacked_events_pos.parquet`
# - `data/03_clean/stacked_events_neg.parquet`
# - `data/03_clean/stacked_events_combined.parquet`
#
# **Key design choices:**
# 1. Each event (cohort) gets its own subsample
# 2. Clean controls: never-treated or not-yet-treated
# 3. Event-time indexing relative to treatment
# 4. Drop overlapping event windows

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

print(f"Project root: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Load Panel with Shocks

# %%
panel_path = DATA_CLEAN / 'panel_with_shocks.parquet'
assert panel_path.exists(), f"Panel data not found: {panel_path}"

df = pd.read_parquet(panel_path)
print(f"Panel: {len(df)} observations, {df['iso3c'].nunique()} countries")
print(f"Years: {sorted(df['year'].unique())}")

# %% [markdown]
# ## 2. Event Study Configuration

# %%
# Event study parameters
EVENT_CONFIG = {
    # Number of pre-event periods (K in paper)
    # Extended from 2 to 4 to use more of the 11 quinquennial periods
    'pre_periods': 4,  # -4, -3, -2, -1 relative to event (20 years of pre-data)
    
    # Number of post-event periods (L in paper)
    # Extended from 4 to 6 to capture longer-run effects
    'post_periods': 6,  # 0, 1, 2, 3, 4, 5, 6 relative to event (30 years of post-data)
    
    # Event variables to use (cooldown-enforced)
    'pos_event_col': 'reform_agg_pos_cd',
    # Note: Use basic shocks (S-) for negative reforms due to sparse sustained reforms
    'neg_event_col': 'shock_agg_neg_cd',  # Alternative: df might have reform_agg_neg_cd_alt
    
    # Control specification
    'control_type': 'not_yet_treated',  # or 'never_treated'
    
    # Minimum observations per event
    'min_obs_per_event': 5,
}

# Derived parameters
K = EVENT_CONFIG['pre_periods']
L = EVENT_CONFIG['post_periods']
window_size = K + L + 1  # Total event window size
YEARS = sorted(df['year'].unique())
QUINQUENNIAL_STEP = 5

print("Event study configuration:")
for key, value in EVENT_CONFIG.items():
    print(f"  {key}: {value}")
print(f"\nEvent window: {-K} to +{L} ({window_size} periods)")

# %% [markdown]
# ## 3. Identify Event Cohorts

# %%
def identify_events(df, event_col):
    """
    Identify all events and their timing.
    
    Returns DataFrame with (iso3c, event_year, event_type)
    """
    events = df[df[event_col] == 1][['iso3c', 'year']].copy()
    events = events.rename(columns={'year': 'event_year'})
    events['event_type'] = event_col
    return events

# Identify positive and negative events
events_pos = identify_events(df, EVENT_CONFIG['pos_event_col'])
events_neg = identify_events(df, EVENT_CONFIG['neg_event_col'])

print(f"Positive reform events: {len(events_pos)}")
print(f"Negative reform events: {len(events_neg)}")

print("\nPositive events by year:")
print(events_pos.groupby('event_year').size())

print("\nNegative events by year:")
print(events_neg.groupby('event_year').size())

# %% [markdown]
# ## 4. Build Stacked Dataset for One Event Type

# %%
def build_stacked_dataset(df, events_df, K=2, L=4, control_type='not_yet_treated'):
    """
    Build stacked event-study dataset.
    
    For each event cohort (event_year, event_country), we create a subsample
    containing the treated unit and appropriate controls, indexed by event time.
    
    Parameters
    ----------
    df : DataFrame
        Full panel data
    events_df : DataFrame
        Events with columns ['iso3c', 'event_year']
    K : int
        Number of pre-periods
    L : int
        Number of post-periods
    control_type : str
        'never_treated' or 'not_yet_treated'
        
    Returns
    -------
    DataFrame in stacked format with cohort identifiers
    """
    stacked_dfs = []
    
    # Get all treated countries
    treated_countries = set(events_df['iso3c'].unique())
    never_treated = set(df['iso3c'].unique()) - treated_countries
    
    # Get all event years
    all_event_years = sorted(events_df['event_year'].unique())
    
    for idx, (_, event) in enumerate(events_df.iterrows()):
        iso = event['iso3c']
        event_year = event['event_year']
        
        # Define event window
        window_start = event_year - K * QUINQUENNIAL_STEP
        window_end = event_year + L * QUINQUENNIAL_STEP
        
        # Create cohort identifier
        cohort_id = f"{iso}_{event_year}"
        
        # Select treated unit
        treated_data = df[
            (df['iso3c'] == iso) & 
            (df['year'] >= window_start) & 
            (df['year'] <= window_end)
        ].copy()
        treated_data['treated'] = 1
        treated_data['cohort_id'] = cohort_id
        treated_data['cohort_year'] = event_year
        
        # Define post-treatment indicator
        treated_data['post'] = (treated_data['year'] >= event_year).astype(int)
        
        # Event time relative to treatment
        treated_data['event_time'] = (treated_data['year'] - event_year) // QUINQUENNIAL_STEP
        
        # Interaction for DiD
        treated_data['treat_post'] = treated_data['treated'] * treated_data['post']
        
        # Select controls
        if control_type == 'never_treated':
            control_countries = never_treated
        else:  # not_yet_treated
            # Countries that haven't been treated by this event year
            later_treated = events_df[events_df['event_year'] > event_year]['iso3c'].unique()
            control_countries = never_treated | set(later_treated)
        
        if len(control_countries) == 0:
            continue
            
        control_data = df[
            (df['iso3c'].isin(control_countries)) & 
            (df['year'] >= window_start) & 
            (df['year'] <= window_end)
        ].copy()
        control_data['treated'] = 0
        control_data['cohort_id'] = cohort_id
        control_data['cohort_year'] = event_year
        control_data['post'] = (control_data['year'] >= event_year).astype(int)
        control_data['event_time'] = (control_data['year'] - event_year) // QUINQUENNIAL_STEP
        control_data['treat_post'] = 0
        
        # Combine
        cohort_data = pd.concat([treated_data, control_data], ignore_index=True)
        stacked_dfs.append(cohort_data)
    
    if len(stacked_dfs) == 0:
        return pd.DataFrame()
    
    # Stack all cohorts
    stacked = pd.concat(stacked_dfs, ignore_index=True)
    
    # Add cohort-specific fixed effects identifier
    stacked['cohort_country'] = stacked['cohort_id'] + '_' + stacked['iso3c']
    
    return stacked

# %% [markdown]
# ## 5. Build Stacked Datasets

# %%
# Build stacked dataset for positive reforms
print("Building stacked dataset for positive reforms...")
stacked_pos = build_stacked_dataset(
    df, events_pos,
    K=K, L=L,
    control_type=EVENT_CONFIG['control_type']
)

print(f"Positive reforms stacked:")
print(f"  Total observations: {len(stacked_pos)}")
print(f"  Cohorts: {stacked_pos['cohort_id'].nunique()}")
print(f"  Countries: {stacked_pos['iso3c'].nunique()}")
print(f"  Event times: {sorted(stacked_pos['event_time'].unique())}")

# %%
# Build stacked dataset for negative reforms
print("\nBuilding stacked dataset for negative reforms...")
stacked_neg = build_stacked_dataset(
    df, events_neg,
    K=K, L=L,
    control_type=EVENT_CONFIG['control_type']
)

print(f"Negative reforms stacked:")
print(f"  Total observations: {len(stacked_neg)}")
print(f"  Cohorts: {stacked_neg['cohort_id'].nunique()}")
print(f"  Countries: {stacked_neg['iso3c'].nunique()}")
print(f"  Event times: {sorted(stacked_neg['event_time'].unique())}")

# %% [markdown]
# ## 6. Create Event-Time Dummies

# %%
def add_event_time_dummies(df, K=2, L=4, omit_period=-1):
    """
    Add event-time dummy variables.
    
    Parameters
    ----------
    df : DataFrame with 'event_time' column
    K : int
        Number of pre-periods
    L : int
        Number of post-periods
    omit_period : int
        Reference period to omit (default: -1)
        
    Returns
    -------
    DataFrame with event-time dummies
    """
    df = df.copy()
    
    for k in range(-K, L+1):
        if k == omit_period:
            continue  # Reference period
        col_name = f'et_{k}' if k < 0 else f'et_p{k}'  # et_-2, et_-1, et_p0, et_p1, ...
        df[col_name] = (df['event_time'] == k).astype(int)
    
    # Create interaction dummies (treated × event_time)
    for k in range(-K, L+1):
        if k == omit_period:
            continue
        dummy_col = f'et_{k}' if k < 0 else f'et_p{k}'
        interact_col = f'treat_et_{k}' if k < 0 else f'treat_et_p{k}'
        df[interact_col] = df['treated'] * df[dummy_col]
    
    return df

# Add event-time dummies
stacked_pos = add_event_time_dummies(stacked_pos, K=K, L=L)
stacked_neg = add_event_time_dummies(stacked_neg, K=K, L=L)

# Check interaction columns
interact_cols = [c for c in stacked_pos.columns if c.startswith('treat_et_')]
print(f"\nTreatment × Event-time interactions ({len(interact_cols)} columns):")
for col in sorted(interact_cols):
    print(f"  {col}: {stacked_pos[col].sum()} treated obs")

# %% [markdown]
# ## 7. Prepare Outcome Variables (Forward Growth)

# %%
def prepare_outcomes(df):
    """
    Prepare outcome variables for LP estimation.
    
    For LP-DiD, we use growth forward from the base period.
    Y_{i,t+h} - Y_{i,t-1} or equivalently sum of growth from t to t+h
    """
    df = df.copy()
    
    # Check outcome availability
    if 'growth' in df.columns:
        # Forward growth rates are already in the data
        # For event-study, we want cumulative growth from t-1 to t+h
        # This requires recomputing based on ln_gdp_pc
        pass
    
    if 'ln_gdp_pc' in df.columns:
        # Compute cumulative log difference from base period (t-1)
        # For each cohort, t-1 is event_time = -1
        
        # Get base period ln_gdp_pc for each unit in each cohort
        base_period = df[df['event_time'] == -1][['cohort_id', 'iso3c', 'ln_gdp_pc']].copy()
        base_period = base_period.rename(columns={'ln_gdp_pc': 'ln_gdp_pc_base'})
        
        df = df.merge(base_period, on=['cohort_id', 'iso3c'], how='left')
        
        # Cumulative outcome: Y_{t+h} - Y_{t-1}
        df['y_cumulative'] = df['ln_gdp_pc'] - df['ln_gdp_pc_base']
    
    return df

stacked_pos = prepare_outcomes(stacked_pos)
stacked_neg = prepare_outcomes(stacked_neg)

print("Outcome variable summary (stacked_pos):")
if 'y_cumulative' in stacked_pos.columns:
    print(stacked_pos.groupby('event_time')['y_cumulative'].describe().round(3))

# %% [markdown]
# ## 8. Validation Checks

# %%
def validate_stacked_data(df, name):
    """Validate stacked dataset integrity."""
    print(f"\nValidation: {name}")
    print("="*50)
    
    # 1. Check for duplicate: each cohort should have unique country-year combinations
    # Note: Same country CAN appear in multiple cohorts (expected in stacked design)
    dup_within_cohort = df.groupby(['cohort_id']).apply(
        lambda x: x.duplicated(subset=['iso3c', 'year']).any()
    )
    n_cohorts_with_dups = dup_within_cohort.sum()
    if n_cohorts_with_dups > 0:
        print(f"⚠ Warning: {n_cohorts_with_dups} cohorts have duplicate country-year")
    else:
        print("✓ No duplicate observations within cohorts")
    
    # 2. Check event_time range
    et_range = df['event_time'].unique()
    print(f"✓ Event-time range: {min(et_range)} to {max(et_range)}")
    
    # 3. Check treatment timing
    treated_at_0 = df[(df['treated'] == 1) & (df['event_time'] == 0)]
    n_cohorts = df['cohort_id'].nunique()
    print(f"✓ Cohorts: {n_cohorts}, treated at t=0: {len(treated_at_0)}")
    
    # 4. Check controls
    n_treated_units = df[df['treated'] == 1]['iso3c'].nunique()
    n_control_units = df[df['treated'] == 0]['iso3c'].nunique()
    print(f"✓ Treated units: {n_treated_units}, Control units: {n_control_units}")
    
    # 5. Check outcome availability
    if 'y_cumulative' in df.columns:
        n_outcome = df['y_cumulative'].notna().sum()
        pct = 100 * n_outcome / len(df)
        print(f"✓ Outcome available: {n_outcome} obs ({pct:.1f}%)")
    
    return True

validate_stacked_data(stacked_pos, "Positive Reforms")
validate_stacked_data(stacked_neg, "Negative Reforms")

# %% [markdown]
# ## 9. Create Combined Dataset with Sign Indicator

# %%
# Add sign indicator
stacked_pos['reform_sign'] = 'positive'
stacked_neg['reform_sign'] = 'negative'

# Combine
stacked_combined = pd.concat([stacked_pos, stacked_neg], ignore_index=True)

print(f"\nCombined stacked dataset:")
print(f"  Total observations: {len(stacked_combined)}")
print(f"  Positive cohorts: {stacked_combined[stacked_combined['reform_sign'] == 'positive']['cohort_id'].nunique()}")
print(f"  Negative cohorts: {stacked_combined[stacked_combined['reform_sign'] == 'negative']['cohort_id'].nunique()}")

# %% [markdown]
# ## 10. Summary Statistics by Event Time

# %%
print("\nSummary by event time (Positive Reforms):")
summary_pos = stacked_pos.groupby('event_time').agg({
    'treated': 'sum',
    'y_cumulative': ['count', 'mean', 'std'],
    'cohort_id': 'nunique'
}).round(3)
summary_pos.columns = ['n_treated', 'n_outcome', 'mean_y', 'std_y', 'n_cohorts']
print(summary_pos)

print("\nSummary by event time (Negative Reforms):")
summary_neg = stacked_neg.groupby('event_time').agg({
    'treated': 'sum',
    'y_cumulative': ['count', 'mean', 'std'],
    'cohort_id': 'nunique'
}).round(3)
summary_neg.columns = ['n_treated', 'n_outcome', 'mean_y', 'std_y', 'n_cohorts']
print(summary_neg)

# %% [markdown]
# ## 11. Save Outputs

# %%
# Save stacked datasets
output_pos = DATA_CLEAN / 'stacked_events_pos.parquet'
output_neg = DATA_CLEAN / 'stacked_events_neg.parquet'
output_combined = DATA_CLEAN / 'stacked_events_combined.parquet'

stacked_pos.to_parquet(output_pos, index=False)
stacked_neg.to_parquet(output_neg, index=False)
stacked_combined.to_parquet(output_combined, index=False)

print(f"\n✓ Saved positive reforms to {output_pos}")
print(f"✓ Saved negative reforms to {output_neg}")
print(f"✓ Saved combined dataset to {output_combined}")

# Save metadata
meta = {
    'config': EVENT_CONFIG,
    'timestamp': datetime.now().isoformat(),
    'positive_reforms': {
        'n_obs': len(stacked_pos),
        'n_cohorts': int(stacked_pos['cohort_id'].nunique()),
        'n_treated': int(stacked_pos[stacked_pos['treated'] == 1]['iso3c'].nunique()),
        'n_control': int(stacked_pos[stacked_pos['treated'] == 0]['iso3c'].nunique()),
    },
    'negative_reforms': {
        'n_obs': len(stacked_neg),
        'n_cohorts': int(stacked_neg['cohort_id'].nunique()),
        'n_treated': int(stacked_neg[stacked_neg['treated'] == 1]['iso3c'].nunique()),
        'n_control': int(stacked_neg[stacked_neg['treated'] == 0]['iso3c'].nunique()),
    },
    'variables': {
        'outcome': 'y_cumulative (ln GDP per capita relative to t-1)',
        'treatment': 'treated × event_time interactions',
        'fixed_effects': 'cohort_id × iso3c (cohort-country FE)',
    }
}

meta_path = OUTPUT_LOGS / 'stacked_events_metadata.json'
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"✓ Saved metadata to {meta_path}")

# %% [markdown]
# ## 12. Final Summary

# %%
print("\n" + "="*60)
print("05_BUILD_STACKED_EVENT_DATA COMPLETE")
print("="*60)
print(f"Event window: t={-K} to t=+{L} ({window_size} periods)")
print(f"Control type: {EVENT_CONFIG['control_type']}")
print(f"Positive reform cohorts: {stacked_pos['cohort_id'].nunique()}")
print(f"Negative reform cohorts: {stacked_neg['cohort_id'].nunique()}")
print(f"Total stacked observations: {len(stacked_combined)}")
print("="*60)
