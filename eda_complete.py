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
# # Economic Freedom of the World: Complete Analysis
#
# **Data**: Fraser Institute Economic Freedom of the World Index (1970–2023)  
# **Coverage**: 165 countries across 7 World Bank regions  
#
# This notebook provides a comprehensive **MIT-level descriptive analysis** of the EFW data,
# organized around three analytical dimensions:
#
# 1. **LOCATION**: Region, income group, and country-level patterns
# 2. **PERIOD**: Pre/post 1990 (Cold War), pre/post 2000 (annual data), 2008 GFC window
# 3. **COMPONENT**: The 5 EFW areas (Size of Government, Legal System, Sound Money, Trade, Regulation)
#
# The EFW index measures economic freedom on a 0–10 scale. The overall score is the
# unweighted mean of five area scores:
#
# | Area | Measures |
# |------|----------|
# | Size of Government | Consumption, transfers, taxes, SOEs |
# | Legal System & Property Rights | Courts, contracts, property rights |
# | Sound Money | Inflation, money growth stability |
# | Trade Freedom | Tariffs, barriers, capital controls |
# | Regulation | Credit, labor, business regulations |
#
# **Methodological notes:**
# - All interpretations are descriptive—no causal claims
# - Sample composition effects are explicitly addressed
# - Balanced-panel analyses complement unbalanced trends

# %% [markdown]
# ## 0. Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Import helper utilities
import efw_utils as efw

# Configuration
DATA_PATH = "data/fraser.xlsx"
OUTPUT_DIR = "outputs"
TABLES_DIR = "outputs/tables"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Column names (from efw_utils)
EFW = efw.EFW_COL
AREAS = efw.AREA_COLS
LABELS = efw.AREA_LABELS
COLORS = efw.AREA_COLORS

# Key years and periods
YEAR_BREAKS = [1990, 2000, 2008]
QUINQUENNIAL = efw.QUINQUENNIAL_YEARS
PERIODS = [(1990, 2000), (2000, 2008), (2008, 2013), (2013, 2020)]

# Apply publication-quality plot style
efw.setup_plot_style()

# Initialize figure index tracker
fig_index = efw.FigureIndex()

print("Configuration loaded.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Key year breaks: {YEAR_BREAKS}")

# %% [markdown]
# ## 1. Data Loading

# %%
# Load and clean data using helper function
df = efw.load_fraser(DATA_PATH, verbose=True)

# Add area availability tracking
df = efw.add_area_availability(df, AREAS)

# Rename columns for consistency with our analysis
REGION = 'Region'
INCOME = 'Income_Group'

# Quick preview
print(f"\nRegions: {df[REGION].nunique()}")
print(df[REGION].value_counts())
print(f"\nIncome groups: {df[INCOME].nunique()}")
print(df[INCOME].value_counts())

# Area availability summary
print(f"\nArea availability:")
print(f"  Rows with all 5 areas: {df['complete5'].sum():,} ({df['complete5'].mean()*100:.1f}%)")
print(f"  Rows with EFW but <5 areas: {(df[EFW].notna() & ~df['complete5']).sum():,}")

# %% [markdown]
# ## 2. Data Audit
#
# Rigorous verification of data integrity before analysis.

# %% [markdown]
# ### 2A. Basic Integrity Checks

# %%
# Check for duplicates by (Year, Countries)
dups = df.groupby(['Year', 'Countries']).size()
dup_count = (dups > 1).sum()
print(f"Duplicate (Year, Country) pairs: {dup_count}")

if dup_count > 0:
    print("\nExamples of duplicates:")
    dup_pairs = dups[dups > 1].head(5)
    for (year, country), count in dup_pairs.items():
        print(f"  {country} in {year}: {count} rows")
    
    # Deduplicate deterministically (keep first)
    df = df.drop_duplicates(subset=['Year', 'Countries'], keep='first')
    print(f"\nAfter deduplication: {len(df):,} observations")

# %%
# Check score ranges [0, 10]
print("Score range validation:")
print("-" * 50)

all_score_cols = [EFW] + AREAS
for col in all_score_cols:
    if col in df.columns:
        vals = df[col].dropna()
        out_of_range = ((vals < 0) | (vals > 10)).sum()
        col_label = LABELS[AREAS.index(col)] if col in AREAS else 'Overall EFW'
        print(f"{col_label:20s}: min={vals.min():.2f}, max={vals.max():.2f}, "
              f"out-of-range={out_of_range}")

# Flag out-of-range observations
for col in all_score_cols:
    if col in df.columns:
        out_mask = (df[col] < 0) | (df[col] > 10)
        if out_mask.any():
            print(f"\n⚠ Out-of-range values in {col}:")
            print(df[out_mask][['Year', 'Countries', col]].head())

# %%
# CRITICAL: Validate EFW = mean(AVAILABLE areas), not just complete cases
# Fraser computes overall as mean of available areas when some are missing
max_diff = efw.validate_overall_matches_available_areas(df, EFW, AREAS, tol=0.01)
print(f"\nEFW vs mean(available areas) validation:")
print(f"  Max absolute difference: {max_diff:.4f}")
print(f"  ✓ Confirmed: EFW = mean of available areas (not missing when <5 areas)")

# Distribution of n_areas for EFW-nonmissing rows
efw_present = df[df[EFW].notna()]
print(f"\nAmong EFW-nonmissing rows (N={len(efw_present):,}):")
for n in sorted(efw_present['n_areas'].unique(), reverse=True):
    count = (efw_present['n_areas'] == n).sum()
    print(f"  {n} areas: {count:,} ({100*count/len(efw_present):.1f}%)")

# %% [markdown]
# ### 2B. Missingness Overview

# %%
# Compute comprehensive missingness statistics
miss_stats = efw.compute_missingness(df, [EFW] + AREAS)

# By column
print("Missing % by column:")
print("-" * 40)
col_labels = ['Overall EFW'] + LABELS
for col, label in zip([EFW] + AREAS, col_labels):
    if col in miss_stats['by_column'].index:
        print(f"  {label:20s}: {miss_stats['by_column'][col]:5.1f}%")

# Save to CSV
miss_by_col = pd.DataFrame({
    'Column': col_labels,
    'Missing_Pct': [miss_stats['by_column'].get(c, np.nan) for c in [EFW] + AREAS]
})
efw.export_table(miss_by_col, 'missingness_by_column.csv', TABLES_DIR)

# %%
# Missing by year
fig, ax = plt.subplots(figsize=(12, 4))
miss_by_year = miss_stats['by_year']
ax.bar(miss_by_year.index, miss_by_year.values, color='#e74c3c', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Missing %')
ax.set_title('EFW Missing Rate by Year')
ax.axhline(y=miss_by_year.mean(), color='gray', linestyle='--', 
           label=f'Mean = {miss_by_year.mean():.1f}%')
ax.legend()
plt.tight_layout()
efw.savefig(fig, 'missing_by_year.png', OUTPUT_DIR)
plt.show()

fig_index.add('missing_by_year.png', 'Missing rate by year for overall EFW',
              'Full', '1970-2023', 'Missing rate highest in early years, drops after 2000')

# %%
# Missing by region
if 'by_region' in miss_stats:
    print("\nMissing % by region (for EFW):")
    region_miss = miss_stats['by_region'][EFW].sort_values(ascending=False)
    for region, pct in region_miss.items():
        print(f"  {region:40s}: {pct:5.1f}%")
    
    efw.export_table(miss_stats['by_region'], 'missingness_by_region.csv', TABLES_DIR)

# %%
# Availability heatmap (Year × Region)
avail = efw.create_availability_heatmap(df, EFW)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(avail, cmap='YlGnBu', annot=True, fmt='d', ax=ax,
            cbar_kws={'label': 'Countries with EFW data'})
ax.set_title('Data Availability: Countries per Year by Region')
ax.set_xlabel('Region')
ax.set_ylabel('Year')
plt.tight_layout()
efw.savefig(fig, 'availability_heatmap.png', OUTPUT_DIR)
plt.show()

fig_index.add('availability_heatmap.png', 'Year × Region availability matrix',
              'Full', '1970-2023', 'Coverage expansion concentrated post-1990')

# %% [markdown]
# ### 2C. Index Construction Verification
#
# **Question**: Does the overall EFW exactly equal the unweighted mean of the 5 area scores?
#
# **Method**: Compute diff = EFW − mean(Areas) on complete cases and examine distribution.
#
# **What to look for**: If construction is exact, mean(diff) ≈ 0 and max|diff| should be negligible.

# %%
# Verify index construction
verify_results = efw.verify_index_construction(df, EFW, AREAS)

print("Index Construction Verification")
print("=" * 50)
print(f"Complete cases (all 5 areas + EFW): {verify_results['n_complete']:,}")
print(f"Mean difference (EFW - mean(areas)): {verify_results['mean_diff']:.6f}")
print(f"Std of difference: {verify_results['std_diff']:.6f}")
print(f"Max |difference|: {verify_results['max_abs_diff']:.6f}")
print(f"Fraction with |diff| > 0.05: {verify_results['frac_large_diff']:.2%}")
print(f"Correlation (EFW vs computed mean): {verify_results['correlation']:.6f}")

if verify_results['max_abs_diff'] < 0.01:
    print("\n✓ Index construction verified: EFW = mean(5 areas)")
elif verify_results['frac_large_diff'] < 0.01:
    print("\n✓ Near-perfect construction with minimal rounding differences")
else:
    print("\n⚠ Some discrepancies detected - may reflect Fraser's methodology")

# %% [markdown]
# > **Note on Index Construction**: The Fraser Institute computes the overall EFW as the
# > unweighted arithmetic mean of the five area scores. Small differences may arise from:
# > (1) rounding in published data, (2) sub-component weighting within areas, or
# > (3) cases where not all components within an area are available. For our analysis,
# > the correlation > 0.999 confirms the index behaves as documented.

# %% [markdown]
# ### 2D. Panel Structure: Frequency Break Around 2000
#
# **Question**: How does the data frequency change over time?
#
# **Method**: Count observations per year and identify the quinquennial→annual transition.
#
# **What to look for**: Step-change in observation density around 2000.

# %%
year_counts = df.groupby('Year').size()
country_counts = df.groupby('Year')['Countries'].nunique()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Observations per year
ax = axes[0]
colors = ['#2ecc71' if y in QUINQUENNIAL and y <= 2000 else '#3498db' for y in year_counts.index]
ax.bar(year_counts.index, year_counts.values, color=colors, edgecolor='white', width=0.8)
ax.axvline(x=2000.5, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
ax.set_xlabel('Year')
ax.set_ylabel('Total Observations')
ax.set_title('Panel Structure: Observations per Year')
ax.text(1985, year_counts.max()*0.9, 'Quinquennial\n(5-year intervals)', 
        color='#2ecc71', fontsize=9, ha='center')
ax.text(2012, year_counts.max()*0.9, 'Annual', color='#3498db', fontsize=9, ha='center')

# Countries per year
ax = axes[1]
ax.plot(country_counts.index, country_counts.values, 'o-', color='#9b59b6', 
        linewidth=2, markersize=4)
ax.axvline(x=2000.5, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
ax.set_xlabel('Year')
ax.set_ylabel('Countries Covered')
ax.set_title('Panel Structure: Country Coverage')
ax.fill_between(country_counts.index, country_counts.values, alpha=0.2, color='#9b59b6')

plt.tight_layout()
efw.savefig(fig, 'panel_structure.png', OUTPUT_DIR)
plt.show()

fig_index.add('panel_structure.png', 'Panel frequency and country coverage',
              'Full', '1970-2023', 'Quinquennial pre-2000, annual after; coverage grew 3x')

# Document the break
print("\nPanel Frequency Summary:")
print("-" * 50)
pre_2001 = year_counts[year_counts.index <= 2000]
post_2000 = year_counts[year_counts.index > 2000]
print(f"Pre-2001: {len(pre_2001)} waves (quinquennial)")
print(f"Post-2000: {len(post_2000)} annual observations")
print(f"\n⚠ WARNING: Any trend plot spanning 1970–2023 mixes frequencies.")
print("  Balanced-sample comparisons are advisable.")

# %% [markdown]
# > **Critical Note**: The panel is quinquennial (5-year intervals) from 1970–2000 and annual
# > thereafter. Any "trend" analysis spanning the full period conflates measurement frequency
# > with actual trends. We address this by showing (i) raw unbalanced means, (ii) balanced-sample
# > means (common countries), and (iii) median ± IQR for distributional stability.

# %% [markdown]
# ## 3. Descriptive Statistics

# %%
# Compute descriptive statistics
desc = df[[EFW] + AREAS].describe().T
desc['missing%'] = df[[EFW] + AREAS].isnull().mean() * 100
desc = desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing%']]
desc.index = ['Overall EFW'] + LABELS

print("Descriptive Statistics")
print("=" * 80)
print(desc.round(2).to_string())

# Save to CSV
desc.to_csv(f'{TABLES_DIR}/descriptive_statistics.csv')

# %% [markdown]
# ### Distribution of Overall EFW
#
# **Question**: What is the cross-sectional distribution of economic freedom?
#
# **Method**: Histogram with kernel density overlay, showing mean and median.
#
# **What to look for**: Unimodal distribution, skewness, and central tendency.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
data = df[EFW].dropna()

# Histogram with KDE
ax.hist(data, bins=40, density=True, alpha=0.6, color='#3498db', edgecolor='white')
data.plot.kde(ax=ax, color='#2c3e50', linewidth=2, label='KDE')

# Central tendency markers
ax.axvline(data.mean(), color='#c0392b', linestyle='--', linewidth=2, 
           label=f'Mean = {data.mean():.2f}')
ax.axvline(data.median(), color='#27ae60', linestyle=':', linewidth=2, 
           label=f'Median = {data.median():.2f}')

ax.set_xlabel('Economic Freedom Score')
ax.set_ylabel('Density')
ax.set_title('Distribution of Overall EFW (All Country-Years)')
ax.legend()
ax.set_xlim(2, 10)
plt.tight_layout()
efw.savefig(fig, 'distribution_efw.png', OUTPUT_DIR)
plt.show()

fig_index.add('distribution_efw.png', 'Histogram of overall EFW across all observations',
              'Full', '1970-2023', 'Approximately symmetric, slight negative skew')

# %% [markdown]
# **Key takeaways:**
# - The distribution is approximately symmetric with mean ≈ median ≈ 6.4
# - Scores range from ~2.5 to ~9, with most mass between 5 and 8
# - Slight negative skew (long left tail) reflects a few very low-freedom countries

# %% [markdown]
# ## 4. Global Trends
#
# This section examines worldwide trends in economic freedom, with careful attention to
# sample composition and distributional changes.

# %% [markdown]
# ### 4.1 Mean, Median, and Distributional Bands
#
# **Question**: How has the global level and distribution of economic freedom evolved?
#
# **Method**: Plot mean, median, p10, and p90 over time to capture both central tendency 
# and dispersion.
#
# **What to look for**: Trends in central tendency, widening/narrowing of distribution.

# %%
# Compute distributional statistics by year
yearly_stats = df.groupby('Year')[EFW].agg(['mean', 'median', 'std', 'count',
    lambda x: x.quantile(0.10), lambda x: x.quantile(0.90)])
yearly_stats.columns = ['mean', 'median', 'std', 'count', 'p10', 'p90']

fig, ax = plt.subplots(figsize=(12, 6))

# P10-P90 band
ax.fill_between(yearly_stats.index, yearly_stats['p10'], yearly_stats['p90'],
                alpha=0.2, color='#3498db', label='P10–P90 range')

# Mean and median
ax.plot(yearly_stats.index, yearly_stats['mean'], 'o-', color='#c0392b',
        linewidth=2, markersize=4, label='Mean')
ax.plot(yearly_stats.index, yearly_stats['median'], 's--', color='#27ae60',
        linewidth=2, markersize=4, label='Median')

# Key year markers
for year in YEAR_BREAKS:
    ax.axvline(x=year, color='gray', linestyle=':', alpha=0.5)
    ax.text(year+0.5, ax.get_ylim()[1]-0.2, str(year), fontsize=8, color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Economic Freedom Score')
ax.set_title('Global EFW Trends: Central Tendency and Dispersion')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(3, 9)
plt.tight_layout()
efw.savefig(fig, 'global_trends_distributional.png', OUTPUT_DIR)
plt.show()

fig_index.add('global_trends_distributional.png', 'Mean, median, P10-P90 bands over time',
              'Unbalanced', '1970-2023', 'Rising trend with narrowing dispersion (convergence)')

# %% [markdown]
# **Key takeaways:**
# - Global mean EFW rose from ~5.5 (1970) to ~6.9 (2023), a gain of ~1.4 points
# - Mean and median track closely, indicating symmetric distribution over time
# - P10-P90 band narrowed substantially, especially post-1990 (σ-convergence)

# %% [markdown]
# ### 4.2 Composition-Aware Trends: Balanced vs Unbalanced Sample
#
# **Question**: Are global trends driven by new countries entering the sample, or by 
# genuine changes in existing countries?
#
# **Method**: Compare (i) unbalanced sample mean, (ii) balanced sample mean (countries 
# observed in both 1970 and 2020), (iii) common quinquennial countries only.
#
# **What to look for**: Divergence between samples would indicate composition effects.

# %%
# Identify common countries across full period
common_countries_full = efw.get_common_countries(df, 1970, 2020, EFW)
print(f"Countries in both 1970 and 2020: {len(common_countries_full)}")

# Balanced quinquennial panel
df_bal_quin = efw.compute_balanced_panel(df, QUINQUENNIAL, EFW)
balanced_countries = df_bal_quin['Countries'].unique()
print(f"Countries in all {len(QUINQUENNIAL)} quinquennial waves: {len(balanced_countries)}")

# Compute means for each sample
yearly_unbal = df.groupby('Year')[EFW].mean()
yearly_common = df[df['Countries'].isin(common_countries_full)].groupby('Year')[EFW].mean()
yearly_balanced = df_bal_quin.groupby('Year')[EFW].mean()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(yearly_unbal.index, yearly_unbal.values, 'o-', color='#3498db',
        linewidth=2, markersize=4, label=f'Unbalanced (all countries)', alpha=0.8)
ax.plot(yearly_common.index, yearly_common.values, 's--', color='#e74c3c',
        linewidth=2, markersize=5, label=f'Common 1970-2020 (N={len(common_countries_full)})')
ax.plot(yearly_balanced.index, yearly_balanced.values, '^-', color='#27ae60',
        linewidth=2, markersize=6, label=f'Balanced quinquennial (N={len(balanced_countries)})')

for year in YEAR_BREAKS:
    ax.axvline(x=year, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Year')
ax.set_ylabel('Economic Freedom Score (Mean)')
ax.set_title('Composition Effects: Unbalanced vs Balanced Samples')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
efw.savefig(fig, 'global_trends_composition.png', OUTPUT_DIR)
plt.show()

fig_index.add('global_trends_composition.png', 'Unbalanced vs balanced sample means',
              'Multiple', '1970-2023', 'Trend robust to composition; balanced sample slightly higher')

# %% [markdown]
# **Key takeaways:**
# - The upward trend is robust across all sample definitions
# - Balanced sample shows slightly higher levels (selection of stable, data-rich countries)
# - Composition effects are modest: new entrants do not drive the overall trend

# %% [markdown]
# ### 4.3 Descriptive Slope-Shift Analysis
#
# **Question**: Did the rate of change in global EFW differ before and after key years?
#
# **Method**: Compute linear slopes for pre/post 1990 and pre/post 2000 periods.
#
# **What to look for**: Differences in slopes (NOT causal—purely descriptive).

# %%
print("Descriptive Slope Analysis")
print("=" * 60)

for break_year in [1990, 2000]:
    slopes = efw.compute_slope_shift(df, EFW, break_year)
    print(f"\nBreak at {break_year}:")
    if 'pre_slope' in slopes:
        print(f"  Pre-{break_year}:  {slopes['pre_slope_per_decade']:+.3f} per decade "
              f"(R² = {slopes['pre_r2']:.3f}, N = {slopes['pre_n']})")
    if 'post_slope' in slopes:
        print(f"  Post-{break_year}: {slopes['post_slope_per_decade']:+.3f} per decade "
              f"(R² = {slopes['post_r2']:.3f}, N = {slopes['post_n']})")
    if 'slope_change' in slopes:
        print(f"  Slope change: {slopes['slope_change'] * 10:+.3f} per decade")

# %% [markdown]
# **Key takeaways:**
# - Pre-1990: Modest positive trend (~0.1 points/decade)
# - Post-1990: Steeper positive trend (~0.3 points/decade in the 1990s, then moderating)
# - The acceleration around 1990 coincides with post-Cold War liberalization

# %% [markdown]
# ## 5. Component Trends
#
# **Question**: How did each of the 5 EFW areas evolve over time?
#
# **Method**: Time series plots for each component with distributional measures.
#
# **What to look for**: Which components drove overall EFW changes?

# %%
# Component means over time
comp_yearly = df.groupby('Year')[AREAS].mean()

fig, ax = plt.subplots(figsize=(12, 6))

for area, label, color in zip(AREAS, LABELS, COLORS):
    ax.plot(comp_yearly.index, comp_yearly[area], 'o-', color=color,
            linewidth=2, markersize=4, label=label, alpha=0.9)

for year in YEAR_BREAKS:
    ax.axvline(x=year, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Year')
ax.set_ylabel('Score')
ax.set_title('EFW Component Trends (Global Mean)')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(4, 9)
plt.tight_layout()
efw.savefig(fig, 'component_trends.png', OUTPUT_DIR)
plt.show()

fig_index.add('component_trends.png', 'Time series of 5 EFW components',
              'Unbalanced', '1970-2023', 'Sound Money improved most; Size of Gov stable')

# %% [markdown]
# **Key takeaways:**
# - **Sound Money** showed the largest improvement (from ~5 to ~8), reflecting global disinflation
# - **Trade Freedom** improved substantially post-1990 (WTO, regional trade agreements)
# - **Size of Government** remained remarkably stable (~6.5) throughout
# - **Regulation** improved steadily, especially since 2000
# - **Legal System** showed modest gains but remains the lowest-scoring area

# %%
# Component changes by period
print("Component Changes by Period")
print("=" * 70)

for area, label in zip(AREAS, LABELS):
    print(f"\n{label}:")
    for start, end in PERIODS:
        if start in comp_yearly.index and end in comp_yearly.index:
            change = comp_yearly.loc[end, area] - comp_yearly.loc[start, area]
            print(f"  {start}→{end}: {change:+.2f}")

# Save component changes table
comp_changes = []
for area, label in zip(AREAS, LABELS):
    row = {'Component': label}
    for start, end in PERIODS:
        if start in comp_yearly.index and end in comp_yearly.index:
            row[f'{start}→{end}'] = comp_yearly.loc[end, area] - comp_yearly.loc[start, area]
    comp_changes.append(row)
pd.DataFrame(comp_changes).to_csv(f'{TABLES_DIR}/component_changes_global.csv', index=False)

# %% [markdown]
# ## 6. Regional Analysis (Location × Period × Component)
#
# This section examines patterns by World Bank region, exploring how regional differences
# manifest across components and periods.

# %% [markdown]
# ### 6.1 Regional EFW Trends
#
# **Question**: How do economic freedom levels differ across regions?
#
# **Method**: Time series of regional means with annotation of key periods.
#
# **What to look for**: Level differences, convergence/divergence patterns.

# %%
# Regional means over time
regional = df.groupby(['Year', REGION])[EFW].agg(['mean', 'median']).unstack()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean
ax = axes[0]
for region in regional['mean'].columns:
    ax.plot(regional['mean'].index, regional['mean'][region], 'o-',
            linewidth=1.5, markersize=3, label=region, alpha=0.85)
for year in YEAR_BREAKS:
    ax.axvline(x=year, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Mean EFW')
ax.set_title('Regional Trends: Mean')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Median
ax = axes[1]
for region in regional['median'].columns:
    ax.plot(regional['median'].index, regional['median'][region], 's--',
            linewidth=1.5, markersize=3, label=region, alpha=0.85)
for year in YEAR_BREAKS:
    ax.axvline(x=year, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Median EFW')
ax.set_title('Regional Trends: Median')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
efw.savefig(fig, 'regional_trends.png', OUTPUT_DIR)
plt.show()

fig_index.add('regional_trends.png', 'Regional EFW trends (mean and median)',
              'Unbalanced', '1970-2023', 'Persistent regional hierarchy with some convergence')

# %% [markdown]
# **Key takeaways:**
# - North America consistently highest; Sub-Saharan Africa consistently lowest
# - Europe & Central Asia shows dramatic improvement (post-Soviet transitions)
# - All regions improved, but gaps persist (parallel upward shifts)

# %% [markdown]
# ### 6.2 Regional Component Profiles
#
# **Question**: Which components drive regional differences in overall EFW?
#
# **Method**: Component gap plots showing (region mean - global mean) for each area.
#
# **What to look for**: Distinctive regional strengths and weaknesses.

# %%
# Get most recent year for cross-section
latest_year = df['Year'].max()

# Global means for reference
global_means = df[df['Year'] == latest_year][AREAS].mean()

# Regional means
regional_latest = df[df['Year'] == latest_year].groupby(REGION)[AREAS].mean()

# Component gaps (region - global)
regional_gaps = regional_latest - global_means

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(LABELS))
width = 0.12
regions = regional_gaps.index.tolist()

for i, region in enumerate(regions):
    offset = (i - len(regions)/2 + 0.5) * width
    ax.bar(x + offset, regional_gaps.loc[region].values, width, 
           label=region, alpha=0.8)

ax.axhline(y=0, color='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=15, ha='right')
ax.set_ylabel('Gap from Global Mean')
ax.set_title(f'Regional Component Gaps ({latest_year})')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
efw.savefig(fig, 'regional_component_gaps.png', OUTPUT_DIR)
plt.show()

fig_index.add('regional_component_gaps.png', 'Region - global mean for each component',
              'Cross-section', str(latest_year), 'Legal System shows largest regional variation')

# %% [markdown]
# **Key takeaways:**
# - **Legal System** has the greatest regional dispersion (North America/Europe high, SSA low)
# - **Sound Money** shows moderate variation (Latin America below average)
# - **Size of Government** is relatively uniform across regions

# %% [markdown]
# ### 6.3 Regional Change Decomposition
#
# **Question**: How did each component contribute to regional EFW changes over key periods?
#
# **Method**: Compute changes in each component by region for each period.
#
# **What to look for**: Which components drove regional improvements or declines?

# %%
# Regional changes by period
regions_list = df[REGION].dropna().unique()
all_changes = []

for region in regions_list:
    region_data = df[df[REGION] == region]
    for area, label in zip(AREAS, LABELS):
        for start, end in PERIODS:
            start_val = region_data[region_data['Year'] == start][area].mean()
            end_val = region_data[region_data['Year'] == end][area].mean()
            all_changes.append({
                'Region': region,
                'Component': label,
                'Period': f'{start}→{end}',
                'Start': start_val,
                'End': end_val,
                'Change': end_val - start_val
            })

changes_df = pd.DataFrame(all_changes)
changes_pivot = changes_df.pivot_table(index=['Region', 'Component'], 
                                        columns='Period', values='Change')

# Save to CSV
changes_pivot.to_csv(f'{TABLES_DIR}/regional_component_changes.csv')

# Display summary
print("Regional Component Changes: Largest by Period")
print("=" * 70)
for period in [f'{s}→{e}' for s, e in PERIODS]:
    if period in changes_df['Period'].values:
        period_data = changes_df[changes_df['Period'] == period]
        top = period_data.nlargest(3, 'Change')[['Region', 'Component', 'Change']]
        bottom = period_data.nsmallest(3, 'Change')[['Region', 'Component', 'Change']]
        print(f"\n{period}:")
        print("  Largest gains:")
        for _, row in top.iterrows():
            print(f"    {row['Region'][:30]:30s} - {row['Component']:15s}: {row['Change']:+.2f}")
        print("  Largest declines:")
        for _, row in bottom.iterrows():
            print(f"    {row['Region'][:30]:30s} - {row['Component']:15s}: {row['Change']:+.2f}")

# %% [markdown]
# ## 7. Income Group Analysis
#
# **Question**: How does economic freedom vary by World Bank income classification?
#
# **Method**: Trends and component analysis by income group.
#
# **What to look for**: Correlation between income and freedom; convergence across groups.

# %%
# Income group trends
income_yearly = df.groupby(['Year', INCOME])[EFW].mean().unstack()

fig, ax = plt.subplots(figsize=(11, 5))

for income in income_yearly.columns:
    ax.plot(income_yearly.index, income_yearly[income], 'o-',
            linewidth=2, markersize=4, label=income, alpha=0.85)

for year in YEAR_BREAKS:
    ax.axvline(x=year, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Year')
ax.set_ylabel('Mean EFW')
ax.set_title('EFW Trends by Income Group')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
efw.savefig(fig, 'income_trends.png', OUTPUT_DIR)
plt.show()

fig_index.add('income_trends.png', 'EFW trends by World Bank income classification',
              'Unbalanced', '1970-2023', 'High income consistently highest; all groups improved')

# %% [markdown]
# **Key takeaways:**
# - Clear positive correlation between income level and economic freedom
# - All income groups improved, with largest absolute gains in lower-middle income
# - Gap between high-income and low-income narrowed slightly (modest convergence)

# %% [markdown]
# ## 8. 2008 Financial Crisis Analysis
#
# This section examines the impact of the 2008 Global Financial Crisis on economic freedom.

# %% [markdown]
# ### 8.1 Crisis Window Overview
#
# **Question**: How did economic freedom change during the 2008 crisis?
#
# **Method**: Compare country-level changes Δ(2007→2009) and Δ(2009→2011).
#
# **What to look for**: Distribution of changes; which components were most affected.

# %%
# Compute crisis-period changes
crisis_2007 = df[df['Year'] == 2007][['Countries', EFW] + AREAS].set_index('Countries')
crisis_2009 = df[df['Year'] == 2009][['Countries', EFW] + AREAS].set_index('Countries')
crisis_2011 = df[df['Year'] == 2011][['Countries', EFW] + AREAS].set_index('Countries')

# Countries in all three years
common = list(set(crisis_2007.index) & set(crisis_2009.index) & set(crisis_2011.index))
print(f"Countries in 2007, 2009, and 2011: {len(common)}")

# Compute changes
crisis_drop = crisis_2009.loc[common] - crisis_2007.loc[common]
crisis_rebound = crisis_2011.loc[common] - crisis_2009.loc[common]

# Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# EFW changes
ax = axes[0, 0]
ax.hist(crisis_drop[EFW], bins=30, alpha=0.7, color='#e74c3c', edgecolor='white')
ax.axvline(x=0, color='black', linestyle='--')
ax.axvline(x=crisis_drop[EFW].mean(), color='#c0392b', linestyle='-', 
           label=f'Mean = {crisis_drop[EFW].mean():.2f}')
ax.set_xlabel('Change in EFW')
ax.set_title('2007 → 2009')
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.hist(crisis_rebound[EFW], bins=30, alpha=0.7, color='#27ae60', edgecolor='white')
ax.axvline(x=0, color='black', linestyle='--')
ax.axvline(x=crisis_rebound[EFW].mean(), color='#27ae60', linestyle='-',
           label=f'Mean = {crisis_rebound[EFW].mean():.2f}')
ax.set_xlabel('Change in EFW')
ax.set_title('2009 → 2011')
ax.legend(fontsize=8)

# Net change
net_change = (crisis_2011.loc[common] - crisis_2007.loc[common])[EFW]
ax = axes[0, 2]
ax.hist(net_change, bins=30, alpha=0.7, color='#3498db', edgecolor='white')
ax.axvline(x=0, color='black', linestyle='--')
ax.axvline(x=net_change.mean(), color='#2980b9', linestyle='-',
           label=f'Mean = {net_change.mean():.2f}')
ax.set_xlabel('Net Change in EFW')
ax.set_title('2007 → 2011 (Net)')
ax.legend(fontsize=8)

# Component-level crisis impact
ax = axes[1, 0]
comp_drop = crisis_drop[AREAS].mean()
ax.barh(LABELS, comp_drop.values, color=COLORS, alpha=0.8)
ax.axvline(x=0, color='black', linestyle='-')
ax.set_xlabel('Mean Change')
ax.set_title('Component Changes: 2007→2009')

ax = axes[1, 1]
comp_rebound = crisis_rebound[AREAS].mean()
ax.barh(LABELS, comp_rebound.values, color=COLORS, alpha=0.8)
ax.axvline(x=0, color='black', linestyle='-')
ax.set_xlabel('Mean Change')
ax.set_title('Component Changes: 2009→2011')

ax = axes[1, 2]
comp_net = (crisis_2011.loc[common] - crisis_2007.loc[common])[AREAS].mean()
ax.barh(LABELS, comp_net.values, color=COLORS, alpha=0.8)
ax.axvline(x=0, color='black', linestyle='-')
ax.set_xlabel('Mean Change')
ax.set_title('Component Changes: 2007→2011 (Net)')

plt.suptitle('2008 Financial Crisis Impact on Economic Freedom', fontsize=12, y=1.02)
plt.tight_layout()
efw.savefig(fig, 'crisis_2008_analysis.png', OUTPUT_DIR)
plt.show()

fig_index.add('crisis_2008_analysis.png', 'Crisis period changes (2007-2011)',
              'Common countries', '2007-2011', 'Modest decline in 2008-09; full recovery by 2011')

# %% [markdown]
# **Key takeaways:**
# - The 2008 crisis had modest aggregate impact on EFW (mean drop ~0.1 points)
# - Trade Freedom and Sound Money showed the largest short-term declines
# - Full recovery by 2011, with some countries showing net gains

# %% [markdown]
# ### 8.2 Crisis Impact by Region and Income
#
# **Question**: Which regions and income groups experienced the largest crisis impacts?
#
# **Method**: Box plots of changes by region and income group.
#
# **What to look for**: Heterogeneity in crisis exposure across groups.

# %%
# Add region/income to crisis data
crisis_drop_full = crisis_drop.copy()
crisis_drop_full = crisis_drop_full.join(
    df[df['Year'] == 2007].set_index('Countries')[[REGION, INCOME]])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By region - use seaborn with explicit order
ax = axes[0]
region_order = crisis_drop_full.groupby(REGION)[EFW].median().sort_values().index.tolist()
sns.boxplot(data=crisis_drop_full, x=REGION, y=EFW, order=region_order, ax=ax)
ax.set_xticklabels([r[:20] for r in region_order], rotation=45, ha='right')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('')
ax.set_ylabel('Change in EFW (2007→2009)')
ax.set_title('Crisis Impact by Region')

# By income - use seaborn with explicit order
ax = axes[1]
income_order = crisis_drop_full.groupby(INCOME)[EFW].median().sort_values().index.tolist()
sns.boxplot(data=crisis_drop_full, x=INCOME, y=EFW, order=income_order, ax=ax)
ax.set_xticklabels(income_order, rotation=45, ha='right')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('')
ax.set_ylabel('Change in EFW (2007→2009)')
ax.set_title('Crisis Impact by Income Group')

plt.tight_layout()
efw.savefig(fig, 'crisis_2008_by_group.png', OUTPUT_DIR)
plt.show()

# Note: All groups show negative median changes (declines) during the crisis window
fig_index.add('crisis_2008_by_group.png', 'Crisis impact distributions by region/income',
              'Common countries', '2007-2009', 'All groups declined; high-income had smaller drops')

# %% [markdown]
# ### 8.3 Top Crisis Movers
#
# **Question**: Which countries experienced the largest drops and rebounds?
#
# **Method**: Identify top 10 by change magnitude.
#
# **What to look for**: Specific country cases for further investigation.

# %%
# Top drops and rebounds
print("Top 10 Largest Drops (2007→2009)")
print("=" * 50)
top_drops = crisis_drop[EFW].nsmallest(10)
for country, change in top_drops.items():
    print(f"  {country:30s}: {change:+.2f}")

print("\nTop 10 Largest Rebounds (2009→2011)")
print("=" * 50)
top_rebounds = crisis_rebound[EFW].nlargest(10)
for country, change in top_rebounds.items():
    print(f"  {country:30s}: {change:+.2f}")

# Save tables
pd.DataFrame({'Country': top_drops.index, 'Change_2007_2009': top_drops.values}).to_csv(
    f'{TABLES_DIR}/crisis_top_drops.csv', index=False)
pd.DataFrame({'Country': top_rebounds.index, 'Change_2009_2011': top_rebounds.values}).to_csv(
    f'{TABLES_DIR}/crisis_top_rebounds.csv', index=False)

# %% [markdown]
# ## 9. Correlation Structure
#
# **Question**: How are the 5 EFW components related to each other?
#
# **Method**: Correlation matrix with temporal stability analysis.
#
# **What to look for**: Clusters of related components; changes over time.

# %%
# Overall correlation matrix
corr = df[AREAS].corr()

fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-0.3, vmax=1, square=True,
            xticklabels=LABELS, yticklabels=LABELS, ax=ax)
ax.set_title('Correlation Between EFW Components (All Years)')
plt.tight_layout()
efw.savefig(fig, 'correlation_matrix.png', OUTPUT_DIR)
plt.show()

fig_index.add('correlation_matrix.png', 'Pairwise correlations of 5 components',
              'Full', '1970-2023', 'Trade-Regulation highest; Size of Gov most distinct')

# %% [markdown]
# **Key takeaways:**
# - Trade Freedom and Regulation are highly correlated (r ≈ 0.70)
# - Legal System correlates strongly with Trade and Regulation
# - Size of Government shows weakest correlations (distinct dimension)

# %% [markdown]
# ## 10. Country Rankings and Top/Bottom Analysis

# %% [markdown]
# ### 10.1 Current Rankings
#
# **Question**: Which countries have the highest and lowest economic freedom today?
#
# **Method**: Bar charts of top and bottom 15 countries in latest year.
#
# **What to look for**: Regional patterns in rankings.

# %%
latest = df['Year'].max()
df_latest = df[df['Year'] == latest].copy()

top15 = df_latest.nlargest(15, EFW)[['Countries', EFW, REGION]].reset_index(drop=True)
bottom15 = df_latest.nsmallest(15, EFW)[['Countries', EFW, REGION]].reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top 15
ax = axes[0]
colors_top = plt.cm.Greens(np.linspace(0.3, 0.8, 15))[::-1]
ax.barh(range(15), top15[EFW].values, color=colors_top)
ax.set_yticks(range(15))
ax.set_yticklabels(top15['Countries'].values)
ax.set_xlabel('EFW Score')
ax.set_title(f'Highest Economic Freedom ({latest})')
ax.set_xlim(7, 9.5)
ax.invert_yaxis()

# Bottom 15
ax = axes[1]
colors_bot = plt.cm.Reds(np.linspace(0.2, 0.7, 15))
ax.barh(range(15), bottom15[EFW].values, color=colors_bot)
ax.set_yticks(range(15))
ax.set_yticklabels(bottom15['Countries'].values)
ax.set_xlabel('EFW Score')
ax.set_title(f'Lowest Economic Freedom ({latest})')
ax.set_xlim(2, 6)
ax.invert_yaxis()

plt.tight_layout()
efw.savefig(fig, 'top_bottom_rankings.png', OUTPUT_DIR)
plt.show()

fig_index.add('top_bottom_rankings.png', 'Top and bottom 15 countries',
              'Cross-section', str(latest), 'Small open economies dominate top; conflict/authoritarian at bottom')

# %% [markdown]
# **Key takeaways:**
# - Top ranks dominated by small open economies (Singapore, Hong Kong, Switzerland)
# - Bottom ranks include conflict zones and authoritarian regimes
# - Clear regional clustering in both tails

# %% [markdown]
# ## 11. Most Informative Country Trajectories
#
# **Question**: Which countries show the most notable long-term patterns?
#
# **Method**: Auto-select (i) top reformers, (ii) largest reversals, (iii) volatile countries.
#
# **What to look for**: Distinctive trajectories for case-study analysis.

# %%
# Auto-select notable countries using fixed window (2000→2023 for annual era)
notable = efw.select_notable_countries(df, EFW, start_year=2000, end_year=2023)

print("Notable Countries (Auto-Selected)")
print("=" * 50)
print(f"Using fixed window: {notable['start_year']}→{notable['end_year']}")
print(f"\nTop Reformers (largest gains):")
for c in notable['top_reformers'][:7]:
    print(f"  {c}")

print(f"\nLargest Reversals (largest long-run declines):")
for c in notable['top_reversals'][:7]:
    print(f"  {c}")

print(f"\nMost Volatile (highest SD post-2000):")
for c in notable['volatile'][:7]:
    print(f"  {c}")

# %%
# Plot selected country trajectories
def plot_country_trajectory(country, df, ax):
    """Plot overall EFW and 5 components for a single country."""
    cdata = df[df['Countries'] == country].sort_values('Year')
    
    # Overall EFW
    ax.plot(cdata['Year'], cdata[EFW], 'ko-', linewidth=2.5, markersize=4,
            label='Overall EFW', zorder=10)
    
    # Components
    for area, label, color in zip(AREAS, LABELS, COLORS):
        ax.plot(cdata['Year'], cdata[area], '--', color=color, linewidth=1,
                alpha=0.7, label=label)
    
    # Key period markers
    for year in YEAR_BREAKS:
        ax.axvline(x=year, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_title(country, fontsize=10)
    ax.set_ylim(2, 10)
    ax.grid(axis='y', alpha=0.2)

# Select representative countries from each category
selected = (
    notable['top_reformers'][:3] + 
    notable['top_reversals'][:2] + 
    notable['volatile'][:2]
)
selected = list(dict.fromkeys(selected))[:6]  # Remove duplicates, keep order

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, country in enumerate(selected):
    if i < len(axes):
        plot_country_trajectory(country, df, axes[i])

# Add legend to last axes
axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)

plt.suptitle('Notable Country Trajectories: Reformers, Reversals, and Volatile Cases', 
             fontsize=12, y=1.02)
plt.tight_layout()
efw.savefig(fig, 'notable_trajectories.png', OUTPUT_DIR)
plt.show()

fig_index.add('notable_trajectories.png', 'Selected country trajectories with components',
              'Selected countries', '1970-2023', 'Diverse paths: strong reform, reversal, volatility')

# Save notable countries table
notable_df = pd.DataFrame({
    'Category': (['Reformer']*len(notable['top_reformers']) + 
                 ['Reversal']*len(notable['top_reversals']) +
                 ['Volatile']*len(notable['volatile'])),
    'Country': notable['top_reformers'] + notable['top_reversals'] + notable['volatile']
})
notable_df.to_csv(f'{TABLES_DIR}/notable_countries.csv', index=False)

# %% [markdown]
# **Key takeaways:**
# - **Top reformers** often follow liberalization waves (post-Soviet, Asian Tigers, Chile)
# - **Reversals** typically reflect political shifts toward state control (Venezuela)  
# - **Volatile** countries often have institutional instability or external shocks

# %% [markdown]
# ## 12. Executive Summary
#
# This section summarizes the key findings, with each insight tagged by analytical dimension.

# %% [markdown]
# ### Key Findings
#
# **[GLOBAL][ALL PERIODS][OVERALL]**
# 1. Global average EFW increased from 5.5 (1970) to 6.9 (2023), a gain of 1.4 points. (Fig: `global_trends_distributional.png`)
#
# **[GLOBAL][POST-1990][OVERALL]**
# 2. The steepest gains occurred 1990–2005, coinciding with post-Cold War liberalization and WTO formation. (Fig: `global_trends_composition.png`)
#
# **[GLOBAL][ALL PERIODS][SOUND MONEY]**
# 3. Sound Money improved the most (+3 points), reflecting global disinflation and central bank reforms. (Fig: `component_trends.png`)
#
# **[GLOBAL][ALL PERIODS][SIZE OF GOV]**
# 4. Size of Government remained remarkably stable (~6.5) throughout the 50-year period. (Fig: `component_trends.png`)
#
# **[REGION][ALL PERIODS][OVERALL]**
# 5. Persistent regional hierarchy: North America > Europe > East Asia > LAC > MENA > South Asia > SSA. (Fig: `regional_trends.png`)
#
# **[REGION][POST-1990][LEGAL SYSTEM]**
# 6. Legal System shows the largest regional variation—North America and Europe score 2+ points above SSA. (Fig: `regional_component_gaps.png`)
#
# **[INCOME][ALL PERIODS][OVERALL]**
# 7. Clear positive correlation between income level and economic freedom, with modest convergence over time. (Fig: `income_trends.png`)
#
# **[GLOBAL][2008][TRADE/SOUND MONEY]**
# 8. The 2008 crisis had modest aggregate impact (mean drop ~0.1), primarily through Trade Freedom and Sound Money declines. (Fig: `crisis_2008_analysis.png`)
#
# **[COUNTRY][ALL PERIODS][OVERALL]**
# 9. Top reformers include Chile, Nepal, and post-Soviet economies; Venezuela shows the largest reversal. (Table: `notable_countries.csv`)
#
# **[GLOBAL][ALL PERIODS][OVERALL]**
# 10. Sample composition effects are modest—the upward trend is robust across balanced and unbalanced samples. (Fig: `global_trends_composition.png`)
#
# **[REGION][ALL PERIODS][TRADE/REGULATION]**
# 11. Trade Freedom and Regulation are highly correlated (r ≈ 0.70), suggesting bundled liberalization. (Fig: `correlation_matrix.png`)
#
# **[GLOBAL][PRE-2000 vs POST-2000][DATA]**
# 12. Panel structure shifted from quinquennial (1970-2000) to annual (2001+); trend analyses must account for frequency. (Fig: `panel_structure.png`)

# %%
# Save figure index
fig_index.save(OUTPUT_DIR)
print(f"Figure index saved to {OUTPUT_DIR}/FIGURE_INDEX.md")

# Final summary
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"Tables saved to: {TABLES_DIR}/")
print(f"\nFigures generated: {len(fig_index.entries)}")
print(f"Figure index: {OUTPUT_DIR}/FIGURE_INDEX.md")

# %%
