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
# # 02. Pull World Bank Data via API
#
# This notebook programmatically downloads World Bank Development Indicators
# via the official API (`api.worldbank.org`).
#
# **Inputs:** None (API calls)
# **Outputs:** 
# - `data/02_intermediate/worldbank_gdppc.parquet`
# - `data/02_intermediate/worldbank_population.parquet`
# - `data/02_intermediate/worldbank_raw.parquet` (combined)
#
# **Key indicators:**
# - NY.GDP.PCAP.KD: GDP per capita (constant 2015 US$)
# - NY.GDP.PCAP.PP.KD: GDP per capita, PPP (constant 2021 international $)
# - SP.POP.TOTL: Population, total

# %%
import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in dir() else Path.cwd().parent
DATA_INTERMEDIATE = PROJECT_ROOT / 'data/02_intermediate'
CACHE_DIR = DATA_INTERMEDIATE / 'worldbank_cache'

# Ensure directories exist
DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Cache directory: {CACHE_DIR}")

# %% [markdown]
# ## 1. World Bank API Configuration

# %%
# API configuration
WB_API_BASE = "https://api.worldbank.org/v2"
API_FORMAT = "json"

# Indicators to download
INDICATORS = {
    'NY.GDP.PCAP.KD': 'gdp_pc_constant',      # GDP per capita, constant 2015 US$
    'NY.GDP.PCAP.PP.KD': 'gdp_pc_ppp',        # GDP per capita, PPP, constant 2021 int'l $
    'SP.POP.TOTL': 'population',               # Total population
}

# Year range (cover all quinquennial years with buffer)
START_YEAR = 1960
END_YEAR = 2023

print("Indicators to download:")
for code, name in INDICATORS.items():
    print(f"  {code}: {name}")

# %% [markdown]
# ## 2. API Helper Functions

# %%
def fetch_wb_indicator(indicator_code: str, start_year: int = 1960, 
                       end_year: int = 2023, per_page: int = 1000) -> pd.DataFrame:
    """
    Fetch a World Bank indicator for all countries with pagination.
    
    Parameters
    ----------
    indicator_code : str
        World Bank indicator code (e.g., 'NY.GDP.PCAP.KD')
    start_year : int
        Start year for data
    end_year : int
        End year for data
    per_page : int
        Number of records per API call
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: iso3c, year, value
    """
    url = f"{WB_API_BASE}/country/all/indicator/{indicator_code}"
    params = {
        'format': 'json',
        'date': f'{start_year}:{end_year}',
        'per_page': per_page,
        'page': 1
    }
    
    all_data = []
    
    # First request to get total pages
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if len(data) < 2 or data[1] is None:
        print(f"  Warning: No data returned for {indicator_code}")
        return pd.DataFrame()
    
    # Get pagination info
    total_pages = data[0]['pages']
    total_records = data[0]['total']
    print(f"  {indicator_code}: {total_records} records across {total_pages} pages")
    
    # Parse first page
    all_data.extend(data[1])
    
    # Fetch remaining pages
    for page in range(2, total_pages + 1):
        time.sleep(0.1)  # Rate limiting
        params['page'] = page
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if len(data) >= 2 and data[1] is not None:
            all_data.extend(data[1])
    
    # Parse to DataFrame
    records = []
    for record in all_data:
        if record['value'] is not None:
            records.append({
                'iso3c': record['countryiso3code'],
                'country_name': record['country']['value'],
                'year': int(record['date']),
                'value': float(record['value']),
            })
    
    df = pd.DataFrame(records)
    return df


def load_or_fetch(indicator_code: str, cache_dir: Path, 
                  force_refresh: bool = False) -> pd.DataFrame:
    """
    Load indicator from cache or fetch from API.
    """
    cache_path = cache_dir / f"{indicator_code.replace('.', '_')}.parquet"
    
    if cache_path.exists() and not force_refresh:
        print(f"  Loading {indicator_code} from cache...")
        return pd.read_parquet(cache_path)
    else:
        print(f"  Fetching {indicator_code} from API...")
        df = fetch_wb_indicator(indicator_code, START_YEAR, END_YEAR)
        if len(df) > 0:
            df.to_parquet(cache_path, index=False)
        return df

# %% [markdown]
# ## 3. Download All Indicators

# %%
# Download all indicators
indicator_dfs = {}

print("Downloading World Bank indicators...")
for code, name in INDICATORS.items():
    print(f"\n{name}:")
    try:
        df = load_or_fetch(code, CACHE_DIR, force_refresh=False)
        if len(df) > 0:
            df = df.rename(columns={'value': name})
            indicator_dfs[name] = df
            print(f"  ✓ {len(df)} observations, {df['iso3c'].nunique()} countries")
            print(f"    Year range: {df['year'].min()} - {df['year'].max()}")
        else:
            print(f"  ✗ No data retrieved")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# %% [markdown]
# ## 4. Combine and Clean Data

# %%
# Start with GDP per capita as base
if 'gdp_pc_constant' in indicator_dfs:
    df_combined = indicator_dfs['gdp_pc_constant'][['iso3c', 'country_name', 'year', 'gdp_pc_constant']].copy()
else:
    raise ValueError("GDP per capita (constant) is required but not available")

# Merge other indicators
for name, df_indicator in indicator_dfs.items():
    if name != 'gdp_pc_constant':
        df_combined = df_combined.merge(
            df_indicator[['iso3c', 'year', name]],
            on=['iso3c', 'year'],
            how='outer'
        )

print(f"\nCombined shape: {df_combined.shape}")
print(f"Countries: {df_combined['iso3c'].nunique()}")
print(f"Year range: {df_combined['year'].min()} - {df_combined['year'].max()}")

# %% [markdown]
# ## 5. Filter to Valid Countries

# %%
# Remove aggregates (World Bank region codes are not valid ISO3)
# Valid ISO3 codes are 3 letters, aggregates often have different patterns

# Common aggregate prefixes to exclude
AGGREGATES = [
    'ARB', 'CSS', 'CEB', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU',
    'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC',
    'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC',
    'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA',
    'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD', 
]

# Filter out aggregates
df_countries = df_combined[~df_combined['iso3c'].isin(AGGREGATES)].copy()

# Also filter out empty ISO codes
df_countries = df_countries[df_countries['iso3c'].notna() & (df_countries['iso3c'] != '')]

print(f"After removing aggregates: {len(df_countries)} observations")
print(f"Countries: {df_countries['iso3c'].nunique()}")

# %% [markdown]
# ## 6. Filter to Quinquennial Years

# %%
QUINQUENNIAL_YEARS = [1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

df_quin = df_countries[df_countries['year'].isin(QUINQUENNIAL_YEARS)].copy()

print(f"Quinquennial observations: {len(df_quin)}")
print(f"Years: {sorted(df_quin['year'].unique())}")

# Coverage by year
print("\nCoverage by quinquennial year:")
for year in QUINQUENNIAL_YEARS:
    n_gdp = df_quin[(df_quin['year'] == year) & df_quin['gdp_pc_constant'].notna()]['iso3c'].nunique()
    print(f"  {year}: {n_gdp} countries with GDP per capita")

# %% [markdown]
# ## 7. Create Log GDP and Growth Variables

# %%
# Sort for proper differencing
df_quin = df_quin.sort_values(['iso3c', 'year'])

# Log GDP per capita
df_quin['ln_gdp_pc'] = np.log(df_quin['gdp_pc_constant'])

# Growth rate (quinquennial log difference)
df_quin['growth'] = df_quin.groupby('iso3c')['ln_gdp_pc'].diff()

# Verify timing
df_quin['year_lag'] = df_quin.groupby('iso3c')['year'].shift(1)
df_quin['year_diff'] = df_quin['year'] - df_quin['year_lag']

# Check all differences are 5 years
valid_growth = df_quin['year_diff'].dropna()
assert (valid_growth == 5).all(), "Non-quinquennial year differences found"
print("✓ All growth calculations use 5-year differences")

# Clean up temp columns
df_quin = df_quin.drop(columns=['year_lag', 'year_diff'])

# Summary statistics
print("\nGrowth statistics (quinquennial log difference):")
print(df_quin['growth'].describe().round(4))

# %% [markdown]
# ## 8. Validation Checks

# %%
print("\n" + "="*60)
print("VALIDATION CHECKS")
print("="*60)

# 1. No duplicate country-year keys
assert not df_quin.duplicated(subset=['iso3c', 'year']).any(), "Duplicate keys found"
print("✓ No duplicate country-year keys")

# 2. GDP per capita in reasonable range
gdp_valid = df_quin['gdp_pc_constant'].dropna()
assert gdp_valid.min() > 100, "GDP per capita too low"
assert gdp_valid.max() < 200000, "GDP per capita too high"
print(f"✓ GDP per capita range: ${gdp_valid.min():,.0f} - ${gdp_valid.max():,.0f}")

# 3. Growth in reasonable range
growth_valid = df_quin['growth'].dropna()
assert growth_valid.min() > -2, "Growth rate implausibly negative"
assert growth_valid.max() < 2, "Growth rate implausibly positive"
print(f"✓ Growth range: {growth_valid.min():.3f} to {growth_valid.max():.3f}")

# 4. Sufficient coverage
assert df_quin['iso3c'].nunique() > 150, "Too few countries"
print(f"✓ Coverage: {df_quin['iso3c'].nunique()} countries")

# %% [markdown]
# ## 9. Save Outputs

# %%
# Save combined quinquennial data
output_path = DATA_INTERMEDIATE / 'worldbank_raw.parquet'
df_quin.to_parquet(output_path, index=False)
print(f"\n✓ Saved World Bank data to {output_path}")
print(f"  Observations: {len(df_quin)}")
print(f"  Countries: {df_quin['iso3c'].nunique()}")

# Save metadata
metadata = {
    'indicators': INDICATORS,
    'quinquennial_years': QUINQUENNIAL_YEARS,
    'source': 'World Bank API (api.worldbank.org)',
    'date_downloaded': pd.Timestamp.now().isoformat(),
    'variables': {
        'gdp_pc_constant': 'GDP per capita, constant 2015 US$',
        'gdp_pc_ppp': 'GDP per capita, PPP, constant 2021 international $',
        'population': 'Total population',
        'ln_gdp_pc': 'Log of gdp_pc_constant',
        'growth': 'Quinquennial log difference of GDP per capita',
    }
}

metadata_path = DATA_INTERMEDIATE / 'worldbank_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata to {metadata_path}")

# %% [markdown]
# ## 10. Summary

# %%
print("\n" + "="*60)
print("02_PULL_WORLDBANK COMPLETE")
print("="*60)
print(f"Downloaded {len(INDICATORS)} indicators from World Bank API")
print(f"Final dataset: {len(df_quin)} obs, {df_quin['iso3c'].nunique()} countries")
print(f"Quinquennial years: {QUINQUENNIAL_YEARS[0]} - {QUINQUENNIAL_YEARS[-1]}")
print("="*60)
