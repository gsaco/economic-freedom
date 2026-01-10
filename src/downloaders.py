"""
Data Downloaders
================
Multi-source downloaders with caching, retries, and validation.

Sources:
- World Bank WDI (primary)
- Penn World Table (fallback/benchmark for GDP/productivity)
- IMF WEO (fallback for fiscal/inflation)
- Chinn-Ito KAOPEN (capital account openness)
- KOF Globalisation Index
- Worldwide Governance Indicators (WGI)
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings
import io
import zipfile

# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DIR = Path("data/raw")
CACHE_METADATA_FILE = RAW_DIR / "download_metadata.json"

# WDI API endpoint
WDI_BASE_URL = "https://api.worldbank.org/v2"

# Indicator definitions with fallback sources
INDICATOR_REGISTRY = {
    # Tier 1: Core macro
    "gdppc_constant": {
        "wdi_code": "NY.GDP.PCAP.KD",
        "name": "GDP per capita (constant 2015 USD)",
        "units": "USD",
        "tier": 1,
        "fallback": "pwt"
    },
    "gdp_growth": {
        "wdi_code": "NY.GDP.MKTP.KD.ZG",
        "name": "GDP growth (annual %)",
        "units": "%",
        "tier": 1
    },
    "inflation_cpi": {
        "wdi_code": "FP.CPI.TOTL.ZG",
        "name": "Inflation, consumer prices (annual %)",
        "units": "%",
        "tier": 1,
        "fallback": "imf_weo"
    },
    "population": {
        "wdi_code": "SP.POP.TOTL",
        "name": "Population, total",
        "units": "persons",
        "tier": 1
    },
    "trade_openness": {
        "wdi_code": "NE.TRD.GNFS.ZS",
        "name": "Trade (% of GDP)",
        "units": "%GDP",
        "tier": 1
    },
    "investment_gfcf": {
        "wdi_code": "NE.GDI.FTOT.ZS",
        "name": "Gross fixed capital formation (% of GDP)",
        "units": "%GDP",
        "tier": 1
    },
    "govt_consumption": {
        "wdi_code": "NE.CON.GOVT.ZS",
        "name": "General govt final consumption (% of GDP)",
        "units": "%GDP",
        "tier": 1
    },
    "unemployment": {
        "wdi_code": "SL.UEM.TOTL.ZS",
        "name": "Unemployment, total (% of labor force)",
        "units": "%",
        "tier": 1,
        "note": "High missingness expected"
    },
    # Tier 2: EFW-relevant comparators
    "private_credit": {
        "wdi_code": "FS.AST.PRVT.GD.ZS",
        "name": "Domestic credit to private sector (% of GDP)",
        "units": "%GDP",
        "tier": 2
    },
    "fdi_inflows": {
        "wdi_code": "BX.KLT.DINV.WD.GD.ZS",
        "name": "FDI net inflows (% of GDP)",
        "units": "%GDP",
        "tier": 2
    },
    "current_account": {
        "wdi_code": "BN.CAB.XOKA.GD.ZS",
        "name": "Current account balance (% of GDP)",
        "units": "%GDP",
        "tier": 2
    },
    "tax_revenue": {
        "wdi_code": "GC.TAX.TOTL.GD.ZS",
        "name": "Tax revenue (% of GDP)",
        "units": "%GDP",
        "tier": 2
    },
    "govt_debt": {
        "wdi_code": "GC.DOD.TOTL.GD.ZS",
        "name": "Central government debt (% of GDP)",
        "units": "%GDP",
        "tier": 2
    },
    "broad_money": {
        "wdi_code": "FM.LBL.BMNY.GD.ZS",
        "name": "Broad money (% of GDP)",
        "units": "%GDP",
        "tier": 2
    },
    # Tier 3: Growth/Development controls (Classic Literature)
    "school_enrollment_secondary": {
        "wdi_code": "SE.SEC.ENRR",
        "name": "School enrollment, secondary (% gross)",
        "units": "%",
        "tier": 3
    },
    "gini_index": {
        "wdi_code": "SI.POV.GINI",
        "name": "Gini index",
        "units": "Index",
        "tier": 3,
        "note": "High missingness expected"
    },
    "life_expectancy": {
        "wdi_code": "SP.DYN.LE00.IN",
        "name": "Life expectancy at birth, total (years)",
        "units": "Years",
        "tier": 3
    },
    "natural_resources_rents": {
        "wdi_code": "NY.GDP.TOTL.RT.ZS",
        "name": "Total natural resources rents (% of GDP)",
        "units": "%GDP",
        "tier": 3
    },
    "labor_force_participation": {
        "wdi_code": "SL.TLF.CACT.ZS",
        "name": "Labor force participation rate, total (% of total population ages 15+)",
        "units": "%",
        "tier": 3
    }
}


def _ensure_raw_dir():
    """Create raw data directory if it doesn't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache_metadata() -> dict:
    """Load download metadata from cache."""
    if CACHE_METADATA_FILE.exists():
        with open(CACHE_METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}


def _save_cache_metadata(metadata: dict):
    """Save download metadata to cache."""
    _ensure_raw_dir()
    with open(CACHE_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def _update_cache_metadata(source: str, indicators: list, date: str, url: str = None):
    """Update cache metadata for a source."""
    metadata = _load_cache_metadata()
    metadata[source] = {
        "pull_date": date,
        "indicators": indicators,
        "url": url
    }
    _save_cache_metadata(metadata)


# =============================================================================
# WORLD BANK WDI DOWNLOADER
# =============================================================================

def download_wdi(
    indicators: List[str] = None,
    start_year: int = 1960,
    end_year: int = None,
    force_download: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download World Bank WDI indicators via API.
    
    Parameters
    ----------
    indicators : list, optional
        List of WDI indicator codes. If None, downloads all Tier 1 indicators.
    start_year : int
        Start year for data
    end_year : int, optional
        End year for data. If None, uses current year.
    force_download : bool
        If True, bypass cache and re-download
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        Long-form panel with columns: iso3, year, variable, value, source
    """
    _ensure_raw_dir()
    cache_file = RAW_DIR / "wdi_data.parquet"
    
    if end_year is None:
        end_year = datetime.now().year
    
    # Default to Tier 1 indicators
    if indicators is None:
        indicators = [v["wdi_code"] for v in INDICATOR_REGISTRY.values() 
                      if "wdi_code" in v]
    
    # Check cache
    if cache_file.exists() and not force_download:
        metadata = _load_cache_metadata()
        cached_inds = set(metadata.get('wdi', {}).get('indicators', []))
        required_inds = set(indicators)
        missing_inds = required_inds - cached_inds
        
        if not missing_inds:
            if verbose:
                print(f"Loading WDI from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            return df
        elif verbose:
            print(f"Cache exists but is missing {len(missing_inds)} indicators. Redownloading...")
            # Fall through to download logic
    
    if verbose:
        print(f"Downloading WDI data for {len(indicators)} indicators...")
        print(f"  Period: {start_year}-{end_year}")
    
    all_data = []
    failed = []
    
    for i, indicator in enumerate(indicators):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(indicators)}")
        
        try:
            df_ind = _fetch_wdi_indicator(indicator, start_year, end_year)
            if df_ind is not None and len(df_ind) > 0:
                all_data.append(df_ind)
        except Exception as e:
            failed.append((indicator, str(e)))
            if verbose:
                print(f"  ⚠ Failed: {indicator} - {e}")
    
    if not all_data:
        raise RuntimeError("No WDI data retrieved. Check network/API.")
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Add source column
    df['source'] = 'WDI'
    
    # Save to cache
    df.to_parquet(cache_file, index=False)
    _update_cache_metadata(
        "wdi",
        indicators,
        datetime.now().isoformat(),
        WDI_BASE_URL
    )
    
    if verbose:
        print(f"✓ Downloaded {len(df):,} observations for {df['variable'].nunique()} indicators")
        print(f"  Countries: {df['iso3'].nunique()}")
        print(f"  Years: {df['year'].min()}-{df['year'].max()}")
        if failed:
            print(f"  Failed indicators: {len(failed)}")
    
    return df


def _fetch_wdi_indicator(
    indicator: str,
    start_year: int,
    end_year: int,
    max_retries: int = 3
) -> Optional[pd.DataFrame]:
    """Fetch a single WDI indicator with retries."""
    
    url = (
        f"{WDI_BASE_URL}/country/all/indicator/{indicator}"
        f"?format=json&per_page=20000&date={start_year}:{end_year}"
    )
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # WDI returns [metadata, data] or just metadata if no data
            if len(data) < 2 or data[1] is None:
                return None
            
            records = data[1]
            
            rows = []
            for rec in records:
                if rec.get('value') is not None:
                    rows.append({
                        'iso3': rec['countryiso3code'],
                        'year': int(rec['date']),
                        'variable': indicator,
                        'value': float(rec['value'])
                    })
            
            if rows:
                return pd.DataFrame(rows)
            return None
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    
    return None


# =============================================================================
# PENN WORLD TABLE DOWNLOADER
# =============================================================================

def download_pwt(
    force_download: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download Penn World Table data.
    
    Uses version 10.01 (latest publicly available).
    
    Returns
    -------
    pd.DataFrame
        Long-form panel with PWT variables
    """
    _ensure_raw_dir()
    cache_file = RAW_DIR / "pwt_data.parquet"
    
    # PWT 10.01 download URL (Excel format)
    PWT_URL = "https://www.rug.nl/ggdc/productivity/pwt/pwt-releases/pwt-10.01/?lang=en"
    PWT_DIRECT_URL = "https://dataverse.nl/api/access/datafile/354098"  # Direct CSV
    
    if cache_file.exists() and not force_download:
        if verbose:
            print(f"Loading PWT from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    if verbose:
        print("Downloading Penn World Table 10.01...")
    
    try:
        # Try direct data file download (Stata format)
        response = requests.get(PWT_DIRECT_URL, timeout=120)
        response.raise_for_status()
        
        # Read Stata file
        df_raw = pd.read_stata(io.BytesIO(response.content))
        
        # Key PWT variables to extract
        pwt_vars = {
            'rgdpna': 'gdp_real_national',      # Real GDP at constant national prices
            'rgdpe': 'gdp_real_expenditure',    # Real GDP expenditure-side
            'rgdpo': 'gdp_real_output',         # Real GDP output-side
            'pop': 'population_pwt',            # Population (millions)
            'emp': 'employment',                # Employment (millions)
            'avh': 'avg_hours_worked',          # Average hours worked
            'hc': 'human_capital',              # Human capital index
            'ctfp': 'tfp',                      # TFP at constant national prices
            'csh_i': 'investment_share',        # Investment share
            'csh_g': 'govt_share',              # Govt consumption share
            'csh_x': 'export_share',            # Export share
            'csh_m': 'import_share',            # Import share
        }
        
        # Reshape to long form
        id_cols = ['countrycode', 'year']
        value_cols = [c for c in pwt_vars.keys() if c in df_raw.columns]
        
        df_long = df_raw[id_cols + value_cols].melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='variable',
            value_name='value'
        )
        
        # Clean up
        df_long = df_long.rename(columns={'countrycode': 'iso3'})
        df_long = df_long.dropna(subset=['value'])
        df_long['source'] = 'PWT'
        
        # Compute GDP per capita
        if 'rgdpe' in df_raw.columns and 'pop' in df_raw.columns:
            df_raw['gdppc_pwt'] = df_raw['rgdpe'] / df_raw['pop']  # millions / millions = units
            gdppc_long = df_raw[['countrycode', 'year', 'gdppc_pwt']].dropna()
            gdppc_long = gdppc_long.rename(columns={'countrycode': 'iso3', 'gdppc_pwt': 'value'})
            gdppc_long['variable'] = 'gdppc_pwt'
            gdppc_long['source'] = 'PWT'
            df_long = pd.concat([df_long, gdppc_long], ignore_index=True)
        
        # Save
        df_long.to_parquet(cache_file, index=False)
        _update_cache_metadata("pwt", list(pwt_vars.values()), datetime.now().isoformat(), PWT_URL)
        
        if verbose:
            print(f"✓ Downloaded PWT: {len(df_long):,} observations")
            print(f"  Variables: {df_long['variable'].nunique()}")
            print(f"  Years: {df_long['year'].min()}-{df_long['year'].max()}")
        
        return df_long
        
    except Exception as e:
        warnings.warn(f"PWT download failed: {e}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['iso3', 'year', 'variable', 'value', 'source'])


# =============================================================================
# KAOPEN (CHINN-ITO) DOWNLOADER
# =============================================================================

def download_kaopen(
    force_download: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download Chinn-Ito KAOPEN capital account openness index.
    
    Source: http://web.pdx.edu/~ito/Chinn-Ito_website.htm
    
    Returns
    -------
    pd.DataFrame
        Long-form panel with KAOPEN variables
    """
    _ensure_raw_dir()
    cache_file = RAW_DIR / "kaopen_data.parquet"
    
    # KAOPEN data URL (Excel)
    KAOPEN_URL = "http://web.pdx.edu/~ito/kaopen_2021.xlsx"
    
    if cache_file.exists() and not force_download:
        if verbose:
            print(f"Loading KAOPEN from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    if verbose:
        print("Downloading Chinn-Ito KAOPEN index...")
    
    try:
        response = requests.get(KAOPEN_URL, timeout=60)
        response.raise_for_status()
        
        # Read Excel
        df_raw = pd.read_excel(io.BytesIO(response.content))
        
        # Expected columns: country_name, ccode (WDI code), year, kaopen, ka_open (normalized)
        # Find the relevant columns
        year_col = [c for c in df_raw.columns if 'year' in c.lower()][0]
        kaopen_col = [c for c in df_raw.columns if c.lower() == 'kaopen' or c.lower() == 'ka_open'][0]
        
        # Try to find ISO3 code - might be named differently
        iso_candidates = ['ccode', 'iso3', 'countrycode', 'iso']
        iso_col = None
        for cand in iso_candidates:
            matches = [c for c in df_raw.columns if cand in c.lower()]
            if matches:
                iso_col = matches[0]
                break
        
        if iso_col is None:
            # Fall back to country name + concordance later
            name_col = [c for c in df_raw.columns if 'country' in c.lower()][0]
            warnings.warn("KAOPEN: ISO3 not found, will need concordance.")
            df_raw['iso3'] = df_raw[name_col]  # Placeholder
        else:
            df_raw['iso3'] = df_raw[iso_col]
        
        # Reshape
        df_long = df_raw[['iso3', year_col, kaopen_col]].copy()
        df_long.columns = ['iso3', 'year', 'value']
        df_long['variable'] = 'kaopen'
        df_long['source'] = 'KAOPEN'
        df_long = df_long.dropna(subset=['value'])
        
        # Save
        df_long.to_parquet(cache_file, index=False)
        _update_cache_metadata("kaopen", ['kaopen'], datetime.now().isoformat(), KAOPEN_URL)
        
        if verbose:
            print(f"✓ Downloaded KAOPEN: {len(df_long):,} observations")
            print(f"  Years: {df_long['year'].min()}-{df_long['year'].max()}")
        
        return df_long
        
    except Exception as e:
        warnings.warn(f"KAOPEN download failed: {e}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['iso3', 'year', 'variable', 'value', 'source'])


# =============================================================================
# KOF GLOBALISATION INDEX DOWNLOADER
# =============================================================================

def download_kof(
    force_download: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download KOF Globalisation Index.
    
    Source: https://kof.ethz.ch/en/forecasts-and-indicators/indicators/kof-globalisation-index.html
    
    Returns
    -------
    pd.DataFrame
        Long-form panel with KOF indices
    """
    _ensure_raw_dir()
    cache_file = RAW_DIR / "kof_data.parquet"
    
    # KOF data URL
    KOF_URL = "https://ethz.ch/content/dam/ethz/special-interest/dual/kof-dam/documents/Globalization/2023/KOFGI_2023_public.xlsx"
    
    if cache_file.exists() and not force_download:
        if verbose:
            print(f"Loading KOF from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    if verbose:
        print("Downloading KOF Globalisation Index...")
    
    try:
        response = requests.get(KOF_URL, timeout=120)
        response.raise_for_status()
        
        # Read Excel - structure varies by year, try common patterns
        xls = pd.ExcelFile(io.BytesIO(response.content))
        
        # Look for the main data sheet
        sheet_name = None
        for name in xls.sheet_names:
            if 'data' in name.lower() or 'kof' in name.lower():
                sheet_name = name
                break
        if sheet_name is None:
            sheet_name = xls.sheet_names[0]
        
        df_raw = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Common structure: first column is country/code, subsequent columns are years
        # Identify the pattern
        if 'code' in df_raw.columns[0].lower() or 'country' in df_raw.columns[0].lower():
            # Wide format: country, year1, year2, ...
            id_col = df_raw.columns[0]
            year_cols = [c for c in df_raw.columns[1:] if str(c).isdigit() or 
                        (isinstance(c, (int, float)) and not pd.isna(c))]
            
            df_long = df_raw[[id_col] + year_cols].melt(
                id_vars=[id_col],
                value_vars=year_cols,
                var_name='year',
                value_name='value'
            )
            df_long = df_long.rename(columns={id_col: 'iso3'})
            df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce').astype(int)
        else:
            # Try long format
            df_long = df_raw.copy()
        
        df_long['variable'] = 'kof_globalization'
        df_long['source'] = 'KOF'
        df_long = df_long.dropna(subset=['value'])
        
        # Save
        df_long.to_parquet(cache_file, index=False)
        _update_cache_metadata("kof", ['kof_globalization'], datetime.now().isoformat(), KOF_URL)
        
        if verbose:
            print(f"✓ Downloaded KOF: {len(df_long):,} observations")
            print(f"  Years: {df_long['year'].min()}-{df_long['year'].max()}")
        
        return df_long
        
    except Exception as e:
        warnings.warn(f"KOF download failed: {e}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['iso3', 'year', 'variable', 'value', 'source'])


# =============================================================================
# MADDISON PROJECT DATABASE DOWNLOADER
# =============================================================================

def download_maddison(
    force_download: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download Maddison Project Database 2020 (Historical GDP p.c.).
    
    Source: https://www.rug.nl/ggdc/historicaldevelopment/maddison/data/mpd2020.xlsx
    
    Returns
    -------
    pd.DataFrame
        Long-form panel
    """
    _ensure_raw_dir()
    cache_file = RAW_DIR / "maddison_data.parquet"
    MPD_URL = "https://www.rug.nl/ggdc/historicaldevelopment/maddison/data/mpd2020.xlsx"
    
    if cache_file.exists() and not force_download:
        if verbose:
            print(f"Loading Maddison from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    if verbose:
        print("Downloading Maddison Project Database 2020...")
        
    try:
        response = requests.get(MPD_URL, timeout=120)
        response.raise_for_status()
        
        # Read Excel (sheet 'Full data')
        df_raw = pd.read_excel(io.BytesIO(response.content), sheet_name='Full data')
        
        # Cols: countrycode, country, year, gdppc, pop
        df_long = df_raw.rename(columns={'countrycode': 'iso3'}).copy()
        
        # Extract GDP per capita (gdppc)
        df_long = df_long[['iso3', 'year', 'gdppc']].dropna()
        df_long['variable'] = 'gdppc_maddison'
        df_long = df_long.rename(columns={'gdppc': 'value'})
        df_long['source'] = 'MPD'
        
        # Save
        df_long.to_parquet(cache_file, index=False)
        _update_cache_metadata("maddison", ['gdppc_maddison'], datetime.now().isoformat(), MPD_URL)
        
        if verbose:
            print(f"✓ Downloaded MPD: {len(df_long):,} observations")
        
        return df_long
        
    except Exception as e:
        warnings.warn(f"Maddison download failed: {e}")
        return pd.DataFrame(columns=['iso3', 'year', 'variable', 'value', 'source'])


# =============================================================================
# BARRO-LEE EDUCATION DOWNLOADER
# =============================================================================

def download_barro_lee(
    force_download: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download Barro-Lee Educational Attainment Data (v3.0, 2021).
    
    Source: http://barrolee.com
    
    Returns
    -------
    pd.DataFrame
        Long-form panel
    """
    _ensure_raw_dir()
    cache_file = RAW_DIR / "barro_lee_data.parquet"
    # Use reliable CSV link for latest version (2021 release)
    BL_URL = "http://www.barrolee.com/data/BL_v3/BL_v3_MF1599.csv"
    
    if cache_file.exists() and not force_download:
        if verbose:
            print(f"Loading Barro-Lee from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    if verbose:
        print("Downloading Barro-Lee Educational Attainment...")
        
    try:
        # User-agent sometimes needed for these academic sites
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(BL_URL, headers=headers, timeout=120)
        response.raise_for_status()
        
        df_raw = pd.read_csv(io.StringIO(response.text))
        
        # Cols: BLcode, country, year, sex, age, yr_sch, ...
        # We focus on Total Population (sex='MF') and Age 15+ (age='1599' or similar logic if implicit in file)
        # The file BL_v3_MF1599.csv is typically MF (Both sexes) 15+ 
        
        # Rename ISO
        # BLcode is usually 3-letter ISO
        
        # Extract Years of Schooling
        if 'BLcode' in df_raw.columns and 'yr_sch' in df_raw.columns:
            df_sub = df_raw[['BLcode', 'year', 'yr_sch']].copy()
            df_sub = df_sub.rename(columns={'BLcode': 'iso3', 'yr_sch': 'value'})
            df_sub['variable'] = 'years_schooling'
            df_sub['source'] = 'BarroLee'
            df_sub = df_sub.dropna()
            
            # Save
            df_sub.to_parquet(cache_file, index=False)
            _update_cache_metadata("barro_lee", ['years_schooling'], datetime.now().isoformat(), BL_URL)
            
            if verbose:
                print(f"✓ Downloaded Barro-Lee: {len(df_sub):,} observations")
            
            return df_sub
            
        else:
            raise ValueError("Unexpected columns in Barro-Lee file")
            
    except Exception as e:
        warnings.warn(f"Barro-Lee download failed: {e}")
        return pd.DataFrame(columns=['iso3', 'year', 'variable', 'value', 'source'])

def download_wgi(
    force_download: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download Worldwide Governance Indicators.
    
    Note: WGI starts in 1996 with biennial data, annual from 2002.
    
    Source: https://info.worldbank.org/governance/wgi/
    
    Returns
    -------
    pd.DataFrame
        Long-form panel with WGI indicators
    """
    _ensure_raw_dir()
    cache_file = RAW_DIR / "wgi_data.parquet"
    
    # WGI indicators available via WDI API
    WGI_INDICATORS = {
        'CC.EST': 'wgi_control_corruption',    # Control of Corruption
        'GE.EST': 'wgi_govt_effectiveness',    # Government Effectiveness
        'PV.EST': 'wgi_political_stability',   # Political Stability
        'RQ.EST': 'wgi_regulatory_quality',    # Regulatory Quality
        'RL.EST': 'wgi_rule_of_law',           # Rule of Law
        'VA.EST': 'wgi_voice_accountability',  # Voice and Accountability
    }
    
    if cache_file.exists() and not force_download:
        if verbose:
            print(f"Loading WGI from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    if verbose:
        print("Downloading World Governance Indicators via WDI API...")
    
    try:
        # WGI is available via WDI API
        all_data = []
        for wdi_code, var_name in WGI_INDICATORS.items():
            df_ind = _fetch_wdi_indicator(wdi_code, 1996, datetime.now().year)
            if df_ind is not None and len(df_ind) > 0:
                df_ind['variable'] = var_name
                all_data.append(df_ind)
        
        if all_data:
            df_long = pd.concat(all_data, ignore_index=True)
            df_long['source'] = 'WGI'
            
            # Save
            df_long.to_parquet(cache_file, index=False)
            _update_cache_metadata("wgi", list(WGI_INDICATORS.values()), 
                                   datetime.now().isoformat(), WDI_BASE_URL)
            
            if verbose:
                print(f"✓ Downloaded WGI: {len(df_long):,} observations")
                print(f"  Years: {df_long['year'].min()}-{df_long['year'].max()}")
            
            return df_long
        else:
            return pd.DataFrame(columns=['iso3', 'year', 'variable', 'value', 'source'])
            
    except Exception as e:
        warnings.warn(f"WGI download failed: {e}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['iso3', 'year', 'variable', 'value', 'source'])


# =============================================================================
# MASTER DOWNLOAD FUNCTION
# =============================================================================

def download_all_sources(
    force_download: bool = False,
    include_secondary: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download data from all available sources and combine.
    
    Parameters
    ----------
    force_download : bool
        If True, bypass cache and re-download all sources
    include_secondary : bool
        If True, include PWT, KAOPEN, KOF, WGI
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        Combined long-form panel with source column
    """
    if verbose:
        print("=" * 60)
        print("DOWNLOADING MACRO DATA FROM MULTIPLE SOURCES")
        print("=" * 60)
        print(f"  Force download: {force_download}")
        print(f"  Include secondary sources: {include_secondary}")
        print()
    
    all_data = []
    
    # 1. World Bank WDI (primary)
    try:
        df_wdi = download_wdi(force_download=force_download, verbose=verbose)
        all_data.append(df_wdi)
    except Exception as e:
        print(f"ERROR: WDI download failed: {e}")
        warnings.warn(f"WDI download failed: {e}")
    
    if include_secondary:
        # 2. Penn World Table
        try:
            df_pwt = download_pwt(force_download=force_download, verbose=verbose)
            if len(df_pwt) > 0:
                all_data.append(df_pwt)
        except Exception as e:
            print(f"ERROR: PWT download failed: {e}")
            warnings.warn(f"PWT download failed: {e}")
        
        # 3. KAOPEN
        try:
            df_kaopen = download_kaopen(force_download=force_download, verbose=verbose)
            if len(df_kaopen) > 0:
                all_data.append(df_kaopen)
        except Exception as e:
            warnings.warn(f"KAOPEN download failed: {e}")
        
        # 4. KOF
        try:
            df_kof = download_kof(force_download=force_download, verbose=verbose)
            if len(df_kof) > 0:
                all_data.append(df_kof)
        except Exception as e:
            warnings.warn(f"KOF download failed: {e}")
        
        # 5. WGI
        try:
            df_wgi = download_wgi(force_download=force_download, verbose=verbose)
            if len(df_wgi) > 0:
                all_data.append(df_wgi)
        except Exception as e:
            print(f"ERROR: WGI download failed: {e}")
            warnings.warn(f"WGI download failed: {e}")

        # 6. Maddison Project (MPD)
        try:
            df_mpd = download_maddison(force_download=force_download, verbose=verbose)
            if len(df_mpd) > 0:
                all_data.append(df_mpd)
        except Exception as e:
            print(f"ERROR: MPD download failed: {e}")
            warnings.warn(f"MPD download failed: {e}")

        # 7. Barro-Lee (Education)
        try:
            df_bl = download_barro_lee(force_download=force_download, verbose=verbose)
            if len(df_bl) > 0:
                all_data.append(df_bl)
        except Exception as e:
            print(f"ERROR: Barro-Lee download failed: {e}")
            warnings.warn(f"Barro-Lee download failed: {e}")
    
    if not all_data:
        raise RuntimeError("No data downloaded from any source.")
    
    df_combined = pd.concat(all_data, ignore_index=True)
    
    if verbose:
        print()
        print("=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Total observations: {len(df_combined):,}")
        print(f"Countries: {df_combined['iso3'].nunique()}")
        print(f"Variables: {df_combined['variable'].nunique()}")
        print(f"Years: {df_combined['year'].min()}-{df_combined['year'].max()}")
        print("\nBy source:")
        print(df_combined.groupby('source').size().to_string())
    
    return df_combined


# =============================================================================
# CACHE UTILITIES
# =============================================================================

def list_cached_data() -> dict:
    """List all cached data files."""
    _ensure_raw_dir()
    cached = {}
    for f in RAW_DIR.glob("*.parquet"):
        cached[f.stem] = {
            "path": str(f),
            "size_mb": f.stat().st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        }
    return cached


def clear_cache(source: str = None):
    """Clear cached data files."""
    _ensure_raw_dir()
    if source:
        cache_file = RAW_DIR / f"{source}_data.parquet"
        if cache_file.exists():
            cache_file.unlink()
            print(f"Cleared cache for {source}")
    else:
        for f in RAW_DIR.glob("*.parquet"):
            f.unlink()
        print("Cleared all cached data")
