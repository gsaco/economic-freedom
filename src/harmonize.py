"""
Data Harmonization
==================
ISO3 concordance, country code resolution, and long-to-wide transformations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

# =============================================================================
# ISO3 CONCORDANCE
# =============================================================================

# Manual concordance for common problem cases
MANUAL_CONCORDANCE = {
    # EFW names → ISO3
    'Hong Kong SAR, China': 'HKG',
    'Hong Kong': 'HKG',
    'Korea, Rep.': 'KOR',
    'Korea, South': 'KOR',
    'South Korea': 'KOR',
    'Korea, North': 'PRK',
    'North Korea': 'PRK',
    'Taiwan, China': 'TWN',
    'Taiwan': 'TWN',
    'Venezuela, RB': 'VEN',
    'Venezuela': 'VEN',
    'Egypt, Arab Rep.': 'EGY',
    'Egypt': 'EGY',
    'Iran, Islamic Rep.': 'IRN',
    'Iran': 'IRN',
    'Syrian Arab Republic': 'SYR',
    'Syria': 'SYR',
    'Yemen, Rep.': 'YEM',
    'Yemen': 'YEM',
    'Congo, Rep.': 'COG',
    'Congo': 'COG',
    'Congo, Dem. Rep.': 'COD',
    'Democratic Republic of the Congo': 'COD',
    'Côte d\'Ivoire': 'CIV',
    'Ivory Coast': 'CIV',
    'Cote d\'Ivoire': 'CIV',
    'Slovak Republic': 'SVK',
    'Slovakia': 'SVK',
    'Czech Republic': 'CZE',
    'Czechia': 'CZE',
    'Russian Federation': 'RUS',
    'Russia': 'RUS',
    'Lao PDR': 'LAO',
    'Laos': 'LAO',
    'Kyrgyz Republic': 'KGZ',
    'Kyrgyzstan': 'KGZ',
    'Myanmar': 'MMR',
    'Burma': 'MMR',
    'Gambia, The': 'GMB',
    'The Gambia': 'GMB',
    'Gambia': 'GMB',
    'Bahamas, The': 'BHS',
    'The Bahamas': 'BHS',
    'Bahamas': 'BHS',
    'Macedonia, FYR': 'MKD',
    'North Macedonia': 'MKD',
    'Micronesia, Fed. Sts.': 'FSM',
    'Micronesia': 'FSM',
    'St. Lucia': 'LCA',
    'Saint Lucia': 'LCA',
    'St. Vincent and the Grenadines': 'VCT',
    'Saint Vincent and the Grenadines': 'VCT',
    'St. Kitts and Nevis': 'KNA',
    'Saint Kitts and Nevis': 'KNA',
    'São Tomé and Príncipe': 'STP',
    'Sao Tome and Principe': 'STP',
    'Brunei Darussalam': 'BRN',
    'Brunei': 'BRN',
    'Cabo Verde': 'CPV',
    'Cape Verde': 'CPV',
    'Türkiye': 'TUR',
    'Turkey': 'TUR',
    'Timor-Leste': 'TLS',
    'East Timor': 'TLS',
    'eSwatini': 'SWZ',
    'Eswatini': 'SWZ',
    'Swaziland': 'SWZ',
    'United States': 'USA',
    'United Kingdom': 'GBR',
    'Germany': 'DEU',
    'France': 'FRA',
    'Italy': 'ITA',
    'Spain': 'ESP',
    'Japan': 'JPN',
    'China': 'CHN',
    'India': 'IND',
    'Brazil': 'BRA',
    'Mexico': 'MEX',
    'Canada': 'CAN',
    'Australia': 'AUS',
    'Netherlands': 'NLD',
    'Switzerland': 'CHE',
    'Poland': 'POL',
    'Belgium': 'BEL',
    'Sweden': 'SWE',
    'Austria': 'AUT',
    'Norway': 'NOR',
    'Denmark': 'DNK',
    'Finland': 'FIN',
    'Ireland': 'IRL',
    'Portugal': 'PRT',
    'Greece': 'GRC',
    'New Zealand': 'NZL',
    'Singapore': 'SGP',
    'Malaysia': 'MYS',
    'Thailand': 'THA',
    'Indonesia': 'IDN',
    'Philippines': 'PHL',
    'Vietnam': 'VNM',
    'South Africa': 'ZAF',
    'Nigeria': 'NGA',
    'Kenya': 'KEN',
    'Chile': 'CHL',
    'Colombia': 'COL',
    'Argentina': 'ARG',
    'Peru': 'PER',
    'Ecuador': 'ECU',
    'Uruguay': 'URY',
    'Paraguay': 'PRY',
    'Bolivia': 'BOL',
    'Panama': 'PAN',
    'Costa Rica': 'CRI',
    'Guatemala': 'GTM',
    'El Salvador': 'SLV',
    'Honduras': 'HND',
    'Nicaragua': 'NIC',
    'Dominican Republic': 'DOM',
    'Jamaica': 'JAM',
    'Trinidad and Tobago': 'TTO',
    'Barbados': 'BRB',
    'Mauritius': 'MUS',
    'Israel': 'ISR',
    'Saudi Arabia': 'SAU',
    'United Arab Emirates': 'ARE',
    'Qatar': 'QAT',
    'Kuwait': 'KWT',
    'Oman': 'OMN',
    'Bahrain': 'BHR',
    'Jordan': 'JOR',
    'Lebanon': 'LBN',
    'Morocco': 'MAR',
    'Tunisia': 'TUN',
    'Algeria': 'DZA',
    'Pakistan': 'PAK',
    'Bangladesh': 'BGD',
    'Sri Lanka': 'LKA',
    'Nepal': 'NPL',
    'Cambodia': 'KHM',
    'Ghana': 'GHA',
    'Uganda': 'UGA',
    'Tanzania': 'TZA',
    'Ethiopia': 'ETH',
    'Rwanda': 'RWA',
    'Zambia': 'ZMB',
    'Zimbabwe': 'ZWE',
    'Botswana': 'BWA',
    'Namibia': 'NAM',
    'Senegal': 'SEN',
    'Cameroon': 'CMR',
    'Hungary': 'HUN',
    'Romania': 'ROU',
    'Bulgaria': 'BGR',
    'Croatia': 'HRV',
    'Serbia': 'SRB',
    'Slovenia': 'SVN',
    'Lithuania': 'LTU',
    'Latvia': 'LVA',
    'Estonia': 'EST',
    'Ukraine': 'UKR',
    'Belarus': 'BLR',
    'Kazakhstan': 'KAZ',
    'Uzbekistan': 'UZB',
    'Georgia': 'GEO',
    'Armenia': 'ARM',
    'Azerbaijan': 'AZE',
    'Mongolia': 'MNG',
    'Iceland': 'ISL',
    'Luxembourg': 'LUX',
    'Malta': 'MLT',
    'Cyprus': 'CYP',
}

# Aggregate regions to exclude (not countries)
EXCLUDE_CODES = {
    'WLD', 'EAS', 'ECS', 'LCN', 'MEA', 'NAC', 'SAS', 'SSF',  # WB regions
    'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'ECA', 'EMU', 'EUU',
    'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX',
    'LAC', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MIC', 'MNA',
    'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SSA', 'SST', 'TEA',
    'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'AFE', 'AFW',
}


def build_iso3_concordance(
    efw_df: pd.DataFrame,
    efw_iso_col: str = 'ISO_Code',
    efw_name_col: str = 'Countries',
    verbose: bool = True
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Build ISO3 concordance from EFW data and manual mappings.
    
    Parameters
    ----------
    efw_df : pd.DataFrame
        EFW DataFrame
    efw_iso_col : str
        Column containing ISO codes in EFW data
    efw_name_col : str
        Column containing country names in EFW data
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    tuple
        (name_to_iso3 dict, concordance DataFrame)
    """
    # Extract unique EFW country entries
    efw_countries = efw_df[[efw_name_col, efw_iso_col]].drop_duplicates()
    efw_countries = efw_countries.dropna(subset=[efw_name_col])
    
    # Build concordance dict
    name_to_iso3 = {}
    
    # First, use EFW's own ISO codes where available
    for _, row in efw_countries.iterrows():
        name = row[efw_name_col]
        iso = row[efw_iso_col]
        if pd.notna(iso) and len(str(iso).strip()) == 3:
            name_to_iso3[name] = str(iso).upper().strip()
    
    # Override with manual concordance
    name_to_iso3.update(MANUAL_CONCORDANCE)
    
    # Build concordance DataFrame
    concordance = []
    for name, iso in name_to_iso3.items():
        concordance.append({
            'country_name': name,
            'iso3': iso,
            'source': 'manual' if name in MANUAL_CONCORDANCE else 'efw'
        })
    concordance_df = pd.DataFrame(concordance)
    
    # Report unresolved names
    unresolved = []
    for name in efw_df[efw_name_col].unique():
        if pd.notna(name) and name not in name_to_iso3:
            unresolved.append(name)
    
    if verbose:
        print(f"ISO3 Concordance built:")
        print(f"  Mapped names: {len(name_to_iso3)}")
        print(f"  From EFW: {(concordance_df['source'] == 'efw').sum()}")
        print(f"  Manual overrides: {(concordance_df['source'] == 'manual').sum()}")
        if unresolved:
            print(f"  ⚠ Unresolved names: {len(unresolved)}")
            for name in unresolved[:5]:
                print(f"      - {name}")
    
    return name_to_iso3, concordance_df


def standardize_iso3(
    df: pd.DataFrame,
    iso_col: str = 'iso3',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Standardize and validate ISO3 codes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ISO3 column
    iso_col : str
        Name of ISO3 column
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized ISO3 codes
    """
    df = df.copy()
    n_before = len(df)
    
    # Standardize format
    df[iso_col] = df[iso_col].astype(str).str.upper().str.strip()
    
    # Filter invalid codes
    valid_mask = (
        (df[iso_col].str.len() == 3) &
        (~df[iso_col].isin(EXCLUDE_CODES)) &
        (df[iso_col] != 'NAN') &
        (df[iso_col] != 'NONE') &
        (~df[iso_col].str.contains(r'[^A-Z]', regex=True, na=False))
    )
    
    df = df[valid_mask].copy()
    n_after = len(df)
    
    if verbose and n_before != n_after:
        print(f"Standardize ISO3: dropped {n_before - n_after:,} rows (invalid/aggregate codes)")
    
    return df


# =============================================================================
# LONG-TO-WIDE TRANSFORMATION
# =============================================================================

def long_to_wide(
    df_long: pd.DataFrame,
    index_cols: List[str] = ['iso3', 'year'],
    variable_col: str = 'variable',
    value_col: str = 'value',
    source_col: str = 'source',
    aggfunc: str = 'first',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Transform long-form panel to wide format.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Long-form data with (iso3, year, variable, value) structure
    index_cols : list
        Columns to use as index
    variable_col : str
        Column containing variable names
    value_col : str
        Column containing values
    source_col : str
        Column containing data source
    aggfunc : str
        Aggregation function for duplicates ('first', 'mean', 'last')
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    pd.DataFrame
        Wide-form panel with variables as columns
    """
    if verbose:
        print(f"Reshaping {len(df_long):,} observations to wide format...")
        print(f"  Variables: {df_long[variable_col].nunique()}")
    
    # Check for duplicates before pivot
    dups = df_long.groupby(index_cols + [variable_col]).size()
    n_dups = (dups > 1).sum()
    
    if n_dups > 0:
        if verbose:
            print(f"  ⚠ Found {n_dups:,} duplicate (iso3, year, variable) combinations")
        
        # Aggregate duplicates
        if aggfunc == 'first':
            df_long = df_long.drop_duplicates(subset=index_cols + [variable_col], keep='first')
        elif aggfunc == 'mean':
            df_long = df_long.groupby(index_cols + [variable_col]).agg({
                value_col: 'mean',
                source_col: 'first'
            }).reset_index()
        elif aggfunc == 'last':
            df_long = df_long.drop_duplicates(subset=index_cols + [variable_col], keep='last')
    
    # Pivot
    df_wide = df_long.pivot_table(
        index=index_cols,
        columns=variable_col,
        values=value_col,
        aggfunc='first'
    ).reset_index()
    
    # Flatten column names
    df_wide.columns = [c if isinstance(c, str) else c[0] for c in df_wide.columns]
    
    if verbose:
        print(f"  Result: {len(df_wide):,} rows × {len(df_wide.columns)} columns")
    
    return df_wide


def combine_sources_for_variable(
    df_long: pd.DataFrame,
    variable_mapping: Dict[str, List[str]],
    prefer_source: Dict[str, str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Combine multiple source variables into unified series with fallback logic.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Long-form data
    variable_mapping : dict
        Mapping from target variable name to list of source variable names
        e.g., {'gdppc': ['NY.GDP.PCAP.KD', 'gdppc_pwt']}
    prefer_source : dict, optional
        Preferred source for each variable (e.g., {'gdppc': 'WDI'})
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    pd.DataFrame
        Long-form data with unified variables
    """
    if prefer_source is None:
        prefer_source = {}
    
    results = []
    
    for target_var, source_vars in variable_mapping.items():
        # Filter to relevant source variables
        mask = df_long['variable'].isin(source_vars)
        df_subset = df_long[mask].copy()
        
        if len(df_subset) == 0:
            continue
        
        # Assign target variable name
        df_subset['variable'] = target_var
        
        # Deduplicate by (iso3, year), preferring specified source
        if target_var in prefer_source:
            pref = prefer_source[target_var]
            df_subset['_pref'] = df_subset['source'] == pref
            df_subset = df_subset.sort_values('_pref', ascending=False)
            df_subset = df_subset.drop('_pref', axis=1)
        
        df_subset = df_subset.drop_duplicates(subset=['iso3', 'year', 'variable'], keep='first')
        results.append(df_subset)
    
    if results:
        df_combined = pd.concat(results, ignore_index=True)
        if verbose:
            print(f"Combined sources: {len(df_combined):,} observations for {len(variable_mapping)} target variables")
        return df_combined
    
    return pd.DataFrame(columns=df_long.columns)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_panel_uniqueness(
    df: pd.DataFrame,
    key_cols: List[str] = ['iso3', 'year'],
    verbose: bool = True
) -> bool:
    """
    Validate that panel has unique observations for each key.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame
    key_cols : list
        Key columns that should uniquely identify rows
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    bool
        True if unique
        
    Raises
    ------
    AssertionError
        If duplicates found
    """
    dups = df.groupby(key_cols).size()
    n_dups = (dups > 1).sum()
    
    if n_dups > 0:
        if verbose:
            print(f"⚠ VALIDATION FAILED: {n_dups:,} duplicate keys found")
            dup_examples = dups[dups > 1].head(5)
            for key, count in dup_examples.items():
                print(f"    {key}: {count} rows")
        raise AssertionError(f"Panel has {n_dups} duplicate ({', '.join(key_cols)}) combinations")
    
    if verbose:
        print(f"✓ Panel uniqueness validated: {len(df):,} unique ({', '.join(key_cols)}) combinations")
    
    return True


def compute_coverage_stats(
    df: pd.DataFrame,
    value_cols: List[str],
    by_decade: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute coverage statistics for variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide-form panel
    value_cols : list
        Columns to compute coverage for
    by_decade : bool
        If True, also compute by decade
    verbose : bool
        Print summary
        
    Returns
    -------
    pd.DataFrame
        Coverage statistics
    """
    # Filter to existing columns
    value_cols = [c for c in value_cols if c in df.columns]
    
    # Overall coverage
    coverage = []
    for col in value_cols:
        nonmiss = df[col].notna().sum()
        total = len(df)
        coverage.append({
            'variable': col,
            'n_obs': nonmiss,
            'n_total': total,
            'coverage_pct': 100 * nonmiss / total if total > 0 else 0,
            'n_countries': df[df[col].notna()]['iso3'].nunique(),
            'year_min': df[df[col].notna()]['year'].min() if nonmiss > 0 else None,
            'year_max': df[df[col].notna()]['year'].max() if nonmiss > 0 else None,
        })
    
    coverage_df = pd.DataFrame(coverage)
    
    if verbose:
        print("\nVariable Coverage Summary:")
        print("-" * 80)
        print(coverage_df.to_string(index=False))
    
    return coverage_df


def compute_decade_coverage(
    df: pd.DataFrame,
    value_cols: List[str]
) -> pd.DataFrame:
    """Compute coverage by decade."""
    value_cols = [c for c in value_cols if c in df.columns]
    
    df = df.copy()
    df['decade'] = (df['year'] // 10) * 10
    
    results = []
    for decade in sorted(df['decade'].unique()):
        decade_df = df[df['decade'] == decade]
        for col in value_cols:
            nonmiss = decade_df[col].notna().sum()
            total = len(decade_df)
            results.append({
                'decade': int(decade),
                'variable': col,
                'coverage_pct': 100 * nonmiss / total if total > 0 else 0,
                'n_countries': decade_df[decade_df[col].notna()]['iso3'].nunique()
            })
    
    return pd.DataFrame(results).pivot(index='variable', columns='decade', values='coverage_pct')


# =============================================================================
# SAVE/LOAD UTILITIES
# =============================================================================

INTERMEDIATE_DIR = Path("data/intermediate")
CLEAN_DIR = Path("data/clean")


def save_concordance(concordance_df: pd.DataFrame, name: str = "iso3_concordance"):
    """Save concordance to intermediate folder."""
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERMEDIATE_DIR / f"{name}.csv"
    concordance_df.to_csv(path, index=False)
    print(f"Saved concordance to {path}")


def save_long_panel(df: pd.DataFrame, name: str = "macro_long"):
    """Save long-form panel to intermediate folder."""
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERMEDIATE_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved long panel to {path}")


def save_wide_panel(df: pd.DataFrame, name: str = "macro_wide"):
    """Save wide-form panel to clean folder."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    path = CLEAN_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved wide panel to {path}")


def load_long_panel(name: str = "macro_long") -> pd.DataFrame:
    """Load long-form panel from intermediate folder."""
    path = INTERMEDIATE_DIR / f"{name}.parquet"
    return pd.read_parquet(path)


def load_wide_panel(name: str = "macro_wide") -> pd.DataFrame:
    """Load wide-form panel from clean folder."""
    path = CLEAN_DIR / f"{name}.parquet"
    return pd.read_parquet(path)
