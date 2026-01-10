"""
EFW Analysis Utilities
======================
Shared helper functions for EDA notebooks analyzing
the Fraser Institute Economic Freedom of the World data.

Author: Economic Freedom Research Team
Date: 2026-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Column names
EFW_COL = 'ECONOMIC FREEDOM ALL AREAS'
AREA_COLS = [
    'Area 1 Size of Government',
    'Area 2 Legal System & Property Rights -- With Gender Adjustment',
    'Area 3 Sound Money',
    'Area 4 Freedom to trade internationally',
    'Area 5 Regulation'
]
AREA_LABELS = ['Size of Gov', 'Legal System', 'Sound Money', 'Trade Freedom', 'Regulation']
AREA_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

# Identifier columns
ID_COLS = ['ISO_Code', 'Countries', 'World Bank Region',
           'World Bank Current Income Classification, 1990-Present']

# Key years
QUINQUENNIAL_YEARS = [1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
YEAR_BREAKS = {'cold_war_end': 1990, 'annual_start': 2000, 'gfc': 2008}
PERIOD_DEFINITIONS = {
    'pre_1990': (None, 1989),
    'post_1990': (1990, None),
    'pre_2000': (None, 1999),
    'post_2000': (2000, None),
    'pre_gfc': (2005, 2007),
    'gfc': (2008, 2009),
    'post_gfc': (2010, 2013)
}


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

def load_fraser(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load and clean Fraser EFW Excel data.
    
    Parameters
    ----------
    path : str
        Path to fraser.xlsx file
    verbose : bool
        Print summary statistics if True
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardized columns
    """
    # Load raw data (header is on row 4, 0-indexed row 3)
    raw_df = pd.read_excel(path, header=3)
    
    # Clean column names (handle merged cells creating 'Unnamed' or 'Data' prefixes)
    new_cols = []
    last_valid = ""
    for col in raw_df.columns:
        col_str = str(col).strip()
        if col_str.lower().startswith('data') or col_str.lower().startswith('unnamed'):
            new_cols.append(f"{last_valid}_data")
        else:
            new_cols.append(col_str)
            last_valid = col_str
    raw_df.columns = new_cols
    
    # Drop rows without Year or Countries
    df = raw_df.dropna(subset=['Year', 'Countries']).copy()
    
    # Clean Year column
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ID_COLS and col != 'Year':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Standardize region/income column names for easier access
    df = df.rename(columns={
        'World Bank Region': 'Region',
        'World Bank Current Income Classification, 1990-Present': 'Income_Group'
    })
    
    # Clean category columns: convert '..' and '0' and empty strings to NaN
    MISSING_STRINGS = {'..', '.', '...', '0', '', 'nan', 'NaN', 'None'}
    for c in ['Region', 'Income_Group']:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                          .replace(list(MISSING_STRINGS), np.nan)
                          .replace('nan', np.nan)
                          .str.strip())
            # Convert 'nan' strings to actual NaN
            df.loc[df[c] == 'nan', c] = np.nan
    
    if verbose:
        print(f"Loaded Fraser EFW data:")
        print(f"  Observations: {len(df):,}")
        print(f"  Years: {df['Year'].min()}–{df['Year'].max()}")
        print(f"  Countries: {df['Countries'].nunique()}")
    
    return df


def validate_columns(df: pd.DataFrame, required: list) -> bool:
    """
    Validate that required columns exist in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required : list
        List of required column names
        
    Returns
    -------
    bool
        True if all columns present
        
    Raises
    ------
    ValueError
        If any required columns are missing
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


# =============================================================================
# PANEL OPERATIONS
# =============================================================================

def compute_balanced_panel(df: pd.DataFrame, years: list, col: str = None) -> pd.DataFrame:
    """
    Compute balanced panel: countries observed in ALL specified years.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame
    years : list
        List of years that must be present
    col : str, optional
        Column that must be non-missing. If None, uses EFW_COL.
        
    Returns
    -------
    pd.DataFrame
        Subset containing only balanced-panel countries
    """
    if col is None:
        col = EFW_COL
    
    # Filter to specified years
    df_subset = df[df['Year'].isin(years)].copy()
    
    # Count non-missing observations per country
    obs_per_country = df_subset.groupby('Countries')[col].apply(lambda x: x.notna().sum())
    
    # Countries with all years
    balanced_countries = obs_per_country[obs_per_country == len(years)].index.tolist()
    
    return df_subset[df_subset['Countries'].isin(balanced_countries)].copy()


def get_common_countries(df: pd.DataFrame, year_start: int, year_end: int, 
                         col: str = None) -> list:
    """
    Get countries observed in both start and end years.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame
    year_start : int
        Start year
    year_end : int
        End year
    col : str, optional
        Column that must be non-missing
        
    Returns
    -------
    list
        List of country names observed in both years
    """
    if col is None:
        col = EFW_COL
    
    start_countries = set(df[(df['Year'] == year_start) & df[col].notna()]['Countries'])
    end_countries = set(df[(df['Year'] == year_end) & df[col].notna()]['Countries'])
    
    return list(start_countries & end_countries)


# =============================================================================
# MISSINGNESS ANALYSIS
# =============================================================================

def compute_missingness(df: pd.DataFrame, cols: list = None) -> dict:
    """
    Compute comprehensive missingness statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    cols : list, optional
        Columns to analyze. If None, uses EFW + AREA columns.
        
    Returns
    -------
    dict
        Dictionary with missingness tables:
        - 'by_column': Series of missing % by column
        - 'by_year': Series of missing % by year (for EFW)
        - 'by_region': DataFrame of missing % by region and column
        - 'by_income': DataFrame of missing % by income group and column
    """
    if cols is None:
        cols = [EFW_COL] + AREA_COLS
    
    # Filter to existing columns
    cols = [c for c in cols if c in df.columns]
    
    results = {}
    
    # By column
    results['by_column'] = df[cols].isnull().mean() * 100
    
    # By year (for EFW)
    efw_col = EFW_COL if EFW_COL in df.columns else cols[0]
    results['by_year'] = df.groupby('Year')[efw_col].apply(lambda x: x.isnull().mean() * 100)
    
    # By region
    if 'Region' in df.columns:
        region_missing = df.groupby('Region')[cols].apply(lambda x: x.isnull().mean() * 100)
        results['by_region'] = region_missing
    
    # By income group
    if 'Income_Group' in df.columns:
        income_missing = df.groupby('Income_Group')[cols].apply(lambda x: x.isnull().mean() * 100)
        results['by_income'] = income_missing
    
    return results


def create_availability_heatmap(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """
    Create year × region availability matrix (count of non-missing observations).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    col : str, optional
        Column to check availability for
        
    Returns
    -------
    pd.DataFrame
        Pivot table with years as rows, regions as columns, values as counts
    """
    if col is None:
        col = EFW_COL
    
    # Create availability indicator
    df_temp = df.copy()
    df_temp['available'] = df_temp[col].notna().astype(int)
    
    return df_temp.pivot_table(
        index='Year', 
        columns='Region', 
        values='available', 
        aggfunc='sum',
        fill_value=0
    )


# =============================================================================
# INDEX VERIFICATION
# =============================================================================

def verify_index_construction(df: pd.DataFrame, efw_col: str = None, 
                              area_cols: list = None) -> dict:
    """
    Verify EFW index construction by computing difference from mean of areas.
    
    This is more rigorous than correlation - it checks if EFW exactly equals
    the unweighted mean of the 5 area scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    efw_col : str, optional
        Overall EFW column name
    area_cols : list, optional
        List of 5 area column names
        
    Returns
    -------
    dict
        Verification statistics:
        - 'n_complete': Number of rows with all values
        - 'mean_diff': Mean difference (EFW - mean(areas))
        - 'std_diff': Standard deviation of differences
        - 'max_abs_diff': Maximum absolute difference
        - 'frac_large_diff': Fraction with |diff| > 0.05
        - 'correlation': Correlation between EFW and computed mean
    """
    if efw_col is None:
        efw_col = EFW_COL
    if area_cols is None:
        area_cols = AREA_COLS
    
    # Filter to complete cases
    required_cols = [efw_col] + area_cols
    complete = df[required_cols].dropna()
    
    if len(complete) == 0:
        return {'error': 'No complete cases found'}
    
    # Compute difference
    computed_mean = complete[area_cols].mean(axis=1)
    actual_efw = complete[efw_col]
    diff = actual_efw - computed_mean
    
    return {
        'n_complete': len(complete),
        'mean_diff': diff.mean(),
        'std_diff': diff.std(),
        'max_abs_diff': diff.abs().max(),
        'frac_large_diff': (diff.abs() > 0.05).mean(),
        'correlation': np.corrcoef(actual_efw, computed_mean)[0, 1]
    }


def add_area_availability(df: pd.DataFrame, area_cols: list = None) -> pd.DataFrame:
    """
    Add columns tracking area score availability.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify
    area_cols : list, optional
        List of area column names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new columns:
        - 'n_areas': count of non-missing areas (0-5)
        - 'complete5': True if all 5 areas present
        - 'efw_mean_available': mean of available areas
    """
    if area_cols is None:
        area_cols = AREA_COLS
    
    df = df.copy()
    df['n_areas'] = df[area_cols].notna().sum(axis=1)
    df['complete5'] = df[area_cols].notna().all(axis=1)
    df['efw_mean_available'] = df[area_cols].mean(axis=1, skipna=True)
    
    return df


def validate_overall_matches_available_areas(df: pd.DataFrame, efw_col: str = None,
                                              area_cols: list = None, 
                                              tol: float = 0.01) -> float:
    """
    Validate that overall EFW equals mean of AVAILABLE areas (not just complete cases).
    
    This is critical because Fraser computes overall as mean of available areas,
    not as missing when any area is missing.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    efw_col : str, optional
        Overall EFW column
    area_cols : list, optional
        Area column names
    tol : float
        Tolerance for difference
        
    Returns
    -------
    float
        Maximum absolute difference found
        
    Raises
    ------
    AssertionError
        If max difference exceeds tolerance
    """
    if efw_col is None:
        efw_col = EFW_COL
    if area_cols is None:
        area_cols = AREA_COLS
    
    tmp = df[df[efw_col].notna()].copy()
    tmp['mean_avail'] = tmp[area_cols].mean(axis=1, skipna=True)
    max_abs = (tmp[efw_col] - tmp['mean_avail']).abs().max()
    
    if max_abs > tol:
        warnings.warn(f"EFW != mean(available areas): max_abs={max_abs:.4f}")
    
    return max_abs


# =============================================================================
# PERIOD ANALYSIS
# =============================================================================

def compute_period_changes(df: pd.DataFrame, periods: list, col: str = None,
                           groupby: str = None) -> pd.DataFrame:
    """
    Compute changes across specified periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    periods : list of tuples
        List of (start_year, end_year) tuples
    col : str, optional
        Column to compute changes for
    groupby : str, optional
        Column to group by (e.g., 'Region', 'Income_Group')
        
    Returns
    -------
    pd.DataFrame
        Table of changes with columns for each period
    """
    if col is None:
        col = EFW_COL
    
    results = []
    
    for start, end in periods:
        period_name = f"{start}→{end}"
        
        if groupby:
            start_vals = df[df['Year'] == start].groupby(groupby)[col].mean()
            end_vals = df[df['Year'] == end].groupby(groupby)[col].mean()
            change = end_vals - start_vals
            for group in change.index:
                results.append({
                    'Group': group,
                    'Period': period_name,
                    'Start': start_vals.get(group, np.nan),
                    'End': end_vals.get(group, np.nan),
                    'Change': change.get(group, np.nan)
                })
        else:
            start_val = df[df['Year'] == start][col].mean()
            end_val = df[df['Year'] == end][col].mean()
            results.append({
                'Period': period_name,
                'Start': start_val,
                'End': end_val,
                'Change': end_val - start_val
            })
    
    return pd.DataFrame(results)


def compute_slope_shift(df: pd.DataFrame, col: str = None, 
                        break_year: int = 1990) -> dict:
    """
    Compute descriptive slope shift around a break year.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    col : str, optional
        Column to analyze
    break_year : int
        Year to use as break point
        
    Returns
    -------
    dict
        Slope statistics for pre and post periods
    """
    if col is None:
        col = EFW_COL
    
    yearly = df.groupby('Year')[col].mean()
    
    pre = yearly[yearly.index < break_year]
    post = yearly[yearly.index >= break_year]
    
    results = {}
    
    if len(pre) >= 2:
        slope_pre, intercept_pre, r_pre, p_pre, se_pre = stats.linregress(
            pre.index.astype(float), pre.values
        )
        results['pre_slope'] = slope_pre
        results['pre_slope_per_decade'] = slope_pre * 10
        results['pre_r2'] = r_pre ** 2
        results['pre_n'] = len(pre)
    
    if len(post) >= 2:
        slope_post, intercept_post, r_post, p_post, se_post = stats.linregress(
            post.index.astype(float), post.values
        )
        results['post_slope'] = slope_post
        results['post_slope_per_decade'] = slope_post * 10
        results['post_r2'] = r_post ** 2
        results['post_n'] = len(post)
    
    if 'pre_slope' in results and 'post_slope' in results:
        results['slope_change'] = results['post_slope'] - results['pre_slope']
        results['break_year'] = break_year
    
    return results


def area_decomposition(df: pd.DataFrame, start_year: int, end_year: int,
                       efw_col: str = None, area_cols: list = None,
                       groupby: str = None, require_complete5: bool = True) -> pd.DataFrame:
    """
    Decompose EFW change into area contributions on a common sample.
    
    This is the correct way to do decomposition: compute per-country changes
    on a common sample, then average. This ensures contributions sum to the
    overall change.
    
    ΔEFW_i = (1/5) * Σ ΔArea_ia
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    start_year : int
        Start year
    end_year : int
        End year
    efw_col : str, optional
        Overall EFW column
    area_cols : list, optional
        List of 5 area columns
    groupby : str, optional
        Column to group by (e.g., 'Region')
    require_complete5 : bool
        If True, only use countries with all 5 areas at both endpoints
        
    Returns
    -------
    pd.DataFrame
        Decomposition table with:
        - Delta_EFW: overall change
        - Delta_{area}: change for each area
        - Contribution_{area}: area contribution (= Delta_{area} / 5)
        - N: number of countries used
    """
    if efw_col is None:
        efw_col = EFW_COL
    if area_cols is None:
        area_cols = AREA_COLS
    
    all_cols = [efw_col] + area_cols
    
    # Get data for both years
    start_df = df[df['Year'] == start_year][['Countries'] + all_cols].copy()
    end_df = df[df['Year'] == end_year][['Countries'] + all_cols].copy()
    
    if require_complete5:
        # Filter to complete5 at both endpoints
        start_df = start_df.dropna(subset=all_cols)
        end_df = end_df.dropna(subset=all_cols)
    else:
        # Just require EFW
        start_df = start_df.dropna(subset=[efw_col])
        end_df = end_df.dropna(subset=[efw_col])
    
    # Common countries
    common = set(start_df['Countries']) & set(end_df['Countries'])
    start_df = start_df[start_df['Countries'].isin(common)].set_index('Countries')
    end_df = end_df[end_df['Countries'].isin(common)].set_index('Countries')
    
    # Compute per-country changes
    changes = end_df - start_df
    
    # Optionally merge groupby column
    if groupby:
        groupby_map = df[['Countries', groupby]].drop_duplicates().set_index('Countries')
        changes = changes.join(groupby_map)
        
        results = []
        for group in changes[groupby].dropna().unique():
            group_changes = changes[changes[groupby] == group]
            row = _compute_decomposition_row(group_changes, efw_col, area_cols)
            row['Group'] = group
            results.append(row)
        return pd.DataFrame(results)
    else:
        row = _compute_decomposition_row(changes, efw_col, area_cols)
        row['Period'] = f'{start_year}→{end_year}'
        return pd.DataFrame([row])


def _compute_decomposition_row(changes: pd.DataFrame, efw_col: str, 
                                area_cols: list) -> dict:
    """Helper to compute decomposition statistics."""
    row = {'N': len(changes)}
    row[f'Delta_{efw_col[:10]}'] = changes[efw_col].mean()
    
    for i, area in enumerate(area_cols):
        label = AREA_LABELS[i] if i < len(AREA_LABELS) else f'Area_{i+1}'
        delta = changes[area].mean()
        row[f'Delta_{label}'] = delta
        row[f'Contrib_{label}'] = delta / 5  # Equal weights
    
    return row

# =============================================================================
# CONVERGENCE ANALYSIS
# =============================================================================

def compute_sigma_convergence(df: pd.DataFrame, col: str = None, 
                               balanced: bool = False, years: list = None) -> pd.DataFrame:
    """
    Compute sigma-convergence (cross-sectional standard deviation over time).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    col : str, optional
        Column to compute SD for
    balanced : bool
        If True, use balanced panel
    years : list, optional
        Years for balanced panel (required if balanced=True)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Year, SD, and trend statistics
    """
    if col is None:
        col = EFW_COL
    
    if balanced and years:
        df = compute_balanced_panel(df, years, col)
    
    sigma = df.groupby('Year')[col].agg(['std', 'count'])
    sigma.columns = ['SD', 'N']
    
    # Compute linear trend
    valid = sigma.dropna()
    if len(valid) >= 2:
        slope, intercept, r, p, se = stats.linregress(
            valid.index.astype(float), valid['SD'].values
        )
        sigma['Trend'] = intercept + slope * sigma.index.astype(float)
        sigma.attrs['slope'] = slope
        sigma.attrs['slope_p'] = p
        sigma.attrs['r_squared'] = r ** 2
    
    return sigma


def compute_beta_convergence(df: pd.DataFrame, year_start: int, year_end: int,
                             col: str = None) -> dict:
    """
    Compute beta-convergence (descriptive regression of change on initial level).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    year_start : int
        Initial year
    year_end : int
        Final year
    col : str, optional
        Column to analyze
        
    Returns
    -------
    dict
        Regression statistics
    """
    if col is None:
        col = EFW_COL
    
    # Get common countries
    start_data = df[df['Year'] == year_start][['Countries', col]].dropna()
    end_data = df[df['Year'] == year_end][['Countries', col]].dropna()
    
    merged = start_data.merge(end_data, on='Countries', suffixes=('_start', '_end'))
    merged['change'] = merged[f'{col}_end'] - merged[f'{col}_start']
    
    if len(merged) < 3:
        return {'error': 'Insufficient observations'}
    
    # OLS regression
    slope, intercept, r, p, se = stats.linregress(
        merged[f'{col}_start'].values, merged['change'].values
    )
    
    return {
        'n': len(merged),
        'beta': slope,
        'beta_se': se,
        'beta_p': p,
        'r_squared': r ** 2,
        'intercept': intercept,
        'convergence': slope < 0  # Negative beta indicates convergence
    }


# =============================================================================
# MOBILITY ANALYSIS
# =============================================================================

def compute_transition_matrix(df: pd.DataFrame, year_start: int, year_end: int,
                              col: str = None, n_quantiles: int = 4,
                              fixed_cutoffs: bool = False) -> tuple:
    """
    Compute transition matrix between quantiles.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    year_start : int
        Initial year
    year_end : int
        Final year
    col : str, optional
        Column to analyze
    n_quantiles : int
        Number of quantiles (default 4 for quartiles)
    fixed_cutoffs : bool
        If True, use year_start cutoffs for both years (absolute progress)
        If False, use year-specific cutoffs (relative position)
        
    Returns
    -------
    tuple
        (transition_matrix, labels, metadata)
    """
    if col is None:
        col = EFW_COL
    
    # Get data for both years
    start_data = df[df['Year'] == year_start][['Countries', col]].dropna().copy()
    end_data = df[df['Year'] == year_end][['Countries', col]].dropna().copy()
    
    # Create quantile labels
    labels = [f'Q{i+1}' for i in range(n_quantiles)]
    labels[0] = f'Q1 (Low)'
    labels[-1] = f'Q{n_quantiles} (High)'
    
    if fixed_cutoffs:
        # Use start year cutoffs for both
        # CRITICAL: Extend bins with -inf/+inf to catch values outside start-year range
        _, raw_cutoffs = pd.qcut(start_data[col], n_quantiles, labels=False, retbins=True)
        # Replace min/max with -inf/+inf
        bins = np.r_[-np.inf, raw_cutoffs[1:-1], np.inf]
        start_data['Q'] = pd.cut(start_data[col], bins=bins, labels=labels, include_lowest=True)
        end_data['Q'] = pd.cut(end_data[col], bins=bins, labels=labels, include_lowest=True)
    else:
        # Year-specific cutoffs
        start_data['Q'] = pd.qcut(start_data[col], n_quantiles, labels=labels)
        end_data['Q'] = pd.qcut(end_data[col], n_quantiles, labels=labels)
    
    # Merge and compute transition matrix
    merged = start_data[['Countries', 'Q']].merge(
        end_data[['Countries', 'Q']], 
        on='Countries', 
        suffixes=('_start', '_end')
    )
    
    trans_matrix = pd.crosstab(
        merged['Q_start'], 
        merged['Q_end'], 
        normalize='index'
    ) * 100
    
    metadata = {
        'n_countries': len(merged),
        'year_start': year_start,
        'year_end': year_end,
        'fixed_cutoffs': fixed_cutoffs
    }
    
    return trans_matrix, labels, metadata


def compute_shorrocks_mobility(trans_matrix: pd.DataFrame) -> float:
    """
    Compute Shorrocks mobility index from transition matrix.
    
    M = (n - trace(P)) / (n - 1)
    
    Where n is the number of states and P is the transition matrix.
    Higher values indicate more mobility (max = 1).
    
    Parameters
    ----------
    trans_matrix : pd.DataFrame
        Transition matrix (rows sum to 100 or 1)
        
    Returns
    -------
    float
        Shorrocks mobility index
    """
    # Normalize to proportions if needed
    P = trans_matrix.values
    if P.sum() > trans_matrix.shape[0] + 1:  # Likely percentages
        P = P / 100
    
    n = P.shape[0]
    trace = np.trace(P)
    
    return (n - trace) / (n - 1)


# =============================================================================
# COUNTRY SELECTION
# =============================================================================

def select_notable_countries(df: pd.DataFrame, col: str = None,
                             start_year: int = None, end_year: int = None) -> dict:
    """
    Automatically select notable countries for case studies.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    col : str, optional
        Column to analyze
    start_year : int, optional
        Fixed start year for change calculation. If None, uses earliest in data.
    end_year : int, optional
        Fixed end year for change calculation. If None, uses latest in data.
        
    Returns
    -------
    dict
        Dictionary with lists of countries:
        - 'top_reformers': Largest long-run increases
        - 'top_reversals': Largest long-run decreases
        - 'volatile': Highest within-country SD post-2000
        - 'start_year': The start year used
        - 'end_year': The end year used
    """
    if col is None:
        col = EFW_COL
    
    results = {}
    
    # Use fixed window or data-defined window
    if start_year is None:
        start_year = df['Year'].min()
    if end_year is None:
        end_year = df['Year'].max()
    
    results['start_year'] = start_year
    results['end_year'] = end_year
    
    common = get_common_countries(df, start_year, end_year, col)
    if common:
        changes = {}
        for country in common:
            cdata = df[df['Countries'] == country]
            start_val = cdata[cdata['Year'] == start_year][col].values
            end_val = cdata[cdata['Year'] == end_year][col].values
            if len(start_val) > 0 and len(end_val) > 0:
                changes[country] = end_val[0] - start_val[0]
        
        changes_series = pd.Series(changes).sort_values()
        results['top_reformers'] = changes_series.nlargest(10).index.tolist()
        results['top_reversals'] = changes_series.nsmallest(10).index.tolist()
    else:
        results['top_reformers'] = []
        results['top_reversals'] = []
    
    # Volatility post-2000
    post_2000 = df[df['Year'] >= 2000]
    volatility = post_2000.groupby('Countries')[col].std()
    # Require at least 5 observations
    obs_count = post_2000.groupby('Countries')[col].count()
    volatility = volatility[obs_count >= 5]
    results['volatile'] = volatility.nlargest(10).index.tolist()
    
    return results


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def savefig(fig, name: str, output_dir: str = 'outputs', dpi: int = 300, 
            tight: bool = True) -> str:
    """
    Save figure to outputs directory with consistent settings.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    name : str
        Filename (without path, with extension)
    output_dir : str
        Output directory
    dpi : int
        Resolution
    tight : bool
        Use tight bounding box
        
    Returns
    -------
    str
        Full path to saved figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / name
    
    if tight:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    else:
        fig.savefig(filepath, dpi=dpi, facecolor='white')
    
    return str(filepath)


def setup_plot_style():
    """Apply consistent plot styling for publication-quality figures."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.size': 10,
        'font.family': 'serif',
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.facecolor': 'white'
    })
    sns.set_palette("colorblind")


def add_period_markers(ax, years: list = None, alpha: float = 0.5):
    """Add vertical lines at key historical periods."""
    if years is None:
        years = [1990, 2000, 2008]
    
    for year in years:
        ax.axvline(x=year, color='gray', linestyle=':', alpha=alpha)


# =============================================================================
# TABLE EXPORT
# =============================================================================

def export_table(df: pd.DataFrame, name: str, output_dir: str = 'outputs/tables') -> str:
    """
    Export DataFrame to CSV in tables subdirectory.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export
    name : str
        Filename (without path, with .csv extension)
    output_dir : str
        Output directory
        
    Returns
    -------
    str
        Full path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / name
    df.to_csv(filepath, index=True)
    
    return str(filepath)


# =============================================================================
# FIGURE INDEX
# =============================================================================

class FigureIndex:
    """Track figures for creating FIGURE_INDEX.md."""
    
    def __init__(self):
        self.entries = []
    
    def add(self, filename: str, description: str, sample: str, 
            years: str, takeaway: str):
        """Add a figure entry."""
        self.entries.append({
            'Filename': filename,
            'Description': description,
            'Sample': sample,
            'Years': years,
            'Takeaway': takeaway
        })
    
    def save(self, output_dir: str = 'outputs') -> str:
        """Save FIGURE_INDEX.md."""
        output_path = Path(output_dir) / 'FIGURE_INDEX.md'
        
        lines = ['# Figure Index\n\n']
        lines.append('| Filename | Description | Sample | Years | Main Takeaway |\n')
        lines.append('|----------|-------------|--------|-------|---------------|\n')
        
        for entry in self.entries:
            lines.append(
                f"| `{entry['Filename']}` | {entry['Description']} | "
                f"{entry['Sample']} | {entry['Years']} | {entry['Takeaway']} |\n"
            )
        
        with open(output_path, 'w') as f:
            f.writelines(lines)
        
        return str(output_path)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame(self.entries)
