"""
EFW Merge Utilities
===================
Functions for merging macro panel with Fraser EFW data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings

EFW_COL = 'ECONOMIC FREEDOM ALL AREAS'
AREA_COLS = [
    'Area 1 Size of Government',
    'Area 2 Legal System & Property Rights -- With Gender Adjustment',
    'Area 3 Sound Money',
    'Area 4 Freedom to trade internationally',
    'Area 5 Regulation'
]
AREA_LABELS = ['Size of Gov', 'Legal System', 'Sound Money', 'Trade Freedom', 'Regulation']
QUINQUENNIAL_YEARS = [1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
CLEAN_DIR = Path("data/clean")


def load_efw(path: str, iso_col: str = 'ISO_Code', year_col: str = 'Year', verbose: bool = True) -> pd.DataFrame:
    """Load and standardize EFW data for merging."""
    path = Path(path)
    
    if path.suffix == '.xlsx':
        try:
            import efw_utils as efw
            df = efw.load_fraser(str(path), verbose=verbose)
        except:
            df = pd.read_excel(path, header=3)
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    
    df = df.rename(columns={iso_col: 'iso3', year_col: 'year'})
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    if 'iso3' in df.columns:
        df['iso3'] = df['iso3'].astype(str).str.upper().str.strip()
        df = df[df['iso3'].str.len() == 3]
    
    if EFW_COL in df.columns:
        df['efw_present'] = df[EFW_COL].notna()
    
    area_cols_present = [c for c in AREA_COLS if c in df.columns]
    if area_cols_present:
        df['n_areas'] = df[area_cols_present].notna().sum(axis=1)
        df['complete5'] = df['n_areas'] == 5
    
    if verbose:
        print(f"Loaded EFW: {len(df):,} obs, {df['iso3'].nunique()} countries")
    
    return df


def get_efw_quinquennial(df: pd.DataFrame) -> pd.DataFrame:
    """Extract quinquennial observations from EFW data."""
    return df[df['year'].isin(QUINQUENNIAL_YEARS)].copy()


def get_efw_annual(df: pd.DataFrame, start_year: int = 2000) -> pd.DataFrame:
    """Extract annual-era observations from EFW data."""
    return df[df['year'] >= start_year].copy()


def merge_efw_macro(efw_df: pd.DataFrame, macro_df: pd.DataFrame, how: str = 'outer', 
                    verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Merge EFW and macro panels with overlap accounting."""
    efw_keys = set(zip(efw_df['iso3'], efw_df['year']))
    macro_keys = set(zip(macro_df['iso3'], macro_df['year']))
    
    overlap_stats = {
        'efw_obs': len(efw_df), 'macro_obs': len(macro_df),
        'keys_both': len(efw_keys & macro_keys),
        'countries_efw': efw_df['iso3'].nunique(),
        'countries_macro': macro_df['iso3'].nunique(),
    }
    
    df = pd.merge(efw_df, macro_df, on=['iso3', 'year'], how=how, suffixes=('', '_macro'))
    df['efw_present'] = df[EFW_COL].notna() if EFW_COL in df.columns else False
    
    macro_cols = [c for c in macro_df.columns if c not in ['iso3', 'year']]
    df['macro_n_vars'] = df[[c for c in macro_cols if c in df.columns]].notna().sum(axis=1)
    df['macro_present'] = df['macro_n_vars'] > 0
    df['merged_present'] = df['efw_present'] & df['macro_present']
    
    if verbose:
        print(f"Merge: {len(df):,} rows, {df['merged_present'].sum():,} with both")
    
    return df, overlap_stats


def create_analysis_samples(df: pd.DataFrame, core_macro_cols: List[str], 
                           verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Create multiple analysis samples with different missingness requirements."""
    samples = {}
    samples['efw_only'] = df[df['efw_present'].fillna(False)].copy()
    samples['efw_any_macro'] = df[df['efw_present'].fillna(False) & df['macro_present'].fillna(False)].copy()
    
    core_present = [c for c in core_macro_cols if c in df.columns]
    if core_present:
        df['complete_macro_core'] = df[core_present].notna().all(axis=1)
        samples['efw_complete_macro'] = df[df['efw_present'].fillna(False) & df['complete_macro_core']].copy()
    
    if 'complete5' in df.columns:
        samples['efw_complete5'] = df[df['complete5'].fillna(False)].copy()
    
    if verbose:
        for name, s in samples.items():
            print(f"  {name}: {len(s):,} obs")
    
    return samples


def build_quinquennial_panel(df: pd.DataFrame, waves: List[int] = None, 
                             verbose: bool = True) -> pd.DataFrame:
    """Build quinquennial panel aligned to EFW waves (no interpolation)."""
    if waves is None:
        waves = QUINQUENNIAL_YEARS
    return df[df['year'].isin(waves)].copy()


def build_balanced_quinquennial(df: pd.DataFrame, waves: List[int] = None, 
                                require_col: str = None, verbose: bool = True) -> pd.DataFrame:
    """Build balanced quinquennial panel with countries in all waves."""
    if waves is None:
        waves = QUINQUENNIAL_YEARS
    if require_col is None:
        require_col = EFW_COL
    
    df_waves = df[df['year'].isin(waves)].copy()
    obs_per_country = df_waves.groupby('iso3')[require_col].apply(lambda x: x.notna().sum())
    balanced = obs_per_country[obs_per_country == len(waves)].index.tolist()
    
    if verbose:
        print(f"Balanced: {len(balanced)} countries in all {len(waves)} waves")
    
    return df_waves[df_waves['iso3'].isin(balanced)].copy()


def save_merged_panel(df: pd.DataFrame, name: str = "efw_macro_merged"):
    """Save merged panel."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_DIR / f"{name}.parquet", index=False)
    df.to_csv(CLEAN_DIR / f"{name}.csv", index=False)
