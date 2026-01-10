"""
Quality Assurance Utilities
===========================
Validation, missingness analysis, and outlier detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import warnings

TABLES_DIR = Path("outputs/tables")


def validate_unique_keys(df: pd.DataFrame, keys: List[str] = ['iso3', 'year']) -> bool:
    """Check that panel has unique keys. Raises if not."""
    dups = df.groupby(keys).size()
    n_dups = (dups > 1).sum()
    if n_dups > 0:
        raise AssertionError(f"Found {n_dups} duplicate keys")
    print(f"✓ Unique keys validated: {len(df):,} rows")
    return True


def compute_missingness_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Compute missingness statistics for columns."""
    cols = [c for c in cols if c in df.columns]
    stats = []
    for col in cols:
        n_total = len(df)
        n_present = df[col].notna().sum()
        stats.append({
            'variable': col,
            'n_present': n_present,
            'n_missing': n_total - n_present,
            'pct_missing': 100 * (n_total - n_present) / n_total if n_total > 0 else 0,
            'coverage_pct': 100 * n_present / n_total if n_total > 0 else 0
        })
    return pd.DataFrame(stats)


def compute_decade_coverage(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Compute coverage by decade."""
    cols = [c for c in cols if c in df.columns]
    df = df.copy()
    df['decade'] = (df['year'] // 10) * 10
    
    results = []
    for decade in sorted(df['decade'].unique()):
        decade_df = df[df['decade'] == decade]
        row = {'decade': int(decade), 'n_obs': len(decade_df)}
        for col in cols:
            row[f'{col}_pct'] = 100 * decade_df[col].notna().mean()
        results.append(row)
    return pd.DataFrame(results)


def flag_outliers(df: pd.DataFrame, col: str, low: float = None, high: float = None,
                  iqr_mult: float = None) -> pd.Series:
    """Flag outlier values. Returns boolean Series."""
    flags = pd.Series(False, index=df.index)
    
    if low is not None:
        flags |= (df[col] < low)
    if high is not None:
        flags |= (df[col] > high)
    if iqr_mult is not None:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        flags |= (df[col] < q1 - iqr_mult * iqr) | (df[col] > q3 + iqr_mult * iqr)
    
    return flags


def create_outlier_report(df: pd.DataFrame, rules: Dict[str, Dict]) -> pd.DataFrame:
    """Create outlier report based on rules dict."""
    reports = []
    for col, rule in rules.items():
        if col not in df.columns:
            continue
        flags = flag_outliers(df, col, **rule)
        n_flagged = flags.sum()
        if n_flagged > 0:
            examples = df[flags][['iso3', 'year', col]].head(5)
            reports.append({
                'variable': col,
                'n_flagged': n_flagged,
                'pct_flagged': 100 * n_flagged / len(df),
                'rule': str(rule),
                'examples': examples.to_dict('records')
            })
    return pd.DataFrame(reports) if reports else pd.DataFrame()


def print_sample_dashboard(samples: Dict[str, pd.DataFrame]):
    """Print sample definition dashboard."""
    print("\n" + "=" * 60)
    print("SAMPLE DEFINITION DASHBOARD")
    print("=" * 60)
    for name, df in samples.items():
        if len(df) > 0:
            print(f"\n{name}:")
            print(f"  Observations: {len(df):,}")
            print(f"  Countries: {df['iso3'].nunique()}")
            print(f"  Years: {df['year'].min()}-{df['year'].max()}")
        else:
            print(f"\n{name}: EMPTY")


def save_table(df: pd.DataFrame, name: str):
    """Save table to outputs/tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    print(f"Saved: {TABLES_DIR}/{name}.csv")


def log_filter(before: int, after: int, reason: str):
    """Log a filter operation."""
    dropped = before - after
    print(f"  Filter ({reason}): {before:,} → {after:,} ({dropped:,} dropped, {100*dropped/before:.1f}%)")
