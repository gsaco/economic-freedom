"""
Plotting Utilities
==================
Standardized plotting helpers for EDA notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple

FIGURES_DIR = Path("outputs/figures")


def setup_plot_style():
    """Set up publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.figsize': (10, 6)
    })


def savefig(fig, name: str, dpi: int = 300):
    """Save figure to outputs/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")


def plot_time_trend(df: pd.DataFrame, col: str, title: str = None, 
                    show_median: bool = True, show_band: bool = True,
                    sample_label: str = None) -> plt.Figure:
    """Plot time trend with mean, median, and percentile bands."""
    stats = df.groupby('year')[col].agg(['mean', 'median', 'std', 'count',
        lambda x: x.quantile(0.10), lambda x: x.quantile(0.90)])
    stats.columns = ['mean', 'median', 'std', 'count', 'p10', 'p90']
    
    fig, ax = plt.subplots(figsize=(11, 5))
    
    if show_band:
        ax.fill_between(stats.index, stats['p10'], stats['p90'], 
                       alpha=0.2, color='#3498db', label='P10-P90')
    
    ax.plot(stats.index, stats['mean'], 'o-', color='#c0392b', 
           linewidth=2, markersize=4, label='Mean')
    
    if show_median:
        ax.plot(stats.index, stats['median'], 's--', color='#27ae60',
               linewidth=2, markersize=4, label='Median')
    
    ax.set_xlabel('Year')
    ax.set_ylabel(col)
    ax.set_title(title or f'{col} Over Time')
    ax.legend()
    
    if sample_label:
        ax.text(0.02, 0.98, sample_label, transform=ax.transAxes, 
               fontsize=8, va='top', ha='left', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, cols: List[str], title: str = None,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """Plot correlation matrix heatmap."""
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
               center=0, vmin=-1, vmax=1, ax=ax, square=True,
               cbar_kws={'shrink': 0.8})
    
    ax.set_title(title or 'Correlation Matrix')
    plt.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str = None,
                color_by: str = None, add_trend: bool = True) -> plt.Figure:
    """Plot scatter with optional trend line."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    if color_by and color_by in df.columns:
        for cat in df[color_by].dropna().unique():
            mask = df[color_by] == cat
            ax.scatter(df.loc[mask, x], df.loc[mask, y], alpha=0.5, label=cat, s=20)
        ax.legend(fontsize=8)
    else:
        ax.scatter(df[x], df[y], alpha=0.4, s=20, color='#3498db')
    
    if add_trend:
        valid = df[[x, y]].dropna()
        if len(valid) > 2:
            z = np.polyfit(valid[x], valid[y], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid[x].min(), valid[x].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f'{y} vs {x}')
    plt.tight_layout()
    return fig


def plot_distribution(df: pd.DataFrame, col: str, title: str = None,
                     bins: int = 40) -> plt.Figure:
    """Plot distribution histogram with KDE."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    data = df[col].dropna()
    ax.hist(data, bins=bins, density=True, alpha=0.6, color='#3498db', edgecolor='white')
    data.plot.kde(ax=ax, color='#2c3e50', linewidth=2)
    
    ax.axvline(data.mean(), color='#c0392b', linestyle='--', linewidth=2,
              label=f'Mean = {data.mean():.2f}')
    ax.axvline(data.median(), color='#27ae60', linestyle=':', linewidth=2,
              label=f'Median = {data.median():.2f}')
    
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.set_title(title or f'Distribution of {col}')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_missingness_heatmap(df: pd.DataFrame, cols: List[str], 
                            title: str = None) -> plt.Figure:
    """Plot missingness by year for multiple variables."""
    cols = [c for c in cols if c in df.columns]
    miss_by_year = df.groupby('year')[cols].apply(lambda x: x.isnull().mean() * 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(miss_by_year.T, cmap='YlOrRd', annot=False, ax=ax,
               cbar_kws={'label': 'Missing %'})
    ax.set_title(title or 'Missingness by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Variable')
    plt.tight_layout()
    return fig


def plot_convergence(sigma_df: pd.DataFrame, title: str = None) -> plt.Figure:
    """Plot sigma-convergence (SD over time)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(sigma_df.index, sigma_df['SD'], 'o-', color='#3498db', 
           linewidth=2, markersize=5, label='Cross-sectional SD')
    
    if 'Trend' in sigma_df.columns:
        ax.plot(sigma_df.index, sigma_df['Trend'], '--', color='#e74c3c',
               linewidth=2, label='Linear trend')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Standard Deviation')
    ax.set_title(title or 'Sigma-Convergence')
    ax.legend()
    plt.tight_layout()
    return fig
