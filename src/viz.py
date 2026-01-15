"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })


def save_figure(fig: plt.Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved figure: {path}")


def plot_missingness_heatmap(df: pd.DataFrame, value_cols: Iterable[str], title: str, path: Path | str):
    data = df[list(value_cols)].isna().astype(int)
    fig, ax = plt.subplots()
    sns.heatmap(data.T, cbar=False, ax=ax, cmap="Reds")
    ax.set_title(title)
    ax.set_xlabel("Row index")
    ax.set_ylabel("Variables")
    save_figure(fig, path)
    plt.close(fig)


def plot_distribution(series: pd.Series, title: str, path: Path | str):
    fig, ax = plt.subplots()
    sns.histplot(series.dropna(), bins=30, kde=True, ax=ax)
    ax.set_title(title)
    save_figure(fig, path)
    plt.close(fig)


def plot_irf(irf_df: pd.DataFrame, title: str, path: Path | str):
    fig, ax = plt.subplots()
    for term, sub in irf_df.groupby("term"):
        sub = sub.sort_values("horizon")
        horizons = sub["horizon"]
        coef = sub["coef"]
        se = sub["std_err"]
        ax.plot(horizons, coef, marker="o", label=term)
        ax.fill_between(horizons, coef - 1.96 * se, coef + 1.96 * se, alpha=0.2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Horizon (steps of 5 years)")
    ax.set_ylabel("Effect")
    ax.legend()
    save_figure(fig, path)
    plt.close(fig)


def plot_event_study(df: pd.DataFrame, title: str, path: Path | str):
    fig, ax = plt.subplots()
    for label, sub in df.groupby("label"):
        sub = sub.sort_values("event_time")
        ax.plot(sub["event_time"], sub["value"], marker="o", label=label)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Event time (years)")
    ax.set_ylabel("GDP per capita growth (log diff)")
    ax.legend()
    save_figure(fig, path)
    plt.close(fig)
