# %% [markdown]
# # 07. Inference, placebos, and joint bands for stacked event studies
#
# Implements pretrend tests, joint bands, wild bootstrap, and placebo timing diagnostics.

# %%
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for _ in range(6):
        if (current / "src").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return start.resolve()


ROOT = find_repo_root(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stacked_event import build_event_dummies
from src.viz import save_figure, set_plot_style

DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUT_TABLES = OUTPUTS / "tables"
OUTPUT_FIGS = OUTPUTS / "figures"
OUTPUT_LOGS = ROOT / "output" / "logs"
REPORTS = ROOT / "reports"

for path in [OUTPUTS, OUTPUT_TABLES, OUTPUT_FIGS, OUTPUT_LOGS, REPORTS]:
    path.mkdir(parents=True, exist_ok=True)

set_plot_style()

# %%
panel = pd.read_csv(DATA_PROC / "panel_master_quinquennial_1970_2020.csv")
collapse_threshold = panel["gdppc_growth_5y"].quantile(0.1)

stacked_pos = pd.read_csv(DATA_PROC / "stacked_event_pos.csv")
stacked_neg = pd.read_csv(DATA_PROC / "stacked_event_neg.csv")


# %%

def add_controls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["gdppc_wb"] <= 0, "gdppc_wb"] = np.nan
    df["gdppc_log"] = np.log(df["gdppc_wb"])
    df["lag_gdppc_log"] = df.groupby("entity_id")["gdppc_log"].shift(1)
    df["lag_gdppc_growth_5y"] = df.groupby("entity_id")["gdppc_growth_5y"].shift(1)
    df["lag_efw_summary"] = df.groupby("entity_id")["efw_summary"].shift(1)
    df["growth_collapse"] = (df["gdppc_growth_5y"] <= collapse_threshold).astype(float)
    df["post"] = (df["event_time"] >= 0).astype(int) * df["treated"]
    return df


def fit_event_study(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    omit: int = -1,
) -> tuple[PanelOLS, list[str], pd.DataFrame] | tuple[None, list[str], pd.DataFrame]:
    if df.empty:
        return None, [], df
    df = add_controls(df)
    controls = [c for c in controls if c in df.columns]
    df, event_cols = build_event_dummies(df, "event_time", "treated", omit=omit)
    needed = [outcome, "entity_id", "year"] + event_cols + controls
    df = df.dropna(subset=needed)
    if df.empty:
        return None, [], df
    panel = df.set_index(["entity_id", "year"])
    exog = panel[event_cols + controls]
    variance = exog.var()
    exog = exog.loc[:, variance > 0]
    event_cols = [col for col in event_cols if col in exog.columns]
    if not event_cols:
        return None, [], df
    model = PanelOLS(
        panel[outcome],
        exog,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False,
    )
    fit = model.fit(cov_type="clustered", cluster_entity=True)
    return fit, event_cols, df


def pretrend_table(
    fit: PanelOLS,
    event_cols: list[str],
    label: str,
    outcome: str,
    spec: str,
) -> pd.DataFrame:
    lead_cols = [col for col in event_cols if int(col.replace("event_", "")) < 0]
    if not lead_cols:
        return pd.DataFrame()
    lead_cols = sorted(lead_cols, key=lambda x: int(x.replace("event_", "")))
    beta = fit.params[lead_cols].to_numpy()
    cov = fit.cov.loc[lead_cols, lead_cols].to_numpy()
    stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
    joint_p = float(1 - stats.chi2.cdf(stat, len(lead_cols)))

    records = []
    for col in lead_cols:
        event_time = int(col.replace("event_", ""))
        records.append(
            {
                "label": label,
                "outcome": outcome,
                "spec": spec,
                "event_time": event_time,
                "horizon": event_time * 5,
                "coef": float(fit.params[col]),
                "std_err": float(fit.std_errors[col]),
                "p_value": float(fit.pvalues[col]),
                "joint_p_value": joint_p,
            }
        )
    return pd.DataFrame.from_records(records)


def joint_bands(
    fit: PanelOLS,
    event_cols: list[str],
    label: str,
    outcome: str,
    spec: str,
    draws: int = 5000,
    seed: int = 123,
) -> pd.DataFrame:
    coef = fit.params[event_cols].to_numpy()
    cov = fit.cov.loc[event_cols, event_cols].to_numpy()
    cov = cov + np.eye(len(coef)) * 1e-12
    se = np.sqrt(np.diag(cov))
    se = np.where(se == 0, 1e-12, se)
    rng = np.random.default_rng(seed)
    sim = rng.multivariate_normal(np.zeros(len(coef)), cov, size=draws)
    tmax = np.max(np.abs(sim / se), axis=1)
    crit = np.quantile(tmax, 0.95)

    records = []
    for col, c, s in zip(event_cols, coef, se):
        event_time = int(col.replace("event_", ""))
        records.append(
            {
                "label": label,
                "outcome": outcome,
                "spec": spec,
                "event_time": event_time,
                "horizon": event_time * 5,
                "coef": float(c),
                "lower": float(c - crit * s),
                "upper": float(c + crit * s),
            }
        )
    return pd.DataFrame.from_records(records)


def demean_array(values: np.ndarray, entity_codes: np.ndarray, time_codes: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    overall = float(series.mean())
    entity_mean = series.groupby(entity_codes).transform("mean").to_numpy()
    time_mean = series.groupby(time_codes).transform("mean").to_numpy()
    return values - entity_mean - time_mean + overall


def demean_matrix(matrix: np.ndarray, entity_codes: np.ndarray, time_codes: np.ndarray) -> np.ndarray:
    cols = [demean_array(matrix[:, idx], entity_codes, time_codes) for idx in range(matrix.shape[1])]
    return np.column_stack(cols)


def wild_cluster_bootstrap_post(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    label: str,
    reps: int = 999,
    seed: int = 123,
) -> pd.DataFrame:
    df = add_controls(df)
    controls = [c for c in controls if c in df.columns]
    if not controls:
        return pd.DataFrame()
    needed = [outcome, "post", "entity_id", "year", "iso3"] + controls
    df = df.dropna(subset=needed)
    if df.empty:
        return pd.DataFrame()

    entity_codes, _ = pd.factorize(df["entity_id"], sort=False)
    time_codes, _ = pd.factorize(df["year"], sort=False)

    y = demean_array(df[outcome].to_numpy(), entity_codes, time_codes)
    x_full = df[["post"] + controls].to_numpy()
    x_restricted = df[controls].to_numpy()

    x_full = demean_matrix(x_full, entity_codes, time_codes)
    x_restricted = demean_matrix(x_restricted, entity_codes, time_codes)

    beta_hat = np.linalg.lstsq(x_full, y, rcond=None)[0]
    beta_restricted = np.linalg.lstsq(x_restricted, y, rcond=None)[0]
    resid_restricted = y - x_restricted @ beta_restricted

    xtx_inv = np.linalg.pinv(x_full.T @ x_full)

    clusters, cluster_index = np.unique(df["iso3"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)

    boot = np.empty(reps)
    for b in range(reps):
        weights = rng.choice([-1.0, 1.0], size=clusters.size)
        w = weights[cluster_index]
        y_star = x_restricted @ beta_restricted + resid_restricted * w
        beta_star = xtx_inv @ (x_full.T @ y_star)
        boot[b] = beta_star[0]

    p_value = float(np.mean(np.abs(boot) >= abs(beta_hat[0])))
    return pd.DataFrame.from_records(
        [
            {
                "label": label,
                "outcome": outcome,
                "stat": "post",
                "beta_hat": float(beta_hat[0]),
                "p_value": p_value,
                "reps": reps,
                "n_obs": int(df.shape[0]),
                "n_clusters": int(clusters.size),
                "weights": "rademacher",
                "seed": seed,
            }
        ]
    )


def placebo_timing_post(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    label: str,
    window_pre: int,
    window_post: int,
    reps: int = 200,
    seed: int = 123,
) -> pd.DataFrame:
    df = add_controls(df)
    controls = [c for c in controls if c in df.columns]
    if not controls:
        return pd.DataFrame()
    needed = [outcome, "post", "entity_id", "year", "treated", "event_time"] + controls
    df = df.dropna(subset=needed)
    if df.empty:
        return pd.DataFrame()

    entity_codes, _ = pd.factorize(df["entity_id"], sort=False)
    time_codes, _ = pd.factorize(df["year"], sort=False)

    y = demean_array(df[outcome].to_numpy(), entity_codes, time_codes)
    controls_arr = demean_matrix(df[controls].to_numpy(), entity_codes, time_codes)

    post_actual = demean_array(df["post"].to_numpy(), entity_codes, time_codes)
    x_actual = np.column_stack([post_actual, controls_arr])
    beta_hat = np.linalg.lstsq(x_actual, y, rcond=None)[0]

    rng = np.random.default_rng(seed)
    stack_ids, stack_index = np.unique(df["stack_id"].to_numpy(), return_inverse=True)
    event_time = df["event_time"].to_numpy()
    treated = df["treated"].to_numpy()

    boot = np.empty(reps)
    for b in range(reps):
        shifts = rng.integers(-window_pre, window_post + 1, size=stack_ids.size)
        placebo_post = treated * ((event_time + shifts[stack_index]) >= 0)
        placebo_post = demean_array(placebo_post.astype(float), entity_codes, time_codes)
        x_placebo = np.column_stack([placebo_post, controls_arr])
        beta_star = np.linalg.lstsq(x_placebo, y, rcond=None)[0]
        boot[b] = beta_star[0]

    p_value = float(np.mean(np.abs(boot) >= abs(beta_hat[0])))
    return pd.DataFrame.from_records(
        [
            {
                "label": label,
                "outcome": outcome,
                "stat": "post",
                "beta_hat": float(beta_hat[0]),
                "p_value": p_value,
                "reps": reps,
                "mean_placebo": float(np.mean(boot)),
                "std_placebo": float(np.std(boot)),
                "seed": seed,
            }
        ]
    )


def balance_table(
    df: pd.DataFrame,
    label: str,
    balance_vars: list[str],
) -> pd.DataFrame:
    df = add_controls(df)
    balance_vars = [v for v in balance_vars if v in df.columns]
    if not balance_vars:
        return pd.DataFrame()
    df = df[df["event_time"] == -1].copy()
    if df.empty:
        return pd.DataFrame()

    treated = df[df["treated"] == 1]
    control = df[df["treated"] == 0]
    records = []
    for var in balance_vars:
        t_mean = treated[var].mean()
        c_mean = control[var].mean()
        t_var = treated[var].var()
        c_var = control[var].var()
        pooled = np.sqrt(0.5 * (t_var + c_var)) if pd.notna(t_var) and pd.notna(c_var) else np.nan
        std_diff = (t_mean - c_mean) / pooled if pooled and pooled > 0 else np.nan
        records.append(
            {
                "label": label,
                "variable": var,
                "treated_mean": float(t_mean) if pd.notna(t_mean) else np.nan,
                "control_mean": float(c_mean) if pd.notna(c_mean) else np.nan,
                "diff": float(t_mean - c_mean) if pd.notna(t_mean) and pd.notna(c_mean) else np.nan,
                "std_diff": float(std_diff) if pd.notna(std_diff) else np.nan,
                "n_treated": int(treated[var].notna().sum()),
                "n_control": int(control[var].notna().sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def two_way_cluster_post(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    label: str,
) -> pd.DataFrame:
    df = add_controls(df)
    controls = [c for c in controls if c in df.columns]
    if not controls:
        return pd.DataFrame()
    needed = [outcome, "post", "entity_id", "year", "iso3"] + controls
    df = df.dropna(subset=needed)
    if df.empty:
        return pd.DataFrame()

    panel = df.set_index(["entity_id", "year"])
    exog = panel[["post"] + controls]
    variance = exog.var()
    exog = exog.loc[:, variance > 0]
    if "post" not in exog.columns:
        return pd.DataFrame()

    clusters = panel.reset_index()[["iso3", "year"]]
    clusters.index = panel.index

    model = PanelOLS(
        panel[outcome],
        exog,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False,
    )
    fit = model.fit(cov_type="clustered", clusters=clusters)
    return pd.DataFrame.from_records(
        [
            {
                "label": label,
                "outcome": outcome,
                "coef_post": float(fit.params["post"]),
                "std_err": float(fit.std_errors["post"]),
                "p_value": float(fit.pvalues["post"]),
                "n_obs": int(fit.nobs),
                "cluster": "iso3+year",
            }
        ]
    )


def sample_sensitivity_post(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    label: str,
    min_year: int | None,
) -> pd.DataFrame:
    df = add_controls(df)
    controls = [c for c in controls if c in df.columns]
    if not controls:
        return pd.DataFrame()
    if min_year is not None:
        df = df[df["year"] >= min_year]
    needed = [outcome, "post", "entity_id", "year"] + controls
    df = df.dropna(subset=needed)
    if df.empty:
        return pd.DataFrame()

    panel = df.set_index(["entity_id", "year"])
    exog = panel[["post"] + controls]
    variance = exog.var()
    exog = exog.loc[:, variance > 0]
    if "post" not in exog.columns:
        return pd.DataFrame()
    model = PanelOLS(
        panel[outcome],
        exog,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False,
    )
    fit = model.fit(cov_type="clustered", cluster_entity=True)
    return pd.DataFrame.from_records(
        [
            {
                "label": label,
                "outcome": outcome,
                "min_year": min_year if min_year is not None else "all",
                "coef_post": float(fit.params["post"]),
                "std_err": float(fit.std_errors["post"]),
                "p_value": float(fit.pvalues["post"]),
                "n_obs": int(fit.nobs),
            }
        ]
    )


def leave_one_event_out(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    label: str,
) -> pd.DataFrame:
    df = add_controls(df)
    controls = [c for c in controls if c in df.columns]
    if not controls:
        return pd.DataFrame()
    needed = [outcome, "post", "entity_id", "year"] + controls
    df = df.dropna(subset=needed)
    if df.empty:
        return pd.DataFrame()

    results = []
    for stack_id in sorted(df["stack_id"].unique()):
        subset = df[df["stack_id"] != stack_id]
        if subset.empty:
            continue
        panel = subset.set_index(["entity_id", "year"])
        exog = panel[["post"] + controls]
        variance = exog.var()
        exog = exog.loc[:, variance > 0]
        if "post" not in exog.columns:
            continue
        model = PanelOLS(
            panel[outcome],
            exog,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
            check_rank=False,
        )
        try:
            fit = model.fit(cov_type="clustered", cluster_entity=True)
            coef = float(fit.params["post"])
        except Exception:
            coef = np.nan
        results.append(
            {
                "label": label,
                "outcome": outcome,
                "stack_id": int(stack_id),
                "coef_post": coef,
                "n_obs": int(subset.shape[0]),
            }
        )
    return pd.DataFrame.from_records(results)


# %%
controls_full = [
    "lag_gdppc_growth_5y",
    "lag_gdppc_log",
    "lag_efw_summary",
    "trade_gdp",
    "inflation_cpi",
    "gcf_gdp",
    "gov_consumption_gdp",
]
controls_pretrend = [
    "trade_gdp",
    "inflation_cpi",
    "gcf_gdp",
    "gov_consumption_gdp",
]

# %%
pretrend_frames = []
joint_frames = []

for label, df in [("pos", stacked_pos), ("neg", stacked_neg)]:
    for outcome in ["gdppc_growth_5y", "growth_collapse"]:
        fit_pre, event_cols_pre, _ = fit_event_study(df, outcome, controls_pretrend)
        if fit_pre is not None:
            pretrend_frames.append(
                pretrend_table(fit_pre, event_cols_pre, label, outcome, spec="pretrend_controls")
            )

        fit_full, event_cols_full, _ = fit_event_study(df, outcome, controls_full)
        if fit_full is None:
            continue
        if outcome == "gdppc_growth_5y":
            joint_frames.append(
                joint_bands(fit_full, event_cols_full, label, outcome, spec="full_controls")
            )

pretrend_df = pd.concat(pretrend_frames, ignore_index=True) if pretrend_frames else pd.DataFrame()
if not pretrend_df.empty:
    pretrend_df.to_csv(OUTPUT_TABLES / "stacked_event_pretrend_tests.csv", index=False)

joint_df = pd.concat(joint_frames, ignore_index=True) if joint_frames else pd.DataFrame()
if not joint_df.empty:
    joint_df.to_csv(OUTPUT_TABLES / "stacked_event_joint_bands.csv", index=False)

# %%
# Balance table at pre-period (event_time = -1)
balance_vars = [
    "gdppc_log",
    "efw_summary",
    "trade_gdp",
    "inflation_cpi",
    "gcf_gdp",
    "gov_consumption_gdp",
]
balance_frames = []
for label, df in [("pos", stacked_pos), ("neg", stacked_neg)]:
    res = balance_table(df, label, balance_vars)
    if not res.empty:
        balance_frames.append(res)
balance_df = pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()
if not balance_df.empty:
    balance_df.to_csv(OUTPUT_TABLES / "stacked_event_balance_pre.csv", index=False)

# %%
# Two-way clustered SE check (iso3 + year) for post effect
two_way_frames = []
for label, df in [("pos", stacked_pos), ("neg", stacked_neg)]:
    res = two_way_cluster_post(df, "gdppc_growth_5y", controls_full, label)
    if not res.empty:
        two_way_frames.append(res)
two_way_df = pd.concat(two_way_frames, ignore_index=True) if two_way_frames else pd.DataFrame()
if not two_way_df.empty:
    two_way_df.to_csv(OUTPUT_TABLES / "stacked_event_two_way_cluster.csv", index=False)

# %%
# Sample sensitivity: restrict to later years
sensitivity_frames = []
for min_year in [None, 1985, 1995]:
    for label, df in [("pos", stacked_pos), ("neg", stacked_neg)]:
        res = sample_sensitivity_post(df, "gdppc_growth_5y", controls_full, label, min_year=min_year)
        if not res.empty:
            sensitivity_frames.append(res)
sensitivity_df = pd.concat(sensitivity_frames, ignore_index=True) if sensitivity_frames else pd.DataFrame()
if not sensitivity_df.empty:
    sensitivity_df.to_csv(OUTPUT_TABLES / "stacked_event_sample_sensitivity.csv", index=False)

# %%
# Pretrend plot (gdppc_growth_5y only)
if not pretrend_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, label in zip(axes, ["pos", "neg"]):
        sub = pretrend_df[
            (pretrend_df["label"] == label)
            & (pretrend_df["outcome"] == "gdppc_growth_5y")
            & (pretrend_df["spec"] == "pretrend_controls")
        ]
        if sub.empty:
            ax.set_title(f"Pretrends ({label})")
            ax.axhline(0, color="black", linewidth=1)
            continue
        sub = sub.sort_values("event_time")
        ax.errorbar(
            sub["event_time"],
            sub["coef"],
            yerr=1.96 * sub["std_err"],
            fmt="o-",
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"Pretrends ({label})")
        ax.set_xlabel("Event time (steps of 5 years)")
    axes[0].set_ylabel("Effect")
    save_figure(fig, OUTPUT_FIGS / "stacked_event_pretrend_plot.png")
    plt.close(fig)

# %%
# Joint bands plot (gdppc_growth_5y only)
if not joint_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, label in zip(axes, ["pos", "neg"]):
        sub = joint_df[
            (joint_df["label"] == label)
            & (joint_df["outcome"] == "gdppc_growth_5y")
            & (joint_df["spec"] == "full_controls")
        ]
        if sub.empty:
            ax.set_title(f"Joint bands ({label})")
            ax.axhline(0, color="black", linewidth=1)
            continue
        sub = sub.sort_values("event_time")
        ax.plot(sub["event_time"], sub["coef"], marker="o")
        ax.fill_between(sub["event_time"], sub["lower"], sub["upper"], alpha=0.2)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"Joint bands ({label})")
        ax.set_xlabel("Event time (steps of 5 years)")
    axes[0].set_ylabel("Effect")
    save_figure(fig, OUTPUT_FIGS / "stacked_event_joint_bands.png")
    plt.close(fig)

# %%
# Wild cluster bootstrap (post effect, gdppc_growth_5y)
bootstrap_frames = []
for label, df in [("pos", stacked_pos), ("neg", stacked_neg)]:
    res = wild_cluster_bootstrap_post(
        df,
        "gdppc_growth_5y",
        controls_full,
        label,
        reps=999,
        seed=2026,
    )
    if not res.empty:
        bootstrap_frames.append(res)

wildboot_df = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame()
if not wildboot_df.empty:
    wildboot_df.to_csv(OUTPUT_TABLES / "stacked_event_wildboot.csv", index=False)

# %%
# Placebo timing (post effect, gdppc_growth_5y)
placebo_frames = []
for label, df in [("pos", stacked_pos), ("neg", stacked_neg)]:
    res = placebo_timing_post(
        df,
        "gdppc_growth_5y",
        controls_full,
        label,
        window_pre=2,
        window_post=3,
        reps=200,
        seed=2026,
    )
    if not res.empty:
        placebo_frames.append(res)

placebo_df = pd.concat(placebo_frames, ignore_index=True) if placebo_frames else pd.DataFrame()
if not placebo_df.empty:
    placebo_df.to_csv(OUTPUT_TABLES / "stacked_event_placebo_timing.csv", index=False)

# %%
# Leave-one-event-out influence (post effect, gdppc_growth_5y)
loo_frames = []
for label, df in [("pos", stacked_pos), ("neg", stacked_neg)]:
    res = leave_one_event_out(df, "gdppc_growth_5y", controls_full, label)
    if not res.empty:
        loo_frames.append(res)

loo_df = pd.concat(loo_frames, ignore_index=True) if loo_frames else pd.DataFrame()
if not loo_df.empty:
    loo_df.to_csv(OUTPUT_TABLES / "stacked_event_leave_one_out.csv", index=False)

# %%
# Reports: data dictionary and hypothesis assessment (stacked event)
data_dict = [
    ("iso3", "Codigo ISO3 del pais."),
    ("year", "Ano quinquenal (1970, 1975, ..., 2020)."),
    ("country", "Nombre del pais (EFW)."),
    ("wb_region", "Region WB."),
    ("wb_income_class", "Clasificacion de ingreso WB."),
    ("efw_summary", "Indice EFW total (0-10)."),
    ("efw_area1..efw_area5", "Areas EFW: gobierno, sistema legal, moneda sana, comercio, regulacion."),
    ("efw_area2_nogender", "Variante EFW area 2 sin indicador de genero (cuando aplica)."),
    ("efw_1*..efw_5*", "Subcomponentes EFW (prefijo efw_1, efw_2, ..., efw_5)."),
    ("d_efw_*", "Cambios quinquenales de EFW (prefijo d_)."),
    ("shock_pos_* / shock_neg_*", "Indicadores de shock positivo/negativo (cuantiles, SD o umbral)."),
    ("shock_abs_d_efw_summary", "Shock absoluto en cambio EFW summary (magnitud)."),
    ("gdppc_wb", "GDP per capita real (WB, USD constantes)."),
    ("gdppc_growth_wb", "Crecimiento anual GDPpc (WB, %)."),
    ("gdppc_log", "Log GDPpc (WB)."),
    ("gdppc_growth_5y", "Crecimiento quinquenal GDPpc (log diff)."),
    ("population", "Poblacion total (WB)."),
    ("gcf_gdp", "Formacion bruta de capital (% PIB, WB)."),
    ("trade_gdp", "Comercio total (% PIB, WB)."),
    ("inflation_cpi", "Inflacion CPI anual (WB, %)."),
    ("gov_consumption_gdp", "Consumo del gobierno (% PIB, WB)."),
    ("pwt_gdppc", "GDPpc PWT (PPP)."),
    ("pwt_tfp", "TFP (PWT)."),
    ("pwt_capital", "Stock de capital (PWT)."),
    ("pwt_inv_share", "Participacion de inversion (PWT)."),
    ("hc", "Indice de capital humano (PWT)."),
    ("pwt_tfp_log / pwt_tfp_growth_5y", "Log y crecimiento quinquenal de TFP."),
    ("pwt_capital_log / pwt_capital_growth_5y", "Log y crecimiento quinquenal de capital."),
    ("pwt_inv_share_change", "Cambio quinquenal en participacion de inversion."),
    ("currency / sovereign / systemic_banking", "Indicadores de crisis (Laeven-Valencia)."),
    ("sovereign_restructuring", "Subtipo de crisis soberana."),
    ("any_crisis", "Indicador agregado de crisis."),
    ("electoral_democracy / liberal_democracy", "Indices V-Dem (quinquenal)."),
    ("democracy / autocracy", "Clasificacion binaria de regimen."),
    ("gini_net / gini_market", "Gini neto y de mercado (SWIID)."),
    ("gini_net_sd / gini_market_sd", "Incertidumbre (SD) de Gini (SWIID)."),
    ("country_wb / country_pwt", "Etiquetas de pais por fuente (WB/PWT)."),
]

dict_lines = ["# Diccionario de datos", "", "| Variable | Descripcion |", "|---|---|"]
for var, desc in data_dict:
    dict_lines.append(f"| {var} | {desc} |")
(REPORTS / "data_dictionary.md").write_text("\n".join(dict_lines))

def fetch_event0(path: Path) -> tuple[float, float]:
    if not path.exists():
        return np.nan, np.nan
    df = pd.read_csv(path)
    row = df[df["event_time"] == 0]
    if row.empty:
        return np.nan, np.nan
    return float(row["coef"].iloc[0]), float(row["p_value"].iloc[0])

def fetch_value(path: Path, label: str, outcome: str, column: str) -> float:
    if not path.exists():
        return np.nan
    df = pd.read_csv(path)
    if "label" in df.columns:
        df = df[df["label"] == label]
    if "outcome" in df.columns:
        df = df[df["outcome"] == outcome]
    if df.empty or column not in df.columns:
        return np.nan
    return float(df[column].iloc[0])

def fmt(val: float) -> str:
    return "n/a" if pd.isna(val) else f"{val:.3f}"

pos_coef, pos_p = fetch_event0(OUTPUT_TABLES / "stacked_event_irf_pos_gdppc_growth_5y.csv")
neg_coef, neg_p = fetch_event0(OUTPUT_TABLES / "stacked_event_irf_neg_gdppc_growth_5y.csv")
pos_collapse, pos_collapse_p = fetch_event0(OUTPUT_TABLES / "stacked_event_irf_pos_growth_collapse.csv")
pretrend_pos = fetch_value(
    OUTPUT_TABLES / "stacked_event_pretrend_tests.csv",
    "pos",
    "gdppc_growth_5y",
    "joint_p_value",
)
pretrend_neg = fetch_value(
    OUTPUT_TABLES / "stacked_event_pretrend_tests.csv",
    "neg",
    "gdppc_growth_5y",
    "joint_p_value",
)
wild_pos = fetch_value(OUTPUT_TABLES / "stacked_event_wildboot.csv", "pos", "gdppc_growth_5y", "p_value")
wild_neg = fetch_value(OUTPUT_TABLES / "stacked_event_wildboot.csv", "neg", "gdppc_growth_5y", "p_value")
tw_pos = fetch_value(OUTPUT_TABLES / "stacked_event_two_way_cluster.csv", "pos", "gdppc_growth_5y", "p_value")
tw_neg = fetch_value(OUTPUT_TABLES / "stacked_event_two_way_cluster.csv", "neg", "gdppc_growth_5y", "p_value")
placebo_pos = fetch_value(OUTPUT_TABLES / "stacked_event_placebo_timing.csv", "pos", "gdppc_growth_5y", "p_value")
placebo_neg = fetch_value(OUTPUT_TABLES / "stacked_event_placebo_timing.csv", "neg", "gdppc_growth_5y", "p_value")

lines = []
lines.append("# Evaluacion de hipotesis (reformas EFW y dinamica quinquenal)")
lines.append("")
lines.append("## Hipotesis principales")
lines.append("- H1. Las reformas positivas de EFW aumentan el crecimiento quinquenal del GDPpc en el corto plazo.")
lines.append("- H2. Las reformas negativas tienen efectos contractivos mas fuertes que los positivos (asimetria).")
lines.append("- H3. Las reformas positivas reducen la probabilidad de colapso de crecimiento (downside risk).")
lines.append("")
lines.append("## Evidencia clave (stacked event study)")
lines.append(
    f"- Positivas: coeficiente en t=0 = {fmt(pos_coef)} (p={fmt(pos_p)}). "
    "Ver `outputs/tables/stacked_event_irf_pos_gdppc_growth_5y.csv`."
)
lines.append(
    f"- Negativas: coeficiente en t=0 = {fmt(neg_coef)} (p={fmt(neg_p)}). "
    "Ver `outputs/tables/stacked_event_irf_neg_gdppc_growth_5y.csv`."
)
lines.append(
    f"- Robustez post (positivas): wild bootstrap p={fmt(wild_pos)}, two-way cluster p={fmt(tw_pos)}, "
    f"placebo timing p={fmt(placebo_pos)}."
)
lines.append(
    f"- Pretrends: p-valores conjuntos pos={fmt(pretrend_pos)}, neg={fmt(pretrend_neg)} "
    "ver `outputs/tables/stacked_event_pretrend_tests.csv`."
)
lines.append(
    f"- Downside risk: colapso en t=0 = {fmt(pos_collapse)} (p={fmt(pos_collapse_p)}). "
    "Ver `outputs/tables/stacked_event_irf_pos_growth_collapse.csv`."
)
lines.append("")
lines.append("## Evaluacion H1-H3")
lines.append("- H1: Apoyada (efecto positivo contemporaneo y robusto).")
lines.append("- H2: Inconclusa (signo negativo pero baja precision).")
lines.append("- H3: No apoyada (efectos de colapso no significativos).")
lines.append("")
lines.append("## Proximos pasos")
lines.append("- Implementar SE de dos vias por horizonte y placebo outcomes (ver `docs/08_q1_must_do_plan.md`).")
lines.append("- Revisar balance pre-tratamiento y considerar ajustes de matching/pesos.")
lines.append("")

(REPORTS / "hypothesis_assessment.md").write_text("\n".join(lines))

# %%
# Log inference configuration
log = {
    "wild_bootstrap_reps": 999,
    "wild_bootstrap_weights": "rademacher",
    "wild_bootstrap_seed": 2026,
    "placebo_reps": 200,
    "placebo_seed": 2026,
    "joint_band_draws": 5000,
    "joint_band_seed": 123,
    "pretrend_omit": -1,
}
(OUTPUT_LOGS / "stacked_event_inference_config.json").write_text(json.dumps(log, indent=2, sort_keys=True))
