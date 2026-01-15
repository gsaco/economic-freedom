# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Q1 candidate evidence: components, crises, downside risk
#
# This notebook runs Modules A–F, logs a spec ledger, and saves figures/tables
# for candidate paper angles.

# %%
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from sklearn.decomposition import PCA

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

from src.analysis_utils import bh_fdr, hash_file
from src.components import build_reform_bundles
from src.lp_models import run_local_projections, summarize_lp_results
from src.viz import plot_event_study, plot_irf, save_figure, set_plot_style

DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

for path in [OUTPUTS, FIGURES, TABLES]:
    path.mkdir(parents=True, exist_ok=True)

set_plot_style()

# %%
panel_path = DATA_PROC / "panel_master_quinquennial_1970_2020.csv"
panel = pd.read_csv(panel_path)
DATA_HASH = hash_file(panel_path)

# %%
panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

def add_lag(df: pd.DataFrame, col: str, lag: int = 1) -> None:
    df[f"lag_{col}"] = df.groupby("iso3")[col].shift(lag)


for col in [
    "gdppc_growth_5y",
    "gdppc_log",
    "efw_summary",
    "efw_area1",
    "efw_area2",
    "efw_area3",
    "efw_area4",
    "efw_area5",
    "pwt_tfp_growth_5y",
    "pwt_capital_growth_5y",
    "pwt_inv_share",
]:
    if col in panel.columns:
        add_lag(panel, col)

shock_cols = ["d_efw_summary", "d_efw_area1", "d_efw_area2", "d_efw_area3", "d_efw_area4", "d_efw_area5"]
for col in shock_cols:
    if col in panel.columns:
        panel[f"{col}_pos"] = panel[col].clip(lower=0)
        panel[f"{col}_neg"] = panel[col].clip(upper=0)

panel["efw_initial"] = (
    panel.sort_values("year").groupby("iso3")["efw_summary"].transform("first")
)
panel["efw_initial_group"] = pd.qcut(panel["efw_initial"], 3, labels=["low", "mid", "high"])

base_controls = [
    "lag_gdppc_growth_5y",
    "lag_gdppc_log",
    "lag_efw_summary",
    "trade_gdp",
    "inflation_cpi",
    "gcf_gdp",
    "gov_consumption_gdp",
]
base_controls = [c for c in base_controls if c in panel.columns]

spec_records: list[dict] = []


def controls_for_outcome(outcome: str) -> list[str]:
    controls = base_controls.copy()
    lag_outcome = f"lag_{outcome}"
    if lag_outcome in panel.columns and lag_outcome not in controls:
        controls.insert(0, lag_outcome)
    return [c for c in controls if c != outcome]


def record_lp_results(
    spec_id: str,
    module: str,
    outcome: str,
    shock_terms: list[str],
    results,
    controls: list[str],
):
    summary = summarize_lp_results(results, shock_terms)
    for _, row in summary.iterrows():
        if row["term"] in panel.columns:
            shock_std = panel[row["term"]].std(skipna=True)
        else:
            shock_std = np.nan
        spec_records.append(
            {
                "spec_id": spec_id,
                "module": module,
                "outcome": outcome,
                "term": row["term"],
                "horizon": row["horizon"],
                "coef": row["coef"],
                "std_err": row["std_err"],
                "p_value": row["p_value"],
                "nobs": row["nobs"],
                "effect_std": row["coef"] * shock_std if pd.notna(shock_std) else np.nan,
                "controls": ",".join(controls),
                "data_hash": DATA_HASH,
            }
        )
    return summary


def event_study(panel_df: pd.DataFrame, event_col: str, outcome_col: str, window_steps: list[int]) -> pd.DataFrame:
    events = panel_df[panel_df[event_col] == 1].copy()
    records = []
    for step in window_steps:
        temp = events.copy()
        temp["event_year"] = temp["year"] + step * 5
        merged = temp.merge(
            panel_df[["iso3", "year", outcome_col]],
            left_on=["iso3", "event_year"],
            right_on=["iso3", "year"],
            how="left",
            suffixes=("", "_event"),
        )
        mean_value = merged[f"{outcome_col}_event"].mean()
        records.append({"event_time": step * 5, "value": mean_value})
    return pd.DataFrame.from_records(records)

# %% [markdown]
# ## Module A — Component dynamics and asymmetries

# %%
outcomes = [
    "gdppc_growth_5y",
    "pwt_tfp_growth_5y",
    "pwt_inv_share_change",
    "inflation_cpi",
]
outcomes = [o for o in outcomes if o in panel.columns]

module_a_tables = []
summary_shock = "d_efw_summary"
area_shocks = [c for c in shock_cols if c.startswith("d_efw_area")]

for outcome in outcomes:
    controls = controls_for_outcome(outcome)

    # Summary shock (one at a time)
    if summary_shock in panel.columns:
        lp_res = run_local_projections(
            panel,
            outcome_col=outcome,
            shock_cols=[summary_shock],
            control_cols=controls,
            horizons=[0, 1, 2, 3],
            add_time_fe=True,
        )
        lp_tbl = record_lp_results(
            spec_id=f"A_linear_{outcome}_{summary_shock}",
            module="A",
            outcome=outcome,
            shock_terms=[summary_shock],
            results=lp_res,
            controls=controls,
        )
        lp_tbl["outcome"] = outcome
        lp_tbl["spec"] = "summary_linear"
        module_a_tables.append(lp_tbl)
        plot_irf(
            lp_tbl,
            f"Module A: IRF summary shock — {outcome}",
            FIGURES / f"moduleA_irf_linear_{outcome}.png",
        )

    # Area shocks (one at a time)
    area_irfs = []
    for shock in area_shocks:
        if shock not in panel.columns:
            continue
        lp_res = run_local_projections(
            panel,
            outcome_col=outcome,
            shock_cols=[shock],
            control_cols=controls,
            horizons=[0, 1, 2, 3],
            add_time_fe=True,
        )
        lp_tbl = record_lp_results(
            spec_id=f"A_linear_{outcome}_{shock}",
            module="A",
            outcome=outcome,
            shock_terms=[shock],
            results=lp_res,
            controls=controls,
        )
        lp_tbl["outcome"] = outcome
        lp_tbl["spec"] = "area_linear"
        module_a_tables.append(lp_tbl)
        area_irfs.append(lp_tbl)

    if area_irfs:
        area_df = pd.concat(area_irfs, ignore_index=True)
        plot_irf(
            area_df,
            f"Module A: IRF area shocks — {outcome}",
            FIGURES / f"moduleA_irf_areas_{outcome}.png",
        )

    # Asymmetry for summary shock only
    if "d_efw_summary_pos" in panel.columns and "d_efw_summary_neg" in panel.columns:
        lp_res_asym = run_local_projections(
            panel,
            outcome_col=outcome,
            shock_cols=["d_efw_summary_pos", "d_efw_summary_neg"],
            control_cols=controls,
            horizons=[0, 1, 2, 3],
            add_time_fe=True,
        )
        lp_tbl_asym = record_lp_results(
            spec_id=f"A_asym_{outcome}",
            module="A",
            outcome=outcome,
            shock_terms=["d_efw_summary_pos", "d_efw_summary_neg"],
            results=lp_res_asym,
            controls=controls,
        )
        lp_tbl_asym["outcome"] = outcome
        lp_tbl_asym["spec"] = "summary_asymmetric"
        module_a_tables.append(lp_tbl_asym)
        plot_irf(
            lp_tbl_asym,
            f"Module A: IRF (asym) — {outcome}",
            FIGURES / f"moduleA_irf_asym_{outcome}.png",
        )

module_a_all = pd.concat(module_a_tables, ignore_index=True)
module_a_all.to_csv(TABLES / "moduleA_lp_coefficients.csv", index=False)

# Component leaderboard
leaderboard = (
    module_a_all[
        module_a_all["term"].isin([c for c in shock_cols if c.startswith("d_efw_area")])
    ]
    .groupby(["outcome", "term"])
    .agg(mean_abs_t=("t_stat", lambda x: np.mean(np.abs(x))))
    .reset_index()
    .sort_values(["outcome", "mean_abs_t"], ascending=[True, False])
)
leaderboard.to_csv(TABLES / "moduleA_component_leaderboard.csv", index=False)

# %% [markdown]
# ## Module B — Complementarities (interactions)

# %%
panel["d_area2_x_area3"] = panel["d_efw_area2"] * panel["lag_efw_area3"]
panel["d_area4_x_area5"] = panel["d_efw_area4"] * panel["lag_efw_area5"]

interaction_specs = [
    {
        "name": "legal_x_money",
        "shock_cols": ["d_efw_area2", "d_area2_x_area3"],
        "state_col": "lag_efw_area3",
    },
    {
        "name": "trade_x_regulation",
        "shock_cols": ["d_efw_area4", "d_area4_x_area5"],
        "state_col": "lag_efw_area5",
    },
]

for spec in interaction_specs:
    outcome = "gdppc_growth_5y"
    controls = controls_for_outcome(outcome)
    lp_res = run_local_projections(
        panel,
        outcome_col=outcome,
        shock_cols=spec["shock_cols"],
        control_cols=controls,
        horizons=[0, 1, 2, 3],
        add_time_fe=True,
    )
    lp_tbl = record_lp_results(
        spec_id=f"B_{spec['name']}",
        module="B",
        outcome=outcome,
        shock_terms=spec["shock_cols"],
        results=lp_res,
        controls=controls,
    )
    plot_irf(
        lp_tbl,
        f"Module B: Interaction IRF — {spec['name']}",
        FIGURES / f"moduleB_irf_{spec['name']}.png",
    )

    # Marginal effect plot at low/high state
    state = panel[spec["state_col"]]
    low_state = state.quantile(0.25)
    high_state = state.quantile(0.75)

    effects = []
    for res in lp_res:
        coef_base = res.fit.params[spec["shock_cols"][0]]
        coef_int = res.fit.params[spec["shock_cols"][1]]
        cov = res.fit.cov
        var_base = cov.loc[spec["shock_cols"][0], spec["shock_cols"][0]]
        var_int = cov.loc[spec["shock_cols"][1], spec["shock_cols"][1]]
        covar = cov.loc[spec["shock_cols"][0], spec["shock_cols"][1]]
        for label, level in [("low", low_state), ("high", high_state)]:
            coef = coef_base + coef_int * level
            var = var_base + (level ** 2) * var_int + 2 * level * covar
            effects.append(
                {
                    "horizon": res.horizon,
                    "level": label,
                    "coef": coef,
                    "std_err": np.sqrt(var),
                }
            )

    effects_df = pd.DataFrame(effects)
    fig, ax = plt.subplots()
    for level, sub in effects_df.groupby("level"):
        sub = sub.sort_values("horizon")
        ax.plot(sub["horizon"], sub["coef"], marker="o", label=level)
        ax.fill_between(
            sub["horizon"],
            sub["coef"] - 1.96 * sub["std_err"],
            sub["coef"] + 1.96 * sub["std_err"],
            alpha=0.2,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Module B marginal effects — {spec['name']}")
    ax.set_xlabel("Horizon (steps of 5 years)")
    ax.set_ylabel("Effect")
    ax.legend()
    save_figure(fig, FIGURES / f"moduleB_marginal_{spec['name']}.png")
    plt.close(fig)

# %% [markdown]
# ## Module C — Reform bundles (clustering)

# %%
change_cols = [f"d_efw_area{k}" for k in range(1, 6)]
if "shock_pos_d_efw_summary_abs" in panel.columns and "shock_neg_d_efw_summary_abs" in panel.columns:
    large_reform = panel["shock_pos_d_efw_summary_abs"] | panel["shock_neg_d_efw_summary_abs"]
else:
    abs_threshold = panel["d_efw_summary"].abs().quantile(0.9)
    large_reform = panel["d_efw_summary"].abs() >= abs_threshold
panel["large_reform"] = large_reform.fillna(False)

bundle_df, centers = build_reform_bundles(panel, change_cols=change_cols, k=4)
bundle_df = bundle_df.merge(panel[["iso3", "year", "large_reform"]], on=["iso3", "year"], how="left")
bundle_large = bundle_df[bundle_df["large_reform"]].copy()

centers.to_csv(TABLES / "moduleC_bundle_centroids.csv")
bundle_large.to_csv(TABLES / "moduleC_bundle_assignments.csv", index=False)

# PCA visualization
pca = PCA(n_components=2, random_state=42)
pca_scores = pca.fit_transform(bundle_df[change_cols])
plot_df = pd.DataFrame(pca_scores, columns=["pc1", "pc2"])
plot_df["bundle"] = bundle_df["bundle"]

fig, ax = plt.subplots()
sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="bundle", palette="tab10", ax=ax)
ax.set_title("Module C: Reform bundles (PCA on area changes)")
save_figure(fig, FIGURES / "moduleC_reform_taxonomy.png")
plt.close(fig)

# LPs by bundle (large reforms)
bundle_large = bundle_large.merge(
    panel,
    on=["iso3", "year"],
    how="left",
    suffixes=("", "_panel"),
)
for k in sorted(bundle_large["bundle"].unique()):
    bundle_large[f"bundle_{k}"] = (bundle_large["bundle"] == k).astype(int)

baseline_bundle = bundle_large["bundle"].value_counts().idxmax()
bundle_cols = [f"bundle_{k}" for k in sorted(bundle_large["bundle"].unique()) if k != baseline_bundle]
bundle_controls = [c for c in ["lag_gdppc_growth_5y", "lag_gdppc_log", "lag_efw_summary"] if c in bundle_large.columns]
lp_res_bundle = run_local_projections(
    bundle_large,
    outcome_col="gdppc_growth_5y",
    shock_cols=bundle_cols,
    control_cols=bundle_controls,
    horizons=[0, 1, 2, 3],
    add_time_fe=True,
    drop_absorbed=True,
    check_rank=False,
)
lp_tbl_bundle = record_lp_results(
    spec_id="C_bundles",
    module="C",
    outcome="gdppc_growth_5y",
    shock_terms=bundle_cols,
    results=lp_res_bundle,
    controls=bundle_controls,
)
lp_tbl_bundle.to_csv(TABLES / "moduleC_bundle_irfs.csv", index=False)
plot_irf(
    lp_tbl_bundle,
    "Module C: IRFs by reform bundle",
    FIGURES / "moduleC_bundle_irfs.png",
)

# %% [markdown]
# ## Module D — Crises and resilience

# %%
event_window = [-2, -1, 0, 1, 2]
for crisis_type in ["systemic_banking", "currency", "sovereign"]:
    if crisis_type not in panel.columns:
        continue
    event_df = event_study(panel, crisis_type, "gdppc_growth_5y", event_window)
    event_df["label"] = crisis_type
    event_df.to_csv(TABLES / f"moduleD_event_study_{crisis_type}.csv", index=False)
    plot_event_study(
        event_df,
        f"Module D: GDP growth around {crisis_type} crises",
        FIGURES / f"moduleD_event_{crisis_type}.png",
    )

# Interaction LP: crisis onset x EFW
panel["crisis_x_efw"] = panel["any_crisis"] * panel["lag_efw_summary"]
controls = controls_for_outcome("gdppc_growth_5y")
lp_res_crisis = run_local_projections(
    panel,
    outcome_col="gdppc_growth_5y",
    shock_cols=["any_crisis", "crisis_x_efw"],
    control_cols=controls,
    horizons=[0, 1, 2, 3],
    add_time_fe=True,
    drop_absorbed=True,
    check_rank=False,
)
lp_tbl_crisis = record_lp_results(
    spec_id="D_crisis_interaction",
    module="D",
    outcome="gdppc_growth_5y",
    shock_terms=["any_crisis", "crisis_x_efw"],
    results=lp_res_crisis,
    controls=controls,
)
plot_irf(
    lp_tbl_crisis,
    "Module D: Crisis x EFW interaction IRF",
    FIGURES / "moduleD_crisis_interaction.png",
)

# Crisis onset logit
logit_df = panel.dropna(
    subset=[
        "any_crisis",
        "lag_efw_summary",
        "d_efw_summary",
        "lag_gdppc_log",
        "trade_gdp",
        "inflation_cpi",
        "gcf_gdp",
    ]
).copy()
logit_model = smf.logit(
    "any_crisis ~ lag_efw_summary + d_efw_summary + lag_gdppc_log + trade_gdp + inflation_cpi + gcf_gdp + C(year)",
    data=logit_df,
)
logit_res = logit_model.fit(
    disp=False,
    cov_type="cluster",
    cov_kwds={"groups": logit_df["iso3"]},
)
param_index = getattr(logit_res.params, "index", logit_model.exog_names)
logit_table = pd.DataFrame(
    {
        "term": list(param_index),
        "coef": np.asarray(logit_res.params),
        "std_err": np.asarray(logit_res.bse),
        "p_value": np.asarray(logit_res.pvalues),
    }
)
logit_table.to_csv(TABLES / "moduleD_crisis_logit.csv", index=False)

# %% [markdown]
# ## Module E — Regime stratification and measurement sensitivity

# %%
for regime, label in [(1.0, "democracy"), (0.0, "autocracy")]:
    sub = panel[panel["democracy"] == regime].copy()
    controls = controls_for_outcome("gdppc_growth_5y")
    lp_res = run_local_projections(
        sub,
        outcome_col="gdppc_growth_5y",
        shock_cols=["d_efw_summary"],
        control_cols=controls,
        horizons=[0, 1, 2, 3],
        add_time_fe=True,
    )
    lp_tbl = record_lp_results(
        spec_id=f"E_{label}",
        module="E",
        outcome="gdppc_growth_5y",
        shock_terms=["d_efw_summary"],
        results=lp_res,
        controls=controls,
    )
    plot_irf(
        lp_tbl,
        f"Module E: IRF summary shock — {label}",
        FIGURES / f"moduleE_irf_{label}.png",
    )

# Measurement error simulation (autocracies)
np.random.seed(42)
sim_results = []
sim_draws = 50
for sigma in [0.1, 0.2]:
    for draw in range(sim_draws):
        sim = panel.copy()
        mask = sim["democracy"] == 0
        noise = np.random.normal(0, sigma, size=int(mask.sum()))
        sim.loc[mask, "gdppc_log_noisy"] = sim.loc[mask, "gdppc_log"] + noise
        sim.loc[~mask, "gdppc_log_noisy"] = sim.loc[~mask, "gdppc_log"]
        sim["gdppc_growth_noisy"] = sim.groupby("iso3")["gdppc_log_noisy"].diff()
        sim["lag_gdppc_growth_noisy"] = sim.groupby("iso3")["gdppc_growth_noisy"].shift(1)
        controls = controls_for_outcome("gdppc_growth_noisy")
        lp_res = run_local_projections(
            sim,
            outcome_col="gdppc_growth_noisy",
            shock_cols=["d_efw_summary"],
            control_cols=controls,
            horizons=[0],
            add_time_fe=True,
        )
        coef = summarize_lp_results(lp_res, ["d_efw_summary"]).iloc[0]["coef"]
        sim_results.append({"sigma": sigma, "coef_h0": coef})

sim_df = pd.DataFrame(sim_results)
sim_summary = sim_df.groupby("sigma")["coef_h0"].agg(["mean", "std", "count"]).reset_index()
sim_summary.to_csv(TABLES / "moduleE_measurement_error_sim.csv", index=False)

# %% [markdown]
# ## Module F — Downside risk and quantiles

# %%
collapse_threshold = panel["gdppc_growth_5y"].quantile(0.1)
panel["growth_collapse"] = (panel["gdppc_growth_5y"] <= collapse_threshold).astype(int)

controls = controls_for_outcome("growth_collapse")
lp_res_collapse = run_local_projections(
    panel,
    outcome_col="growth_collapse",
    shock_cols=["d_efw_summary"],
    control_cols=controls,
    horizons=[0, 1, 2, 3],
    add_time_fe=True,
)
lp_tbl_collapse = record_lp_results(
    spec_id="F_collapse_lp",
    module="F",
    outcome="growth_collapse",
    shock_terms=["d_efw_summary"],
    results=lp_res_collapse,
    controls=controls,
)
plot_irf(
    lp_tbl_collapse,
    "Module F: Growth collapse probability (LP)",
    FIGURES / "moduleF_collapse_irf.png",
)

# Quantile regression (pooled with year FE)
quant_df = panel.dropna(
    subset=["gdppc_growth_5y", "d_efw_summary", "lag_gdppc_growth_5y", "lag_efw_summary"]
).copy()
quant_formula = (
    "gdppc_growth_5y ~ d_efw_summary + lag_gdppc_growth_5y + lag_efw_summary + "
    "trade_gdp + inflation_cpi + gcf_gdp + gov_consumption_gdp + C(year)"
)

quant_results = []
for q in [0.1, 0.5, 0.9]:
    model = smf.quantreg(quant_formula, quant_df)
    res = model.fit(q=q)
    quant_results.append(
        {
            "quantile": q,
            "coef": res.params.get("d_efw_summary", np.nan),
            "std_err": res.bse.get("d_efw_summary", np.nan),
            "p_value": res.pvalues.get("d_efw_summary", np.nan),
        }
    )
quant_table = pd.DataFrame(quant_results)
quant_table.to_csv(TABLES / "moduleF_quantile_regression.csv", index=False)

# %% [markdown]
# ## Spec ledger and research log

# %%
ledger = pd.DataFrame(spec_records)
ledger["q_value"] = bh_fdr(ledger["p_value"].fillna(1.0))
ledger.to_csv(OUTPUTS / "spec_ledger.csv", index=False)

top_hits = (
    ledger.sort_values("p_value")
    .groupby("module")
    .head(5)
    .loc[:, ["module", "spec_id", "outcome", "term", "horizon", "coef", "p_value", "q_value"]]
)

lines = [
    "# Research log",
    "",
    "## Top hits by module (lowest p-values)",
    top_hits.to_markdown(index=False),
    "",
    "## Notes",
    "- Spec ledger stored in outputs/spec_ledger.csv with BH/FDR q-values.",
    f"- Dataset hash: {DATA_HASH}",
]
log_path = OUTPUTS / "research_log.md"
log_path.write_text("\n".join(lines))
print("Spec ledger saved:", OUTPUTS / "spec_ledger.csv")
print("Research log saved:", log_path)
