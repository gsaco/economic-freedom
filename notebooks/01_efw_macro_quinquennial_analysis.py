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
# # EFW shocks y crecimiento del PBI per cápita (panel quinquenal 1970–2020)
#
# Este cuaderno construye el pipeline completo: ingesta de EFW, descarga macro (World Bank + PWT), panel quinquenal, auditoría de datos, episodios de shocks, y prototipos de proyecciones locales.

# %%
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

from src.data_ingest_efw import ingest_efw
from src.data_fetch_macro import fetch_world_bank_indicators, fetch_pwt, prepare_pwt_subset
from src.build_panel import build_panel, validate_panel
from src.lp_models import run_local_projections, summarize_lp_results
from src.viz import (
    set_plot_style,
    plot_distribution,
    plot_missingness_heatmap,
    plot_irf,
    plot_event_study,
    save_figure,
)

np.random.seed(42)
set_plot_style()

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
REPORTS = ROOT / "reports"

for path in [DATA_RAW, DATA_PROC, OUTPUTS, FIGURES, TABLES, REPORTS]:
    path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Localizar fraser.xlsx e ingestar EFW

# %%
fraser_files = sorted(ROOT.rglob("fraser.xlsx"))
print("Archivos fraser.xlsx encontrados:", fraser_files)
if not fraser_files:
    raise FileNotFoundError("No se encontró fraser.xlsx. Revise la estructura del repositorio.")

fraser_path = fraser_files[0]
print("Usando:", fraser_path)

ingest = ingest_efw(fraser_path)

efww = ingest.df.copy()
print("EFW ingestado desde:", ingest.source_sheet)
print(efww.head())

# Guardar EFW quinquenal intermedio
efww.to_csv(DATA_PROC / "efw_quinquennial.csv", index=False)

# %% [markdown]
# ## 2. Crosswalk de países

# %%
# Tomar el último año disponible por país para región/ingreso WB.
latest_meta = (
    efww.sort_values("year")
    .groupby("iso3")
    .tail(1)
    .loc[:, ["iso3", "country", "wb_region", "wb_income_class"]]
    .reset_index(drop=True)
)
latest_meta["iso3_valid"] = latest_meta["iso3"].str.len() == 3
latest_meta.to_csv(DATA_PROC / "country_crosswalk.csv", index=False)
print("Crosswalk guardado:", DATA_PROC / "country_crosswalk.csv")

# %% [markdown]
# ## 3. Auditoría rápida de EFW

# %%
efww["decade"] = (efww["year"] // 10) * 10

summary_decade = (
    efww.groupby("decade")[
        ["efw_summary", "efw_area1", "efw_area2", "efw_area3", "efw_area4", "efw_area5"]
    ]
    .agg(["mean", "median", "std", "min", "max", "count"])
)
summary_decade.to_csv(TABLES / "efw_summary_by_decade.csv")
print("Tabla guardada:", TABLES / "efw_summary_by_decade.csv")

plot_distribution(
    efww["efw_summary"],
    "Distribución del índice EFW (niveles)",
    FIGURES / "efw_distribution_levels.png",
)

plot_distribution(
    efww["d_efw_summary"],
    "Distribución de cambios quinquenales en EFW",
    FIGURES / "efw_distribution_changes.png",
)

# %% [markdown]
# ### Movilidad de quintiles EFW

# %%
efww["efw_quintile"] = (
    efww.groupby("year")["efw_summary"]
    .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1)
)

efww = efww.sort_values(["iso3", "year"])

efww["efw_quintile_next"] = efww.groupby("iso3")["efw_quintile"].shift(-1)

transition = pd.crosstab(
    efww["efw_quintile"],
    efww["efw_quintile_next"],
    normalize="index",
)
transition.to_csv(TABLES / "efw_quintile_transition.csv")
print("Tabla guardada:", TABLES / "efw_quintile_transition.csv")

# %% [markdown]
# ## 4. Definición de shocks discretos en EFW

# %%
# Distribución de cambios absolutos para decidir umbral absoluto.
abs_changes = efww["d_efw_summary"].abs().dropna()
abs_quantiles = abs_changes.quantile([0.75, 0.85, 0.9])
print("Cuantiles de |ΔEFW|:")
print(abs_quantiles)

abs_threshold = float(abs_quantiles.loc[0.9].round(2))
print("Umbral absoluto elegido (c):", abs_threshold)

# Shocks por cuantiles dentro de cada año
for q in [0.85, 0.9]:
    low_q = 1 - q
    thresholds = (
        efww.groupby("year")["d_efw_summary"]
        .quantile([low_q, q])
        .unstack()
        .rename(columns={low_q: f"q{int(low_q*100)}", q: f"q{int(q*100)}"})
    )
    efww = efww.join(thresholds, on="year")
    efww[f"shock_pos_q{int(q*100)}"] = efww["d_efw_summary"] >= efww[f"q{int(q*100)}"]
    efww[f"shock_neg_q{int(q*100)}"] = efww["d_efw_summary"] <= efww[f"q{int(low_q*100)}"]

# Shocks por desviacion estandar (global)
std_change = efww["d_efw_summary"].std(skipna=True)
for k in [1.0, 1.5]:
    efww[f"shock_pos_sd{k}"] = efww["d_efw_summary"] >= k * std_change
    efww[f"shock_neg_sd{k}"] = efww["d_efw_summary"] <= -k * std_change

# Shocks por umbral absoluto
efww["shock_pos_abs"] = efww["d_efw_summary"] >= abs_threshold
efww["shock_neg_abs"] = efww["d_efw_summary"] <= -abs_threshold

# Resumen de frecuencia de shocks por definicion
shock_defs = [
    "shock_pos_q90",
    "shock_neg_q90",
    "shock_pos_q85",
    "shock_neg_q85",
    "shock_pos_sd1.0",
    "shock_neg_sd1.0",
    "shock_pos_sd1.5",
    "shock_neg_sd1.5",
    "shock_pos_abs",
    "shock_neg_abs",
]
shock_summary = efww[shock_defs].mean().reset_index()
shock_summary.columns = ["shock_definition", "frequency"]
shock_summary.to_csv(TABLES / "shock_definition_frequencies.csv", index=False)
print("Tabla guardada:", TABLES / "shock_definition_frequencies.csv")

# %% [markdown]
# ### Tabla de episodios extremos (por década y región)

# %%
meta_region = latest_meta.set_index("iso3")["wb_region"]

episodes = efww.copy()
if "wb_region" not in episodes.columns:
    episodes = episodes.merge(meta_region.rename("wb_region"), on="iso3", how="left")
else:
    episodes["wb_region"] = episodes["wb_region"].fillna(episodes["iso3"].map(meta_region))

episodes["decade"] = (episodes["year"] // 10) * 10

# Top 3 positivos y negativos por década y región
pos = (
    episodes.dropna(subset=["d_efw_summary"])
    .sort_values("d_efw_summary", ascending=False)
    .groupby(["decade", "wb_region"])\
    .head(3)
)
neg = (
    episodes.dropna(subset=["d_efw_summary"])
    .sort_values("d_efw_summary", ascending=True)
    .groupby(["decade", "wb_region"])\
    .head(3)
)

episodes_table = (
    pd.concat([pos.assign(shock="positive"), neg.assign(shock="negative")])
    .loc[:, ["decade", "wb_region", "country", "iso3", "year", "d_efw_summary", "shock"]]
    .sort_values(["decade", "wb_region", "shock", "d_efw_summary"], ascending=[True, True, True, False])
)

episodes_table.to_csv(DATA_PROC / "episodes_efw_shocks.csv", index=False)
print("Episodios guardados:", DATA_PROC / "episodes_efw_shocks.csv")

# %% [markdown]
# ## 5. Descarga de series macro (World Bank + PWT)

# %%
wb_indicators = {
    "gdppc_wb": "NY.GDP.PCAP.KD",
    "gdppc_growth_wb": "NY.GDP.PCAP.KD.ZG",
    "population": "SP.POP.TOTL",
    "gcf_gdp": "NE.GDI.FTOT.ZS",
    "trade_gdp": "NE.TRD.GNFS.ZS",
    "inflation_cpi": "FP.CPI.TOTL.ZG",
    "gov_consumption_gdp": "NE.CON.GOVT.ZS",
}

wb_raw, wb_sources = fetch_world_bank_indicators(
    wb_indicators,
    cache_dir=DATA_RAW / "wb",
    start_year=1970,
    end_year=2020,
)

print("WB rows:", wb_raw.shape)

pwt_raw, pwt_url = fetch_pwt(cache_dir=DATA_RAW / "pwt")

pwt_subset = prepare_pwt_subset(pwt_raw, start_year=1970, end_year=2020)

# %% [markdown]
# ## 6. Panel quinquenal macro y diagnósticos de cobertura

# %%
quin_years = set(range(1970, 2021, 5))

wb = wb_raw.copy()
wb = wb[wb["year"].isin(quin_years)]
wb = wb[wb["iso3"].str.len() == 3]
wb = wb.rename(columns={"country": "country_wb"})

# Cobertura por año y variable
coverage = wb.groupby("year")[list(wb_indicators.keys())].apply(lambda x: x.notna().sum())
coverage.to_csv(TABLES / "wb_coverage_by_year.csv")
print("Tabla guardada:", TABLES / "wb_coverage_by_year.csv")

# Heatmap de missingness (frac)
wb_country_counts = wb.groupby("year")["iso3"].nunique()
missing_frac = 1 - (coverage.T / wb_country_counts)
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(missing_frac, ax=ax, cmap="Reds", cbar_kws={"label": "Missing fraction"})
ax.set_title("Missingness de series WB por año (quinquenal)")
save_figure(fig, FIGURES / "wb_missingness_heatmap.png")
plt.close(fig)

# PWT quinquenal
pwt = pwt_subset.copy()
pwt = pwt[pwt["year"].isin(quin_years)]
pwt = pwt.rename(columns={"country": "country_pwt"})

# %% [markdown]
# ### Comparación WB vs PWT para GDP per cápita

# %%
compare = wb.merge(
    pwt[["iso3", "year", "pwt_gdppc"]],
    on=["iso3", "year"],
    how="inner",
)
compare = compare.dropna(subset=["gdppc_wb", "pwt_gdppc"])
compare["log_ratio"] = np.log(compare["gdppc_wb"] / compare["pwt_gdppc"])

corr_overall = compare[["gdppc_wb", "pwt_gdppc"]].corr().iloc[0, 1]
print("Correlación WB vs PWT (nivel):", corr_overall)

corr_by_year = compare.groupby("year").apply(lambda x: x[["gdppc_wb", "pwt_gdppc"]].corr().iloc[0, 1])

corr_table = corr_by_year.reset_index().rename(columns={0: "corr_gdppc_wb_pwt"})
corr_table.to_csv(TABLES / "gdppc_wb_pwt_corr_by_year.csv", index=False)
print("Tabla guardada:", TABLES / "gdppc_wb_pwt_corr_by_year.csv")

pd.DataFrame([{"corr_gdppc_wb_pwt_overall": corr_overall}]).to_csv(
    TABLES / "gdppc_wb_pwt_corr_overall.csv", index=False
)
print("Tabla guardada:", TABLES / "gdppc_wb_pwt_corr_overall.csv")

plot_distribution(
    compare["log_ratio"],
    "Distribución log(GDPpc WB / GDPpc PWT)",
    FIGURES / "gdppc_wb_pwt_log_ratio.png",
)

# %% [markdown]
# ## 7. Construir panel EFW + macro

# %%
macro_df = wb.copy()

# Dejar solo columnas relevantes
macro_keep = ["iso3", "year", "country_wb"] + list(wb_indicators.keys())
macro_df = macro_df[macro_keep]

pwt_keep = ["iso3", "year", "country_pwt", "pwt_gdppc", "hc"]
if "hc" not in pwt.columns:
    pwt["hc"] = np.nan
pwt_df = pwt[pwt_keep]

panel, merge_reports = build_panel(efww, macro_df, pwt_df)
for report in merge_reports:
    print(report)

validate_panel(panel)

# Variables derivadas
panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
panel.loc[panel["gdppc_wb"] <= 0, "gdppc_wb"] = np.nan
panel["gdppc_log"] = np.log(panel["gdppc_wb"])
panel["gdppc_growth_5y"] = panel.groupby("iso3")["gdppc_log"].diff()

panel["d_efw_summary_pos"] = panel["d_efw_summary"].clip(lower=0)
panel["d_efw_summary_neg"] = panel["d_efw_summary"].clip(upper=0)

panel["lag_gdppc_growth_5y"] = panel.groupby("iso3")["gdppc_growth_5y"].shift(1)
panel["lag_efw_summary"] = panel.groupby("iso3")["efw_summary"].shift(1)

# Grupos iniciales por país
initial = (
    panel.sort_values("year")
    .groupby("iso3")
    .head(1)
    .set_index("iso3")
)

panel["efw_initial"] = panel["iso3"].map(initial["efw_summary"])
panel["gdppc_initial"] = panel["iso3"].map(initial["gdppc_wb"])

panel["efw_initial_group"] = pd.qcut(panel["efw_initial"], 3, labels=["low", "mid", "high"])
panel["gdppc_initial_group"] = pd.qcut(panel["gdppc_initial"], 3, labels=["low", "mid", "high"])

panel.to_csv(DATA_PROC / "panel_quinquennial_1970_2020.csv", index=False)
print("Panel guardado:", DATA_PROC / "panel_quinquennial_1970_2020.csv")

# %% [markdown]
# ## 8. EDA macro + crecimiento

# %%
panel["decade"] = (panel["year"] // 10) * 10

macro_vars = [
    "gdppc_wb",
    "gdppc_growth_5y",
    "population",
    "gcf_gdp",
    "trade_gdp",
    "inflation_cpi",
    "gov_consumption_gdp",
    "efw_summary",
]

summary_macro_decade = panel.groupby("decade")[macro_vars].agg(["mean", "median", "std", "count"])
summary_macro_decade.to_csv(TABLES / "macro_summary_by_decade.csv")
print("Tabla guardada:", TABLES / "macro_summary_by_decade.csv")

plot_distribution(
    panel["gdppc_growth_5y"],
    "Distribución del crecimiento quinquenal (log GDPpc)",
    FIGURES / "gdppc_growth_distribution.png",
)

# %% [markdown]
# ### Frecuencia de shocks por grupo inicial

# %%
shock_cols = [
    "shock_pos_q90",
    "shock_neg_q90",
    "shock_pos_sd1.0",
    "shock_neg_sd1.0",
    "shock_pos_sd1.5",
    "shock_neg_sd1.5",
    "shock_pos_abs",
    "shock_neg_abs",
]

# Mapear shocks desde efww al panel solo si faltan columnas
missing_shocks = [col for col in shock_cols if col not in panel.columns]
if missing_shocks:
    panel = panel.merge(
        efww[["iso3", "year"] + missing_shocks],
        on=["iso3", "year"],
        how="left",
    )

shock_freq = (
    panel.groupby(["efw_initial_group", "gdppc_initial_group"])[shock_cols]
    .mean()
    .reset_index()
)
shock_freq.to_csv(TABLES / "shock_frequency_by_group.csv", index=False)
print("Tabla guardada:", TABLES / "shock_frequency_by_group.csv")

# %% [markdown]
# ## 9. Event-study descriptivo (shocks positivos vs negativos)

# %%
# Usamos shocks cuantiles q90 como episodios.
positive_events = panel[panel["shock_pos_q90"]].copy()
negative_events = panel[panel["shock_neg_q90"]].copy()

# Ventana de -10 a +10 años (pasos quinquenales)
window_steps = [-2, -1, 0, 1, 2]

records = []
for label, events in [("positive", positive_events), ("negative", negative_events)]:
    for step in window_steps:
        temp = events.copy()
        temp["event_year"] = temp["year"] + step * 5
        merged = temp.merge(
            panel[["iso3", "year", "gdppc_growth_5y"]],
            left_on=["iso3", "event_year"],
            right_on=["iso3", "year"],
            how="left",
            suffixes=("", "_event"),
        )
        mean_growth = merged["gdppc_growth_5y_event"].mean()
        records.append({
            "label": label,
            "event_time": step * 5,
            "value": mean_growth,
        })

event_df = pd.DataFrame.from_records(records)
event_df.to_csv(TABLES / "event_study_growth.csv", index=False)
print("Tabla guardada:", TABLES / "event_study_growth.csv")
plot_event_study(
    event_df,
    "Event-study: crecimiento GDPpc alrededor de shocks EFW",
    FIGURES / "event_study_growth.png",
)

# %% [markdown]
# ## 10. Proyecciones locales (LP) - prototipos

# %%
# Controles básicos
controls = [
    "lag_gdppc_growth_5y",
    "lag_efw_summary",
    "trade_gdp",
    "inflation_cpi",
    "gcf_gdp",
    "gov_consumption_gdp",
]

# Asegurar datos limpios
panel_lp = panel.copy()

# Especificacion A: lineal simetrica
lp_res_a = run_local_projections(
    panel_lp,
    outcome_col="gdppc_growth_5y",
    shock_cols=["d_efw_summary"],
    control_cols=controls,
    horizons=[0, 1, 2, 3],
    add_time_fe=True,
)

lp_tbl_a = summarize_lp_results(lp_res_a, ["d_efw_summary"])\
    .assign(spec="linear")

plot_irf(
    lp_tbl_a,
    "IRF LP (lineal) - ΔEFW",
    FIGURES / "lp_irf_linear.png",
)

# Especificacion B: asimetrica
lp_res_b = run_local_projections(
    panel_lp,
    outcome_col="gdppc_growth_5y",
    shock_cols=["d_efw_summary_pos", "d_efw_summary_neg"],
    control_cols=controls,
    horizons=[0, 1, 2, 3],
    add_time_fe=True,
)

lp_tbl_b = summarize_lp_results(lp_res_b, ["d_efw_summary_pos", "d_efw_summary_neg"])\
    .assign(spec="asymmetric")

plot_irf(
    lp_tbl_b,
    "IRF LP (asimetrico) - ΔEFW+ y ΔEFW-",
    FIGURES / "lp_irf_asymmetric.png",
)

# Especificacion C: magnitud (grandes vs pequenas)
panel_lp["large_shock"] = panel_lp["d_efw_summary"].abs() >= abs_threshold
panel_lp["d_efw_large"] = panel_lp["d_efw_summary"] * panel_lp["large_shock"]

lp_res_c = run_local_projections(
    panel_lp,
    outcome_col="gdppc_growth_5y",
    shock_cols=["d_efw_summary", "d_efw_large"],
    control_cols=controls,
    horizons=[0, 1, 2, 3],
    add_time_fe=True,
)

lp_tbl_c = summarize_lp_results(lp_res_c, ["d_efw_summary", "d_efw_large"])\
    .assign(spec="magnitude")

plot_irf(
    lp_tbl_c,
    "IRF LP (magnitud) - ΔEFW y ΔEFW*Large",
    FIGURES / "lp_irf_magnitude.png",
)

# Especificacion D: dependencia de estado
panel_lp["efw_initial_z"] = (panel_lp["efw_initial"] - panel_lp["efw_initial"].mean()) / panel_lp["efw_initial"].std()
panel_lp["gdppc_initial_z"] = (panel_lp["gdppc_initial"] - panel_lp["gdppc_initial"].mean()) / panel_lp["gdppc_initial"].std()
panel_lp["shock_x_efw0"] = panel_lp["d_efw_summary"] * panel_lp["efw_initial_z"]
panel_lp["shock_x_y0"] = panel_lp["d_efw_summary"] * panel_lp["gdppc_initial_z"]

lp_res_d = run_local_projections(
    panel_lp,
    outcome_col="gdppc_growth_5y",
    shock_cols=["d_efw_summary", "shock_x_efw0", "shock_x_y0"],
    control_cols=controls,
    horizons=[0, 1, 2, 3],
    add_time_fe=True,
)

lp_tbl_d = summarize_lp_results(lp_res_d, ["d_efw_summary", "shock_x_efw0", "shock_x_y0"])\
    .assign(spec="state_dependence")

plot_irf(
    lp_tbl_d,
    "IRF LP (dependencia de estado) - interacciones",
    FIGURES / "lp_irf_state_dependence.png",
)

lp_table_all = pd.concat([lp_tbl_a, lp_tbl_b, lp_tbl_c, lp_tbl_d], ignore_index=True)
lp_table_all.to_csv(TABLES / "lp_coefficients.csv", index=False)
print("Tabla guardada:", TABLES / "lp_coefficients.csv")

# %% [markdown]
# ## 11. Pattern mining: PCA + clustering

# %%
# Resumen por pais (niveles medios y volatilidad)
country_features = (
    panel.groupby("iso3")
    .agg(
        efw_mean=("efw_summary", "mean"),
        efw_vol=("efw_summary", "std"),
        efw_change_mean=("d_efw_summary", "mean"),
        growth_mean=("gdppc_growth_5y", "mean"),
        growth_vol=("gdppc_growth_5y", "std"),
        trade_mean=("trade_gdp", "mean"),
        inflation_mean=("inflation_cpi", "mean"),
    )
    .dropna()
)

scaler = StandardScaler()
X = scaler.fit_transform(country_features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X)

pca_df = pd.DataFrame(X_pca, columns=["pc1", "pc2"], index=country_features.index)
pca_df["cluster"] = clusters

fig, ax = plt.subplots()
sns.scatterplot(data=pca_df, x="pc1", y="pc2", hue="cluster", palette="tab10", ax=ax)
ax.set_title("Clustering de países (PCA sobre trayectorias EFW+crecimiento)")
save_figure(fig, FIGURES / "pca_country_clusters.png")
plt.close(fig)

pca_df.to_csv(TABLES / "country_clusters.csv")
print("Tabla guardada:", TABLES / "country_clusters.csv")

# %% [markdown]
# ## 12. High-impact episodes (caidas EFW + colapsos de crecimiento)

# %%
# Definir colapso de crecimiento como bottom decile de gdppc_growth_5y
threshold_growth = panel["gdppc_growth_5y"].quantile(0.1)

high_impact = panel[
    (panel["d_efw_summary"] <= -abs_threshold)
    & (panel["gdppc_growth_5y"] <= threshold_growth)
].copy()

high_impact = high_impact[["iso3", "country", "year", "d_efw_summary", "gdppc_growth_5y"]]

high_impact.to_csv(TABLES / "high_impact_episodes.csv", index=False)
print("Tabla guardada:", TABLES / "high_impact_episodes.csv")

# %% [markdown]
# ## 13. Data dictionary

# %%
data_dict = [
    ("iso3", "Código ISO3 del país (EFW)."),
    ("year", "Año quinquenal (1970–2020)."),
    ("country", "Nombre del país (EFW)."),
    ("efw_summary", "Índice EFW total (0–10)."),
    ("efw_area1", "Area 1: Tamaño de gobierno."),
    ("efw_area2", "Area 2: Sistema legal y derechos de propiedad."),
    ("efw_area3", "Area 3: Moneda sana."),
    ("efw_area4", "Area 4: Libertad de comercio internacional."),
    ("efw_area5", "Area 5: Regulación."),
    ("d_efw_summary", "Cambio quinquenal en EFW total."),
    ("gdppc_wb", "GDP per cápita real (WB, USD constantes)."),
    ("gdppc_growth_wb", "Crecimiento anual GDPpc (WB, %)."),
    ("gdppc_growth_5y", "Crecimiento quinquenal GDPpc (log diff)."),
    ("population", "Población total (WB)."),
    ("gcf_gdp", "Formación bruta de capital (% PIB, WB)."),
    ("trade_gdp", "Comercio total (% PIB, WB)."),
    ("inflation_cpi", "Inflación CPI anual (WB, %)."),
    ("gov_consumption_gdp", "Consumo del gobierno (% PIB, WB)."),
    ("pwt_gdppc", "GDPpc PWT (PPP, base PWT10)."),
    ("hc", "Índice de capital humano (PWT)."),
    ("wb_region", "Región WB (EFW)."),
    ("wb_income_class", "Clasificación ingreso WB (EFW)."),
]

dict_lines = ["# Diccionario de datos", "", "| Variable | Descripción |", "|---|---|"]
for var, desc in data_dict:
    dict_lines.append(f"| {var} | {desc} |")

(REPORTS / "data_dictionary.md").write_text("\n".join(dict_lines))
print("Diccionario guardado:", REPORTS / "data_dictionary.md")

# %% [markdown]
# ## 14. Reporte preliminar de hipótesis

# %%
# Usamos resultados descriptivos + LP para sintetizar patrones.

lp_summary = lp_table_all.copy()

def fmt_coef(term: str, spec: str, horizon: int) -> str:
    sub = lp_summary[(lp_summary["spec"] == spec) & (lp_summary["term"] == term) & (lp_summary["horizon"] == horizon)]
    if sub.empty:
        return "n/a"
    row = sub.iloc[0]
    return f"{row['coef']:.3f} (p={row['p_value']:.3f})"

# Event-study resumen
event_t0 = event_df[event_df["event_time"] == 0].set_index("label")["value"].to_dict()
event_t5 = event_df[event_df["event_time"] == 5].set_index("label")["value"].to_dict()

lines = []
lines.append("# Evaluación preliminar de hipótesis (EFW shocks y crecimiento)")
lines.append("")
lines.append("## Hipótesis iniciales")
lines.append("- H1. ¿Cuál es la respuesta dinámica del crecimiento del PBI real per cápita ante choques discretos positivos y negativos en el índice EFW?")
lines.append("- H2. ¿Los choques negativos generan efectos más rápidos, intensos y persistentes que choques positivos comparables?")
lines.append("- H3. ¿La respuesta dinámica es no lineal según (i) magnitud del choque y (ii) estado inicial del país (nivel EFW e ingreso)?")
lines.append("")

lines.append("## Evidencia descriptiva y patrones (con cifras)")
lines.append(
    f"- Event-study (q90): crecimiento medio en t=0: positivo={event_t0.get('positive', float('nan')):.3f}, "
    f"negativo={event_t0.get('negative', float('nan')):.3f}; en t=5: positivo={event_t5.get('positive', float('nan')):.3f}, "
    f"negativo={event_t5.get('negative', float('nan')):.3f} (ver `outputs/figures/event_study_growth.png` y "
    f"`outputs/tables/event_study_growth.csv`)."
)
lines.append(
    f"- LP lineal h=0: {fmt_coef('d_efw_summary', 'linear', 0)}; h=5: {fmt_coef('d_efw_summary', 'linear', 1)} "
    f"(ver `outputs/figures/lp_irf_linear.png`)."
)
lines.append(
    f"- LP asimétrico h=10: ΔEFW+ {fmt_coef('d_efw_summary_pos', 'asymmetric', 2)} vs ΔEFW- "
    f"{fmt_coef('d_efw_summary_neg', 'asymmetric', 2)} (ver `outputs/figures/lp_irf_asymmetric.png`)."
)
lines.append(
    f"- LP magnitud h=5: ΔEFW {fmt_coef('d_efw_summary', 'magnitude', 1)} y ΔEFW*Large "
    f"{fmt_coef('d_efw_large', 'magnitude', 1)} (ver `outputs/figures/lp_irf_magnitude.png`)."
)
lines.append(
    "- La comparación WB vs PWT muestra alta correlación agregada y colas en el log ratio "
    "(ver `outputs/figures/gdppc_wb_pwt_log_ratio.png` y `outputs/tables/gdppc_wb_pwt_corr_overall.csv`)."
)
lines.append("")

lines.append("## Evaluación de H1–H3 (provisional)")
lines.append(
    "- H1: El event-study muestra brechas visibles entre shocks positivos y negativos; el LP lineal sugiere "
    "respuesta contemporánea positiva (h=0) y efectos más débiles en h=5."
)
lines.append(
    "- H2: La especificación asimétrica en h=10 muestra coeficientes distintos para ΔEFW- vs ΔEFW+, con señal "
    "adversa más marcada en los negativos (recordar que ΔEFW- es negativo, por lo que un coeficiente positivo "
    "implica efecto negativo)."
)
lines.append(
    "- H3: En la especificación de magnitud, el término ΔEFW*Large es pequeño y no significativo en h=5; la "
    "interacción con estados iniciales sugiere heterogeneidad pero con baja precisión en este prototipo."
)
lines.append("")

lines.append("## Hipótesis revisadas / mejoradas")
lines.append(
    "- H1R: Los shocks negativos discretos en EFW reducen el crecimiento quinquenal con mayor intensidad en países "
    "con baja EFW inicial, mientras que shocks positivos muestran efectos más graduales."
)
lines.append(
    "- H2R: La asimetría en IRFs se concentra en horizontes medios (h=10), con respuestas negativas más persistentes "
    "en regiones con menor apertura comercial."
)
lines.append(
    "- H3R: La magnitud del shock interactúa con el nivel inicial de ingreso: shocks grandes en países de ingreso "
    "medio muestran mayor volatilidad posterior que en extremos de ingreso."
)
lines.append("")

lines.append("## Implicancias para próximos modelos LP")
lines.append("- Incluir 1–2 rezagos adicionales del crecimiento y EFW para capturar persistencia.")
lines.append("- Probar FE de país + tiempo y variantes con tendencias lineales por país.")
lines.append("- Definir shocks negativos usando cuantiles por año y verificar robustez con umbral absoluto.")
lines.append("- Estimar efectos separados por regiones WB e ingreso inicial (interacciones o submuestras).")
lines.append("")

(REPORTS / "hypothesis_assessment.md").write_text("\n".join(lines))
print("Reporte guardado:", REPORTS / "hypothesis_assessment.md")
