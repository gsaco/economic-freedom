# Step 0 Orientation

## Repo tree (top 3 levels)
Command (tree unavailable): `find . -maxdepth 3 -print | rg -v '^\./\.git'`

```
.
./tools
./tools/build_report.py
./tools/run_pipeline.py
./tools/run_all.py
./.DS_Store
./requirements.txt
./output
./output/tables
./output/tables/moduleD_event_study_sovereign.csv
./output/tables/wb_coverage_by_year.csv
./output/tables/gdppc_wb_pwt_corr_by_year.csv
./output/tables/lp_coefficients.csv
./output/tables/moduleE_measurement_error_sim.csv
./output/tables/efw_shock_thresholds_global.csv
./output/tables/gdppc_wb_pwt_corr_by_regime.csv
./output/tables/high_impact_episodes.csv
./output/tables/moduleF_quantile_regression.csv
./output/tables/efw_quintile_transition.csv
./output/tables/shock_definition_frequencies.csv
./output/tables/moduleD_crisis_logit.csv
./output/tables/macro_summary_by_decade.csv
./output/tables/shock_frequency_by_group.csv
./output/tables/efw_summary_by_decade.csv
./output/tables/gdppc_wb_pwt_corr_overall.csv
./output/tables/country_clusters.csv
./output/tables/gdppc_wb_pwt_logratio_by_regime.csv
./output/tables/efw_shock_decomposition.csv
./output/tables/moduleA_lp_coefficients.csv
./output/tables/moduleC_bundle_irfs.csv
./output/tables/efw_shock_frequencies_core.csv
./output/tables/moduleD_event_study_systemic_banking.csv
./output/tables/moduleA_component_leaderboard.csv
./output/tables/master_coverage_by_year.csv
./output/tables/efw_shock_thresholds_by_year.csv
./output/tables/efw_shock_decomposition_top_area.csv
./output/tables/moduleC_bundle_centroids.csv
./output/tables/moduleC_bundle_assignments.csv
./output/tables/moduleD_event_study_currency.csv
./output/tables/event_study_growth.csv
./output/report.json
./output/report.md
./output/logs
./output/logs/run_all.json
./output/figures
./output/figures/moduleC_reform_taxonomy.png
./output/figures/lp_irf_magnitude.png
./output/figures/moduleA_irf_areas_gdppc_growth_5y.png
./output/figures/moduleB_marginal_legal_x_money.png
./output/figures/moduleD_crisis_interaction.png
./output/figures/event_study_growth.png
./output/figures/moduleA_irf_areas_pwt_tfp_growth_5y.png
./output/figures/moduleB_irf_legal_x_money.png
./output/figures/moduleA_irf_linear_pwt_tfp_growth_5y.png
./output/figures/moduleE_irf_autocracy.png
./output/figures/moduleA_irf_linear_pwt_inv_share_change.png
./output/figures/moduleA_irf_areas_inflation_cpi.png
./output/figures/moduleD_event_systemic_banking.png
./output/figures/gdppc_wb_pwt_log_ratio.png
./output/figures/moduleC_bundle_irfs.png
./output/figures/moduleA_irf_asym_gdppc_growth_5y.png
./output/figures/moduleA_irf_areas_pwt_inv_share_change.png
./output/figures/master_missingness_heatmap.png
./output/figures/gdppc_growth_distribution.png
./output/figures/moduleD_event_currency.png
./output/figures/pca_country_clusters.png
./output/figures/efw_distribution_changes.png
./output/figures/moduleB_marginal_trade_x_regulation.png
./output/figures/moduleD_event_sovereign.png
./output/figures/moduleA_irf_asym_inflation_cpi.png
./output/figures/lp_irf_state_dependence.png
./output/figures/moduleE_irf_democracy.png
./output/figures/moduleA_irf_linear_inflation_cpi.png
./output/figures/wb_missingness_heatmap.png
./output/figures/lp_irf_asymmetric.png
./output/figures/moduleF_collapse_irf.png
./output/figures/moduleA_irf_asym_pwt_tfp_growth_5y.png
./output/figures/efw_distribution_levels.png
./output/figures/moduleA_irf_linear_gdppc_growth_5y.png
./output/figures/moduleB_irf_trade_x_regulation.png
./output/figures/lp_irf_linear.png
./output/figures/moduleA_irf_asym_pwt_inv_share_change.png
./README.md
./ vri
./ vri/CV_Saco.pdf
./ vri/GabrielSaco.pdf
./ vri/Carta DW.pdf
./ vri/Presupuesto.pdf
./ vri/Presentacion.pdf
./data
./data/fraser.xlsx
./data/.DS_Store
./data/processed
./data/processed/panel_quinquennial_1970_2020.csv
./data/processed/panel_master_quinquennial_1970_2020.csv
./data/processed/efw_quinquennial.csv
./data/processed/episodes_efw_shocks.csv
./data/processed/country_crosswalk.csv
./data/raw
./data/raw/wb
./data/raw/crisis
./data/raw/pwt
./data/raw/swiid
./data/raw/regime
./outputs
./outputs/tables
./outputs/tables/moduleD_event_study_sovereign.csv
./outputs/tables/wb_coverage_by_year.csv
./outputs/tables/gdppc_wb_pwt_corr_by_year.csv
./outputs/tables/lp_coefficients.csv
./outputs/tables/moduleE_measurement_error_sim.csv
./outputs/tables/efw_shock_thresholds_global.csv
./outputs/tables/gdppc_wb_pwt_corr_by_regime.csv
./outputs/tables/high_impact_episodes.csv
./outputs/tables/moduleF_quantile_regression.csv
./outputs/tables/efw_quintile_transition.csv
./outputs/tables/shock_definition_frequencies.csv
./outputs/tables/moduleD_crisis_logit.csv
./outputs/tables/macro_summary_by_decade.csv
./outputs/tables/shock_frequency_by_group.csv
./outputs/tables/efw_summary_by_decade.csv
./outputs/tables/gdppc_wb_pwt_corr_overall.csv
./outputs/tables/country_clusters.csv
./outputs/tables/gdppc_wb_pwt_logratio_by_regime.csv
./outputs/tables/efw_shock_decomposition.csv
./outputs/tables/moduleA_lp_coefficients.csv
./outputs/tables/moduleC_bundle_irfs.csv
./outputs/tables/efw_shock_frequencies_core.csv
./outputs/tables/moduleD_event_study_systemic_banking.csv
./outputs/tables/moduleA_component_leaderboard.csv
./outputs/tables/master_coverage_by_year.csv
./outputs/tables/efw_shock_thresholds_by_year.csv
./outputs/tables/efw_shock_decomposition_top_area.csv
./outputs/tables/moduleC_bundle_centroids.csv
./outputs/tables/moduleC_bundle_assignments.csv
./outputs/tables/moduleD_event_study_currency.csv
./outputs/tables/event_study_growth.csv
./outputs/literature_sources.json
./outputs/research_log.md
./outputs/figures
./outputs/figures/moduleC_reform_taxonomy.png
./outputs/figures/lp_irf_magnitude.png
./outputs/figures/moduleA_irf_areas_gdppc_growth_5y.png
./outputs/figures/moduleB_marginal_legal_x_money.png
./outputs/figures/moduleD_crisis_interaction.png
./outputs/figures/event_study_growth.png
./outputs/figures/moduleA_irf_areas_pwt_tfp_growth_5y.png
./outputs/figures/moduleB_irf_legal_x_money.png
./outputs/figures/moduleA_irf_linear_pwt_tfp_growth_5y.png
./outputs/figures/moduleE_irf_autocracy.png
./outputs/figures/moduleA_irf_linear_pwt_inv_share_change.png
./outputs/figures/moduleA_irf_areas_inflation_cpi.png
./outputs/figures/moduleD_event_systemic_banking.png
./outputs/figures/gdppc_wb_pwt_log_ratio.png
./outputs/figures/moduleC_bundle_irfs.png
./outputs/figures/moduleA_irf_asym_gdppc_growth_5y.png
./outputs/figures/moduleA_irf_areas_pwt_inv_share_change.png
./outputs/figures/master_missingness_heatmap.png
./outputs/figures/gdppc_growth_distribution.png
./outputs/figures/moduleD_event_currency.png
./outputs/figures/pca_country_clusters.png
./outputs/figures/efw_distribution_changes.png
./outputs/figures/moduleB_marginal_trade_x_regulation.png
./outputs/figures/moduleD_event_sovereign.png
./outputs/figures/moduleA_irf_asym_inflation_cpi.png
./outputs/figures/lp_irf_state_dependence.png
./outputs/figures/moduleE_irf_democracy.png
./outputs/figures/moduleA_irf_linear_inflation_cpi.png
./outputs/figures/wb_missingness_heatmap.png
./outputs/figures/lp_irf_asymmetric.png
./outputs/figures/moduleF_collapse_irf.png
./outputs/figures/moduleA_irf_asym_pwt_tfp_growth_5y.png
./outputs/figures/efw_distribution_levels.png
./outputs/figures/moduleA_irf_linear_gdppc_growth_5y.png
./outputs/figures/moduleB_irf_trade_x_regulation.png
./outputs/figures/lp_irf_linear.png
./outputs/figures/moduleA_irf_asym_pwt_inv_share_change.png
./outputs/spec_ledger.csv
./tesis.pdf
./notebooks
./notebooks/02_components_bundles_crises.ipynb
./notebooks/01_efw_macro_quinquennial_analysis.py
./notebooks/02_components_bundles_crises.executed.ipynb
./notebooks/03_q1_candidate_evidence.executed.ipynb
./notebooks/03_q1_candidate_evidence.py
./notebooks/03_q1_candidate_evidence.ipynb
./notebooks/01_efw_macro_quinquennial_analysis.executed.ipynb
./notebooks/01_efw_macro_quinquennial_analysis.ipynb
./notebooks/02_components_bundles_crises.py
./reports
./reports/data_dictionary.md
./reports/q1_paper_pitch.md
./reports/literature_map.md
./reports/bibliography.bib
./reports/hypothesis_assessment.md
./src
./src/regime_data.py
./src/data_ingest_efw.py
./src/data_fetch_macro.py
./src/__pycache__
./src/__pycache__/viz.cpython-311.pyc
./src/__pycache__/inequality_data.cpython-311.pyc
./src/__pycache__/lp_models.cpython-311.pyc
./src/__pycache__/components.cpython-311.pyc
./src/__pycache__/crisis_data.cpython-311.pyc
./src/__pycache__/regime_data.cpython-311.pyc
./src/__pycache__/build_panel.cpython-311.pyc
./src/__pycache__/data_fetch_macro.cpython-311.pyc
./src/__pycache__/data_ingest_efw.cpython-311.pyc
./src/__pycache__/analysis_utils.cpython-311.pyc
./src/__pycache__/country_utils.cpython-311.pyc
./src/analysis_utils.py
./src/components.py
./src/viz.py
./src/inequality_data.py
./src/lp_models.py
./src/build_panel.py
./src/crisis_data.py
./src/country_utils.py
```

## Directory map
- `notebooks/`: primary pipeline and analysis notebooks (`01_...`, `02_...`, `03_...`).
- `tools/`: pipeline runners (`run_pipeline.py`, new `run_all.py`) and report builder (`build_report.py`).
- `data/`: raw downloads in `data/raw/`, processed panels in `data/processed/`.
- `outputs/`: original figures/tables/spec ledger.
- `output/`: mirror created for required inspection (`output/figures`, `output/tables`, `output/logs`, `output/report.md`).
- `reports/`: narrative drafts (will be superseded by docs/ per new requirements).
- `paper/`: not present (note for later packaging).

## Main empirical design (implemented files)
- Shock construction:
  - Summary shocks: `notebooks/01_efw_macro_quinquennial_analysis.py` (q90/q85, SD, absolute thresholds).
  - Component shocks: `notebooks/02_components_bundles_crises.py` via `src/components.py`.
- Stacked cohort / event construction:
  - Simple event-study windows (not stacked cohorts): `notebooks/01_efw_macro_quinquennial_analysis.py` and `notebooks/03_q1_candidate_evidence.py`.
  - No stacked cohort dataset or clean control definition implemented yet.
- Main estimation:
  - Local projections in `src/lp_models.py`, called in `notebooks/01_...` and `notebooks/03_...`.
- Inference / bands / placebos:
  - Country-clustered SEs only; no wild bootstrap, joint bands, or placebo timing yet.
- State dependence / nonlinearity:
  - State interactions in `notebooks/01_...` (shock x initial conditions) and `notebooks/03_...` (Module B interactions).
- Dimensions / bundles:
  - Clustering of component changes in `notebooks/03_q1_candidate_evidence.py` (Module C), with `src/components.py` bundle helper.
- SCM case studies:
  - None implemented (no synthetic control code found).

## Pipeline run (Step 0)
- Required command: `python tools/run_all.py`.
- Initial failure: missing `tools/run_all.py` (fixed by adding `tools/run_all.py` and `tools/build_report.py`).
- Successful run: generated `output/report.md`, `output/figures/`, `output/tables/`, and `output/logs/run_all.json`.

## Report summary (from `output/report.md`)
- Figures: 37
- Tables: 31
- Logs: 1
- Report: `output/report.md` lists all generated artifacts.
