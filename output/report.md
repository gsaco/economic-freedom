# Economic Freedom and Growth: Analysis Report

This document contains run reports and PhD-level analysis of the
econometric pipeline for studying EFW-growth dynamics.

---

## Run Report — 2026-01-07 13:37

---

### Executive Summary

- **Run timestamp:** 2026-01-07T13:32:57.772330
- **Total notebooks:** 11
- **Successful:** 11
- **Failed:** 0
- **Total runtime:** 4m 4s
- **Git commit:** 67a9ebc

### Pipeline Status

| # | Notebook | Status | Runtime | Exit | Artifacts |
|---|----------|--------|---------|------|-----------|
| 00 | `00_env_setup` | ✓ | 1.9s | 0 | 2 |
| 01 | `01_ingest_fraser` | ✓ | 1.1s | 0 | 1 |
| 02 | `02_pull_worldbank` | ✓ | 0.6s | 0 | 1 |
| 03 | `03_build_quinquennial_panel` | ✓ | 0.5s | 0 | 1 |
| 04 | `04_construct_shocks` | ✓ | 1.4s | 0 | 2 |
| 05 | `05_build_stacked_event_data` | ✓ | 1.3s | 0 | 2 |
| 06 | `06_estimate_lp_stacked` | ✓ | 1m 38s | 0 | 5 |
| 07 | `07_inference_bands_placebos` | ✓ | 51.2s | 0 | 7 |
| 08 | `08_state_dependence_nonlinearity` | ✓ | 51.0s | 0 | 8 |
| 09 | `09_dimensions_bundles` | ✓ | 31.0s | 0 | 6 |
| 10 | `10_scm_major_reforms` | ✓ | 6.1s | 0 | 3 |

### Event Counts (from shock construction)

| Event Type | Count (raw) | Count (cooldown) |
|------------|-------------|------------------|
| Positive Shocks (S+) | 62 | 52 |
| Negative Shocks (S-) | 23 | 20 |
| Sustained Positive (R+) | 57 | 48 |
| Sustained Negative (R-) | 7 | 6 |

### Main IRF Results

| Horizon | Positive β | SE | p-val | Negative β | SE | p-val |
|---------|-----------|-----|-------|-----------|-----|-------|
| t-2 | -0.0075 | 0.0396 | 0.850 | -0.0357 | 0.0559 | 0.523 |
| t-1 | 0.0000 | 0.0000 | nan | 0.0000 | 0.0000 | nan |
| t0 | 0.0845*** | 0.0297 | 0.004 | -0.0345 | 0.0345 | 0.318 |
| t1 | 0.2224*** | 0.0401 | 0.000 | -0.0046 | 0.0489 | 0.925 |
| t2 | 0.3515*** | 0.0457 | 0.000 | -0.0520 | 0.0799 | 0.515 |
| t3 | 0.4608*** | 0.0581 | 0.000 | -0.1208 | 0.1047 | 0.248 |
| t4 | 0.5372*** | 0.0620 | 0.000 | 0.0923 | 0.1138 | 0.418 |

*Notes: \*\*\* p<0.01, \*\* p<0.05, \* p<0.1. Standard errors clustered at country level.*

### Results Interpretation

#### Magnitude Interpretation

- **Quinquennial mapping:** Each horizon h corresponds to 5×h years from treatment
  - h=0: Treatment year (0-5 years)
  - h=1: 5-10 years post
  - h=4: 20-25 years post
- **Log points to percent:** A coefficient of 0.10 ≈ 10% higher GDP per capita
- **Cumulative interpretation:** These are Y_{t+h} - Y_{t-1}, not period-by-period growth

#### Headline Findings

**Positive reforms (liberalization):**
- Impact effect (h=0): 0.084 log points (8.4%)
- Peak effect: 0.537 log points (53.7%) at h=4
- Effects are highly significant (p<0.01) at all post-treatment horizons

**Negative reforms (deterioration):**
- Impact effect (h=0): -0.034 log points
- Peak effect: -0.121 log points at h=3
- Effects are NOT statistically significant (p>0.10 at all horizons)
- **⚠ Small sample concern:** Only ~20 negative events vs ~48 positive

#### Pre-trends Assessment

Pre-treatment coefficient at h=-2:
- Positive reforms: β = -0.0075 (p = 0.850)
- Negative reforms: β = -0.0357 (p = 0.523)

**Assessment:** Pre-trends appear parallel (coefficients small and insignificant).
However, with only 2 pre-periods (limited by quinquennial data), this test has low power.

---

## Notebook-by-Notebook Analysis

### 00_env_setup

**Purpose:** Environment verification and configuration

**Status:** ✓ Success (exit code 0, runtime 1.9s)

**Outputs Created/Modified:**
- `logs/run_00_env_setup.log` (new)
- `logs/spec_ledger.json` (modified)

**Key Checks:**
- Package versions
- Directory structure
- Data availability

### 01_ingest_fraser

**Purpose:** Load and process Fraser EFW data from Excel

**Status:** ✓ Success (exit code 0, runtime 1.1s)

**Outputs Created/Modified:**
- `logs/run_01_ingest_fraser.log` (new)

**Key Checks:**
- Data completeness
- Missing value detection
- Quinquennial filtering

<details>
<summary>Output (last 40 lines)</summary>

```

Changes (Δ):
       d_efw_aggregate  d_efw_area1  ...  d_efw_area4  d_efw_area5
count          1311.00      1408.00  ...      1133.00      1418.00
mean              0.13         0.11  ...         0.20         0.10
std               0.50         1.03  ...         1.17         0.60
min              -1.94        -5.46  ...        -9.27        -3.63
25%              -0.15        -0.36  ...        -0.26        -0.17
50%               0.09         0.03  ...         0.04         0.05
75%               0.38         0.53  ...         0.56         0.35
max               2.84         9.32  ...         6.27         3.62

[8 rows x 6 columns]

Distribution of aggregate EFW changes:
  Mean: 0.127
  Std:  0.501
  Negative: 524 (40.0%)
  Positive: 781 (59.6%)
  |Δ| ≥ 1.0: 85
  |Δ| ≥ 1.5: 18
  |Δ| ≥ 2.0: 3

✓ Saved quinquennial EFW panel to /Users/gabrielsaco/Documents/GitHub/economic-freedom/data/02_intermediate/efw_panel_raw.parquet
  Observations: 1815
  Countries: 165
  Years: [np.int64(1970), np.int64(1975), np.int64(1980), np.int64(1985), np.int64(1990), np.int64(1995), np.int64(2000), np.int64(2005), np.int64(2010), np.int64(2015), np.int64(2020)]
✓ Saved metadata to /Users/gabrielsaco/Documents/GitHub/economic-freedom/data/02_intermediate/efw_metadata.json

============================================================
FINAL VALIDATION CHECKS
============================================================
✓ No duplicate country-year keys
✓ All years are quinquennial
✓ EFW values in valid range [0, 10]
✓ Panel size: 1815 obs, 165 countries

============================================================
01_INGEST_FRASER COMPLETE
============================================================
```
</details>

### 02_pull_worldbank

**Purpose:** Download World Bank indicators via API

**Status:** ✓ Success (exit code 0, runtime 0.6s)

**Outputs Created/Modified:**
- `logs/run_02_pull_worldbank.log` (new)

**Key Checks:**
- API pagination
- Country filtering
- GDP construction

### 03_build_quinquennial_panel

**Purpose:** Merge EFW and World Bank data into analysis panel

**Status:** ✓ Success (exit code 0, runtime 0.5s)

**Outputs Created/Modified:**
- `logs/run_03_build_quinquennial_panel.log` (new)

**Key Checks:**
- Merge quality
- Lag/lead construction
- State dependence vars

<details>
<summary>Output (last 40 lines)</summary>

```
  d_efw_area3: 1342 non-missing
  d_efw_area4: 1133 non-missing
  d_efw_area5: 1418 non-missing

outcomes:
  gdp_pc_constant: 1661 non-missing
  gdp_pc_ppp: 1125 non-missing
  ln_gdp_pc: 1661 non-missing
  growth: 1497 non-missing
  population: 1804 non-missing

lagged_controls:
  ln_gdp_pc_lag1: 1498 non-missing
  growth_lag1: 1334 non-missing
  efw_aggregate_lag1: 1311 non-missing
  efw_area1_lag1: 1408 non-missing
  efw_area2_lag1: 1551 non-missing
  efw_area3_lag1: 1342 non-missing
  efw_area4_lag1: 1133 non-missing
  efw_area5_lag1: 1418 non-missing

leads_for_classification:
  efw_aggregate_lead1: 1379 non-missing
  efw_aggregate_lead2: 1272 non-missing

state_dependence:
  ln_gdp_pc_initial: 1804 non-missing
  efw_initial: 1815 non-missing
  gdp_quartile: 1498 non-missing
  efw_quartile: 1311 non-missing

groups:
  region: 1815 non-missing
  income_group: 1815 non-missing
  wb_region: 1476 non-missing
  wb_income: 1044 non-missing

============================================================
03_BUILD_QUINQUENNIAL_PANEL COMPLETE
============================================================
```
</details>

### 04_construct_shocks

**Purpose:** Define reform episodes using shock classification

**Status:** ✓ Success (exit code 0, runtime 1.4s)

**Outputs Created/Modified:**
- `logs/run_04_construct_shocks.log` (new)
- `logs/event_counts.json` (modified)

**Key Checks:**
- Shock definition (τ=1.0)
- Maintenance rule
- Cooldown enforcement

<details>
<summary>Output (last 40 lines)</summary>

```
============================================================
✓ No duplicate country-year keys
✓ All 50 shock/reform indicators are binary
✓ Cooldown properly enforced
✓ Sustained positive reforms: 48
⚠ WARNING: Only 6 sustained negative reforms (maintenance rule is strict)
  Using basic shocks (S-_cd) for negative analysis: 20
✓ Basic shocks (for robustness): S+_cd=52, S-_cd=20

✓ Saved panel with shocks to /Users/gabrielsaco/Documents/GitHub/economic-freedom/data/03_clean/panel_with_shocks.parquet
  Observations: 1815
  Countries: 165
  Total columns: 105

New shock/reform columns (62):
  d_efw_aggregate
  d_efw_area1
  d_efw_area2
  d_efw_area3
  d_efw_area4
  d_efw_area5
  d_efw_bin
  d_efw_category
  is_bundled_reform
  is_clean_reform
  n_areas_shocked
  reform_agg_any
  reform_agg_neg
  reform_agg_neg_cd
  reform_agg_neg_cd_alt
  reform_agg_pos
  reform_agg_pos_cd
  reform_area1_any
  reform_area1_neg
  reform_area1_neg_cd
  ... and 42 more

============================================================
04_CONSTRUCT_SHOCKS COMPLETE
============================================================
```
</details>

### 05_build_stacked_event_data

**Purpose:** Build stacked event-study design for LP-DiD

**Status:** ✓ Success (exit code 0, runtime 1.3s)

**Outputs Created/Modified:**
- `logs/run_05_build_stacked_event_data.log` (new)
- `logs/stacked_events_metadata.json` (modified)

**Key Checks:**
- Cohort construction
- Control selection
- Event-time indexing

<details>
<summary>Output (last 40 lines)</summary>

```
  Total observations: 63592
  Positive cohorts: 48
  Negative cohorts: 20

Summary by event time (Positive Reforms):
            n_treated  n_outcome  mean_y  std_y  n_cohorts
event_time                                                
-2                 49       5650  -0.044  0.221         48
-1                 49       6023   0.000  0.000         48
 0                 49       6023   0.049  0.237         48
 1                 49       6023   0.133  0.339         48
 2                 49       6022   0.239  0.388         48
 3                 48       5898   0.328  0.444         47
 4                 40       4954   0.411  0.498         39

Summary by event time (Negative Reforms):
            n_treated  n_outcome  mean_y  std_y  n_cohorts
event_time                                                
-2                 14       1788  -0.100  0.185         13
-1                 23       2652   0.000  0.000         20
 0                 23       2650   0.103  0.161         20
 1                 21       2356   0.194  0.270         18
 2                 21       2353   0.235  0.354         18
 3                 18       1907   0.291  0.467         15
 4                 16       1604   0.338  0.572         13

✓ Saved positive reforms to /Users/gabrielsaco/Documents/GitHub/economic-freedom/data/03_clean/stacked_events_pos.parquet
✓ Saved negative reforms to /Users/gabrielsaco/Documents/GitHub/economic-freedom/data/03_clean/stacked_events_neg.parquet
✓ Saved combined dataset to /Users/gabrielsaco/Documents/GitHub/economic-freedom/data/03_clean/stacked_events_combined.parquet
✓ Saved metadata to /Users/gabrielsaco/Documents/GitHub/economic-freedom/output/logs/stacked_events_metadata.json

============================================================
05_BUILD_STACKED_EVENT_DATA COMPLETE
============================================================
Event window: t=-2 to t=+4 (7 periods)
Control type: not_yet_treated
Positive reform cohorts: 48
Negative reform cohorts: 20
Total s
```
</details>

### 06_estimate_lp_stacked

**Purpose:** Main LP-DiD estimation with clustered SEs

**Status:** ✓ Success (exit code 0, runtime 1m 38s)

**Outputs Created/Modified:**
- `logs/run_06_estimate_lp_stacked.log` (new)
- `tables/tab_main_results.tex` (modified)
- `logs/lp_did_results_main.json` (modified)
- `figures/fig_irf_combined.pdf` (modified)
- `figures/fig_irf_signed_main.pdf` (modified)

**Key Checks:**
- Pre-trends
- Coefficient magnitudes
- Clustering

### 07_inference_bands_placebos

**Purpose:** Wild cluster bootstrap and placebo tests

**Status:** ✓ Success (exit code 0, runtime 51.2s)

**Outputs Created/Modified:**
- `logs/run_07_inference_bands_placebos.log` (new)
- `tables/tab_pretrends_placebos.tex` (modified)
- `logs/inference_results.json` (modified)
- `figures/fig_irf_pos_bootstrap.pdf` (modified)
- `figures/fig_irf_neg_bootstrap.pdf` (modified)
- `figures/fig_placebo_neg.pdf` (modified)

**Key Checks:**
- Bootstrap validity
- Joint coverage
- Pre-trend tests

### 08_state_dependence_nonlinearity

**Purpose:** Heterogeneity by initial income and EFW

**Status:** ✓ Success (exit code 0, runtime 51.0s)

**Outputs Created/Modified:**
- `logs/run_08_state_dependence_nonlinearity.log` (new)
- `logs/state_dependence_results.json` (modified)
- `figures/fig_irf_state_income_neg.pdf` (modified)
- `figures/fig_irf_state_income_pos.pdf` (modified)
- `figures/fig_irf_magnitude_neg.pdf` (modified)
- `figures/fig_irf_state_efw_neg.pdf` (modified)

**Key Checks:**
- Sample splits
- Magnitude heterogeneity

### 09_dimensions_bundles

**Purpose:** Area-specific reform analysis (5 EFW dimensions)

**Status:** ✓ Success (exit code 0, runtime 31.0s)

**Outputs Created/Modified:**
- `logs/run_09_dimensions_bundles.log` (new)
- `tables/tab_area_effects.tex` (modified)
- `logs/dimensions_bundles_results.json` (modified)
- `figures/fig_irf_areas_pos.pdf` (modified)
- `figures/fig_irf_areas_neg.pdf` (modified)
- `figures/fig_cumulative_by_area.pdf` (modified)

**Key Checks:**
- Area-specific events
- Bundle vs clean reforms

### 10_scm_major_reforms

**Purpose:** Synthetic control for major case studies

**Status:** ✓ Success (exit code 0, runtime 6.1s)

**Outputs Created/Modified:**
- `logs/run_10_scm_major_reforms.log` (new)
- `logs/scm_results.json` (modified)
- `figures/fig_scm_top_positive.pdf` (modified)

**Key Checks:**
- Donor pool balance
- Pre-treatment fit
- Placebo inference

---

## Identification Discussion

### Design: Stacked Local Projection DiD

The estimation follows Cengiz et al. (2019) and Baker et al. (2022):
1. Each reform event defines a cohort
2. Each cohort gets its own subsample with treated unit + clean controls
3. Controls are 'not-yet-treated' (reform later or never)
4. Stacking avoids TWFE contamination from heterogeneous treatment timing

### Threats to Identification

| Threat | Concern Level | Mitigation |
|--------|--------------|------------|
| **Reverse causality** | Medium | EFW changes → growth is plausible, but growth → reforms also possible |
| **Anticipation effects** | Medium | Reforms may be announced before implementation; h=-1 normalization helps |
| **Selection into treatment** | High | Countries that reform may differ systematically (crisis-driven?) |
| **Contemporaneous shocks** | Medium | Oil shocks, global crises may coincide with reforms |
| **Spillovers** | Low | Trade/investment spillovers to control countries |
| **Measurement error** | Medium | EFW constructed with judgment; may lag true policy |

### Shock Definition Sensitivity

Current specification uses τ = 1.0 EFW point threshold. Recommend testing:
- τ ∈ {0.75, 1.0, 1.25, 1.5} for robustness
- Percentile-based thresholds (P90, P95 of |ΔEFW|)
- Continuous treatment (magnitude instead of binary)

## Inference Discussion

### What Is Implemented

- **Country-clustered standard errors** via statsmodels `cov_type='cluster'`
- **Wild cluster bootstrap** (Rademacher weights) — 999 replications
- **Sup-t joint confidence bands** for simultaneous coverage across horizons
- **Pre-trend joint tests** (Wald statistic on pre-period coefficients)

### Recommendations for Publication

1. **Two-way clustering:** Consider clustering by country AND time (Petersen, 2009)
2. **More bootstrap replications:** Increase to 9999 for tight CI bounds
3. **Wild bootstrap version:** Consider Webb six-point distribution for small N
4. **Multiple testing correction:** Apply Bonferroni or Romano-Wolf for horizon-specific tests
5. **Exact cluster bootstrap:** With ~160 clusters, asymptotic approximation reasonable

### Small Sample Concern (Negative Reforms)

With only ~20 negative reform events:
- Clustered SEs may be downward biased (Cameron & Miller, 2015)
- Consider pooling + sign interactions: β+ = β_base + β_pos × Positive
- Consider Bayesian partial pooling for sign-specific effects
- Report confidence intervals rather than point estimates

## Robustness Agenda

### Priority 1 (Essential)
- [ ] Alternative threshold τ ∈ {0.75, 1.25, 1.5}
- [ ] Placebo timing test (randomly permute reform years)
- [ ] Drop crisis years (2008-2010, 2020)

### Priority 2 (Strongly Recommended)
- [ ] Two-way clustered SEs (country × time)
- [ ] State-dependent effects by income quartile
- [ ] Drop single countries with multiple events
- [ ] Balance tests on pre-treatment observables

### Priority 3 (Extension)
- [ ] Continuous treatment (EFW magnitude, not binary)
- [ ] Area-specific reforms (which dimension matters most?)
- [ ] Synthetic control for top 5 events (external validation)
- [ ] Longer event window if data permits (h=5, h=6)

## Reproducibility Manifest

### Environment

- **Python:** 3.11.14 | packaged by conda-forge | (main, Oct 22 2025, 22:56:31) [Clang 19.1.7 ]
- **Platform:** macOS-15.6.1-arm64-arm-64bit
- **Git commit:** 67a9ebc (main)

### Key Dependencies

```
pandas>=1.5
numpy>=1.20
statsmodels>=0.14
matplotlib>=3.5
pyarrow>=10.0
jupytext>=1.15
requests>=2.28
pycountry>=22.0
scipy>=1.10
tqdm>=4.65
```

### How to Reproduce

```bash
# Clone repository
git clone <repo_url>
cd economic-freedom

# Install dependencies
pip install -r requirements.txt

# Ensure fraser.xlsx is in data/01_raw/

# Run full pipeline
python tools/run_all.py

# View results
cat output/report.md
```

## Artifact Index

### Figures

- [fig_cumulative_by_area.pdf](./figures/fig_cumulative_by_area.pdf)
- [fig_irf_areas_neg.pdf](./figures/fig_irf_areas_neg.pdf)
- [fig_irf_areas_pos.pdf](./figures/fig_irf_areas_pos.pdf)
- [fig_irf_combined.pdf](./figures/fig_irf_combined.pdf)
- [fig_irf_magnitude_neg.pdf](./figures/fig_irf_magnitude_neg.pdf)
- [fig_irf_magnitude_pos.pdf](./figures/fig_irf_magnitude_pos.pdf)
- [fig_irf_neg_bootstrap.pdf](./figures/fig_irf_neg_bootstrap.pdf)
- [fig_irf_pos_bootstrap.pdf](./figures/fig_irf_pos_bootstrap.pdf)
- [fig_irf_signed_main.pdf](./figures/fig_irf_signed_main.pdf)
- [fig_irf_state_efw_neg.pdf](./figures/fig_irf_state_efw_neg.pdf)
- [fig_irf_state_efw_pos.pdf](./figures/fig_irf_state_efw_pos.pdf)
- [fig_irf_state_income_neg.pdf](./figures/fig_irf_state_income_neg.pdf)
- [fig_irf_state_income_pos.pdf](./figures/fig_irf_state_income_pos.pdf)
- [fig_placebo_neg.pdf](./figures/fig_placebo_neg.pdf)
- [fig_placebo_pos.pdf](./figures/fig_placebo_pos.pdf)
- [fig_scm_top_positive.pdf](./figures/fig_scm_top_positive.pdf)

### Tables

- [tab_area_effects.tex](./tables/tab_area_effects.tex)
- [tab_main_results.tex](./tables/tab_main_results.tex)
- [tab_pretrends_placebos.tex](./tables/tab_pretrends_placebos.tex)

### Logs

- [artifacts_00_env_setup.json](./logs/artifacts_00_env_setup.json)
- [artifacts_01_ingest_fraser.json](./logs/artifacts_01_ingest_fraser.json)
- [artifacts_02_pull_worldbank.json](./logs/artifacts_02_pull_worldbank.json)
- [artifacts_03_build_quinquennial_panel.json](./logs/artifacts_03_build_quinquennial_panel.json)
- [artifacts_04_construct_shocks.json](./logs/artifacts_04_construct_shocks.json)
- [artifacts_05_build_stacked_event_data.json](./logs/artifacts_05_build_stacked_event_data.json)
- [artifacts_06_estimate_lp_stacked.json](./logs/artifacts_06_estimate_lp_stacked.json)
- [artifacts_07_inference_bands_placebos.json](./logs/artifacts_07_inference_bands_placebos.json)
- [artifacts_08_state_dependence_nonlinearity.json](./logs/artifacts_08_state_dependence_nonlinearity.json)
- [artifacts_09_dimensions_bundles.json](./logs/artifacts_09_dimensions_bundles.json)
- [artifacts_10_scm_major_reforms.json](./logs/artifacts_10_scm_major_reforms.json)
- [dimensions_bundles_results.json](./logs/dimensions_bundles_results.json)
- [event_counts.json](./logs/event_counts.json)
- [inference_results.json](./logs/inference_results.json)
- [lp_did_results_main.json](./logs/lp_did_results_main.json)
- [run_ledger.json](./logs/run_ledger.json)
- [scm_results.json](./logs/scm_results.json)
- [spec_ledger.json](./logs/spec_ledger.json)
- [stacked_events_metadata.json](./logs/stacked_events_metadata.json)
- [state_dependence_results.json](./logs/state_dependence_results.json)

## Open Issues / TODO

### Critical
- [ ] **Small N for negative reforms:** Only ~6 sustained, ~20 basic negative shocks
- [ ] **Missing EFW data 1970-1995:** ~30-40% of countries lack coverage

### High Priority
- [ ] Implement two-way clustered SEs
- [ ] Add placebo timing tests
- [ ] Document World Bank indicator sources in detail

### Medium Priority
- [ ] Clean up notebook 07-10 execution (currently scaffolds)
- [ ] Add balance tables for treated vs control
- [ ] Expand SCM to more events

---

*Report generated by `tools/build_report.py`*
