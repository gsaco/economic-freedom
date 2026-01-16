# Inference, placebos, and joint bands (stacked event studies)

## Overview
These procedures apply to the stacked event-study design implemented in:
- `notebooks/04_construct_shocks.py`
- `notebooks/05_build_stacked_event_data.py`
- `notebooks/06_estimate_lp_stacked.py`
- `notebooks/07_inference_bands_placebos.py`

Outputs are written to `outputs/` (tables/figures) and `output/logs/` (configs).

## Baseline standard errors
- Baseline event-study regressions use `PanelOLS` with two-way FE (`entity_id` and `year`) and clustered SE by entity (stack-country pair).
- Implemented in `notebooks/06_estimate_lp_stacked.py`.

## Pretrend tests (joint lead test)
- **Procedure:** Wald test of joint null that all lead coefficients (event_time < 0) are zero. To preserve lead variation, pretrend regressions use contemporaneous controls only (`trade_gdp`, `inflation_cpi`, `gcf_gdp`, `gov_consumption_gdp`), excluding lagged controls that eliminate the k = -2 observations.
- **Test statistic:** \(\beta' V^{-1} \beta\) with \(\chi^2\) reference distribution, df = #leads.
- **Output:**
  - `outputs/tables/stacked_event_pretrend_tests.csv` (lead coefficients + joint p-value)
  - `outputs/figures/stacked_event_pretrend_plot.png` (lead coefficients with 95% CI)

## Joint horizon bands (sup-t)
- **Procedure:** Draw 5,000 samples from \(\mathcal{N}(0, V)\), where \(V\) is the clustered covariance of event-time coefficients. Compute the sup \(|t|\) across horizons to obtain a 95% critical value for joint bands.
- **Output:**
  - `outputs/tables/stacked_event_joint_bands.csv`
  - `outputs/figures/stacked_event_joint_bands.png`

## Few-treated robustness: wild cluster bootstrap
- **Target estimand:** post-period average effect (`post = treated × 1[event_time >= 0]`) for `gdppc_growth_5y`.
- **Procedure:**
  - Two-way demeaning removes entity and time FE.
  - Wild cluster bootstrap with Rademacher weights at the country level (`iso3`).
  - Null-imposing bootstrap: residuals from the restricted model (controls only) are resampled.
  - Reps: 999.
- **Output:** `outputs/tables/stacked_event_wildboot.csv`.

## Placebo timing (random shifts within window)
- **Target estimand:** post-period average effect (`post`) for `gdppc_growth_5y`.
- **Procedure:**
  - For each stacked event, draw a random shift \(s\in[-2,3]\) and redefine the post indicator as `1[event_time + s >= 0]`.
  - Two-way demeaning applied; estimate coefficient on the placebo post indicator.
  - Reps: 200.
- **Output:** `outputs/tables/stacked_event_placebo_timing.csv`.

## Leave-one-event-out influence checks
- **Procedure:** Re-estimate the post-effect model dropping one stacked event (stack_id) at a time.
- **Output:** `outputs/tables/stacked_event_leave_one_out.csv`.

## Balance diagnostics (treated vs controls)
- **Procedure:** Compare treated vs control means at `event_time = -1` for core covariates; report standardized differences.
- **Output:** `outputs/tables/stacked_event_balance_pre.csv`.

## Two-way clustering check
- **Procedure:** Re-estimate post-effect models with two-way clustering by country (`iso3`) and calendar year.
- **Output:** `outputs/tables/stacked_event_two_way_cluster.csv`.

## Sample sensitivity (early-year coverage)
- **Procedure:** Re-estimate post-effect models for later subsamples (year >= 1985, year >= 1995).
- **Output:** `outputs/tables/stacked_event_sample_sensitivity.csv`.

## Config log
- `output/logs/stacked_event_inference_config.json` (bootstrap reps, placebo reps, seeds, joint band draws).
