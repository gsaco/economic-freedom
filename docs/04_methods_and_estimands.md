# Methods and Estimands (plan.tex → implementation)

This document maps the estimands and identification strategy in `plan.tex` to concrete code artifacts.

## Part 1: RD reduced‑form and first stage
- **Running variable (primary):** market‑bloc vote share minus 0.5, built from CLEA votes and V‑Party ideology.
  - Implementation: `notebooks/11_construct_close_elections_rd_sample.py` (CLEA branch), using `src/clea.py` + `src/ideology.py`.
  - Market bloc: parties with V‑Party left‑right (`v2pariglef_mean`/`v2pariglef`) > 0.
- **Fallback running variable:** signed top‑two vote margin when bloc vote shares are unavailable.
- **Treatment:** `D = 1` if more‑market bloc wins, `D = 0` if less‑market bloc wins.
  - Incumbency orientation uses DPI (`execrlc`/`gov1rlc`) when party‑level data are missing.
- **EFW first stage:**
  - $\Delta \mathrm{EFW}_{ce}(h) = \mathrm{EFW}_{c,t+h} - \mathrm{EFW}_{c,t-1}$
  - Implemented via `src/shocks.build_event_panel` with `efw_path_h{h}`.
- **EFW components:** `efw_area1..efw_area5` paths computed as `{area}_path_h{h}`.
- **EFW excluding Sound Money:** average of areas 1,2,4,5 computed as `efw_ex_sound_path_h{h}`.
- **Macro outcomes:**
  - $\Delta Y_{ce}(h) = Y_{c,t+h} - Y_{c,t-1}$
  - `log_gdp_cum_h{h}`, `inv_share_avg_h{h}`, `inflation_path_h{h}`.
- **Tail outcomes:** `worst_growth_h{h}`, `infl_spike_{20,40}_h{h}`, `max_drawdown_h{h}`, `crisis_start_h{h}`.

## RD specification (primary)
- **Estimator:** robust bias‑corrected local polynomial RD (CCT) via `rdrobust`.
- **Kernel:** triangular; **bandwidth:** MSE‑optimal (`mserd`) unless specified.
- **Inference:** robust standard errors, clustered by country where available.

## Part 2: RD‑IV (EFW → macro)
- **Wald ratio:**
  - $\hat{\beta}^{IV}(h) = \hat{\tau}_{Y}(h) / \hat{\tau}_{EFW}(h)$
  - Implemented in `notebooks/22_rd_iv_main_results.py` using `rd_first_stage.csv` and `rd_reduced_form.csv`.
- **2SLS within bandwidth:** implemented in `src/lpiv.py` for interaction models and robustness.

## LP fallback (if IV weak)
- **Local projection (LP) reduced‑form:** dynamic winner effects on macro outcomes without IV interpretation.
- Implemented in `notebooks/22b_lp_reduced_form.py` (added as fallback per mission requirement).

## Exclusion sensitivity (required)
- **Direct‑effect bounds:** `notebooks/22_exclusion_sensitivity_bounds.py` computes $\beta(h)$ over a grid of $\delta(h)$ calibrated from pre‑trend RD estimates.

## Module A: Asymmetry
- **Positive vs negative EFW shocks:** sample split by pre‑election incumbency (pre‑treatment) and IV IRFs.
- Outputs: `output/paper_tables/asymmetry_tests.csv` and `output/paper_figures/asymmetry/*`.

## Module B: State dependence
- **Predetermined regimes:** baseline EFW and income splits used for IV heterogeneity.
- Outputs: `output/paper_tables/nonlinear_effects.csv` and `output/paper_figures/nonlinear/*`.

## Module C: Nonlinearities
- **Magnitude splits:** large vs small shifts.
- **Tail outcomes:** IV effects on downside risks.

## Claims discipline
- **Tier 1:** RD winner effects on EFW and macro outcomes.
- **Tier 2:** EFW → macro causal interpretation with explicit exclusion assumptions and bounds.
- **Tier 3:** heterogeneity/nonlinearity with multiplicity discipline (Holm) in robustness reporting.

## Overlap safeguards
- Inflation outcomes are paired with EFW excluding Sound Money.
- Component‑wise results reported for transparency.

## Threats and failure modes (plan‑aligned)
- **Exclusion/direct effects:** addressed with exclusion‑bounds table (`rd_exclusion_bounds*`).
- **Weak first stage:** first‑stage tables + IV estimates reported only where `tau_EFW` non‑zero; LP fallback reported if IV is infeasible.
- **Measurement error (EFW/ideology):** robustness across area components; orientation pipeline documented.
- **Sign‑specific instruments:** asymmetry results reported with bootstrap diagnostics; weak‑IV sensitivity noted.
- **External validity:** CLEA global sample is primary; ParlGov reported as Tier‑1 high‑precision fallback.
