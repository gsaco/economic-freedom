# Flagship specs: Asymmetric reforms and downside risk (stacked LP-DiD)

## A) Main stacked event-study equation
**Model (stacked event-study / LP-DiD):**

\[
Y_{i,e,t} = \sum_{k \in \mathcal{K}, k \neq -1} \beta_k \cdot 1[\text{event\_time}=k] \cdot \text{treated}_{i,e} + \gamma' X_{i,e,t-1} + \alpha_{i,e} + \lambda_t + \varepsilon_{i,e,t}
\]

- **Outcome definitions (quinquennial grid):**
  - `gdppc_growth_5y`: 5-year log GDPpc growth.
  - `growth_collapse`: indicator = 1 if `gdppc_growth_5y` <= 10th percentile of the pooled sample.
- **Event-time indicators:** `event_time = (year - event_year) / 5` (integer). Omitted period is `k = -1` (one quinquennial period pre-event).
- **Fixed effects:**
  - `alpha_{i,e}` = entity FE for stacked unit `entity_id = stack_id:iso3`.
  - `lambda_t` = calendar year FE.
- **Controls (lagged, t-1):** `lag_gdppc_growth_5y`, `lag_gdppc_log`, `lag_efw_summary`, plus macro controls if available (`trade_gdp`, `inflation_cpi`, `gcf_gdp`, `gov_consumption_gdp`).
- **Control group (clean controls):** not-yet-treated countries with **no** EFW event in the event window (exclude any country with events in the window). Implemented in `src/stacked_event.py` and `notebooks/05_build_stacked_event_data.py` with `exclude_any_event=True`.
- **Event window (quinquennial):** `K = 2` pre-periods, `L = 3` post-periods.
  - Mapping: `K=2` => 10 years pre; `L=3` => 15 years post.

## B) Treatment definitions
- **Change variable:** `d_efw_summary = efw_summary(t) - efw_summary(t-5)`.
- **Positive reform event (pos):** `d_efw_summary >= tau_pos`.
- **Negative reform event (neg):** `d_efw_summary <= tau_neg`.
- **Thresholds (baseline):** `tau_pos` = 0.90 quantile; `tau_neg` = 0.10 quantile, computed on pooled `d_efw_summary`.
- **Sustained rule (epsilon):** `epsilon = 0.2` at horizon `t+5`.
  - Positive event must satisfy `efw_summary_{t+5} >= efw_summary_t - epsilon`.
  - Negative event must satisfy `efw_summary_{t+5} <= efw_summary_t + epsilon`.
- **Cooldown:** minimum 10 years between events of the same type in the same country.
- **Small-N fallback:** if either pos or neg events < 20, relax thresholds to `tau_pos = 0.85`, `tau_neg = 0.15` (logged in `output/logs/stacked_event_config.json`).

## C) Required figures/tables (pipeline outputs)
**Tables (definitions and diagnostics):**
- `outputs/tables/stacked_event_thresholds.csv` (quantiles, thresholds, epsilon, window, fallback flag).
- `outputs/tables/stacked_event_counts.csv` (pos/neg event counts).
- `outputs/tables/stacked_event_panel_summary.csv` (stacked panel sizes and coverage).

**Tables (IRF estimates):**
- `outputs/tables/stacked_event_irf_pos_gdppc_growth_5y.csv`
- `outputs/tables/stacked_event_irf_neg_gdppc_growth_5y.csv`
- `outputs/tables/stacked_event_irf_pos_growth_collapse.csv`
- `outputs/tables/stacked_event_irf_neg_growth_collapse.csv`

**Figures (IRFs):**
- `outputs/figures/stacked_event_irf_pos_gdppc_growth_5y.png`
- `outputs/figures/stacked_event_irf_neg_gdppc_growth_5y.png`
- `outputs/figures/stacked_event_irf_pos_growth_collapse.png`
- `outputs/figures/stacked_event_irf_neg_growth_collapse.png`

**Data artifacts:**
- `data/processed/efw_reform_events.csv`
- `data/processed/stacked_event_pos.csv`
- `data/processed/stacked_event_neg.csv`

## D) Minimum robustness set (Q1 referee standard)
- **Pretrends and joint lead tests:** implement in `notebooks/07_inference_bands_placebos.py`, save to `outputs/tables/stacked_event_pretrend_tests.csv` and `outputs/figures/stacked_event_pretrend_plot.png`.
- **Few-treated inference (wild cluster bootstrap):** implement in `notebooks/07_inference_bands_placebos.py`, save to `outputs/tables/stacked_event_wildboot.csv`.
- **Randomization inference / placebo timing:** implement in `notebooks/07_inference_bands_placebos.py`, save to `outputs/tables/stacked_event_placebo_timing.csv`.
- **Joint bands (sup-t across horizons):** implement in `notebooks/07_inference_bands_placebos.py`, save to `outputs/figures/stacked_event_joint_bands.png`.
- **Clean-control sensitivity:** alternative control definitions in `notebooks/05_build_stacked_event_data.py`, save to `outputs/tables/stacked_event_control_sensitivity.csv`.
