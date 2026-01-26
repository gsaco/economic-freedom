# Results Manifest

This file lists generated tables/figures and the commands that produce them.

## Tables
- `output/paper_tables/rd_first_stage.csv` — RD first stage (EFW overall + areas + EFW ex‑Sound Money)
- `output/paper_tables/rd_reduced_form.csv` — RD reduced‑form macro outcomes
- `output/paper_tables/rd_tail_outcomes.csv` — RD tail outcomes (worst growth, inflation spikes, drawdown, crisis start)
- `output/paper_tables/rd_iv_main.csv` — RD‑IV (Wald ratio) estimates (primary horizons)
- `output/paper_tables/lp_reduced_form.csv` — LP reduced‑form (winner effects; fallback when IV weak)
- `output/paper_tables/rd_exclusion_bounds.csv` — exclusion sensitivity grid (delta → beta)
- `output/paper_tables/rd_exclusion_bounds_summary.csv` — exclusion bounds summary
- `output/paper_tables/asymmetry_tests.csv` — asymmetry tests
- `output/paper_tables/nonlinear_effects.csv` — nonlinearity/state dependence
- `output/paper_tables/robustness_suite.csv` — donut/placebo/bandwidth robustness

## Figures
- `output/paper_figures/first_stage/rd_efw_path_h1.*` — RD plot of EFW change at h=1
- `output/paper_figures/first_stage/rd_log_gdp_cum_h4.*` — RD plot of log GDP per‑capita (h=4)
- `output/paper_figures/asymmetry/peak_persistence_bars.*` — asymmetry summary
- `output/paper_figures/nonlinear/irf_by_magnitude.*` — nonlinearity by magnitude
- `output/paper_figures/nonlinear/irf_by_state.*` — state dependence
- `output/paper_figures/robustness/exclusion_bounds_h4.*` — exclusion sensitivity plot

## Generation commands
```bash
make run
```
