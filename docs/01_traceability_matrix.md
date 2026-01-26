# Traceability Matrix (plan.tex → artifacts → verification)

Status legend: TODO / DOING / DONE.

| Plan item (section) | Artifact(s) to create/verify | Verification method | Status |
|---|---|---|---|
| 0. Executive pitch + core claim discipline | `docs/04_methods_and_estimands.md` (claim ladder) | Doc review | DONE |
| 1. Estimands (Part 1: RD reduced form + first stage) | `src/shocks.py`, `notebooks/21_rd_first_stage_and_reduced_form.py`, `output/paper_tables/rd_first_stage.csv`, `output/paper_tables/rd_reduced_form.csv` | `make run` | DONE |
| 1. Tail outcomes definitions | `src/shocks.py`, `output/paper_tables/rd_tail_outcomes.csv` | `make run` | DONE |
| 1. Part‑2 RD‑IV estimand (Wald ratio) | `notebooks/22_rd_iv_main_results.py`, `output/paper_tables/rd_iv_main.csv` | `make run` | DONE |
| 1. LP reduced‑form fallback (if IV weak) | `notebooks/22b_lp_reduced_form.py`, `output/paper_tables/lp_reduced_form.csv` | `make run` | DONE |
| 1. Asymmetry estimands (+/‑ EFW) | `notebooks/31_asymmetry_tests.py`, `output/paper_tables/asymmetry_tests.csv` | `make run` | DONE |
| 1. State dependence estimands | `notebooks/32_nonlinearity_magnitude_state.py`, `output/paper_tables/nonlinear_effects.csv` | `make run` | DONE |
| 1. Nonlinearity estimands (threshold + tails) | `notebooks/32_nonlinearity_magnitude_state.py`, `output/paper_figures/nonlinear/*` | `make run` | DONE |
| 2. Running variable + treatment definition | `src/clea.py`, `src/ideology.py`, `notebooks/11_construct_close_elections_rd_sample.py` | `make run` | DONE |
| 2. Orientation rules (market vs non‑market) | `src/ideology.py`, `docs/04_methods_and_estimands.md` | Doc + code review | DONE |
| 2. Sample restrictions (clean margins, data availability) | `notebooks/11_construct_close_elections_rd_sample.py` | `make run` | DONE |
| 2. RD improvements: bias‑corrected local polynomial, clustered SE | `src/rd.py` (rdrobust) | `make run` | DONE |
| 2. Local randomization robustness | `src/rd_localrand.py`, `output/paper_tables/rd_balance.csv` | `make run` | DONE |
| 2. Overlapping elections contamination handling | `output/paper_tables/*switch*`, `notebooks/35_switch_only_variant.py` | `make run` | DONE |
| 2. Falsification suite (density, balance, placebo, donut) | `output/paper_tables/rd_density_*.csv`, `rd_placebo_cutoffs.csv`, `rd_donut.csv` | `make run` | DONE |
| 3. Data sources documentation | `docs/03_data_sources.md` | Doc review | DONE |
| 3. EFW access + component handling | `notebooks/01_ingest_fraser.py` | `make run` | DONE |
| 3. Elections/margins access (CLEA primary; ParlGov/DPI/V‑Party fallbacks) | `notebooks/10_ingest_clea.py`, `src/clea.py`, `docs/03_data_sources.md` | `make run` + doc | DONE |
| 3. Macro outcomes (WDI/PWT) | `notebooks/02_pull_worldbank.py`, `notebooks/03b_build_annual_panel.py` | `make run` | DONE |
| 3. Crisis data (Laeven‑Valencia) | `src/crisis.py`, `data/01_raw/crisis/*` | `make run` | DONE |
| 4. Empirical specs + horizons | `src/shocks.py`, `docs/04_methods_and_estimands.md` | Doc + code review | DONE |
| 5. Part‑2 identification strategies + sensitivity bounds | `output/paper_tables/rd_exclusion_bounds*.csv`, `output/paper_figures/robustness/exclusion_bounds_h4.*` | `make run` | DONE |
| 6. Module A asymmetry outputs | `output/paper_tables/asymmetry_tests.csv`, `output/paper_figures/asymmetry/*` | `make run` | DONE |
| 7. Module B state dependence outputs | `output/paper_tables/nonlinear_effects.csv`, `output/paper_figures/nonlinear/*` | `make run` | DONE |
| 8. Module C nonlinearity outputs | `output/paper_figures/nonlinear/*`, `output/paper_tables/rd_tail_outcomes.csv` | `make run` | DONE |
| 9. Novelty reference set | `references.bib` | grep for required citations | DONE |
| 10. Threats/failure modes + mitigation | `docs/04_methods_and_estimands.md` | Doc review | DONE |
| 11. Core tables/figures (14 items) | `output/paper_tables/*`, `output/paper_figures/*` | `make run` + `docs/06_results_manifest.md` | DONE |
| Appendix diagnostics checklist | `output/paper_tables/rd_*`, `output/paper_figures/rd_*` | `make run` | DONE |
| Build system targets (setup/test/lint/run/repro/clean) | `Makefile` | `make setup/test/lint/run/repro` | DONE |
| Execution log + reproducibility | `docs/05_execution_log.md` | Doc review | DONE |
