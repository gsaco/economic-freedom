# Changelog (minimum-change / maximum-coverage)

## 2026-01-15
- Ignored large data artifacts and removed them from version control to enable pushes.
  - Files: `.gitignore`, `data/raw/*`, `data/processed/stacked_event_*.csv`, `output/*`, `outputs/*`, `tesis.pdf` (untracked).
  - Why (referee concern): keep repository lean and reproducible with regenerable artifacts excluded from VCS.
  - Test: `git status --short` (tracked deletions staged).
- Added pipeline runner and report builder to satisfy required `python tools/run_all.py` and `output/report.md` expectations.
  - Files: `tools/run_all.py`, `tools/build_report.py`.
  - Why (referee concern): reproducibility and one-command rebuild with explicit report artifacts.
  - Test: `python tools/run_all.py` (success; outputs mirrored to `output/`).
- Created `output/` mirror and log artifacts via `run_all.py` (figures/tables copied from `outputs/`).
  - Files: `output/report.md`, `output/report.json`, `output/logs/run_all.json` (generated).
  - Why: required inspection of `output/figures`, `output/tables`, `output/logs` per instructions.
  - Test: `python tools/run_all.py`.
- Orientation doc started (see `docs/00_orientation.md`) with pipeline failure note (missing run_all) and subsequent fix.

- Pipeline execution refreshed executed notebooks and updated one table artifact.
  - Files: `notebooks/01_efw_macro_quinquennial_analysis.ipynb`, `notebooks/02_components_bundles_crises.ipynb`, `notebooks/03_q1_candidate_evidence.ipynb`, `outputs/tables/moduleC_bundle_centroids.csv`.
  - Why: capture fresh-kernel run outputs for reproducibility.
  - Test: `python tools/run_all.py`.
- SWIID metadata refreshed during pipeline run.
  - File: `data/raw/swiid/swiid9_91.dta.meta`.
  - Why: ensure raw data provenance metadata is present/updated.
  - Test: `python tools/run_all.py`.

- Added Rondón (2025) thesis baseline summary to avoid duplication.
  - File: `docs/01_thesis_baseline_rondon2025.md`.
  - Why: explicit anti-duplication baseline for Q1 positioning.
  - Test: `pdftotext -layout tesis.pdf /tmp/tesis_rondon_2025.txt` (source extraction).

- Added component-by-component repo audit ledger.
  - File: `docs/02_repo_audit_ledger.md`.
  - Why: Q1 readiness assessment and missing-module map.
  - Test: `rg -n "construct.*shock|stacked|event|lp|local projection|bootstrap|placebo|synthetic" -S .`

- Added candidate paper decision memo with scoring and rejection rationale.
  - File: `docs/03_candidate_papers_and_decision.md`.
  - Why: enforce non-duplication vs thesis and select flagship design.
  - Test: manual review against `docs/01_thesis_baseline_rondon2025.md` and `outputs/figures/moduleF_collapse_irf.png`.

- Implemented stacked event-study pipeline for asymmetric EFW reforms (flagship spec).
  - Files: `src/stacked_event.py`, `notebooks/04_construct_shocks.py`, `notebooks/05_build_stacked_event_data.py`, `notebooks/06_estimate_lp_stacked.py`, `tools/run_pipeline.py`.
  - Why: clean-control stacked LP-DiD to address endogeneity and asymmetry concerns for Q1 referees.
  - Test: `python tools/run_all.py` (success; new stacked-event tables/figures in `output/`).

- Wrote exact model/specification memo for flagship design.
  - File: `docs/04_flagship_specs_models_identification.md`.
  - Why: enforce explicit estimand and treatment definitions with logged thresholds and windows.
  - Test: `python tools/build_report.py` (doc appears in `output/report.md`).

- Added inference, placebo, and joint-band diagnostics for stacked events.
  - Files: `notebooks/07_inference_bands_placebos.py`, `docs/05_inference_placebos_joint_bands.md`, `tools/run_pipeline.py`.
  - Outputs: `outputs/tables/stacked_event_pretrend_tests.csv`, `outputs/figures/stacked_event_pretrend_plot.png`, `outputs/tables/stacked_event_joint_bands.csv`, `outputs/figures/stacked_event_joint_bands.png`, `outputs/tables/stacked_event_wildboot.csv`, `outputs/tables/stacked_event_placebo_timing.csv`, `outputs/tables/stacked_event_leave_one_out.csv`.
  - Why: Q1-grade inference (pretrends, joint bands, bootstrap, placebo timing, LOO influence).
  - Test: `python tools/run_all.py` (success).

- Added balance, two-way clustering, and sample-sensitivity diagnostics; documented open issues.
  - Files: `notebooks/07_inference_bands_placebos.py`, `docs/06_open_issues_and_fixes.md`.
  - Outputs: `outputs/tables/stacked_event_balance_pre.csv`, `outputs/tables/stacked_event_two_way_cluster.csv`, `outputs/tables/stacked_event_sample_sensitivity.csv`.
  - Why: address referee concerns about selection, clustering, and early-year coverage.
  - Test: `python tools/run_all.py` (success).

- Added literature positioning draft (web verification pending).
  - File: `docs/07_literature_positioning.md`.
  - Why: map methodological and EFW literatures to planned contribution and referee critiques.
  - Test: `python tools/build_report.py` (doc appears in `output/report.md`).

- Added Q1 must-do plan checklist.
  - File: `docs/08_q1_must_do_plan.md`.
  - Why: prioritize remaining work by referee objections and required artifacts.
  - Test: `python tools/build_report.py` (doc appears in `output/report.md`).

- Added replication package checklist.
  - File: `docs/09_replication_package_checklist.md`.
  - Why: ensure submission-ready reproducibility and provenance compliance.
  - Test: `python tools/build_report.py` (doc appears in `output/report.md`).

- Added final deliverable summary.
  - File: `docs/FINAL_DELIVERABLE.md`.
  - Why: provide submission-ready synthesis with model, inference, results map, and plan.
  - Test: `python tools/build_report.py` (doc appears in `output/report.md`).

- Shifted report generation to stacked-event notebook to avoid overwriting updated reports.
  - Files: `notebooks/01_efw_macro_quinquennial_analysis.py`, `notebooks/07_inference_bands_placebos.py`.
  - Outputs: `reports/data_dictionary.md`, `reports/hypothesis_assessment.md` regenerated with stacked-event results.
  - Why: ensure final reports reflect flagship stacked-event evidence, not early LP prototypes.
  - Test: `python tools/run_all.py` (success; reports updated).
