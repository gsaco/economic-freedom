# Open issues and fixes (Q1 referee readiness)

## Implemented fixes (Tier 1)

| Issue | Why it matters (referee concern) | Exact edit location | Fix / output artifact | Validation & pass/fail criteria | Status |
| --- | --- | --- | --- | --- | --- |
| Treated vs control balance | Selection bias concerns in stacked DiD; referees ask whether treated countries look different pre-reform. | `notebooks/07_inference_bands_placebos.py` (`balance_table`) | `outputs/tables/stacked_event_balance_pre.csv` with standardized differences at event_time = -1. | Pass if file exists and has nonzero `n_treated` and `n_control` for both `pos` and `neg`. | Implemented |
| Two-way clustering check | Common shocks and serial correlation can understate SE; need iso3+year clustering check. | `notebooks/07_inference_bands_placebos.py` (`two_way_cluster_post`) | `outputs/tables/stacked_event_two_way_cluster.csv` (post effect, iso3+year clusters). | Pass if file exists and includes both `pos` and `neg` rows. | Implemented |
| Placebo timing | Validates timing not spurious; standard event-study diagnostic. | `notebooks/07_inference_bands_placebos.py` (`placebo_timing_post`) | `outputs/tables/stacked_event_placebo_timing.csv` (random timing shifts). | Pass if file exists and p-values are not NA. | Implemented |
| Missing early coverage (1970–1995) | Early quinquennial coverage can be thin; sensitivity is required. | `notebooks/07_inference_bands_placebos.py` (`sample_sensitivity_post`) | `outputs/tables/stacked_event_sample_sensitivity.csv` with min_year = 1985, 1995. | Pass if file includes rows for both min_year values and both labels. | Implemented |
| Small-N negative reforms | Low power can distort asymmetry; need fallback. | `notebooks/04_construct_shocks.py` (fallback quantiles), `output/logs/stacked_event_config.json` | Fallback to q85/q15 if event count < 20 (not triggered in current data). | Pass if log reports `event_count_neg >= 20`; if not, fallback must be `true`. | Implemented |

## Pending / Tier 2 issues

| Issue | Why it matters | Proposed location | Proposed fix | Validation | Status |
| --- | --- | --- | --- | --- | --- |
| Two-way clustering for full event-time coefficients | Current two-way clustering is only for the post summary effect; referees may want event-time SEs with two-way clustering. | `notebooks/06_estimate_lp_stacked.py` or `notebooks/07_inference_bands_placebos.py` | Add two-way clustered SEs for event-time coefficients and store in `outputs/tables/stacked_event_irf_*_two_way.csv`. | Pass if new tables exist with matching horizons. | Pending |
| Placebo outcomes (optional) | Additional falsification beyond timing can strengthen credibility. | `notebooks/07_inference_bands_placebos.py` | Add placebo outcomes (e.g., future growth) if available. | Pass if placebo coefficient centered at 0. | Pending |
