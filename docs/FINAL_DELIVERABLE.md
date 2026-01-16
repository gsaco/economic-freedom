1) Chosen Paper Thesis (title + 150-250 word abstract)
Title: Asymmetric Economic Freedom Reforms and Short-Run Growth: Evidence from Clean-Control Stacked Event Studies

Abstract (approx. 180 words):
This paper studies the short-run macro effects of economic freedom reforms using a quinquennial (1970-2020) cross-country panel constructed from Fraser EFW and standard macro sources. Large reforms are defined as the top and bottom deciles of 5-year EFW changes, subject to a sustain rule and a 10-year cooldown. We build clean-control stacked event panels (no reforms in the event window for controls) and estimate dynamic event-time effects with entity and year fixed effects, plus lagged macro controls. Positive reforms are associated with a 0.064 log-point increase in contemporaneous 5-year GDP per capita growth (p=0.006), and a post-period summary effect of 0.056 (wild bootstrap p=0.007; two-way clustered p=0.028; placebo timing p=0.035). Negative reforms are negative but imprecise. Joint lead tests show no evidence of pretrends. Tail-risk effects on growth-collapse probability are negative but not statistically significant in the stacked design. Results are stable to later-sample restrictions (1985+, 1995+), but balance diagnostics indicate treated units differ in baseline EFW levels, so claims remain associative. The contribution reframes economic freedom as a short-run growth stabilizer under credible event-study diagnostics while reporting null results for downside risk and negative reforms.

2) Novelty vs Rondon (2025) thesis (explicit bullets)
- Uses clean-control stacked event studies with explicit reform timing, not TWFE or component-first regressions.
- Focuses on asymmetric reforms (positive vs negative) as the main estimand; components are appendix-only.
- Adds Q1-grade inference (joint bands, wild cluster bootstrap, placebo timing, leave-one-event-out).
- Provides explicit balance, two-way clustering, and sample-sensitivity diagnostics.

3) Model + Identification (exact equations and definitions)
Main stacked event-study equation:
Y_{i,e,t} = sum_{k != -1} beta_k * 1[event_time=k] * treated_{i,e} + gamma' X_{i,e,t-1} + alpha_{i,e} + lambda_t + epsilon_{i,e,t}

Definitions:
- event_time = (year - event_year) / 5, k in {-2, -1, 0, 1, 2, 3}, omit k=-1.
- treated_{i,e} = 1 for the treated country in event stack e.
- alpha_{i,e} are stack-country fixed effects; lambda_t are year fixed effects.
- Controls X include lagged gdppc growth, lagged gdppc log, lagged EFW level, and macro controls (trade, inflation, investment, government consumption).

Treatment definition:
- d_efw_summary = efw_summary(t) - efw_summary(t-5).
- Positive reform: d_efw_summary >= q90; negative reform: d_efw_summary <= q10.
- Sustain rule: efw_summary(t+5) must not reverse by more than epsilon=0.2.
- Cooldown: minimum 10 years between same-type events.
- Window: K=2 pre periods, L=3 post periods.

4) Inference + Placebos + Bands (exact procedures)
- Baseline SE: clustered by stack-country entity (PanelOLS, entity and year FE).
- Pretrend test: Wald test of all lead coefficients (event_time < 0) using clustered covariance.
- Joint bands: sup-t critical value from 5,000 draws of N(0, V) where V is clustered covariance of event-time coefficients.
- Wild cluster bootstrap: Rademacher weights at iso3 level, 999 reps, null imposed via restricted model (controls only), target = post indicator effect.
- Placebo timing: random shifts s in [-2, 3] within each stack, 200 reps, re-estimate post effect.
- Leave-one-event-out: drop each stack and re-estimate post effect.

5) Results map (tables/figures list; what each proves)
- outputs/figures/stacked_event_irf_pos_gdppc_growth_5y.png: positive reform dynamics on growth.
- outputs/tables/stacked_event_irf_pos_gdppc_growth_5y.csv: coefficient values and SEs (event-time effects).
- outputs/tables/stacked_event_wildboot.csv: few-treated robustness for post effect.
- outputs/tables/stacked_event_placebo_timing.csv: placebo timing distribution and p-values.
- outputs/figures/stacked_event_joint_bands.png: joint horizon bands across event times.
- outputs/tables/stacked_event_pretrend_tests.csv: joint lead tests (pretrend diagnostics).
- outputs/tables/stacked_event_two_way_cluster.csv: iso3+year clustered SE check.
- outputs/tables/stacked_event_sample_sensitivity.csv: early-year coverage sensitivity.
- outputs/tables/stacked_event_balance_pre.csv: treated vs control balance at t=-1.
- outputs/tables/stacked_event_irf_pos_growth_collapse.csv: tail-risk (collapse) results (null/weak).

6) Threats to validity + Mitigations (tied to repo artifacts)
- Endogeneity of reforms: mitigate with clean controls and pretrend tests, but interpret as associative (see outputs/tables/stacked_event_pretrend_tests.csv).
- Baseline imbalance (treated vs controls): quantify with balance table (outputs/tables/stacked_event_balance_pre.csv); consider matching as future work.
- Few treated negative reforms: event counts documented (outputs/tables/stacked_event_counts.csv); fallback quantiles logged (output/logs/stacked_event_config.json).
- Common shocks / serial correlation: two-way clustering check (outputs/tables/stacked_event_two_way_cluster.csv).
- Early coverage concerns: sample sensitivity (outputs/tables/stacked_event_sample_sensitivity.csv).
- Multiple testing: spec ledger with BH/FDR (outputs/spec_ledger.csv).

7) Q1 Must-Do Plan (Tier 1/2/3 checklist with repo edits + commands)
Tier 1:
- Two-way clustered event-time SEs: implement in notebooks/06_estimate_lp_stacked.py; run `python tools/run_all.py`; expect outputs/tables/stacked_event_irf_*_two_way.csv; pass if horizons populated.
- Paper-ready narrative: update reports/q1_paper_pitch.md and reports/hypothesis_assessment.md; run `python tools/build_report.py`; pass if every claim references outputs/ tables/figures.
- README replication: ensure `python tools/run_all.py` is documented; pass if output/report.md regenerates cleanly.

Tier 2:
- Placebo outcomes: add to notebooks/07_inference_bands_placebos.py; run `python tools/run_all.py`; expect outputs/tables/stacked_event_placebo_outcomes.csv; pass if effects are null.
- Collapse outcome two-way SEs: extend event-time SEs for growth_collapse; pass if sign/timing stable.

Tier 3:
- Reform bundles stacked design: integrate moduleC bundles into stacked events; pass if at least one bundle has stable post effects without pretrends.
- Night-lights robustness: add alternative outcome for misreporting concerns; pass if signs align with GDP results.

8) Literature positioning (web-cited references + paragraph; NEEDS WEB VERIFICATION if needed)
NEEDS WEB VERIFICATION: Sun and Abraham (2021); Callaway and Sant'Anna (2021); Goodman-Bacon (2021); Jorda (2005); Cameron, Gelbach, Miller (2008); Roodman et al. (2019); Gwartney et al. (EFW); Hall and Lawson (2014); Martinez (2022). See docs/07_literature_positioning.md for full list and referee attack map.

Positioning paragraph: The paper uses clean-control stacked event studies to address post-2020 critiques of TWFE event studies and reframes economic freedom as a short-run growth stabilizer rather than a long-run growth engine. Robust inference (joint bands, wild bootstrap, placebo timing) and transparent null results for tail risk differentiate the contribution from component-heavy prior work.

9) Replication package checklist (submission-ready items)
- One-command rebuild: python tools/run_all.py.
- Data provenance: raw data cached in data/raw with .meta files.
- Environment pinning: requirements.txt.
- Seeds/configs logged: output/logs/stacked_event_config.json and output/logs/stacked_event_inference_config.json.
- Outputs mirrored: outputs/ and output/ with report listing.
- Thesis benchmark replication labeled as appendix only.
