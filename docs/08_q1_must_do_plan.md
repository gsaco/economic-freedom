# Q1 must-do plan (prioritized checklist)

## TIER 1 — Non‑negotiable

1) **Two-way clustered event-time SEs**
- Goal (referee objection): SEs understated due to common shocks/time clustering across countries.
- Repo locations: `notebooks/06_estimate_lp_stacked.py`, `notebooks/07_inference_bands_placebos.py`.
- Command: `python tools/run_all.py`
- Expected artifacts: `outputs/tables/stacked_event_irf_pos_gdppc_growth_5y_two_way.csv`, `outputs/tables/stacked_event_irf_neg_gdppc_growth_5y_two_way.csv`.
- Pass/Fail: Pass if two-way tables exist with non-NA SEs for all horizons.

2) **Paper-ready narrative and evidence map**
- Goal: referees require a clear contribution and explicit evidence-to-claim mapping.
- Repo locations: `reports/q1_paper_pitch.md`, `reports/hypothesis_assessment.md`.
- Command: `python tools/build_report.py`
- Expected artifacts: updated `reports/q1_paper_pitch.md` citing `outputs/figures/stacked_event_*` and `outputs/tables/stacked_event_*`.
- Pass/Fail: Pass if every main claim in the pitch references at least one output artifact.

3) **Replication-ready README + run command**
- Goal: replication package compliance.
- Repo locations: `README.md`, `tools/run_all.py`.
- Command: `python tools/run_all.py`
- Expected artifacts: README section “How to run everything,” `output/report.md` updated with docs.
- Pass/Fail: Pass if a clean run regenerates all figures/tables listed in `output/report.md`.

## TIER 2 — Major credibility upgrades

1) **Placebo outcomes**
- Goal: falsification tests beyond timing.
- Repo locations: `notebooks/07_inference_bands_placebos.py`.
- Command: `python tools/run_all.py`
- Expected artifacts: `outputs/tables/stacked_event_placebo_outcomes.csv`.
- Pass/Fail: Pass if placebo effects are statistically indistinguishable from zero.

2) **Event-time two-way clustering for collapse outcome**
- Goal: robustness of tail-risk findings.
- Repo locations: `notebooks/06_estimate_lp_stacked.py`.
- Command: `python tools/run_all.py`
- Expected artifacts: `outputs/tables/stacked_event_irf_*_growth_collapse_two_way.csv`.
- Pass/Fail: Pass if sign and timing are stable vs baseline.

3) **Alternative shock thresholds sensitivity**
- Goal: results not driven by q90/q10 choice.
- Repo locations: `notebooks/04_construct_shocks.py` (config) and `outputs/spec_ledger.csv`.
- Command: `python tools/run_all.py`
- Expected artifacts: `outputs/tables/stacked_event_thresholds.csv` updates + spec ledger entry.
- Pass/Fail: Pass if main coefficients remain within 1 SE under q85/q15.

## TIER 3 — Nice‑to‑have extensions

1) **Reform bundle–specific stacked events**
- Goal: tie asymmetry to concrete reform packages.
- Repo locations: `notebooks/03_q1_candidate_evidence.py`, new stacked design in `notebooks/07_inference_bands_placebos.py`.
- Command: `python tools/run_all.py`
- Expected artifacts: `outputs/figures/moduleC_bundle_irfs.png` updated with stacked design.
- Pass/Fail: Pass if at least one bundle shows clear post dynamics without pretrends.

2) **Night-lights robustness (if data available)**
- Goal: alleviate GDP misreporting concerns.
- Repo locations: new `src/nightlights_data.py`, `notebooks/03_q1_candidate_evidence.py`.
- Command: `python tools/run_all.py`
- Expected artifacts: `outputs/figures/moduleE_nightlights_irf.png`.
- Pass/Fail: Pass if night-lights and GDP-based results align in sign and timing.
