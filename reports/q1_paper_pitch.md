# Q1 paper pitch: Asymmetric economic freedom reforms and short-run growth dynamics (1970-2020, quinquennial)

## 1) Main research question
Do positive vs. negative economic freedom reforms have asymmetric effects on quinquennial growth and downside risk, once we use clean-control stacked event studies and modern inference?

## 2) Contribution vs literature
- Shifts the contribution away from component disaggregation as a main result; components are a benchmark appendix only.
- Uses clean-control stacked event studies with explicit reform timing, avoiding TWFE staggered-adoption bias.
- Emphasizes asymmetry and downside risk rather than average correlations.
- Implements joint bands, wild cluster bootstrap, placebo timing, and leave-one-event-out diagnostics.

## 3) Identification strategy (what variation identifies what)
- Reform timing is defined by large quinquennial changes in EFW (q90/q10) with a sustain rule and cooldown.
- Stacked event-study design uses not-yet-treated clean controls (no events in the window), with country-stack and year FE.
- Effects interpreted as dynamic within-country associations; causality is framed cautiously given potential endogeneity.

## 4) Main results (with evidence refs)
- **Positive reforms**: event_time 0 effect on 5y GDPpc growth is 0.064 (p=0.006). See `outputs/tables/stacked_event_irf_pos_gdppc_growth_5y.csv` and `outputs/figures/stacked_event_irf_pos_gdppc_growth_5y.png`.
- **Negative reforms**: event_time 0 effect is -0.099 (p=0.230) and not precisely estimated. See `outputs/tables/stacked_event_irf_neg_gdppc_growth_5y.csv`.
- **Post effect robustness** (positive reforms): wild bootstrap p=0.007, two-way cluster p=0.028, placebo timing p=0.035. See `outputs/tables/stacked_event_wildboot.csv`, `outputs/tables/stacked_event_two_way_cluster.csv`, `outputs/tables/stacked_event_placebo_timing.csv`.
- **Pretrends**: joint lead tests show no evidence of pretrends (p>0.79). See `outputs/tables/stacked_event_pretrend_tests.csv` and `outputs/figures/stacked_event_pretrend_plot.png`.
- **Downside risk**: growth-collapse effects are negative but not significant (pos event_time 0: -0.078, p=0.228). See `outputs/tables/stacked_event_irf_pos_growth_collapse.csv`.

## 5) Robustness checks passed/failed
Passed:
- Clean-control stacked design with joint bands (see `outputs/figures/stacked_event_joint_bands.png`).
- Wild cluster bootstrap, placebo timing, and leave-one-event-out influence checks (see `outputs/tables/stacked_event_wildboot.csv`, `outputs/tables/stacked_event_placebo_timing.csv`, `outputs/tables/stacked_event_leave_one_out.csv`).
- Sample sensitivity (year >= 1985, 1995) preserves positive reform effect (see `outputs/tables/stacked_event_sample_sensitivity.csv`).

Not yet passed / weak:
- Negative reform effects are imprecise (low power / heterogeneity).
- Tail-risk (collapse) effects are not robust in stacked design.
- Balance table shows meaningful pre-differences in EFW levels; interpretation remains associative (see `outputs/tables/stacked_event_balance_pre.csv`).

## 6) Limitations and next steps
- Endogeneity remains; the design improves timing and inference but is not a causal instrument.
- Augment with external instruments or reform narratives (future work).
- Add placebo outcomes and two-way clustered event-time SEs (see `docs/08_q1_must_do_plan.md`).

## Proposed paper angle (selected)
Primary angle: **positive economic freedom reforms are followed by higher quinquennial growth in clean-control stacked event studies, while negative reforms are imprecise and tail-risk effects are weak**. The paper frames EFW as a short-run growth stabilizer under credible event-study diagnostics, with components relegated to a benchmark appendix.
