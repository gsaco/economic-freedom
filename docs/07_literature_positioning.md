# Literature positioning (NEEDS WEB VERIFICATION)

## Key references (12–20; links need verification)
1) Sun, L. and Abraham, S. (2021). “Estimating dynamic treatment effects in event studies with heterogeneous treatment effects.” *Journal of Econometrics*. **NEEDS WEB VERIFICATION**
   - Relevance: Modern event-study corrections; motivates clean-control stacked designs.
   - Link: NEEDS WEB VERIFICATION

2) Callaway, B. and Sant’Anna, P. (2021). “Difference-in-Differences with multiple time periods.” *Journal of Econometrics*. **NEEDS WEB VERIFICATION**
   - Relevance: Staggered adoption DiD with heterogeneous effects; informs cohort definitions.
   - Link: NEEDS WEB VERIFICATION

3) Goodman-Bacon, A. (2021). “Difference-in-differences with variation in treatment timing.” *Journal of Econometrics*. **NEEDS WEB VERIFICATION**
   - Relevance: Decomposes TWFE DiD; supports stacked/event-study alternatives.
   - Link: NEEDS WEB VERIFICATION

4) Borusyak, K., Jaravel, X., and Spiess, J. (2021). “Revisiting event study designs.” **NEEDS WEB VERIFICATION**
   - Relevance: Modern inference for dynamic effects; informs placebo and pretrend tests.
   - Link: NEEDS WEB VERIFICATION

5) Jordà, O. (2005). “Estimation and inference of impulse responses by local projections.” *American Economic Review*. **NEEDS WEB VERIFICATION**
   - Relevance: LP foundation for dynamic effects.
   - Link: NEEDS WEB VERIFICATION

6) Plagborg-Møller, M. and Montiel Olea, J. (2019). “Local projection inference is robust to misspecification.” **NEEDS WEB VERIFICATION**
   - Relevance: LP inference justification and robustness.
   - Link: NEEDS WEB VERIFICATION

7) Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008). “Bootstrap-based improvements for inference with clustered errors.” *Review of Economics and Statistics*. **NEEDS WEB VERIFICATION**
   - Relevance: Wild cluster bootstrap for few-treated settings.
   - Link: NEEDS WEB VERIFICATION

8) Roodman, D., MacKinnon, J. G., Nielsen, M. Ø., and Webb, M. D. (2019). “Fast and wild: bootstrap inference for few clusters.” **NEEDS WEB VERIFICATION**
   - Relevance: Few-cluster inference guidance.
   - Link: NEEDS WEB VERIFICATION

9) Laeven, L. and Valencia, F. (2013, updates). “Systemic banking crises database.” **NEEDS WEB VERIFICATION**
   - Relevance: Crisis timing for resilience/stress-test designs.
   - Link: NEEDS WEB VERIFICATION

10) Gwartney, J., Lawson, R., and Hall, J. (annual). *Economic Freedom of the World* report. **NEEDS WEB VERIFICATION**
   - Relevance: Primary EFW data source and methodology.
   - Link: NEEDS WEB VERIFICATION

11) Hall, J. C. and Lawson, R. A. (2014). “Economic freedom of the world: an accounting of the literature.” **NEEDS WEB VERIFICATION**
   - Relevance: Survey of EFW-growth evidence and gaps.
   - Link: NEEDS WEB VERIFICATION

12) De Haan, J., Lundström, S., and Sturm, J.-E. (2006/2007). “Market-oriented institutions and policies and economic growth: a critical survey.” **NEEDS WEB VERIFICATION**
   - Relevance: Institutional-growth debates and endogeneity concerns.
   - Link: NEEDS WEB VERIFICATION

13) Acemoglu, D., Johnson, S., and Robinson, J. (2001). “Colonial origins of comparative development.” *AER*. **NEEDS WEB VERIFICATION**
   - Relevance: Institutions-growth causal identification benchmark.
   - Link: NEEDS WEB VERIFICATION

14) Rodrik, D., Subramanian, A., and Trebbi, F. (2004). “Institutions rule.” **NEEDS WEB VERIFICATION**
   - Relevance: Institutions vs trade as drivers of growth; contextualizes EFW mechanisms.
   - Link: NEEDS WEB VERIFICATION

15) Henderson, J. V., Storeygard, A., and Weil, D. (2012). “Measuring economic growth from outer space.” *AER*. **NEEDS WEB VERIFICATION**
   - Relevance: Alternative proxies (night lights) for misreporting concerns.
   - Link: NEEDS WEB VERIFICATION

16) Martínez, L. (2022). “How much should we trust the dictator’s GDP growth estimates?” **NEEDS WEB VERIFICATION**
   - Relevance: GDP misreporting in autocracies; motivates regime-based sensitivity.
   - Link: NEEDS WEB VERIFICATION

17) Alesina, A., Ardagna, S., and coauthors (various). “Large reforms / fiscal adjustments and growth.” **NEEDS WEB VERIFICATION**
   - Relevance: Reform packages and bundle effects; informs clustered “reform bundle” design.
   - Link: NEEDS WEB VERIFICATION

18) Cengiz, D., Dube, A., Lindner, A., and Zipperer, B. (2019). “The effect of minimum wages on low-wage jobs.” *QJE*. **NEEDS WEB VERIFICATION**
   - Relevance: Stacked event-study design template.
   - Link: NEEDS WEB VERIFICATION

## Positioning paragraph
This project positions economic freedom reforms within the modern staggered-adoption/event-study literature, emphasizing dynamic asymmetry and downside risk rather than average growth correlations. By implementing a clean-control stacked event study with joint bands, wild cluster bootstrap, placebo timing, and leave-one-event-out influence checks, the design directly addresses the post–Sun-Abraham/Callaway-Sant’Anna critiques of naïve TWFE event studies. The contribution is to reframe institutional reforms as drivers of tail risk and macro resilience while maintaining transparent limitations around endogeneity and measurement error.

## Referee attack map (critique → reference → repo evidence)
| Likely critique | Reference(s) to cite | Repo evidence/output |
| --- | --- | --- |
| Staggered adoption bias in event studies | Sun & Abraham (2021); Goodman-Bacon (2021) | `outputs/figures/stacked_event_irf_pos_gdppc_growth_5y.png`, `outputs/tables/stacked_event_pretrend_tests.csv` |
| Few treated events and weak inference | Cameron et al. (2008); Roodman et al. (2019) | `outputs/tables/stacked_event_wildboot.csv`, `outputs/tables/stacked_event_leave_one_out.csv` |
| Multiple horizons inflate false positives | Plagborg-Møller & Montiel Olea (2019) | `outputs/figures/stacked_event_joint_bands.png` |
| Selection bias / treated vs controls differ | Standard DiD diagnostics | `outputs/tables/stacked_event_balance_pre.csv` |
| Early-year data coverage concerns | EFW documentation + sample sensitivity | `outputs/tables/stacked_event_sample_sensitivity.csv` |
| GDP misreporting in autocracies | Martínez (2022); Henderson et al. (2012) | `outputs/figures/moduleE_irf_autocracy.png`, `outputs/tables/moduleE_measurement_error_sim.csv` |
