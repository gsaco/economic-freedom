# Candidate papers and decision (non-duplication vs Rondón 2025)

## Candidate A (flagship): Asymmetric reforms and downside risk
**Working title:** Asymmetric economic freedom reforms and the tail risk of growth

**Contribution (5 bullets):**
- Shifts the focus from mean growth to downside risk (growth collapse probability and quantile effects).
- Separates positive vs negative EFW shocks to test asymmetry in short-run dynamics.
- Uses quinquennial LPs with country/time FE and explicit pretrend/placebo tests (to be implemented).
- Reports multi-horizon joint inference and few-treated robustness (to be implemented).
- Frames EFW as a stabilizer of tail risk rather than a long-run growth engine.

**What is new vs Rondón (2025):**
- Thesis does not study tail risk or quantile effects.
- Thesis uses symmetric shocks and reform dummies; no explicit asymmetry tests.
- Thesis emphasis is on component-level effects; here components are secondary/appendix.

**Core estimand + identification (one paragraph):**
Estimate dynamic effects of positive vs negative EFW shocks on growth and growth-collapse risk using quinquennial local projections with country and time fixed effects. Identification uses within-country variation in quinquennial EFW changes; event-window checks and placebo timing will test for pretrends. Tail-risk outcomes (collapse indicator; quantile regression) highlight asymmetric impacts not visible in mean regressions.

**Key threats + how neutralized (linked to repo files/outputs):**
- Endogeneity of reforms: address with placebo timing and pretrend checks in new `notebooks/07_inference_bands_placebos.*` (to be created).
- Horizon-wise multiple testing: BH/FDR already in `outputs/spec_ledger.csv`; extend to joint bands.
- Measurement error in autocracies: regime splits and WB vs PWT checks in `notebooks/03_q1_candidate_evidence.py` (Module E), plus new robustness tests.

**Feasibility score (0–10):** 8 (tail-risk module already exists: `outputs/figures/moduleF_collapse_irf.png`, `outputs/tables/moduleF_quantile_regression.csv`; requires inference upgrades).

**Decision score (0–20):**
- Identification credibility: 2
- Inference robustness: 1
- Novelty vs thesis: 3
- Power/support: 3
- Mechanism plausibility + SCM coherence: 3
**Total: 12**

---

## Candidate B: Crisis resilience and recovery
**Working title:** Economic freedom as a crisis buffer: event studies around systemic crises

**Contribution (5 bullets):**
- Uses crisis onsets as stress tests to assess resilience and recovery speed.
- Implements stacked event studies around crises with clean controls (to be built).
- Tests interaction between pre-crisis EFW and crisis shocks.
- Adds robustness: exclude extreme outliers; placebo crisis timing.
- Connects institutions to macro stability, not just growth.

**What is new vs Rondón (2025):**
- Thesis does not use crisis events or resilience metrics.
- No stacked event study design in thesis.

**Core estimand + identification (one paragraph):**
Estimate event-time path of GDPpc growth around crisis onset, stratified by pre-crisis EFW levels. Identification leverages exogenous crisis timing (Laeven-Valencia) with clean control cohorts (not-yet-crisis countries) and stacked event-study design.

**Key threats + how neutralized:**
- Crisis timing endogeneity: use pretrend checks + placebo timing.
- Sparse events: implement few-treated inference (wild bootstrap, RI).
- Measurement error: WB vs PWT robustness for outcomes.

**Feasibility score (0–10):** 6 (crisis data exists, but stacked design and inference upgrades are missing).

**Decision score (0–20):** 11

---

## Candidate C: Regime mismeasurement and EFW-growth bias
**Working title:** Economic freedom and growth under data manipulation: regime heterogeneity

**Contribution (5 bullets):**
- Tests whether EFW-growth links are driven by autocracy measurement bias.
- Compares WB vs PWT growth; runs measurement-error simulations.
- Uses regime splits to bound plausible effects.
- Provides credibility checks for growth under-reporting/misreporting.
- Connects to GDP misreporting literature.

**What is new vs Rondón (2025):**
- Thesis does not examine regime-based measurement bias or alternative GDP sources.

**Core estimand + identification (one paragraph):**
Compare dynamic LP estimates between democracies and autocracies, and between WB and PWT outcomes, to assess sensitivity to measurement error. Identification is diagnostic rather than causal, focusing on robustness bounds.

**Key threats + how neutralized:**
- Diagnostic nature: treat as robustness appendix unless strong differential patterns emerge.
- Measurement uncertainty: incorporate SWIID uncertainty and alternative proxies (future extension).

**Feasibility score (0–10):** 7 (module E already exists; needs stronger external validation).

**Decision score (0–20):** 10

---

## Decision (KEEP THE BEST OPTION)
**Chosen flagship:** Candidate A (asymmetry + downside risk).

Rationale: highest total score (12/20), clear novelty vs thesis, and already supported by existing outputs in `outputs/figures/moduleA_irf_asym_gdppc_growth_5y.png` and `outputs/figures/moduleF_collapse_irf.png`. Candidate B is a strong backup if asymmetry/tail-risk effects fail pretrend or joint-band tests.

## Decision appendix (why others were not chosen)
- Candidate B rejected as flagship because crisis interactions are currently weak and stacked-event infrastructure is missing (risk of low power).
- Candidate C rejected as flagship because it is primarily diagnostic; without external proxies (night lights) it is unlikely to sustain a Q1 causal contribution.
