# Q1 paper pitch: Components of economic freedom, short-run growth, and downside risk (1970-2020, quinquennial)

## 1) Main research question
Which components of economic freedom (EFW) drive short-run growth dynamics and downside risk, and are these relationships robust to regime measurement concerns and crisis episodes?

## 2) Contribution vs literature
- Component-first panel (EFW areas + subcomponents) aligned to a strict quinquennial grid (1970-2020) with transparent shock definitions and decomposition.
- Dynamic evidence using local projections with country and time FE, asymmetry (pos/neg shocks), regime splits, and downside-risk outcomes.
- Reproducibility and multiple-testing discipline via a spec ledger with BH/FDR (`outputs/spec_ledger.csv`).

## 3) Identification strategy (what variation identifies what)
- Within-country variation in quinquennial changes of EFW components (country FE + time FE) identifies short-run associations net of common shocks.
- Dynamic LPs (h = 0,1,2,3) capture contemporaneous vs forward effects at 5-year horizons.
- Crisis event studies align crisis timing to quinquennial bins for stress-test diagnostics, not causal claims.

## 4) Main results (with evidence refs)
- Summary EFW change has a strong contemporaneous association with GDPpc growth (h=0: 0.0587, p=1.6e-08). See `outputs/figures/moduleA_irf_linear_gdppc_growth_5y.png` and `outputs/tables/moduleA_lp_coefficients.csv`.
- Component effects concentrate in area2 (legal system), area3 (sound money), and area5 (regulation) at h=0; effects decay at longer horizons. See `outputs/figures/moduleA_irf_areas_gdppc_growth_5y.png` and `outputs/tables/moduleA_component_leaderboard.csv`.
- Asymmetry: negative shocks are more contractionary in implied terms (pos vs neg coefficients at h=0). See `outputs/figures/moduleA_irf_asym_gdppc_growth_5y.png`.
- Downside risk: EFW changes reduce the probability of growth collapse at h=0 (p=0.013), with weak persistence. See `outputs/figures/moduleF_collapse_irf.png`.
- Regime heterogeneity: stronger h=0 effects in autocracies than democracies (autocracy 0.071 vs democracy 0.030). See `outputs/figures/moduleE_irf_autocracy.png` and `outputs/figures/moduleE_irf_democracy.png`.
- Crisis event studies show contemporaneous growth drops for systemic banking and sovereign crises, but crisis x EFW interactions are not significant. See `outputs/figures/moduleD_event_systemic_banking.png` and `outputs/figures/moduleD_crisis_interaction.png`.

## 5) Robustness checks passed/failed
Passed:
- Country FE + time FE; lagged outcomes and controls.
- Multiple-testing control via BH/FDR in the spec ledger.
- Regime splits and GDP measurement-error simulations (Module E).
- Alternative outcomes: TFP growth, investment share, inflation (Module A).

Not yet passed / weak:
- Crisis interactions (Module D) and reform bundles (Module C) are not robust.
- Complementarities (Module B) weak and sensitive to specification.
- Persistence beyond h=0 is limited across most outcomes.

## 6) Limitations and next steps
- Endogeneity remains: contemporaneous effects may reflect simultaneous reforms and growth rather than causal impacts.
- Measurement issues in autocracies require stronger external validation (night lights, alternative GDP sources).
- Expand robustness to alternative shock definitions (year-specific quantiles vs pooled) and re-estimate growth using PWT GDP.
- Improve identification by exploring reform narratives or external instruments (trade shocks, monetary anchor adoption) if data allow.

## Proposed paper angle (selected)
Primary angle: EFW components (especially sound money and legal system) drive short-run growth and downside-risk reductions, with limited persistence and possible regime-dependent measurement bias. This is a cautious but publishable contribution that reframes EFW effects as short-run stabilization rather than long-run growth engines.
