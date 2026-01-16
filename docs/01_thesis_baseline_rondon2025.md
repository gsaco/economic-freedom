# Rondón (2025) thesis baseline (anti-duplication)

Source: `tesis.pdf` (found in repo root). Required search path `/mnt/data` does not exist on this machine.

## 1) One-page bullet summary
- Research question: causal impact of economic freedom (EFW, Fraser) on economic growth, and whether components matter differently.
- Data/sample:
  - EFW index (aggregate + 5 areas) for ~165 countries, 1970-2022.
  - EFW data 1970-2000 is quinquennial; 2000-2022 annual.
  - Growth outcome: log real GDP per capita (2015 USD).
  - Controls: WDI macro variables; IMF Primary Commodity Terms of Trade (constructed); region and income-quintile trends.
- Identification strategy:
  - Local projections (panel LP) with country FE and time FE.
  - Shock-style variables: (i) EFW differences; (ii) “reform” dummy based on Grier and Grier (2021) (large, sustained increases).
  - Assumes large discrete reforms are quasi-exogenous; uses lag-augmented LP (two lags of outcome) to mitigate serial correlation.
- Shock definitions:
  - EFW differences as observed shocks (large jumps interpreted as exogenous).
  - Reform dummy: increase >= 1.25 between t-5 and t; no decline > 0.2 in t+5 and t+10; 10-year cooldown before re-activation.
- Models estimated:
  - LP in levels and long-difference.
  - Lag-augmented LP as robustness (two lags of outcome).
  - Clustered SEs by country; Driscoll-Kraay SEs as robustness.
- Main results (aggregate):
  - Positive, significant effects of EFW shocks and reform dummy on growth; stronger in long-difference; horizons up to h=10 years.
- Main results (components):
  - Areas with strongest effects: legal system/property rights, regulation, and trade; size of government weaker.
  - Reform dummies by area show persistence for legal system and regulation; sound money effects shorter horizon.
- Additional analysis:
  - Quinquenal regressions for 1970-2000 to recover historical reforms (Washington Consensus era), with positive effects at 0–15 years.
- Novelty claim in thesis:
  - Uses LP with observed shocks (EFW differences and reform dummy) and decomposes effects by area.

## 2) Do NOT repeat (already done in thesis)
- Component-level EFW analysis as a main contribution (areas and their differential effects).
- Reform dummy construction based on Grier and Grier (2021) as a headline identification strategy.
- LP comparisons in levels vs long-difference (including lag-augmented LP with two lags).
- Quinquenal regression for 1970-2000 to recover historical reforms as a centerpiece result.
- Claims that legal system/regulation/trade are the dominant drivers (already a central thesis claim).

## 3) Safe to reuse as benchmark appendix (non-primary)
- Replication of aggregate LP results using EFW differences and reform dummy.
- Comparison of levels vs long-difference LP estimates as a robustness appendix.
- Quinquenal 1970-2000 regression as a historical check (appendix only).

## 4) What is missing / Q1 referee attack points
- Treatment timing bias in staggered reforms: no stacked DiD or clean control definition.
- Joint inference across horizons (no sup-t bands) and few-treated inference (no wild bootstrap/RI).
- Pretrend diagnostics for reform episodes (beyond narrative justification).
- Measurement error in GDP (autocracy misreporting) not addressed beyond standard data sources.
- Potential endogeneity of EFW differences (policy reforms may be contemporaneous with growth shocks).
- Limited falsification/placebo tests and leave-one-event-out influence checks.
- No SCM/SynthDiD validation for the largest reforms or case studies.
