# Assumptions and Conservative Resolutions

This file records ambiguities in `plan.tex` and the conservative resolutions used to execute the plan without asking for clarifications.

## Data access and licensing
- **Credentialed sources (V-Dem/V-Party, Manifesto Project, some CLEA variants)**: these require registration or keys. I do **not** attempt automated downloads without credentials. The pipeline is wired to use local files if present and logs missing inputs; documentation lists how to obtain and where to place files.
- **Primary run uses available open data** in the repo (`data/01_raw`, `data/raw`) and open URLs (e.g., ParlGov, WDI, PWT) when downloads are permitted.

## Sample construction and scope
- **Baseline sample** uses the **ParlGov parliamentary elections** workflow because it is the only fully scripted and open-access election source currently integrated. This implements the plan’s Tier‑1 “high-precision” sample.
- **Expanded global samples (CLEA/DPI/V‑Party)** are treated as **optional extensions** and are documented with fallback instructions; they are not required to run the baseline pipeline unless data are already present.
- **Presidential elections** are included only where a scripted variant exists (e.g., NED-based robustness) and are reported as robustness, not the baseline.

## Orientation (“more‑market vs less‑market”)
- **Primary orientation** uses ParlGov party left‑right positions to classify market vs non‑market blocs in parliamentary systems, with the market threshold pre‑specified in code.
- **Ranked fallback order** (V‑Party → DPI → MARPOR) is preserved in documentation, but only executed if the corresponding datasets are present locally.

## Term alignment and overlapping elections
- **Term alignment** uses actual election timing (next election year) when available; otherwise a **fixed 4‑year term** is assumed for horizon mapping.
- **Overlapping elections** are treated as ITT for fixed‑calendar horizons and as “term‑aligned” when the next election year is observed.

## RD estimation and inference
- The plan requires **robust bias‑corrected local polynomial RD (CCT)**. Where a dedicated RD package is available (e.g., `rdrobust`), it is used; otherwise a conservative **local‑linear WLS with triangular kernel** is used and explicitly labeled as a fallback in methods documentation.
- **Local randomization** uses balance‑based window choice; randomization inference is implemented via permutation tests when feasible. If sample sizes are too small for stable permutation inference, results are reported as descriptive robustness only.

## IV and sensitivity/bounds
- **Weak‑IV robust inference** uses Anderson–Rubin style tests if supported by the toolchain; otherwise, delta‑method and bootstrap CIs are reported with explicit caveats.
- **Exclusion sensitivity** follows the plan’s direct‑effect‑bound framing using a user‑specified grid of $
\delta(h)
$ values; identified sets are computed analytically where possible.

## Outcomes and overlapping components
- For **inflation outcomes**, “EFW excluding Sound Money” is constructed as the average of the remaining four EFW areas, consistent with the plan.
- **Crisis start** uses Laeven–Valencia (1970–2011) when present locally; post‑2011 is proxied by pre‑specified drawdown/inflation‑spike rules.

## Multiplicity control
- Multiple‑testing control uses **Holm adjustment** within the pre‑specified outcome families and horizons in `plan.tex`.

## Documentation policy
- All assumptions above are carried into `docs/02_repo_contract.md` and `docs/04_methods_and_estimands.md` with explicit flags so that reviewers can see which elements are baseline vs optional.
