# Assumptions and Conservative Resolutions

This file records ambiguities in `plan.tex` and the conservative resolutions used to execute the plan without asking for clarifications.

## Data access and licensing
- **Credentialed sources (V-Dem/V-Party, Manifesto Project, some CLEA variants)**: these require registration or keys. I do **not** attempt automated downloads without credentials. The pipeline uses local files when present and logs missing inputs.
- **CLEA access**: CLEA is provided locally as `data/01_raw/clea/clea_lc_20251015.sav` with a manifest mapping column names.

## Sample construction and scope
- **Primary elections source** is **CLEA** (global legislative coverage) to align with EFW’s 165 jurisdictions.
- **EFW alignment**: elections are restricted to countries covered by EFW in the annual panel.

## Orientation (“more‑market vs less‑market”)
- **Primary ideology mapping** uses **V‑Party** party positions (`v2pariglef` / `v2pariglef_mean`) matched by country‑year and normalized party name.
- **Fallback ideology mapping** uses **DPI executive ideology** (DPI2012 in this environment) to classify incumbency where party‑level positions are unavailable or unmatched.
- **Threshold** for market vs non‑market is set at 0 on the V‑Party left‑right scale (positive = more‑market; negative = less‑market). Ties at 0 are treated as missing.

## DPI vintage
- The available DPI file in this repo is `dpi2012.xls`. The plan references DPI2020; I proceed with DPI2012 and document the vintage in data‑source docs.

## RD and LP fallback
- **RD estimation** uses robust bias‑corrected local polynomial RD (CCT) when possible.
- **LP fallback** provides reduced‑form dynamics if RD‑IV is weak or infeasible.

## Outcomes and overlapping components
- For inflation outcomes, “EFW excluding Sound Money” is constructed as the average of the remaining four EFW areas.
- Crisis start uses Laeven–Valencia (1970–2011) when present locally; post‑2011 uses pre‑specified drawdown/inflation‑spike proxies.

## Documentation policy
- All assumptions above are carried into `docs/02_repo_contract.md` and `docs/04_methods_and_estimands.md` with explicit flags so reviewers can see baseline vs fallback paths.
