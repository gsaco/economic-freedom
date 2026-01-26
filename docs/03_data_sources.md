# Data Sources (provider, access, constraints, fallbacks)

This file documents the datasets referenced in `plan.tex`, how the pipeline accesses them, and the ranked fallbacks. It **does not** describe data cleaning beyond minimal parsing required for access.

## Economic Freedom of the World (EFW)
- **Provider / link:** Fraser Institute, Economic Freedom of the World dataset (Annual Report dataset download).
- **Coverage:** Up to ~165 jurisdictions; 1970–present (quinquennial early years, annual later).
- **Unit:** jurisdiction‑year.
- **Access method:** Manual download into `data/01_raw/efw/` (CSV/XLSX). The pipeline reads cached raw files if present.
- **License/constraints:** Academic use with citation to Fraser Institute annual report.
- **Limitations:** Composite index; component definitions evolve; overlap with inflation in Sound Money.
- **Ranked fallbacks:**
  1. Use area‑specific indices and exclude overlapping components.
  2. Use EFW ranks/percentiles where level comparability is questioned.
  3. Treat 5 areas as separate latent dimensions rather than a single scalar.

## Elections and margins
### 1) CLEA (global legislative)
- **Provider / link:** Constituency‑Level Elections Archive (electiondataarchive.org).
- **Coverage:** Global constituency‑level returns; legislative elections.
- **Unit:** constituency‑level returns (aggregable to national vote/seat totals).
- **Access method:** Manual download to `data/raw/elections/clea/` (not currently scripted).
- **License/constraints:** Heterogeneous source licensing; attribution required.
- **Limitations:** Legislative focus; coalition mapping required; varying electoral rules.

### 2) ParlGov (EU/OECD parliamentary)
- **Provider / link:** ParlGov data releases / GitHub/Dataverse.
- **Coverage:** EU/OECD democracies 1900–2023.
- **Unit:** country‑election, party vote/seat shares, cabinets.
- **Access method:** Automated download in `notebooks/10_ingest_parlgov.py` (cached to `data/01_raw/parlgov/`).
- **License/constraints:** Open academic data; verify per release.
- **Limitations:** EU/OECD only; coalition mapping required.

### 3) DPI (political institutions)
- **Provider / link:** World Bank DPI (DPI2020 catalog).
- **Coverage:** 1975–2020, ~180 countries.
- **Unit:** country‑year.
- **Access method:** Manual download to `data/01_raw/dpi/` (not currently scripted).
- **Limitations:** Limited margin information; ideology measures coarse.

**Ranked fallback options for elections/margins**
1. Restrict to clean margin systems (top‑2 presidential or single‑party parliamentary).
2. Use ParlGov for EU/OECD as Tier‑1; treat global extensions as appendix.
3. Reconcile multiple sources by agreement rule; exclude discordant cases.

## Market‑orientation scoring (more‑market vs less‑market)
1. **V‑Dem V‑Party (preferred):** party economic positions; requires download and ID matching.
2. **DPI ideology categories:** coarse left/center/right labels; broader coverage.
3. **Manifesto Project (MARPOR):** manifestoR/API key required.

**Access note:** V‑Party and MARPOR require credentials. The pipeline expects local files if present and logs missing inputs.

## Macro outcomes
### World Development Indicators (WDI)
- **Provider / link:** World Bank WDI API.
- **Coverage:** broad annual macro indicators.
- **Unit:** country‑year.
- **Access method:** API pull with caching (`notebooks/02_pull_worldbank.py` → `data/01_raw/wdi_cache/`).
- **License:** CC BY 4.0 per World Bank terms.
- **Core indicators:** GDP per capita, GDP growth, CPI inflation, investment share, trade openness, population.

### Penn World Table (PWT)
- **Provider / link:** Groningen Growth and Development Centre.
- **Coverage:** PWT 10.0 (1950–2019), PWT 11.0 (through 2023).
- **Unit:** country‑year.
- **Access method:** Manual download to `data/raw/pwt/`.
- **License:** CC BY 4.0 (PWT 10.0).

## Crisis and tail events
- **Laeven–Valencia Systemic Banking Crises Database** (1970–2011 update).
- **Access method:** Manual download to `data/raw/crisis/`.
- **Fallback:** construct tail proxies from macro series (inflation spikes, drawdowns).

## Pre‑treatment state variables
- **V‑Dem country‑year dataset (v15, March 2025):** democracy and institutional constraints.
- **Governance/State capacity:** WGI or tax‑to‑GDP proxies where available.
- **Openness and vulnerability:** trade openness, external debt, inflation history (WDI).

## Local data paths (expected)
- Raw: `data/01_raw/` and `data/raw/`
- Intermediate: `data/02_intermediate/`
- Cleaned/analysis: `data/03_clean/`, `data/04_analysis/`

## Ranked fallbacks summary
- **Orientation:** V‑Party → DPI → MARPOR.
- **Elections/margins:** CLEA → ParlGov (Tier‑1 precision) → DPI‑assisted extensions.
- **Macro:** WDI primary, PWT for robustness.
- **Crises:** Laeven–Valencia where available → macro‑constructed tail proxies.
