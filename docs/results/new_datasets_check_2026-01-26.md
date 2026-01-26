# New dataset checks (Jan 26, 2026)

Scope: Evaluate newly added datasets as potential replacements for ParlGov in the close-election pipeline. Per request, CLEA ingest is deferred.

## 1) NED v2 (presidential_elections_v2.dta, parliamentary_elections_v2.dta)

### Raw coverage
- Presidential: 1,409 elections, 123 countries, 1789–2023; 39 party columns with vote_share1_* fields.
- Parliamentary: 4,900 elections, 211 countries, 1789–2023; 74 party columns with seat_share_* fields.

### Pipeline test (current code)
- Command: `python tools/build_ned_samples.py`
- Outputs:
  - `data/04_analysis/ned_presidential_sample_cov80.parquet` (2 rows)
  - `data/04_analysis/ned_parliamentary_sample_cov80.parquet` (20 rows)
  - `data/04_analysis/ned_presidential_sample_cov50.parquet` (2 rows)
  - `data/04_analysis/ned_parliamentary_sample_cov50.parquet` (39 rows)

### Coverage results (ParlGov party-map, current script)
- Presidential: 2 elections, 1 country (2009–2019), coverage ~0.88.
- Parliamentary: 20 elections (cov>=0.8), 39 elections (cov>=0.5), 4–8 countries.

### Quick V-Party name-match check (no code changes)
Using V-Party party names with simple normalization (exact string match by iso3-year-name):
- Presidential: 16 elections (cov>=0.8), 28 elections (cov>=0.5), 11–16 countries.
- Parliamentary: 60 elections (cov>=0.8), 79 elections (cov>=0.5), 18–26 countries.
- Mean coverage remains ~0.02, so most elections have little matched ideology.

### Interpretation
- NED provides vote/seat shares and could supply the running variable, but ideology mapping is the bottleneck.
- With the current ParlgGov-based party map, NED samples are extremely small and **cannot** replace ParlGov in the main design.
- Even with direct V-Party matching, coverage is still too low for a global replacement; would require improved party-name reconciliation or a curated mapping.

## 2) ETAD v1.0.0 (ETAD_v_1_0_0.csv)

- Rows: 3,133 elections; 148 countries; 1945–2023.
- Content: election timing (held vs scheduled dates), election type, term length.
- No vote shares, seat shares, or party results.

### Interpretation
- Useful for timing diagnostics (early/late elections), but **cannot** construct margins or market-winner coding.
- Not a replacement for ParlGov in the close-election RD pipeline.

## 3) NELDA 6.0 (nelda_id_q_wide.tab)

- Rows: 2,974 elections; 1945–2011.
- Content: extensive election attributes/flags, no vote or seat margins.

### Interpretation
- Useful for sample restrictions (e.g., competitive elections, irregularities), but **cannot** build the running variable.
- Not a replacement for ParlGov in the close-election RD pipeline.

## 4) CLEA (clea_lc_20251015.sav)

### Fast ingest approach (to avoid slow .sav reads)
- New converter: `tools/convert_clea_sav_to_parquet.py` writes a column-subset parquet in chunks.
- Sidecar output: `data/01_raw/clea/clea_lc_20251015_subset.parquet`.
- `src/clea.py` now prefers a `.parquet` sidecar when the source is `.sav` and reads only needed columns.
- Elections table now de-duplicates by `election_id` before ISO3 mapping to avoid a 1.3M-row country lookup.

### CLEA ingest outputs
- `data/02_intermediate/clea_elections.parquet`: 2,304 elections, 181 countries.
- `data/02_intermediate/clea_results.parquet`: 189,520 party-election rows (constituency totals aggregated).

### Close-election sample (CLEA branch)
- `data/04_analysis/close_elections_vote_margin.parquet` (CLEA):
  - Rows: 782, Countries: 173, Years: 2000–2021.
  - `running_var_vote` missing rate: **65.7%** (V-Party ideology mapping bottleneck).
  - `running_var_top2` missing rate: **1.0%** (vote-share coverage is strong).
  - Countries with any non-missing `running_var_vote`: 92; with ≥50% non-missing: 66.

### Baseline comparison (ParlGov branch)
- Rows: 295, Countries: 36, Years: 2000–2021.
- Missing `running_var_vote`: 0%.

### Interpretation
- CLEA **greatly expands coverage** (173 vs 36 countries) and provides reliable top-2 vote margins.
- The **binding constraint** is ideology mapping for party names; the V-Party exact match leaves ~2/3 of elections without a market-vs-nonmarket running variable.
- As-is, CLEA is usable for:
  - broader coverage on **closeness/margin diagnostics**, and
  - a **partial market-orientation sample** (92 countries with some coverage),
  - but **not** yet a full replacement for ParlGov in the main RD design.

## Summary
- **NED**: promising for raw vote/seat shares, but ideology mapping coverage is too low for replacement with current methods.
- **ETAD**: timing only; no margins.
- **NELDA**: election attributes only; no margins.
- **CLEA**: now ingested and fast to load; expands coverage substantially but needs better party-ideology matching to serve as the main RD running variable source.

## Next steps (if you want)
1) Improve CLEA party-to-ideology mapping (fuzzy matching + manual mapping table) and re-run coverage.
2) Add NELDA/ETAD as filters/metadata (not as margin sources).
3) Consider using CLEA top-2 margins for a “closeness only” appendix, while keeping ParlGov for the main market-orientation RD.
