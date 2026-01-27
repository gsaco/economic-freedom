# Elections Dataset Build

## Entry point
Primary CLI:
```
python elections_pipeline.py --convert-clea
```

Optional raw extraction (if you have the original zip files in the repo root):
```
python elections_pipeline.py --extract-raw --convert-clea
```

Include external datasets (uses existing files under `data/external`):
```
python elections_pipeline.py --convert-clea
```

Download external datasets before the build:
```
python elections_pipeline.py --download-external --convert-clea
```

Fetch API data only (no build):
```
python elections_pipeline.py --fetch-wdi --no-build
python elections_pipeline.py --fetch-electionguide --no-build
```

Fail fast on validation warnings:
```
python elections_pipeline.py --convert-clea --strict
```

## Outputs
### Processed datasets
- `data/processed/elections_ned.parquet`
- `data/processed/elections_clea.parquet`
- `data/processed/elections_master.parquet`
- `data/processed/elections_master_full.parquet`
- `data/processed/elections_master_external.parquet` (only if external inputs are present)
- `data/processed/elections_final.parquet`

### Reports and diagnostics
- `data/interim/coverage_ned.csv`
- `data/interim/coverage_clea.csv`
- `reports/coverage_summary.csv`
- `reports/missing_efw_countries.csv`
- `reports/merge_strategy_metrics.csv`
- `reports/merge_strategy_choice.txt`
- `reports/election_combinations_metrics.csv`
- `reports/missing_efw_countries_clean.csv`
- `reports/external_country_map.csv`
- `reports/external_combo_metrics.csv`
- `reports/elections_final_summary.csv`

## Merge + validation rules (core)
- **Country harmonization**: ISO3 mapping is built from EFW country names with manual overrides and fuzzy matching (min score 70). ISO3 values are kept as uppercase strings.
- **CLEA aggregation**: constituency data are summed to election totals; vote shares are primary, seat shares are fallback for top-two ranking.
- **Share normalization**: if max share ≤ 1.5, shares are multiplied by 100; values outside [0, 100] are set to missing.
- **Top-two ordering**: rows are reordered to enforce `share_1 >= share_2`, swapping party/candidate fields consistently.
- **Party matching**: NED/CLEA use normalized names within country; the master rebuild uses stopword-stripped names with fuzzy matching.
- **Ideology assignment**: V-Party ideology uses closest available year (≤ 4 years); ideology quality labels use match score + year distance.
- **Merge strategies**: four keys are evaluated (`date_exact`, `date_fallback`, `year_month`, `year`), scored for coverage/completeness, and the top strategy is selected. Preferred rows are flagged via `merge_preferred` (no row drops).
- **Quality flags**: `clean_flag` requires non-missing `iso3/year/share_1/share_2` and `share_1 >= share_2`; `sample_competitive` applies NELDA + NED irregularity flags.

## External merge rules
- **Country maps**: ISO3→country names are matched via fuzzy matching to CHES/ELFF/ParlGov/DES/DPI.
- **Ideology priority**: `vparty → parlgov → ches → elff` (stored as `ideo_final*` + `ideo_source*`).
- **Electoral systems**: DES matched to nearest election date; IDEA matched by `iso3-year`.
- **Institutional controls**: DPI matched by normalized country name + year.

## Validation + diagnostics
- All joins log key coverage, duplicate counts, and match rates.
- Right-side joins are deduplicated by key using the most complete row (logged warnings when triggered).
- Use `--strict` to turn validation warnings into failures.

