# Data Overview

## Sources
- Economic Freedom of the World (Fraser Institute), quinquennial extracts.
- World Development Indicators (World Bank API), indicator-by-indicator cached pulls.
- Natural Earth geodata for global maps.

## Output Artifacts
- `data/02_intermediate/efw_quinquennial.parquet`
- `data/02_intermediate/wdi_annual_selected.parquet`
- `data/02_intermediate/wdi_quinquennial_selected.parquet`
- `data/03_clean/panel_quinquennial_atlas.parquet`
- `data/03_clean/panel_quinquennial_atlas_metadata.json`

## Coverage
Coverage tables and plots are saved under:
- `output/tables/efw_coverage_by_year.csv`
- `output/tables/wdi_coverage_matrix.csv`
- `output/figures/coverage/`

Notes
- Coverage is sparse in early decades for some macro indicators; missingness is reported explicitly.
- No imputation is applied to EFW scores by default.
