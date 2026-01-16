# External Commitment Devices → EFW → Macro (Quinquennial Global Panel)

## Project Goal
Build a Q1-grade empirical pipeline for a global quinquennial panel (1970-2020) to estimate how external commitment devices (EU/WTO) affect institutional quality (EFW) and macro outcomes, with reform vs reversal asymmetries, state dependence, nonlinearities, and episode robustness.

## Quickstart
```bash
python -m pip install -r requirements.txt
make all
```

## Reproduce Results
```bash
make data
make build
make estimate
make docs
make notebooks
```

## Folder Map
- `data/raw/`: cached raw inputs (EFW, EU, WTO, ACDB, WDI/PWT)
- `data/processed/`: quinquennial panels and merged datasets
- `outputs/figures/`: saved figures
- `outputs/tables/`: saved tables
- `docs/results/`: markdown reports with embedded figures
- `notebooks/`: jupytext-paired notebooks
- `src/`: ingestion, build, estimators, analysis, viz
- `tools/`: pipeline orchestration
- `tests/`: unit + integration tests
- `legacy/`: archived old pipeline content

## Data: Auto vs Manual
Auto-download attempts:
- EFW master dataset (Fraser Institute)
- WDI indicators via World Bank API
- PWT (Penn World Table)
- WTO accession list via public web table (fallback to manual CSV)

Manual placement if download is blocked:
- `data/raw/efw/` (EFW CSV/XLSX)
- `data/raw/macro/` (PWT CSV/XLSX)
- `data/raw/wto/wto_accessions.csv` (WTO accessions)
- `data/raw/wto_acdb/` (ACDB commitments export)

See `docs/results/01_data_overview.md` for coverage and treatment lists.
