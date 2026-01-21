# Quinquennial EFW Atlas (Descriptive)

## Project Goal
Build a descriptive quinquennial atlas (1970-2020) of Economic Freedom of the World (EFW) and macro indicators.
The pipeline reports coverage, distributions, ranks, maps, and co-movements without causal inference.
See `docs/atlas_scope.md` for scope rules.

## Quickstart
```bash
python -m pip install -r requirements.txt
make all
```

## Reproduce Results
```bash
make data
make build
make notebooks
make docs
```

## Folder Map
- `data/01_raw/`: cached raw inputs (EFW, WDI cache, geodata)
- `data/02_intermediate/`: cleaned intermediate extracts + metadata
- `data/03_clean/`: final atlas panel + metadata
- `output/figures/`: figures (PNG 500 dpi + PDF)
- `output/tables/`: tables
- `output/logs/`: run ledger and environment snapshots
- `docs/results/`: descriptive atlas writeups
- `notebooks/`: jupytext-paired atlas notebooks
- `src/`: atlas utilities (paths, qc, io, maps, viz style)
- `tools/`: pipeline orchestration
- `archive/`: deprecated inference pipeline

## Data: Auto vs Manual
Auto-download attempts:
- EFW master dataset (Fraser Institute)
- WDI indicators via World Bank API
- Natural Earth geodata for maps

Manual placement if download is blocked:
- `data/01_raw/efw/` (EFW CSV/XLSX)

See `docs/results/01_data_overview.md` for coverage and dataset notes.
