# Repository Contract

## Purpose
This repository executes `plan.tex` end‑to‑end, producing the close‑election RD spine, RD‑IV extensions, and publication‑ready tables/figures.

## Supported environments
- macOS and Linux
- Python 3.10+ (tested via `requirements.txt`)

## Dependencies
- Python packages in `requirements.txt` (includes `rdrobust` for CCT RD inference and `pycountry` for ISO mapping)
- System libraries for geospatial stacks (GDAL/GEOS/PROJ) if running map notebooks

## Primary commands
```bash
make setup
make lint
make test
make run
make repro
make clean
```

### Command semantics
- `make setup`: install Python dependencies.
- `make lint`: static checks (ruff on `src` and `tests`).
- `make test`: unit/integration tests (pytest).
- `make run`: execute the full plan pipeline (paper notebooks + outputs).
- `make repro`: clean generated artifacts and rebuild outputs from scratch.
- `make clean`: remove generated outputs/intermediate artifacts only.

## Expected outputs (baseline)
- `output/paper_tables/`: CSV tables for RD, IV, asymmetry, nonlinearity, robustness
- `output/paper_figures/`: PDF/PNG figures for RD plots, IRFs, sensitivity
- `output/paper_logs/`: metadata and run logs
- `data/02_intermediate/`, `data/03_clean/`, `data/04_analysis/`: intermediate pipeline products

## Elections data contract (CLEA‑first)
- Primary elections source is **CLEA**. Current inputs: `data/01_raw/clea/clea_lc_20251015.sav` + `data/01_raw/clea/clea_manifest.json`.
- V‑Party is used for party ideology: `data/01_raw/vparty/CPD_V-Party_CSV_v2/V-Dem-CPD-Party-V2.csv`.
- DPI (currently `data/01_raw/dpi/dpi2012.xls`) is used as an incumbency fallback when party‑level positions are unavailable.
- If CLEA is missing, the pipeline **falls back to ParlGov** and labels outputs as Tier‑1 (high‑precision) only. This does **not** satisfy the global 165‑jurisdiction target.

## Determinism and seeds
- Most steps are deterministic given fixed input files.
- Where resampling/permutation is used, explicit seeds are set in code and logged in `output/paper_logs/`.

## Reproducibility notes
- Some data sources require credentials or manual download (V‑Dem/V‑Party, Manifesto, CLEA). The pipeline will use local files when present and log missing inputs.
- The primary pipeline expects CLEA to reach the plan’s cross‑country coverage target.
