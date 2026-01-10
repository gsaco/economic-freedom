# Macro Panel Pipeline - README

## Overview

This pipeline downloads, harmonizes, and merges macroeconomic data from multiple sources with the Fraser Institute's Economic Freedom of the World (EFW) index for comprehensive analysis.

## Folder Structure

```
economic-freedom/
├── src/                          # Shared pipeline modules
│   ├── __init__.py
│   ├── downloaders.py            # Multi-source data downloaders (WDI, PWT, KAOPEN, KOF, WGI)
│   ├── harmonize.py              # ISO3 concordance, long-to-wide transformations
│   ├── merge_efw.py              # EFW loading and merge utilities
│   ├── quality.py                # Validation, missingness, outlier detection
│   └── plots.py                  # Standardized plotting helpers
│
├── annual_eda.py                 # Annual EDA notebook (2000-present)
├── annual_eda.ipynb              # (Auto-generated from .py via Jupytext)
├── quinquennial_eda.py           # Quinquennial EDA notebook (1970-2020)
├── quinquennial_eda.ipynb        # (Auto-generated from .py via Jupytext)
│
├── data/
│   ├── raw/                      # Cached downloads (parquet)
│   │   ├── wdi_data.parquet
│   │   └── download_metadata.json
│   ├── intermediate/             # Harmonized long panels, concordances
│   │   ├── iso3_concordance.csv
│   │   └── wdi_annual_long.parquet
│   └── clean/                    # Final merged analysis panels
│       ├── annual_merged.parquet/.csv
│       └── macro_annual_wide.parquet
│
├── outputs/
│   ├── figures/                  # Generated plots (300 DPI)
│   └── tables/                   # Generated CSV tables
│
└── efw_utils.py                  # Original EFW helper utilities
```

## Installation

```bash
# Minimal dependencies
pip install pandas numpy matplotlib seaborn scipy openpyxl requests jupytext scikit-learn
```

## Usage

### Run Pipeline End-to-End

```bash
# 1. Run annual EDA (2000-present)
python annual_eda.py

# 2. Run quinquennial EDA (1970-2020 waves)
python quinquennial_eda.py

# 3. Sync notebooks after editing .py files
jupytext --sync annual_eda.py quinquennial_eda.py
```

### Force Re-download Data

```python
from src import downloaders
df = downloaders.download_wdi(force_download=True)
```

### Clear Cache

```python
from src import downloaders
downloaders.clear_cache()  # All sources
downloaders.clear_cache('wdi')  # Just WDI
```

## Data Sources

| Source | Variables | Coverage | Status |
|--------|-----------|----------|--------|
| World Bank WDI | GDP, inflation, trade, investment, govt consumption | 1960-2024 | ✓ Implemented |
| Penn World Table | Real GDP, productivity, TFP | 1950-2019 | ✓ Implemented (optional) |
| Chinn-Ito KAOPEN | Capital account openness | 1970-2021 | ✓ Implemented (optional) |
| KOF Globalisation | Globalisation indices | 1970-2021 | ✓ Implemented (optional) |
| WGI | Rule of law, regulatory quality | 1996-2023 | ✓ Implemented (via WDI API) |

## Key Outputs

### Annual EDA (2000-present)
- Sample: 3,865 country-year observations, 164 countries
- Key correlation: EFW vs Log GDP/capita = +0.68
- Figures: trends, correlations, scatter plots, missingness dashboards

### Quinquennial EDA (1970-2020)
- Sample: 1,477 observations across 11 waves
- Balanced panel: 98 countries observed in all waves
- Key findings:
  - σ-convergence confirmed (declining cross-sectional SD)
  - β-convergence confirmed (β = -0.53, initial level predicts change)
  - High rank persistence (Spearman > 0.85 across all wave pairs)

## Validation Checks (Built-in)

- ✓ Unique (iso3, year) keys in all panels
- ✓ Score range validation [0, 10] for EFW
- ✓ Missingness tables by variable, decade, region
- ✓ Merge overlap accounting (rows gained/lost)
- ✓ Outlier flagging (high inflation, growth collapse)

---

*Generated: 2026-01-08*
