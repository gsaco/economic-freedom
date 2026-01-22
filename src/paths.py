from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "01_raw"
INTERMEDIATE_DIR = DATA_DIR / "02_intermediate"
CLEAN_DIR = DATA_DIR / "03_clean"
ANALYSIS_DIR = DATA_DIR / "04_analysis"

OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
LOGS_DIR = OUTPUT_DIR / "logs"
PAPER_FIGURES_DIR = OUTPUT_DIR / "paper_figures"
PAPER_TABLES_DIR = OUTPUT_DIR / "paper_tables"
PAPER_LOGS_DIR = OUTPUT_DIR / "paper_logs"

DOCS_RESULTS_DIR = BASE_DIR / "docs" / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

QUINQUENNIAL_YEARS = list(range(1970, 2021, 5))


def ensure_directories() -> None:
    paths = [
        RAW_DIR / "efw",
        RAW_DIR / "wdi_cache",
        RAW_DIR / "world_bank",
        RAW_DIR / "geodata",
        INTERMEDIATE_DIR,
        CLEAN_DIR,
        ANALYSIS_DIR,
        OUTPUT_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        LOGS_DIR,
        PAPER_FIGURES_DIR,
        PAPER_TABLES_DIR,
        PAPER_LOGS_DIR,
        DOCS_RESULTS_DIR,
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
