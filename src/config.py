from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"
DOCS_RESULTS_DIR = BASE_DIR / "docs" / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

QUINQUENNIAL_YEARS = list(range(1970, 2021, 5))


def ensure_directories() -> None:
    for path in [
        RAW_DIR / "efw",
        RAW_DIR / "eu",
        RAW_DIR / "wto",
        RAW_DIR / "wto_acdb",
        RAW_DIR / "macro",
        RAW_DIR / "crosswalk",
        PROCESSED_DIR,
        INTERIM_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        DIAGNOSTICS_DIR,
        DOCS_RESULTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
