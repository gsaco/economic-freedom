from pathlib import Path

import pandas as pd
import pytest

from src.paths import QUINQUENNIAL_YEARS
from src.qc import assert_year_grid


def test_year_grid():
    path = Path("data/03_clean/panel_quinquennial_atlas.parquet")
    if not path.exists():
        pytest.skip("Atlas panel not built")
    df = pd.read_parquet(path)
    assert_year_grid(df, "year", QUINQUENNIAL_YEARS)
