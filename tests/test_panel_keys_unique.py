from pathlib import Path

import pandas as pd
import pytest

from src.qc import assert_unique_key


def test_panel_keys_unique():
    path = Path("data/03_clean/panel_quinquennial_atlas.parquet")
    if not path.exists():
        pytest.skip("Atlas panel not built")
    df = pd.read_parquet(path)
    assert_unique_key(df, ["iso3c", "year"])
