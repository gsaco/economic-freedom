from pathlib import Path

import pandas as pd
import pytest


def test_shock_presence():
    paths = [
        Path("data/04_analysis/rd_event_panel_pos.parquet"),
        Path("data/04_analysis/rd_event_panel_neg.parquet"),
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("Event panels not built")
    for path in paths:
        df = pd.read_parquet(path)
        assert "shock_efw" in df.columns
        assert df["shock_efw"].notna().mean() > 0
