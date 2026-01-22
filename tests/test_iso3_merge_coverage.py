from pathlib import Path

import pandas as pd
import pytest


def test_iso3_merge_coverage():
    path = Path("data/04_analysis/close_elections_sample.parquet")
    if not path.exists():
        pytest.skip("Close-election sample not built")
    df = pd.read_parquet(path)
    missing_share = df["iso3c"].isna().mean()
    assert missing_share < 0.01
    non_missing = df["iso3c"].dropna()
    assert (non_missing.str.len() == 3).all()
