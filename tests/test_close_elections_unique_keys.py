from pathlib import Path

import pandas as pd
import pytest

from src.qc import assert_unique_key


def test_close_elections_unique_keys():
    path = Path("data/04_analysis/close_elections_vote_margin.parquet")
    if not path.exists():
        pytest.skip("Close-election sample not built")
    df = pd.read_parquet(path)
    assert_unique_key(df, ["iso3c", "election_year"])
