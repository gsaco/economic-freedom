from pathlib import Path

import pandas as pd
import pytest


def test_running_variable_continuity_basic():
    path = Path("data/04_analysis/close_elections_vote_margin.parquet")
    if not path.exists():
        pytest.skip("Close-election sample not built")
    df = pd.read_parquet(path)
    running = df["running_var_vote"].dropna()
    assert running.std() > 0
    share_zero = (running.abs() < 1e-6).mean()
    assert share_zero < 0.1
    assert running.nunique() > 20
