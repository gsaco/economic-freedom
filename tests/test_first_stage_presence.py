from pathlib import Path

import pandas as pd
import pytest


def test_first_stage_presence():
    paths = [
        Path("output/paper_tables/irf_first_stage_pos.csv"),
        Path("output/paper_tables/irf_first_stage_neg.csv"),
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("LP-IV first-stage outputs not built")
    for path in paths:
        df = pd.read_csv(path)
        assert not df.empty
