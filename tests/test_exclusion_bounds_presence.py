from pathlib import Path

import pandas as pd
import pytest


def test_exclusion_bounds_presence():
    path = Path("output/paper_tables/rd_exclusion_bounds.csv")
    if not path.exists():
        pytest.skip("Exclusion bounds outputs not built")
    df = pd.read_csv(path)
    assert not df.empty
