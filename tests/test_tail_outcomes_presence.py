from pathlib import Path

import pandas as pd
import pytest


def test_tail_outcomes_presence():
    path = Path("output/paper_tables/rd_tail_outcomes.csv")
    if not path.exists():
        pytest.skip("Tail outcome outputs not built")
    df = pd.read_csv(path)
    assert not df.empty
