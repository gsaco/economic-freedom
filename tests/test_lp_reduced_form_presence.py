from pathlib import Path

import pytest


def test_lp_reduced_form_presence():
    path = Path("output/paper_tables/lp_reduced_form.csv")
    if not path.exists():
        pytest.skip("LP reduced-form output not built")
    assert path.stat().st_size > 0
