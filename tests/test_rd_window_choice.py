import json
from pathlib import Path

import pytest


def test_rd_window_choice_logged():
    path = Path("output/paper_logs/rd_window_choice.json")
    if not path.exists():
        pytest.skip("RD window choice not built")
    data = json.loads(path.read_text())
    assert "positive" in data and "negative" in data
    assert "window" in data["positive"] and "window" in data["negative"]
