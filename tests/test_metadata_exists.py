from pathlib import Path

import pytest


def test_metadata_exists():
    panel_path = Path("data/03_clean/panel_quinquennial_atlas.parquet")
    if not panel_path.exists():
        pytest.skip("Atlas panel not built")
    required = [
        Path("data/02_intermediate/efw_metadata.json"),
        Path("data/02_intermediate/wdi_metadata.json"),
        Path("data/03_clean/panel_quinquennial_atlas_metadata.json"),
    ]
    missing = [path for path in required if not path.exists()]
    assert not missing, f"Missing metadata files: {missing}"
