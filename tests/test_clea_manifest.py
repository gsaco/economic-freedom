from pathlib import Path

import json
import pytest


def test_clea_manifest_schema():
    manifest_path = Path("data/01_raw/clea/clea_manifest.json")
    if not manifest_path.exists():
        pytest.skip("CLEA manifest not present")
    manifest = json.loads(manifest_path.read_text())
    for key in ["election_file", "result_file", "election_columns", "result_columns"]:
        assert key in manifest, f"Missing {key} in CLEA manifest"
