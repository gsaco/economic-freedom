import subprocess
from pathlib import Path


def test_run_all_ci():
    result = subprocess.run([
        "python",
        "tools/run_all.py",
        "--stage",
        "ci",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    assert Path("data/03_clean/panel_quinquennial_atlas.parquet").exists()
    assert Path("output/logs/run_ledger.jsonl").exists()
    assert any(Path("output/figures").rglob("*.png"))
