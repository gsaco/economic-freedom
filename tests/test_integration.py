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

    assert Path("data/processed/panel_quinquennial.parquet").exists()
    assert any(Path("outputs/figures").glob("fig_*.png"))
