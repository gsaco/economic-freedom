from pathlib import Path

import pytest


def test_figures_exported():
    fig_dir = Path("output/figures")
    if not fig_dir.exists():
        pytest.skip("Figures not built")
    pngs = list(fig_dir.rglob("*.png"))
    if not pngs:
        pytest.skip("No figures built")
    missing = [path for path in pngs if not path.with_suffix(".pdf").exists()]
    assert not missing, f"Missing PDF exports for: {missing[:5]}"
