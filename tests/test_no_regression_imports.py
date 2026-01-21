import subprocess


def test_no_regression_imports():
    result = subprocess.run(
        ["python", "tools/qc_scan_no_inference.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
