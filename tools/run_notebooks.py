from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import papermill as pm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import LOGS_DIR

BANNED_IMPORTS = (
    "statsmodels",
    "linearmodels",
    "econml",
    "causalml",
    "pystata",
    "synthetic_control",
    "cvxpy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_ref() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _assert_no_inference_imports(py_path: Path) -> None:
    text = py_path.read_text(encoding="utf-8")
    for banned in BANNED_IMPORTS:
        if f"import {banned}" in text or f"from {banned}" in text:
            raise RuntimeError(f"Banned import '{banned}' found in {py_path}")


def sync_notebooks(py_paths: list[Path]) -> None:
    for py_path in py_paths:
        subprocess.run(["jupytext", "--sync", str(py_path)], check=True)


def execute_notebooks(nb_paths: list[Path]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = LOGS_DIR / "run_ledger.jsonl"
    git_ref = _git_ref()
    for nb_path in nb_paths:
        start = time.time()
        status = "success"
        error = ""
        try:
            pm.execute_notebook(
                str(nb_path),
                str(nb_path),
                log_output=True,
                kernel_name="python3",
            )
        except Exception as exc:  # pragma: no cover - bubbled up after logging
            status = "failed"
            error = str(exc)
            raise
        finally:
            end = time.time()
            entry = {
                "notebook": str(nb_path),
                "status": status,
                "started_at_utc": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
                "finished_at_utc": datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
                "duration_sec": round(end - start, 3),
                "sha256": _sha256(nb_path),
                "python": sys.version.split()[0],
                "git_ref": git_ref,
                "error": error,
            }
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")


def _resolve_notebook_paths(notebooks_dir: Path, stems: list[str]) -> tuple[list[Path], list[Path]]:
    py_paths = []
    nb_paths = []
    for stem in stems:
        py_path = notebooks_dir / f"{stem}.py"
        nb_path = notebooks_dir / f"{stem}.ipynb"
        if not py_path.exists():
            raise FileNotFoundError(f"Notebook script missing: {py_path}")
        py_paths.append(py_path)
        nb_paths.append(nb_path)
    return py_paths, nb_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebooks",
        default="",
        help="Comma-separated list of notebook stems to run in order.",
    )
    args = parser.parse_args()

    notebooks_dir = Path("notebooks")
    if args.notebooks:
        stems = [stem.strip() for stem in args.notebooks.split(",") if stem.strip()]
    else:
        stems = [path.stem for path in sorted(notebooks_dir.glob("*.py"))]

    py_paths, nb_paths = _resolve_notebook_paths(notebooks_dir, stems)
    for py_path in py_paths:
        _assert_no_inference_imports(py_path)

    sync_notebooks(py_paths)
    execute_notebooks(nb_paths)


if __name__ == "__main__":
    main()
