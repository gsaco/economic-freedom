#!/usr/bin/env python
"""Run full pipeline and build output report."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def sync_dir(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
            count += 1
    return count


def git_commit_hash(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "output"
    outputs_dir = root / "outputs"
    logs_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run([sys.executable, str(root / "tools" / "run_pipeline.py")], cwd=root)

    figures_count = sync_dir(outputs_dir / "figures", output_dir / "figures")
    tables_count = sync_dir(outputs_dir / "tables", output_dir / "tables")

    log = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit_hash(root),
        "figures": figures_count,
        "tables": tables_count,
        "source_outputs": str(outputs_dir),
    }
    (logs_dir / "run_all.json").write_text(json.dumps(log, indent=2, sort_keys=True))

    run([sys.executable, str(root / "tools" / "build_report.py")], cwd=root)


if __name__ == "__main__":
    main()
