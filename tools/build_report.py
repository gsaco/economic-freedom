#!/usr/bin/env python
"""Build a lightweight pipeline report in output/report.md."""

from __future__ import annotations

import json
from pathlib import Path
import time


def list_files(path: Path, pattern: str) -> list[str]:
    if not path.exists():
        return []
    return sorted([p.name for p in path.glob(pattern) if p.is_file()])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "output"
    outputs_dir = root / "outputs"
    docs_dir = root / "docs"

    output_dir.mkdir(parents=True, exist_ok=True)

    figures = list_files(output_dir / "figures", "*")
    tables = list_files(output_dir / "tables", "*")
    logs = list_files(output_dir / "logs", "*.json")

    docs = []
    if docs_dir.exists():
        docs = sorted([p.name for p in docs_dir.glob("*.md") if p.is_file()])

    lines = [
        "# Pipeline report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Outputs (mirrored to output/)",
        f"- figures: {len(figures)}",
        f"- tables: {len(tables)}",
        f"- logs: {len(logs)}",
        "",
        "## Source outputs",    
        f"- outputs/ exists: {outputs_dir.exists()}",
        "",
    ]

    if figures:
        lines.append("### Figures")
        lines.extend([f"- {name}" for name in figures])
        lines.append("")

    if tables:
        lines.append("### Tables")
        lines.extend([f"- {name}" for name in tables])
        lines.append("")

    if logs:
        lines.append("### Logs")
        lines.extend([f"- {name}" for name in logs])
        lines.append("")

    if docs:
        lines.append("## Docs")
        lines.extend([f"- docs/{name}" for name in docs])
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "figures": len(figures),
        "tables": len(tables),
        "logs": len(logs),
        "docs": len(docs),
    }
    meta_path = output_dir / "report.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
