from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import LOGS_DIR

BANNED_PATTERNS = [
    "statsmodels",
    "linearmodels",
    "econml",
    "causalml",
    "pystata",
    "synthetic_control",
    "cvxpy",
    "PanelOLS",
    "event_study",
    "eventstudy",
    "synthdid",
    "cs_did",
    "sdid",
]

EXCLUDE_DIRS = {
    "archive",
    "output",
    "outputs",
    "data",
    ".venv",
    "__pycache__",
    ".git",
}


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return not parts.isdisjoint(EXCLUDE_DIRS)


def scan_repo(root: Path) -> list[dict]:
    matches = []
    for path in root.rglob("*"):
        if path.is_dir() or _should_skip(path):
            continue
        if path.suffix not in {".py", ".ipynb", ".R"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            for pattern in BANNED_PATTERNS:
                if pattern in stripped:
                    matches.append(
                        {
                            "file": str(path),
                            "line": line_no,
                            "pattern": pattern,
                            "text": stripped,
                        }
                    )
    return matches


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = LOGS_DIR / "regression_code_manifest.json"

    matches = scan_repo(root)
    manifest_path.write_text(json.dumps({"matches": matches}, indent=2))

    if matches:
        raise SystemExit("Inference patterns detected. See regression_code_manifest.json")


if __name__ == "__main__":
    main()
