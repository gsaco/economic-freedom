from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elections.audit import run_audit_path, write_output_manifest
from elections.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write output manifest for processed datasets")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.root)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or run_audit_path(cfg.paths.audit, run_id, "outputs_manifest.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_output_manifest(cfg.paths.processed, out_path, root_dir=args.root, run_id=run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
