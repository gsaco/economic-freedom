from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spec_search.runner import append_results, run_grid, write_spec_matrix
from src.spec_search.specs import Spec, spec_from_dict


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_specs(config_path: Path) -> list[Spec]:
    payload = yaml.safe_load(config_path.read_text())
    defaults = payload.get("defaults", {}) if isinstance(payload, dict) else {}
    specs_payload = payload.get("specs", []) if isinstance(payload, dict) else []
    specs: list[Spec] = []
    for spec_dict in specs_payload:
        merged = _deep_merge(defaults, spec_dict)
        specs.append(spec_from_dict(merged))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/spec_search.yaml",
        help="Path to spec grid YAML config.",
    )
    parser.add_argument(
        "--output",
        default="output/spec_search",
        help="Output directory for spec search results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of specs to run.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Optional path to an election sample parquet to override default.",
    )
    parser.add_argument(
        "--panel",
        default=None,
        help="Optional path to annual panel parquet to override default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing spec outputs.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    specs = load_specs(config_path)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    write_spec_matrix(specs, output_root=output_root)
    panel_path = Path(args.panel) if args.panel else None
    sample_path = Path(args.sample) if args.sample else None

    summaries = run_grid(
        specs,
        output_root=output_root,
        allow_overwrite=args.overwrite,
        limit=args.limit,
        panel_path=panel_path,
        sample_path=sample_path,
    )

    append_results(summaries, output_root=output_root)


if __name__ == "__main__":
    main()
