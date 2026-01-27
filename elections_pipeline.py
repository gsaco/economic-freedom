from __future__ import annotations

import argparse
from pathlib import Path
import sys

from elections_core import (
    PipelinePaths,
    build_elections_clea,
    build_elections_ned,
    build_master_outputs,
    compare_coverage,
    convert_clea_to_parquet,
    extract_raw,
    setup_logger,
)
from elections_external import (
    build_combo_metrics,
    build_external_master,
    build_final,
    download_external,
    fetch_electionguide,
    fetch_wdi,
)


REQUIRED_EXTERNAL_FILES = [
    "partyfacts_external_parties.csv",
    "ches_1999_2024.csv",
    "elff_partypos_summaries.csv",
    "parlgov.zip",
    "des_es_data_v50.zip",
    "idea_export_electoral_system_design_database.xlsx",
    "dpi/DPI2020/dpi2020.csv",
]


def external_inputs_available(paths: PipelinePaths) -> bool:
    missing = []
    for rel in REQUIRED_EXTERNAL_FILES:
        if not (paths.external / rel).exists():
            missing.append(rel)
    return len(missing) == 0


def build_core(paths: PipelinePaths, logger, strict: bool) -> None:
    build_elections_ned(paths, logger, strict=strict)
    build_elections_clea(paths, logger, strict=strict)
    compare_coverage(paths, logger)
    build_master_outputs(paths, logger, strict=strict)


def build_external(paths: PipelinePaths, logger, strict: bool) -> None:
    df = build_external_master(paths, logger, strict=strict)
    metrics = build_combo_metrics(df)
    metrics.to_csv(paths.reports / "external_combo_metrics.csv", index=False)


def build_all(
    paths: PipelinePaths,
    logger,
    run_extract: bool = False,
    run_convert: bool = False,
    run_external: bool = True,
    download_ext: bool = False,
    strict: bool = False,
) -> None:
    paths.ensure_dirs()

    if run_extract:
        extract_raw(paths, logger)

    if run_convert:
        convert_clea_to_parquet(paths, logger)

    build_core(paths, logger, strict=strict)

    if download_ext:
        download_external(paths, logger)

    if run_external:
        if external_inputs_available(paths):
            build_external(paths, logger, strict=strict)
        else:
            logger.warning("External datasets missing; skipping external merge.")

    build_final(paths, logger)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build elections dataset pipeline.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Repo root path")
    parser.add_argument("--extract-raw", action="store_true", help="Extract raw inputs from zip files")
    parser.add_argument("--convert-clea", action="store_true", help="Convert CLEA .sav files to parquet")
    parser.add_argument("--skip-external", action="store_true", help="Skip external dataset merge")
    parser.add_argument("--download-external", action="store_true", help="Download external datasets")
    parser.add_argument("--fetch-wdi", action="store_true", help="Fetch WDI indicators")
    parser.add_argument("--fetch-electionguide", action="store_true", help="Fetch ElectionGuide API data")
    parser.add_argument("--no-build", action="store_true", help="Only run fetch tasks and exit")
    parser.add_argument("--strict", action="store_true", help="Fail on validation warnings")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logger = setup_logger()
    paths = PipelinePaths(args.root)

    if args.fetch_wdi:
        fetch_wdi(paths, logger)
    if args.fetch_electionguide:
        fetch_electionguide(paths, logger)
    if args.no_build:
        return 0

    build_all(
        paths,
        logger,
        run_extract=args.extract_raw,
        run_convert=args.convert_clea,
        run_external=not args.skip_external,
        download_ext=args.download_external,
        strict=args.strict,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
