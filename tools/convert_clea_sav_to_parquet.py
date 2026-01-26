from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing CLEA manifest: {path}")
    return json.loads(path.read_text())


def _collect_usecols(manifest: dict) -> list[str]:
    election_cols = set(manifest["election_columns"].values())
    result_cols = set(manifest["result_columns"].values())
    return sorted(election_cols | result_cols)


def _convert_sav(path: Path, out_path: Path, *, usecols: list[str], chunk_size: int) -> None:
    if out_path.exists():
        print(f"Output already exists; skipping: {out_path}")
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    _, meta = pyreadstat.read_sav(path, metadataonly=True)
    total_rows = meta.number_rows or 0

    writer: pq.ParquetWriter | None = None
    for offset in range(0, total_rows, chunk_size):
        df, _ = pyreadstat.read_sav(
            path,
            usecols=usecols,
            row_offset=offset,
            row_limit=chunk_size,
            apply_value_formats=False,
            formats_as_category=False,
            formats_as_ordered_category=False,
            disable_datetime_conversion=True,
        )
        if df.empty:
            break
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"Rows {offset:,}-{min(offset + chunk_size, total_rows):,} / {total_rows:,}")

    if writer is None:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError("No rows read from CLEA .sav file.")

    writer.close()
    tmp_path.replace(out_path)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CLEA .sav to a parquet subset for faster ingest.")
    parser.add_argument(
        "--manifest",
        default="data/01_raw/clea/clea_manifest.json",
        help="Path to CLEA manifest JSON.",
    )
    parser.add_argument(
        "--out",
        default="data/01_raw/clea/clea_lc_20251015_subset.parquet",
        help="Output parquet path.",
    )
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Rows per chunk.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)

    raw_dir = manifest_path.resolve().parent
    election_file = raw_dir / manifest["election_file"]
    result_file = raw_dir / manifest["result_file"]

    if not election_file.exists():
        raise FileNotFoundError(f"Missing CLEA source file: {election_file}")
    if election_file != result_file:
        raise ValueError("Expected CLEA elections/results in a single source file.")

    usecols = _collect_usecols(manifest)
    _convert_sav(election_file, Path(args.out), usecols=usecols, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
