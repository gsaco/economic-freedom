import hashlib
import shutil
import tempfile
from pathlib import Path

from elections_core import PipelinePaths, setup_logger
from elections_pipeline import build_all
from tests.fixtures_pipeline import write_fixtures


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_outputs(paths: PipelinePaths) -> dict[str, str]:
    outputs = [
        paths.processed / "elections_ned.parquet",
        paths.processed / "elections_clea.parquet",
        paths.processed / "elections_master.parquet",
        paths.processed / "elections_master_external.parquet",
        paths.processed / "elections_final.parquet",
    ]
    return {p.name: sha256(p) for p in outputs if p.exists()}


def test_build_determinism() -> None:
    tmpdir = tempfile.TemporaryDirectory()
    root = Path(tmpdir.name)
    paths = PipelinePaths(root)
    paths.ensure_dirs()
    write_fixtures(root)

    logger = setup_logger()
    logger.setLevel("ERROR")

    build_all(
        paths,
        logger,
        run_convert=True,
        run_external=True,
        download_ext=False,
        strict=True,
    )
    first_hashes = hash_outputs(paths)

    shutil.rmtree(paths.processed, ignore_errors=True)
    shutil.rmtree(paths.reports, ignore_errors=True)
    paths.ensure_dirs()

    build_all(
        paths,
        logger,
        run_convert=True,
        run_external=True,
        download_ext=False,
        strict=True,
    )
    second_hashes = hash_outputs(paths)

    assert first_hashes == second_hashes
    tmpdir.cleanup()

