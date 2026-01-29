import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from elections_core import PipelinePaths, setup_logger
from elections_pipeline import build_all
from tests.fixtures_pipeline import write_fixtures


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.paths = PipelinePaths(self.root)
        self.paths.ensure_dirs()
        write_fixtures(self.root)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_end_to_end_pipeline(self) -> None:
        logger = setup_logger()
        logger.setLevel(logging.ERROR)

        build_all(
            self.paths,
            logger,
            run_convert=True,
            run_external=True,
            download_ext=False,
            strict=True,
        )

        ned = pd.read_parquet(self.paths.processed / "elections_ned.parquet")
        clea = pd.read_parquet(self.paths.processed / "elections_clea.parquet")
        master_full = pd.read_parquet(self.paths.processed / "elections_master_full.parquet")
        master = pd.read_parquet(self.paths.processed / "elections_master.parquet")
        external = pd.read_parquet(self.paths.processed / "elections_master_external.parquet")
        final = pd.read_parquet(self.paths.processed / "elections_final.parquet")

        self.assertEqual(len(ned), 2)
        self.assertEqual(len(clea), 2)
        self.assertEqual(len(master_full), 4)
        self.assertEqual(len(master), 3)
        self.assertEqual(len(external), 3)
        self.assertEqual(len(final), 3)

        self.assertEqual(master["merge_key"].nunique(), len(master))
        self.assertEqual(int(master_full["merge_preferred"].sum()), len(master))

        share_mask = master["share_1"].notna() & master["share_2"].notna()
        self.assertTrue((master.loc[share_mask, "share_1"] >= master.loc[share_mask, "share_2"]).all())
        self.assertTrue(master["iso3"].notna().all())
        self.assertTrue(master["year"].notna().all())

        self.assertIn("analysis_ready", final.columns)
        self.assertTrue(final["analysis_ready"].any())
        france = final[final["iso3"] == "FRA"]
        self.assertTrue((france["analysis_ready"] == False).all())

        self.assertTrue((external[external["iso3"] == "FRE"]["margin_market_ext"].notna()).all())


if __name__ == "__main__":
    unittest.main()
