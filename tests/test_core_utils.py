import unittest

import numpy as np
import pandas as pd

from elections_core import (
    MERGE_STRATEGIES,
    apply_best_merge,
    normalize_name,
    reorder_top_two,
    standardize_shares,
)


class TestCoreUtils(unittest.TestCase):
    def test_normalize_name(self) -> None:
        self.assertEqual(normalize_name("Côte d'Ivoire"), "cote d ivoire")
        self.assertEqual(normalize_name("  National & Democratic Party  "), "national and democratic party")

    def test_standardize_shares(self) -> None:
        df = pd.DataFrame({"share_1": [0.6, 0.4], "share_2": [0.4, 0.2]})
        out = standardize_shares(df, ["share_1", "share_2"])
        self.assertAlmostEqual(out.loc[0, "share_1"], 60.0)
        self.assertAlmostEqual(out.loc[0, "share_2"], 40.0)

    def test_reorder_top_two(self) -> None:
        df = pd.DataFrame({
            "share_1": [40.0],
            "share_2": [60.0],
            "party_1_name": ["Alpha"],
            "party_2_name": ["Beta"],
        })
        out = reorder_top_two(df)
        self.assertEqual(out.loc[0, "share_1"], 60.0)
        self.assertEqual(out.loc[0, "party_1_name"], "Beta")

    def test_apply_best_merge_prefers_ned(self) -> None:
        df = pd.DataFrame({
            "iso3": ["UTO", "UTO"],
            "office_type": ["parliamentary", "parliamentary"],
            "date": pd.to_datetime(["2000-01-01", "2000-01-01"]),
            "year": pd.Series([2000, 2000], dtype="Int64"),
            "month": pd.Series([1, 1], dtype="Int64"),
            "share_1": [60.0, 55.0],
            "share_2": [40.0, 45.0],
            "party_1_name": ["Alpha", "Alpha"],
            "party_2_name": ["Beta", "Beta"],
            "ideo_party1": [2.0, 2.0],
            "ideo_party2": [4.0, 4.0],
            "country_cow": [100, 100],
            "margin_market": [-20.0, -10.0],
            "source": ["NED", "CLEA"],
        })
        full, best, metrics, strategy = apply_best_merge(df, {"UTO"})
        self.assertIn(strategy, MERGE_STRATEGIES)
        self.assertEqual(len(best), 1)
        self.assertEqual(best.iloc[0]["source"], "NED")
        self.assertTrue(full["merge_preferred"].any())
        self.assertTrue(set(metrics["strategy"]).issuperset(MERGE_STRATEGIES))


if __name__ == "__main__":
    unittest.main()
