import pandas as pd

from elections.party_match import build_party_match_results


def test_party_tie_break_year_midpoint() -> None:
    elections = pd.DataFrame({
        "iso3": ["UTO"],
        "election_year": [2005],
        "party_1_name_raw": ["Example Party"],
        "party_2_name_raw": [pd.NA],
    })
    partyfacts = pd.DataFrame({
        "partyfacts_id": [1, 2],
        "country": ["UTO", "UTO"],
        "name": ["Example Party", "Example Party"],
        "name_english": [pd.NA, pd.NA],
        "name_short": [pd.NA, pd.NA],
        "name_other": [pd.NA, pd.NA],
        "year_first": [1950, 2000],
        "year_last": [1960, 2010],
        "technical": [False, False],
    })

    res = build_party_match_results(elections, partyfacts, min_score=90, tie_delta=0.5)
    assert res.loc[0, "partyfacts_id"] == 2
    assert res.loc[0, "match_status"] in {"matched_tiebreak", "matched"}
