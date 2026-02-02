import pandas as pd

from elections.party_match import build_party_match_results


def test_party_variants_stopword_match() -> None:
    elections = pd.DataFrame({
        "iso3": ["ARG"],
        "election_year": [2007],
        "party_1_name_raw": ["Alianza Frente para la Victoria"],
        "party_2_name_raw": [pd.NA],
    })
    partyfacts = pd.DataFrame({
        "partyfacts_id": [1],
        "country": ["ARG"],
        "name": ["Frente para la Victoria"],
        "name_english": [pd.NA],
        "name_short": [pd.NA],
        "name_other": [pd.NA],
        "year_first": [2003],
        "year_last": [2015],
        "technical": [False],
    })

    res = build_party_match_results(elections, partyfacts, min_score=90, tie_delta=0.5)
    assert res.loc[0, "partyfacts_id"] == 1
    assert res.loc[0, "match_status"] in {"matched", "matched_tiebreak"}
