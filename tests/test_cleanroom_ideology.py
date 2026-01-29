import pandas as pd

from elections.ideology import assign_ideology_to_elections


def test_assign_ideology_prefers_common_source():
    elections = pd.DataFrame({
        "party_1_id": [1],
        "party_2_id": [2],
        "election_year": [2000],
        "share_1": [55.0],
        "share_2": [45.0],
    })
    obs = pd.DataFrame({
        "partyfacts_id": [1, 2, 1, 2],
        "source": ["vparty", "vparty", "ches", "ches"],
        "year": [2000, 2000, 2000, 2000],
        "ideo_raw": [0.2, 0.8, 1.0, 5.0],
        "ideo_std": [0.2, 0.8, 0.1, 0.9],
        "iso3": ["AAA", "AAA", "AAA", "AAA"],
        "ideo_scale_note": ["v", "v", "c", "c"],
    })
    out = assign_ideology_to_elections(elections, obs, source_priority=["vparty", "ches"], year_windows={"vparty": 4, "ches": 4})
    assert out.loc[0, "ideo_source"] == "vparty"
    assert out.loc[0, "margin_market"] == -10.0
    assert out.loc[0, "D_market_win"] == 0
