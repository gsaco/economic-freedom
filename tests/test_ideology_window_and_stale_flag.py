import pandas as pd

from elections.ideology import assign_ideology_to_elections


def test_ideology_window_and_stale_flag() -> None:
    elections = pd.DataFrame({
        "party_1_id": [1],
        "party_2_id": [pd.NA],
        "election_year": [2020],
        "share_1": [60.0],
        "share_2": [40.0],
    })
    obs = pd.DataFrame({
        "partyfacts_id": [1],
        "iso3": ["UTO"],
        "source": ["vparty"],
        "year": [2010],
        "ideo_raw": [0.5],
        "ideo_std": [0.5],
        "ideo_scale_note": ["vparty_v2pariglef"],
    })

    out = assign_ideology_to_elections(
        elections,
        obs,
        source_priority=["vparty"],
        year_windows={"vparty": 10},
        stale_after_years=5,
    )
    assert out.loc[0, "ideo_std_1"] == 0.5
    assert out.loc[0, "ideology_1_year_gap"] == 10
    assert bool(out.loc[0, "ideology_1_stale_flag"]) is True
