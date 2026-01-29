import pandas as pd

from elections.standardize import standardize_share, reorder_top_two


def test_standardize_share_proportion():
    s = pd.Series([0.2, 0.55, 1.0])
    out = standardize_share(s)
    assert out.max() > 1.5
    assert round(out.iloc[0], 2) == 20.0


def test_reorder_top_two():
    df = pd.DataFrame({
        "share_1": [40.0, 30.0],
        "share_2": [60.0, 20.0],
        "party_1": ["A", "C"],
        "party_2": ["B", "D"],
    })
    out = reorder_top_two(df, ("share_1", "share_2"), [("share_1", "share_2"), ("party_1", "party_2")])
    assert out.loc[0, "share_1"] == 60.0
    assert out.loc[0, "party_1"] == "B"
    assert out.loc[1, "share_1"] == 30.0
    assert out.loc[1, "party_1"] == "C"
