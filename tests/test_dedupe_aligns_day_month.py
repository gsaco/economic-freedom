import pandas as pd

from elections.dedupe import dedupe_events


def test_dedupe_aligns_day_and_month() -> None:
    df = pd.DataFrame({
        "record_id": ["ned:1", "clea:1"],
        "source": ["ned_parl", "clea_lc"],
        "iso3": ["UTO", "UTO"],
        "office_type": ["parliamentary", "parliamentary"],
        "election_date": [pd.Timestamp("2001-05-10"), pd.Timestamp("2001-05-01")],
        "election_year": [2001, 2001],
        "election_month": [5, 5],
        "election_day": [10, 1],
        "date_precision": ["day", "month"],
        "party_1_id": [1, 1],
        "party_2_id": [2, 2],
    })

    sources, canonical = dedupe_events(df, ["ned_parl", "clea_lc"], margin_diff_max=5)
    assert sources.loc[sources["record_id"] == "ned:1", "election_id"].iloc[0] == sources.loc[sources["record_id"] == "clea:1", "election_id"].iloc[0]
    assert len(canonical) == 1
