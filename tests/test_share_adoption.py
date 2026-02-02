import pandas as pd

from elections.dedupe import dedupe_events, attach_primary_secondary_shares


def test_share_adoption_from_clea() -> None:
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
        "party_1_name_raw": ["Party A", "Party A"],
        "party_2_name_raw": ["Party B", "Party B"],
        "share_1": [55.0, 60.0],
        "share_2": [45.0, 40.0],
        "share_metric": ["seat_share", "vote_share"],
        "margin": [10.0, 20.0],
    })

    sources, canonical = dedupe_events(df, ["ned_parl", "clea_lc"], margin_diff_max=5)
    out = attach_primary_secondary_shares(canonical, sources)
    row = out.iloc[0]
    assert row["share_metric_primary"] == "vote_share"
    assert row["share_1"] == 60.0
    assert row["share_metric_secondary"] == "seat_share"
    assert row["share_primary_source"] == "clea_lc"
    assert row["share_alignment_status"] == "party_id_aligned"
