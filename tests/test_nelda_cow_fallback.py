import pandas as pd

from elections.build_sources import merge_nelda


def test_nelda_fallback_to_cow_code_when_modal_unmatched() -> None:
    sources = pd.DataFrame({
        "record_id": ["rec1"],
        "iso3": ["UTO"],
        "nelda_ccode_from_nelda": [999],
        "cow_code": [100],
        "election_year": [2000],
        "election_month": [1],
        "election_date": [pd.Timestamp("2000-01-15")],
        "date_precision": ["day"],
        "office_type": ["presidential"],
    })

    nelda = pd.DataFrame({
        "ccode": [100],
        "year": [2000],
        "mmdd": [115],
        "types": ["Executive"],
        "electionid": [54321],
        "nelda3": ["yes"],
        "nelda4": ["yes"],
        "nelda5": ["yes"],
        "country": ["Utopia"],
    })

    out, report = merge_nelda(sources, nelda)
    assert report.loc[0, "nelda_match_status"] in {"matched", "matched_outside_tolerance"}
    assert out.loc[0, "nelda_electionid"] == 54321
    assert out.loc[0, "nelda_ccode_used_source"] == "cow_fallback"
