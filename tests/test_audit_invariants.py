import pandas as pd

from elections_audit import invariant_checks


def test_invariants_pass_on_clean_data() -> None:
    df = pd.DataFrame({
        "election_id": ["UTO-pres-2000-01-1"],
        "iso3": ["UTO"],
        "office_type": ["presidential"],
        "year": [2000],
        "month": [1],
        "share_1": [60.0],
        "share_2": [40.0],
        "nelda_known": [True],
        "competitive_nelda": [True],
        "country": ["Utopia"],
    })
    report = invariant_checks(df, iso3_valid={"UTO"})
    issues = report.issues
    assert issues["dup_election_id"] == 0
    assert issues["dup_rows_iso_office_year_month_seq"] == 0
    assert issues["share_out_of_bounds_rows"] == 0
    assert issues["winner_lt_runnerup_rows"] == 0
    assert issues["iso3_invalid_pattern_rows"] == 0
    assert issues["iso3_invalid_membership_rows"] == 0
    assert issues["nelda_tristate_violations"] == 0
    assert issues["impossible_iso_mappings"] == 0


def test_invariants_catch_errors() -> None:
    df = pd.DataFrame({
        "election_id": ["UTO-pres-2000-01-1", "UTO-pres-2000-01-1"],
        "iso3": ["UT", "KOR"],
        "office_type": ["presidential", "presidential"],
        "year": [2000, 2000],
        "month": [1, 1],
        "share_1": [120.0, 40.0],
        "share_2": [10.0, 60.0],
        "nelda_known": [True, False],
        "competitive_nelda": [pd.NA, True],
        "country": ["North Korea", "North Korea"],
    })
    report = invariant_checks(df, iso3_valid={"UTO", "KOR", "PRK"})
    issues = report.issues
    assert issues["dup_election_id"] == 1
    assert issues["share_out_of_bounds_rows"] == 1
    assert issues["winner_lt_runnerup_rows"] == 1
    assert issues["iso3_invalid_pattern_rows"] >= 1
    assert issues["nelda_tristate_violations"] == 2
    assert issues["impossible_iso_mappings"] >= 1

